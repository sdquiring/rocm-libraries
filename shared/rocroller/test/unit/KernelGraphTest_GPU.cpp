/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <common/CommonGraphs.hpp>

#include "GPUContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;

namespace KernelGraphTest
{

    class KernelGraphTestGPU : public CurrentGPUContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::SaveAssembly, true);
            CurrentGPUContextFixture::SetUp();

            fastArith = Expression::FastArithmetic(m_context);
        }

        void GPU_SAXPBY(bool reload);
    };

    void KernelGraphTestGPU::GPU_SAXPBY(bool reload)
    {
        RandomGenerator random(1263u);

        size_t nx = 64;

        auto a = random.vector<int>(nx, -100, 100);
        auto b = random.vector<int>(nx, -100, 100);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device(b);
        auto d_c     = make_shared_device<int>(nx);
        auto d_alpha = make_shared_device<int>();

        int alpha = 22;
        int beta  = 33;

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto example = rocRollerTest::Graphs::VectorAdd<int>(true);
        auto command = example.getCommand();
        auto runtimeArgs
            = example.getRuntimeArguments(nx, d_alpha.get(), beta, d_a.get(), d_b.get(), d_c.get());

        CommandKernel commandKernel(command, testKernelName());
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        // launch again, using saved assembly
        auto assemblyFileName = m_context->assemblyFileName();

        if(reload)
        {
            commandKernel.loadKernelFromAssembly(assemblyFileName, testKernelName());
            commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        }

        std::vector<int> r(nx);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        double rnorm = relativeNormL2(r, example.referenceSolution(alpha, beta, a, b));

        ASSERT_LT(rnorm, 1.e-12);

        if(reload)
        {
            // load, using bad kernel name
            EXPECT_THROW(commandKernel.loadKernelFromAssembly(assemblyFileName, "SAXPBY_BAD"),
                         FatalError);

            // load, using non-existant file
            EXPECT_THROW(
                commandKernel.loadKernelFromAssembly(assemblyFileName + "_bad", testKernelName()),
                FatalError);

            std::filesystem::remove(assemblyFileName);
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBY)
    {
        GPU_SAXPBY(false);
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBYDebug)
    {
        // Make sure Debug mode doesn't introduce bad pointer
        // references in observers
        auto settings = Settings::getInstance();
        settings->set(Settings::LogLvl, LogLevel::Debug);
        GPU_SAXPBY(false);
        settings->reset();
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBYLoadAssembly)
    {
        GPU_SAXPBY(true);
    }

    TEST_F(KernelGraphTestGPU, GPU_LeakyRelu)
    {
        auto command = std::make_shared<rocRoller::Command>();

        constexpr auto dataType = DataType::Float;

        auto xTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        auto xLoadTag   = command->addOperation(rocRoller::Operations::T_Load_Linear(xTensorTag));

        auto alphaScalarTag = command->addOperation(
            rocRoller::Operations::Scalar({dataType, PointerType::PointerGlobal}));
        auto alphaLoadTag
            = command->addOperation(rocRoller::Operations::T_Load_Scalar(alphaScalarTag));

        auto zeroLiteralTag = command->addOperation(rocRoller::Operations::Literal(0.0f));

        auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
        auto condTag
            = execute.addXOp(rocRoller::Operations::E_GreaterThan(xLoadTag, zeroLiteralTag));
        auto productTag = execute.addXOp(rocRoller::Operations::E_Mul(xLoadTag, alphaLoadTag));
        auto reluTag
            = execute.addXOp(rocRoller::Operations::E_Conditional(condTag, xLoadTag, productTag));
        command->addOperation(std::move(execute));

        auto reluTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        command->addOperation(rocRoller::Operations::T_Store_Linear(reluTag, reluTensorTag));

        CommandKernel commandKernel(command, "LeakyRelu");
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        size_t nx    = 64;
        float  alpha = 0.9;

        RandomGenerator random(135679u);
        auto            a = random.vector<float>(nx, -5, 5);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device<float>(nx);
        auto d_alpha = make_shared_device<float>();

        std::vector<float> r(nx), x(nx);

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(xTensorTag, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(xTensorTag, ArgumentType::Limit, nx);
        commandArgs.setArgument(xTensorTag, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(xTensorTag, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(alphaScalarTag, ArgumentType::Value, d_alpha.get());
        commandArgs.setArgument(reluTensorTag, ArgumentType::Value, d_b.get());
        commandArgs.setArgument(reluTensorTag, ArgumentType::Limit, nx);
        commandArgs.setArgument(reluTensorTag, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(reluTensorTag, ArgumentType::Stride, 0, (size_t)1);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
        {
            x[i] = a[i] > 0 ? a[i] : alpha * a[i];
        }

        double rnorm = relativeNormL2(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_LinearCopy)
    {
        auto command = std::make_shared<rocRoller::Command>();

        Operations::Tensor tensor_A(1, DataType::Int32);
        auto               tagTensorA
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(tensor_A)));

        Operations::Tensor tensor_B(1, DataType::Int32);
        auto               tagTensorB
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(tensor_B)));

        Operations::T_Load_Linear load_A(tagTensorA);
        auto                      tagLoadStore
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));
        Operations::T_Store_Linear store_B(tagLoadStore, tagTensorB);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_B)));

        CommandKernel commandKernel(command, "LinearCopy");
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        size_t nx = 64;

        RandomGenerator random(135679u);
        auto            a = random.vector<int>(nx, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, nx);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_b.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Limit, nx);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i];

        double rnorm = relativeNormL2(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    template <typename T>
    void CopyStrideOverride(CommandKernelPtr& commandKernel, bool override = false)
    {
        auto example = rocRollerTest::Graphs::TileCopy<T>();

        example.setTileSize(16, 4);
        example.setSubTileSize(4, 2);

        if(override)
        {
            example.setLiteralStrides({(size_t)0, (size_t)1});
        }

        size_t nx = 256;
        size_t ny = 128;

        RandomGenerator random(193674u);
        auto            ax = static_cast<T>(-100.);
        auto            ay = static_cast<T>(100.);
        auto            a  = random.vector<T>(nx * ny, ax, ay);

        std::vector<T> r(nx * ny, 0.);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<T>(nx * ny);

        auto command     = example.getCommand();
        auto params      = example.getCommandParameters(nx, ny);
        auto launch      = example.getCommandLaunchParameters(nx, ny);
        auto runtimeArgs = example.getRuntimeArguments(nx, ny, d_a.get(), d_b.get());

        std::string colName    = (override) ? "ColOverride" : "";
        std::string kernelName = "TensorTileCopy" + colName + TypeInfo<T>::Name();

        commandKernel = std::make_shared<CommandKernel>(command, kernelName);
        commandKernel->setContext(Context::ForDefaultHipDevice(kernelName));
        commandKernel->setCommandParameters(params);
        commandKernel->generateKernel();

        commandKernel->setLaunchParameters(launch);
        commandKernel->launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * ny * sizeof(T), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto   ref   = example.referenceSolution(a);
        double rnorm = relativeNormL2(r, ref);

        std::ostringstream msg;
        for(int i = 0; i < r.size(); i++)
        {
            msg << i << " " << r[i] << " | " << ref[i];
            if(r[i] != ref[i])
                msg << " * ";
            msg << std::endl;
        }

        ASSERT_LT(rnorm, 1.e-12) << msg.str();
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopy)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<int>(commandKernel);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideHalf)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<Half>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dword "))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dword "))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideFloat)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<float>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dwordx2"))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dwordx2"))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideDouble)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<double>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dwordx4"))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dwordx4"))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileAdd)
    {
        size_t nx = 256; // tensor size x
        size_t ny = 512; // tensor size y

        RandomGenerator random(129674u);

        auto a = random.vector<int>(nx * ny, -100, 100);
        auto b = random.vector<int>(nx * ny, -100, 100);
        auto r = random.vector<int>(nx * ny, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);
        auto d_c = make_shared_device<int>(nx * ny);

        auto example = rocRollerTest::Graphs::TileDoubleAdd<int>();
        example.setTileSize(8, 64);
        example.setSubTileSize(2, 8);

        auto command     = example.getCommand();
        auto runtimeArgs = example.getRuntimeArguments(nx, ny, d_a.get(), d_b.get(), d_c.get());
        auto params      = example.getCommandParameters(nx, ny);
        auto launch      = example.getCommandLaunchParameters(nx, ny);

        CommandKernel commandKernel(command, "TensorTileAdd");
        commandKernel.setContext(m_context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        commandKernel.setLaunchParameters(launch);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        double rnorm = relativeNormL2(r, example.referenceSolution(a, b));

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_Conditional)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto kernel = kgraph.control.addElement(Kernel());
        auto unit   = Expression::literal(1);
        auto zero   = Expression::literal(0);

        auto test = m_context->kernel()->addArgument({"foo", DataType::Int32});

        auto                    destReg = kgraph.coordinates.addElement(Linear());
        Expression::DataFlowTag destRegTag{destReg, Register::Type::Vector, DataType::Int32};

        auto beforeConditionalAssign
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(0)});
        kgraph.control.addElement(Body(), {kernel}, {beforeConditionalAssign});

        auto conditional
            = kgraph.control.addElement(ConditionalOp{test < unit, "Test Conditional"});

        kgraph.control.addElement(Sequence(), {beforeConditionalAssign}, {conditional});

        auto trueOp    = kgraph.control.addElement(Assign{Register::Type::Vector, unit});
        auto trueBody  = kgraph.control.addElement(Body(), {conditional}, {trueOp});
        auto falseOp   = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        auto falseBody = kgraph.control.addElement(Else(), {conditional}, {falseOp});

        kgraph.mapper.connect(beforeConditionalAssign, destReg, NaryArgument::DEST);
        kgraph.mapper.connect(trueOp, destReg, NaryArgument::DEST);
        kgraph.mapper.connect(falseOp, destReg, NaryArgument::DEST);

        kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        m_context->schedule(m_context->kernel()->preamble());
        m_context->schedule(m_context->kernel()->prolog());

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        if(m_context->targetArchitecture().HasCapability(GPUCapability::WorkgroupIdxViaTTMP))
        {
            EXPECT_THAT(output(), testing::HasSubstr("s_cmp_lt_i32 s2, 1"));
        }
        else
        {
            EXPECT_THAT(output(), testing::HasSubstr("s_cmp_lt_i32 s3, 1"));
        }
        EXPECT_THAT(output(), testing::HasSubstr("s_cbranch_scc0")); //Branch for False
        EXPECT_THAT(output(), testing::HasSubstr("s_branch")); //Branch after True
        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v1, 1")); //True Body
        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v1, 0")); //False Body
    }

    TEST_F(KernelGraphTestGPU, GPU_ConditionalExecute)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        std::vector<int> testValues = {22, 66};

        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto two   = Expression::literal(2u);
        auto three = Expression::literal(3u);

        auto k = m_context->kernel();

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(1);
        k->setWorkitemCount({three, one, one});
        k->setWorkgroupSize({1, 1, 1});
        k->setDynamicSharedMemBytes(zero);
        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        // global result
        auto user = kgraph.coordinates.addElement(User({}, "result"));
        auto wg   = kgraph.coordinates.addElement(Workgroup());
        kgraph.coordinates.addElement(PassThrough(), {wg}, {user});

        // result
        auto dstVGPR = kgraph.coordinates.addElement(VGPR());

        // set result to testValues[0]
        auto assignTrueBranch1 = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[0])});
        kgraph.mapper.connect(assignTrueBranch1, dstVGPR, NaryArgument::DEST);
        auto assignTrueBranch2 = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[0])});
        kgraph.mapper.connect(assignTrueBranch2, dstVGPR, NaryArgument::DEST);

        // set result to testValues[1]
        auto assignFalseBranch = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[1])});
        kgraph.mapper.connect(assignFalseBranch, dstVGPR, NaryArgument::DEST);

        auto workgroupExpr = k->workgroupIndex().at(0)->expression();
        auto firstConditional
            = kgraph.control.addElement(ConditionalOp{workgroupExpr < one, "First Conditional"});
        auto secondConditional = kgraph.control.addElement(
            ConditionalOp{(workgroupExpr > one) && (workgroupExpr <= two), "Second Conditional"});

        auto storeIndex = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeIndex, user);
        kgraph.mapper.connect<VGPR>(storeIndex, dstVGPR);

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {firstConditional});
        kgraph.control.addElement(Body(), {firstConditional}, {assignTrueBranch1});
        kgraph.control.addElement(Else(), {firstConditional}, {secondConditional});
        kgraph.control.addElement(Body(), {secondConditional}, {assignTrueBranch2});
        kgraph.control.addElement(Else(), {secondConditional}, {assignFalseBranch});
        kgraph.control.addElement(Sequence(), {firstConditional}, {storeIndex});

        kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(isLocalDevice())
        {
            auto d_result = make_shared_device<int>(3);

            KernelArguments kargs;
            kargs.append("result", d_result.get());

            KernelInvocation kinv;
            kinv.workitemCount = {3, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<int> result(3);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), result.size() * sizeof(int), hipMemcpyDefault),
                HasHipSuccess(0));
            EXPECT_EQ(result[0], testValues[0]);
            EXPECT_EQ(result[1], testValues[1]);
            EXPECT_EQ(result[2], testValues[0]);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_DoWhileExecute)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto three = Expression::literal(3u);

        auto k = m_context->kernel();

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(1);
        k->setWorkitemCount({three, one, one});
        k->setWorkgroupSize({1, 1, 1});
        k->setDynamicSharedMemBytes(zero);
        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        // global result
        auto user = kgraph.coordinates.addElement(User({}, "result"));
        auto wg   = kgraph.coordinates.addElement(Workgroup());
        kgraph.coordinates.addElement(PassThrough(), {wg}, {user});

        // result
        auto dstVGPR = kgraph.coordinates.addElement(VGPR());

        auto dfa = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{dstVGPR, Register::Type::Vector, DataType::UInt32});
        auto assignVGPR = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        kgraph.mapper.connect(assignVGPR, dstVGPR, NaryArgument::DEST);
        auto assignBody = kgraph.control.addElement(Assign{Register::Type::Vector, dfa + one});
        kgraph.mapper.connect(assignBody, dstVGPR, NaryArgument::DEST);
        auto workgroupExpr = k->workgroupIndex().at(0)->expression();

        auto condVGPR = kgraph.coordinates.addElement(VGPR());
        auto condDFT  = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{condVGPR, Register::Type::Vector, DataType::UInt32});

        auto assignCond = kgraph.control.addElement(Assign{Register::Type::Vector, workgroupExpr});
        kgraph.mapper.connect(assignCond, condVGPR, NaryArgument::DEST);

        auto doWhile = kgraph.control.addElement(DoWhileOp{dfa < condDFT, "Test DoWhile"});

        auto storeIndex = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeIndex, user);
        kgraph.mapper.connect<VGPR>(storeIndex, dstVGPR);

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {assignVGPR});
        kgraph.control.addElement(Body(), {kernel}, {assignCond});
        kgraph.control.addElement(Sequence(), {assignCond}, {doWhile});
        kgraph.control.addElement(Sequence(), {assignVGPR}, {doWhile});
        kgraph.control.addElement(Body(), {doWhile}, {assignBody});
        kgraph.control.addElement(Sequence(), {doWhile}, {storeIndex});

        kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(isLocalDevice())
        {
            auto d_result = make_shared_device<int>(3);

            KernelArguments kargs;
            kargs.append("result", d_result.get());

            KernelInvocation kinv;
            kinv.workitemCount = {3, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<int> result(3);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), result.size() * sizeof(int), hipMemcpyDefault),
                HasHipSuccess(0));
            EXPECT_EQ(result[0], 1);
            EXPECT_EQ(result[1], 1);
            EXPECT_EQ(result[2], 2);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }
}
