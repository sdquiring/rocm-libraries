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

#include "../GPUContextFixture.hpp"
#include "../SourceMatcher.hpp"
#include "../Utilities.hpp"

#include "KernelGraphTestGPUFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;

namespace KernelGraphTest
{
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

}
