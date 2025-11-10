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
