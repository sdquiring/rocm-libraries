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

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Reindexer.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <common/CommonGraphs.hpp>

#include "../GenericContextFixture.hpp"
#include "../SourceMatcher.hpp"
#include "../Utilities.hpp"

#include "KernelGraphTestFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;

namespace KernelGraphTest
{
    TEST_F(KernelGraphTest, ReindexConditionalOpExpression)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto unit = Expression::literal(1);

        auto kernel = kgraph.control.addElement(Kernel());

        auto loadA = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadA});

        auto user0 = kgraph.coordinates.addElement(User({}, "user0"));
        auto vgprA = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprA});
        kgraph.mapper.connect<VGPR>(loadA, vgprA);

        auto exprA = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{vgprA, Register::Type::Scalar, DataType::Int32});
        auto conditional = kgraph.control.addElement(ConditionalOp{exprA > unit, "conditional"});
        kgraph.control.addElement(Sequence(), {loadA}, {conditional});

        auto loadB = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadB});
        auto vgprB = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprB});
        kgraph.mapper.connect<VGPR>(loadB, vgprB);

        kgraph.control.addElement(Sequence(), {loadB}, {conditional});

        GraphReindexer reindexer;
        reindexer.coordinates.emplace(vgprA, vgprB);
        reindexExpressions(kgraph, conditional, reindexer);

        auto condition = kgraph.control.get<ConditionalOp>(conditional)->condition;
        auto lhs       = std::get<Expression::GreaterThan>(*condition).lhs;
        auto tag       = std::get<Expression::DataFlowTag>(*lhs).tag;
        EXPECT_EQ(tag, vgprB);
    }

    TEST_F(KernelGraphTest, ReindexAssertOpExpression)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto unit = Expression::literal(1);

        auto kernel = kgraph.control.addElement(Kernel());

        auto loadA = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadA});

        auto user0 = kgraph.coordinates.addElement(User(Operations::OperationTag(0), "user0"));
        auto vgprA = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprA});
        kgraph.mapper.connect<VGPR>(loadA, vgprA);

        auto exprA = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{vgprA, Register::Type::Scalar, DataType::Int32});
        auto assertOp = kgraph.control.addElement(AssertOp{"assert", exprA > unit});
        kgraph.control.addElement(Sequence(), {loadA}, {assertOp});

        auto loadB = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadB});
        auto vgprB = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprB});
        kgraph.mapper.connect<VGPR>(loadB, vgprB);

        kgraph.control.addElement(Sequence(), {loadB}, {assertOp});

        GraphReindexer reindexer;
        reindexer.coordinates.emplace(vgprA, vgprB);
        reindexExpressions(kgraph, assertOp, reindexer);

        auto condition = kgraph.control.get<AssertOp>(assertOp)->condition;
        auto lhs       = std::get<Expression::GreaterThan>(*condition).lhs;
        auto tag       = std::get<Expression::DataFlowTag>(*lhs).tag;
        EXPECT_EQ(tag, vgprB);
    }
}
