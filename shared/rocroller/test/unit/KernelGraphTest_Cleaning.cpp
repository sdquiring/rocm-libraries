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
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <common/CommonGraphs.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

#include "KernelGraphTestFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;

namespace KernelGraphTest
{
    TEST_F(KernelGraphTest, CleanExpression)
    {
        VariableType doubleVal{DataType::Double, PointerType::Value};
        auto         command = std::make_shared<Command>();

        auto aTag = command->allocateTag();
        auto a    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value));
        auto bTag = command->allocateTag();
        auto b    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value));

        m_context->kernel()->addCommandArguments(command->getArguments());

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        auto clean_expr = cleanArguments(expr2, m_context->kernel());

        EXPECT_EQ(
            Expression::toString(clean_expr),
            "Multiply(user_Int32_Value_1:I, Add(user_Int32_Value_0:I, user_Int32_Value_1:I)I)I");
    }

    TEST_F(KernelGraphTest, CleanArguments)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto command = example.getCommand();

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto cleanArgumentsTransform = std::make_shared<CleanArguments>(m_context, command);

        auto kgraph = translate(command);

        auto beforePredicates = {HasSubstr("SubDimension{0, CommandArgument(Tensor_0_size_0)I64}"),
                                 HasSubstr("SubDimension{0, CommandArgument(Tensor_2_size_0)I64}"),
                                 HasSubstr("Linear{CommandArgument(Tensor_0_size_0)I64}"),
                                 HasSubstr("Linear{CommandArgument(Tensor_2_size_0)I64}")};

        // Note that these searches do not include the close braces ("}").  This is because the
        // argument name will have a number appended which is subject to change
        // (Load_Linear_0_size_0 might become Load_Linear_0_size_0_2).
        auto afterPredicates = {
            HasSubstr("SubDimension{0, Tensor_0_size_0"),
            HasSubstr("SubDimension{0, Tensor_2_size_0"),
            HasSubstr("Linear{Tensor_0_size_0"),
            HasSubstr("Linear{Tensor_2_size_0"),
        };

        {
            auto dot = kgraph.toDOT();

            for(auto const& pred : beforePredicates)
                EXPECT_THAT(dot, pred);

            for(auto const& pred : afterPredicates)
                EXPECT_THAT(dot, Not(pred));
        }

        kgraph = kgraph.transform(cleanArgumentsTransform);

        {
            auto dot = kgraph.toDOT();

            for(auto const& pred : beforePredicates)
                EXPECT_THAT(dot, Not(pred));

            for(auto const& pred : afterPredicates)
                EXPECT_THAT(dot, pred);
        }
    }

}
