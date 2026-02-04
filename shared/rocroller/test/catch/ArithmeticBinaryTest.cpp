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

#include "ArithmeticTestKernel.hpp"

#include "CustomMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <common/TestValues.hpp>
#include <common/Utilities.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>

namespace ArithmeticTest
{
    using namespace rocRoller;

    bool validForDivision(std::vector<CommandArgumentValue> operands)
    {
        auto notZero = [](auto const& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr(std::integral<T> || std::floating_point<T>)
            {
                return arg != 0;
            }
            else
            {
                return false;
            }
        };

        auto withinDivisionDomain = [](auto const& arg) {
            using T = std::decay_t<decltype(arg)>;

            if constexpr(std::integral<T> && !std::same_as<T, bool>)
            {
                using SignedT = typename std::make_signed<T>::type;
                return arg <= std::numeric_limits<SignedT>::max();
            }
            else
            {
                return true;
            }
        };

        return std::visit(notZero, operands.at(1))
               && std::visit(withinDivisionDomain, operands.at(0))
               && std::visit(withinDivisionDomain, operands.at(1));
    }

    bool validForShift(std::vector<CommandArgumentValue> operands)
    {
        auto lhsType = resultVariableType(Expression::literal(operands.at(0)));
        auto lhsBits = DataTypeInfo::Get(lhsType.dataType).elementBits;

        auto valid = [lhsBits](auto const& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr(std::integral<T>)
            {
                return arg > 0 && arg < lhsBits;
            }
            else
            {
                return false;
            }
        };

        return std::visit(valid, operands.at(0)) && std::visit(valid, operands.at(1));
    }

    std::vector<std::tuple<Register::Type, Register::Type>>
        inputTypePairs(Register::Type resultRegisterType)
    {
        if(resultRegisterType == Register::Type::Scalar)
        {
            return {{Register::Type::Scalar, Register::Type::Scalar}};
        }
        else
        {
            return {{Register::Type::Scalar, Register::Type::Vector},
                    {Register::Type::Vector, Register::Type::Scalar},
                    {Register::Type::Vector, Register::Type::Vector}};
        }
    }

    /**
     * Adds [reg] op [literal] and [literal] op [reg] for the given operation.
     */
    void addBothOps(
        ArithmeticTestKernel&   kernel,
        DataType                resultDataType,
        CommandArgumentValue    literalValue,
        std::string             name,
        auto                    opFunc,
        ArithmeticOp::Predicate pred = [](std::vector<CommandArgumentValue>) { return true; })
    {
        auto goodLiteralValue = Expression::evaluate(Expression::literal(12, resultDataType));

        auto literalExpression = Expression::literal(literalValue);

        if(pred({literalValue, goodLiteralValue}))
        {
            auto predLHS = [literalValue, pred](std::vector<CommandArgumentValue> operands) {
                operands.insert(operands.begin(), literalValue);
                return pred(operands);
            };

            kernel.addOp(ArithmeticOp{
                name + " LHS Literal",
                [literalExpression, opFunc](
                    std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return opFunc(literalExpression, operands.at(0));
                },
                predLHS});
        }

        if(pred({goodLiteralValue, literalValue}))
        {
            auto predLHS = [literalValue, pred](std::vector<CommandArgumentValue> operands) {
                operands.insert(operands.end(), literalValue);
                return pred(operands);
            };

            kernel.addOp(ArithmeticOp{
                name + " RHS Literal",
                [literalExpression, opFunc](
                    std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return opFunc(operands.at(0), literalExpression);
                }});
        }
    }

    TEST_CASE("Binary Arithmetic Generators work for integral types",
              "[arithmetic][expression][codegen][gpu]")
    {
        auto resultDataType = GENERATE(DataType::Int32, DataType::UInt32);
        auto aType          = GENERATE(DataType::Int32, DataType::UInt32);
        auto bType          = GENERATE(DataType::Int32, DataType::UInt32);

        auto resultRegisterType = GENERATE(Register::Type::Scalar, Register::Type::Vector);

        auto [aRegisterType, bRegisterType]
            = GENERATE_COPY(from_range(inputTypePairs(resultRegisterType)));

        DYNAMIC_SECTION("resultDataType=" << resultDataType //
                                          << ", aType=" << aType //
                                          << ", bType=" << bType //
                                          << ", resultRegisterType=" << resultRegisterType //
                                          << ", aRegisterType=" << aRegisterType //
                                          << ", bRegisterType=" << bRegisterType)
        {
            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}},
                                                      resultDataType,
                                                      aType,
                                                      bType,
                                                      resultRegisterType,
                                                      aRegisterType,
                                                      bRegisterType);

            auto k = context->kernel();

            ArithmeticTestKernel kernel(context.get(), 2);
            kernel.setRegisterTypes(resultRegisterType, {aRegisterType, bRegisterType});
            kernel.setDataTypes(resultDataType, {aType, bType});

            kernel.addOp(ArithmeticOp{
                "Add",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) + operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "Subtract",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) - operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "Multiply",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) * operands.at(1);
                }});

            if(resultDataType == aType && resultDataType == bType)
            {
                kernel.addOp(ArithmeticOp{
                    "Divide",
                    [](std::vector<Expression::ExpressionPtr> operands)
                        -> Expression::ExpressionPtr { return operands.at(0) / operands.at(1); },
                    validForDivision});
                kernel.addOp(ArithmeticOp{
                    "Modulo",
                    [](std::vector<Expression::ExpressionPtr> operands)
                        -> Expression::ExpressionPtr { return operands.at(0) % operands.at(1); },
                    validForDivision});

                kernel.addOp(ArithmeticOp{"MultiplyHigh",
                                          [](std::vector<Expression::ExpressionPtr> operands)
                                              -> Expression::ExpressionPtr {
                                              return multiplyHigh(operands.at(0), operands.at(1));
                                          }});
            }

            kernel.addOp(ArithmeticOp{
                "ShiftL",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) << operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "LogicalShiftR",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return logicalShiftR(operands.at(0), operands.at(1));
                }});
            kernel.addOp(ArithmeticOp{
                "ArithmeticShiftR",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return arithmeticShiftR(operands.at(0), operands.at(1));
                }});

            kernel.addOp(ArithmeticOp{
                "BitwiseAnd",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) & operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "BitwiseOr",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) | operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "BitwiseXor",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) ^ operands.at(1);
                }});

            REQUIRE_NOTHROW(kernel.getAssembledKernel());
            for(auto a : TestValues::byType(aType))
            {
                for(auto b : TestValues::byType(bType))
                {
                    kernel.runAndValidate({a, b});
                }
            }
        }
    }

    TEST_CASE("Binary Arithmetic Generators work for integral types with literal operands.",
              "[arithmetic][literal][expression][codegen][gpu]")
    {
        auto resultDataType  = GENERATE(DataType::Int32, DataType::UInt32);
        auto operandDataType = GENERATE(DataType::Int32, DataType::UInt32);

        auto registerType = GENERATE(Register::Type::Scalar, Register::Type::Vector);

        CommandArgumentValue literalValue
            = GENERATE_COPY(from_range(TestValues::byType(resultDataType)));

        DYNAMIC_SECTION("resultDataType=" << resultDataType //
                                          << ", operandDataType=" << operandDataType //
                                          << ", registerType=" << registerType //
                                          << ", literalValue=" << literalValue)
        {
            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}},
                                                      resultDataType,
                                                      operandDataType,
                                                      registerType,
                                                      literalValue);

            ArithmeticTestKernel kernel(context.get(), 1);
            kernel.setRegisterTypes(registerType, {registerType});
            kernel.setDataTypes(resultDataType, {operandDataType});

            using BinaryOpFunc = Expression::ExpressionPtr (*)(Expression::ExpressionPtr,
                                                               Expression::ExpressionPtr);

            addBothOps(kernel, resultDataType, literalValue, "Add", Expression::operator+);
            addBothOps(kernel,
                       resultDataType,
                       literalValue,
                       "Subtract",
                       static_cast<BinaryOpFunc>(Expression::operator-));
            addBothOps(kernel, resultDataType, literalValue, "Multiply", Expression::operator*);

            if(resultDataType == operandDataType)
            {
                addBothOps(kernel,
                           resultDataType,
                           literalValue,
                           "Divide",
                           Expression::operator/,
                           validForDivision);
                addBothOps(kernel,
                           resultDataType,
                           literalValue,
                           "Modulo",
                           Expression::operator%,
                           validForDivision);

                addBothOps(
                    kernel, resultDataType, literalValue, "MultiplyHigh", Expression::multiplyHigh);
            }

            addBothOps(kernel, resultDataType, literalValue, "BitwiseAnd", Expression::operator&);
            addBothOps(kernel, resultDataType, literalValue, "BitwiseOr", Expression::operator|);
            addBothOps(kernel, resultDataType, literalValue, "BitwiseXor", Expression::operator^);

            addBothOps(kernel,
                       resultDataType,
                       literalValue,
                       "ShiftL",
                       static_cast<BinaryOpFunc>(Expression::operator<<),
                       validForShift);
            addBothOps(kernel,
                       resultDataType,
                       literalValue,
                       "LogicalShiftR",
                       Expression::logicalShiftR,
                       validForShift);
            addBothOps(kernel,
                       resultDataType,
                       literalValue,
                       "ArithmeticShiftR",
                       Expression::arithmeticShiftR,
                       validForShift);

            REQUIRE_NOTHROW(kernel.getAssembledKernel());
            for(auto operand : TestValues::byType(operandDataType))
            {
                kernel.runAndValidate({operand});
            }
        }
    }

    TEST_CASE("Binary Comparison Generators work for integral types",
              "[arithmetic][expression][codegen][gpu]")
    {
        auto resultDataType = GENERATE(DataType::Bool, DataType::Bool64);
        auto operandType    = GENERATE(DataType::Int32, DataType::UInt32);

        auto resultRegisterType = Register::Type::Scalar;

        auto [aRegisterType, bRegisterType] = GENERATE_COPY(from_range(inputTypePairs(
            resultDataType == DataType::Bool ? Register::Type::Scalar : Register::Type::Vector)));

        DYNAMIC_SECTION("resultDataType=" << resultDataType //
                                          << ", operandType=" << operandType //
                                          << ", resultRegisterType=" << resultRegisterType //
                                          << ", aRegisterType=" << aRegisterType //
                                          << ", bRegisterType=" << bRegisterType)
        {
            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}},
                                                      resultDataType,
                                                      operandType,
                                                      resultRegisterType,
                                                      aRegisterType,
                                                      bRegisterType);

            CAPTURE(context->assemblyFileName());

            ArithmeticTestKernel kernel(context.get(), 2);
            kernel.setRegisterTypes(resultRegisterType, {aRegisterType, bRegisterType});
            kernel.setDataTypes(resultDataType, {operandType, operandType});

            kernel.setStoreType(resultDataType == DataType::Bool64 ? DataType::UInt64
                                                                   : DataType::UInt32);

            kernel.addOp(ArithmeticOp{
                "Equal",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) == operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "NotEqual",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) != operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "GreaterThan",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) > operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "GreaterThanEqual",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) >= operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "LessThan",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) < operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "LessThanEqual",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) <= operands.at(1);
                }});

            for(auto a : TestValues::byType(operandType))
            {
                for(auto b : TestValues::byType(operandType))
                {
                    kernel.runAndValidate({a, b});
                }
            }
        }
    }

    TEST_CASE("Binary Comparison Generators work for integral types with literal operands.",
              "[arithmetic][literal][expression][codegen][gpu]")
    {
        auto resultDataType  = GENERATE(DataType::Bool, DataType::Bool64);
        auto operandDataType = GENERATE(DataType::Int32, DataType::UInt32);

        auto resultRegisterType = Register::Type::Scalar;

        auto inputRegisterType = resultDataType == DataType::Bool ? Register::Type::Scalar : Register::Type::Vector;

        CommandArgumentValue literalValue
            = GENERATE_COPY(from_range(TestValues::byType(operandDataType)));

        DYNAMIC_SECTION("resultDataType=" << resultDataType //
                                          << ", operandDataType=" << operandDataType //
                                          << ", resultRegisterType=" << resultRegisterType //
                                          << ", inputRegisterType=" << inputRegisterType //
                                          << ", literalValue=" << literalValue)
        {
            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}},
                                                      resultDataType,
                                                      operandDataType,
                                                      resultRegisterType,
                                                      inputRegisterType);

            ArithmeticTestKernel kernel(context.get(), 1);
            kernel.setRegisterTypes(resultRegisterType, {inputRegisterType});
            kernel.setDataTypes(resultDataType, {operandDataType});
            
            auto storeType = resultDataType == DataType::Bool64 ? DataType::UInt64
                                                                   : DataType::UInt32;
            kernel.setStoreType(storeType);

            addBothOps(kernel, storeType, literalValue, "Equal", Expression::operator==);
            addBothOps(kernel, storeType, literalValue, "NotEqual", Expression::operator!=);
            addBothOps(kernel, storeType, literalValue, "GreaterThan", Expression::operator>);
            addBothOps(kernel, storeType, literalValue, "GreaterThanEqual", Expression::operator>=);
            addBothOps(kernel, storeType, literalValue, "LessThan", Expression::operator<);
            addBothOps(kernel, storeType, literalValue, "LessThanEqual", Expression::operator<=);

            REQUIRE_NOTHROW(kernel.getAssembledKernel());

            for(auto value: TestValues::byType(operandDataType))
            {
                kernel.runAndValidate({value});
            }

        }
    }

    TEST_CASE("Binary Arithmetic Generators work for floating point types",
              "[arithmetic][expression][codegen][gpu]")
    {
        auto dataType = GENERATE(DataType::Float, DataType::Double);

        auto resultRegisterType = Register::Type::Vector;

        std::vector<Register::Type> aRegisterTypes = {Register::Type::Scalar};
        if(resultRegisterType == Register::Type::Vector)
        {
            aRegisterTypes.push_back(Register::Type::Vector);
        }

        auto aRegisterType = GENERATE_COPY(from_range(aRegisterTypes));

        std::vector<Register::Type> bRegisterTypes;
        if(resultRegisterType == Register::Type::Vector)
        {
            bRegisterTypes.push_back(Register::Type::Vector);
            if(aRegisterType == Register::Type::Vector)
                bRegisterTypes.push_back(Register::Type::Scalar);
        }

        auto bRegisterType = GENERATE_COPY(from_range(bRegisterTypes));

        DYNAMIC_SECTION("dataType=" << dataType //
                                    << ", resultRegisterType=" << resultRegisterType //
                                    << ", aRegisterType=" << aRegisterType //
                                    << ", bRegisterType=" << bRegisterType)
        {
            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}},
                                                      dataType,
                                                      resultRegisterType,
                                                      aRegisterType,
                                                      bRegisterType);

            auto k = context->kernel();

            ArithmeticTestKernel kernel(context.get(), 2);
            kernel.setRegisterTypes(resultRegisterType, {aRegisterType, bRegisterType});
            kernel.setDataTypes(dataType, {dataType, dataType});

            kernel.addOp(ArithmeticOp{
                "Add",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) + operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "Subtract",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) - operands.at(1);
                }});
            kernel.addOp(ArithmeticOp{
                "Multiply",
                [](std::vector<Expression::ExpressionPtr> operands) -> Expression::ExpressionPtr {
                    return operands.at(0) * operands.at(1);
                }});

            REQUIRE_NOTHROW(kernel.getAssembledKernel());
            for(auto a : TestValues::byType(dataType))
            {
                for(auto b : TestValues::byType(dataType))
                {
                    kernel.runAndValidate({a, b});
                }
            }
        }
    }
}