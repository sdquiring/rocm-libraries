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

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>

namespace ArithmeticBinaryTest
{
    using namespace rocRoller;

    struct BinaryOp
    {
        std::string name;
        std::function<Expression::ExpressionPtr(Expression::ExpressionPtr,
                                                Expression::ExpressionPtr)>
            createExpression;

        std::function<bool(CommandArgumentValue, CommandArgumentValue)> isValid
            = [](CommandArgumentValue, CommandArgumentValue) { return true; };

    };

    class BinaryArithmeticTestKernel : public AssemblyTestKernel
    {
    public:
        BinaryArithmeticTestKernel(rocRoller::ContextPtr context)
            : AssemblyTestKernel(context)
        {
        }

        void setRegisterTypes(Register::Type resultRegType,
                              Register::Type aRegType,
                              Register::Type bRegType)
        {
            m_resultRegType = resultRegType;
            m_aRegType      = aRegType;
            m_bRegType      = bRegType;
        }

        void setDataTypes(DataType resultType, DataType aType, DataType bType)
        {
            m_resultType = resultType;
            m_aType      = aType;
            m_bType      = bType;
        }

        void addOp(BinaryOp op)
        {
            m_ops.push_back(op);
        }

        size_t numOps() const
        {
            return m_ops.size();
        }

        std::vector<BinaryOp> const& ops() const
        {
            return m_ops;
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->addArgument(
                {"result_", {m_resultType, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            k->addArgument({"a", m_aType});
            k->addArgument({"b", m_bType});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr resultPtr, a, b;
                co_yield m_context->argLoader()->getValue("result_", resultPtr);
                co_yield m_context->argLoader()->getValue("a", a);
                co_yield m_context->argLoader()->getValue("b", b);

                auto resultReg
                    = Register::Value::Placeholder(m_context, m_resultRegType, m_resultType, 1);

                co_yield m_context->copier()->ensureType(a, a, m_aRegType);
                co_yield m_context->copier()->ensureType(b, b, m_bRegType);
                co_yield m_context->copier()->ensureType(resultPtr, resultPtr, m_resultRegType);

                auto resultInfo = DataTypeInfo::Get(m_resultType);

                for(size_t i = 0; i < m_ops.size(); ++i)
                {
                    auto expr = m_ops[i].createExpression(a->expression(), b->expression());

                    co_yield Expression::generate(resultReg, expr, m_context);
                    co_yield m_context->mem()->store(
                        m_resultRegType == Register::Type::Scalar
                            ? MemoryInstructions::MemoryKind::Scalar
                            : MemoryInstructions::MemoryKind::Global,
                        resultPtr,
                        resultReg,
                        Register::Value::Literal(i * resultInfo.elementBytes),
                        resultInfo.elementBytes);
                }

                if(m_resultRegType == Register::Type::Scalar)
                {
                    co_yield Instruction::Wait(
                        WaitCount::DSCnt(m_context->targetArchitecture(), 0));
                    co_yield Instruction("s_dcache_wb", {}, {}, {}, "");
                }
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

        template <typename ResultType>
        void runAndValidate_impl(CommandArgumentValue a, CommandArgumentValue b)
        {
            auto d_results = make_shared_device<ResultType>(numOps());

            (*this)({}, d_results.get(), a, b);
            CAPTURE(a, b);

            using ResultValueType
                = std::conditional_t<std::is_same_v<ResultType, bool>, uint32_t, ResultType>;

            std::vector<ResultValueType> h_results(numOps());
            REQUIRE_THAT(hipMemcpy(h_results.data(),
                                   d_results.get(),
                                   h_results.size() * sizeof(ResultValueType),
                                   hipMemcpyDeviceToHost),
                         HasHipSuccess());

            auto aExpr = Expression::literal(a);
            auto bExpr = Expression::literal(b);

            for(size_t i = 0; i < numOps(); ++i)
            {
                CAPTURE(i, ops()[i].name);
                if(ops()[i].isValid(CommandArgumentValue(a), CommandArgumentValue(b)))
                {
                    auto expr = convert(m_resultType, ops()[i].createExpression(aExpr, bExpr));
                    CHECK(Expression::evaluate(expr) == CommandArgumentValue(h_results[i]));
                }
            }
        }

        template <int Index = 0>
        void runAndValidate_idx(CommandArgumentValue a, CommandArgumentValue b)
        {
            using IdxType = std::variant_alternative_t<Index, CommandArgumentValue>;

            if constexpr(CHasTypeInfo<IdxType> && CArithmeticType<IdxType>)
            {
                if(TypeInfo<IdxType>::Var.dataType == m_resultType)
                {
                    runAndValidate_impl<IdxType>(a, b);
                    return;
                }
            }

            if constexpr(Index + 1 < std::variant_size_v<CommandArgumentValue>)
            {
                runAndValidate_idx<Index + 1>(a, b);
            }
            else
            {
                FAIL("Invalid DataType for result type: " << m_resultType);
            }
        }

        void runAndValidate(CommandArgumentValue a, CommandArgumentValue b)
        {
            runAndValidate_idx(a, b);
        }

    protected:
        std::vector<BinaryOp> m_ops;

        DataType m_resultType;
        DataType m_aType;
        DataType m_bType;

        Register::Type m_resultRegType;
        Register::Type m_aRegType;
        Register::Type m_bRegType;
    };

    TEST_CASE("Arithmetic Generators work for integral types",
              "[arithmetic][expression][codegen][gpu]")
    {
        auto resultDataType = GENERATE(DataType::Int32, DataType::UInt32);
        auto aType          = GENERATE(DataType::Int32, DataType::UInt32);
        auto bType          = GENERATE(DataType::Int32, DataType::UInt32);

        auto resultRegisterType = GENERATE(Register::Type::Scalar, Register::Type::Vector);

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
        else
        {
            bRegisterTypes.push_back(Register::Type::Scalar);
        }

        auto bRegisterType = GENERATE_COPY(from_range(bRegisterTypes));

        DYNAMIC_SECTION("resultDataType=" << resultDataType //
                                          << ", aType=" << aType //
                                          << ", bType=" << bType //
                                          << ", resultRegisterType=" << resultRegisterType //
                                          << ", aRegisterType=" << aRegisterType //
                                          << ", bRegisterType=" << bRegisterType)
        {
            CAPTURE(resultDataType, aType, bType, resultRegisterType, aRegisterType, bRegisterType);

            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}},
                                                      resultDataType,
                                                      aType,
                                                      bType,
                                                      resultRegisterType,
                                                      aRegisterType,
                                                      bRegisterType);

            auto k = context->kernel();

            BinaryArithmeticTestKernel kernel(context.get());
            kernel.setRegisterTypes(resultRegisterType, aRegisterType, bRegisterType);
            kernel.setDataTypes(resultDataType, aType, bType);

            auto validForDivision = [](CommandArgumentValue a, CommandArgumentValue b) {
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

                return std::visit(notZero, b) && std::visit(withinDivisionDomain, a)
                       && std::visit(withinDivisionDomain, b);
            };

            kernel.addOp(BinaryOp{"Add",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a + b; }});
            kernel.addOp(BinaryOp{"Subtract",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a - b; }});
            kernel.addOp(BinaryOp{"Multiply",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a * b; }});

            if(resultDataType == aType && resultDataType == bType)
            {
                kernel.addOp(BinaryOp{"Divide",
                                      [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                          -> Expression::ExpressionPtr { return a / b; },
                                      validForDivision});
                kernel.addOp(BinaryOp{"Modulo",
                                      [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                          -> Expression::ExpressionPtr { return a % b; },
                                      validForDivision});

                kernel.addOp(BinaryOp{"MultiplyHigh",
                                      [](Expression::ExpressionPtr a,
                                         Expression::ExpressionPtr b) -> Expression::ExpressionPtr {
                                          return multiplyHigh(a, b);
                                      }});
            }

            kernel.addOp(BinaryOp{"ShiftL",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a << b; }});
            kernel.addOp(BinaryOp{"LogicalShiftR",
                                  [](Expression::ExpressionPtr a,
                                     Expression::ExpressionPtr b) -> Expression::ExpressionPtr {
                                      return logicalShiftR(a, b);
                                  }});
            kernel.addOp(BinaryOp{"ArithmeticShiftR",
                                  [](Expression::ExpressionPtr a,
                                     Expression::ExpressionPtr b) -> Expression::ExpressionPtr {
                                      return arithmeticShiftR(a, b);
                                  }});

            kernel.addOp(BinaryOp{"BitwiseAnd",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a & b; }});
            kernel.addOp(BinaryOp{"BitwiseOr",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a | b; }});
            kernel.addOp(BinaryOp{"BitwiseXor",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a ^ b; }});

            REQUIRE_NOTHROW(kernel.getAssembledKernel());
            for(auto a : TestValues::byType(aType))
            {
                for(auto b : TestValues::byType(bType))
                {
                    kernel.runAndValidate(a, b);
                }
            }
        }
    }

    TEST_CASE("Arithmetic Generators work for floating point types",
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
            CAPTURE(dataType, resultRegisterType, aRegisterType, bRegisterType);

            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}},
                                                      dataType,
                                                      resultRegisterType,
                                                      aRegisterType,
                                                      bRegisterType);

            auto k = context->kernel();

            BinaryArithmeticTestKernel kernel(context.get());
            kernel.setRegisterTypes(resultRegisterType, aRegisterType, bRegisterType);
            kernel.setDataTypes(dataType, dataType, dataType);

            kernel.addOp(BinaryOp{"Add",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a + b; }});
            kernel.addOp(BinaryOp{"Subtract",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a - b; }});
            kernel.addOp(BinaryOp{"Multiply",
                                  [](Expression::ExpressionPtr a, Expression::ExpressionPtr b)
                                      -> Expression::ExpressionPtr { return a * b; }});

            REQUIRE_NOTHROW(kernel.getAssembledKernel());
            for(auto a : TestValues::byType(dataType))
            {
                for(auto b : TestValues::byType(dataType))
                {
                    kernel.runAndValidate(a, b);
                }
            }
        }
    }

}