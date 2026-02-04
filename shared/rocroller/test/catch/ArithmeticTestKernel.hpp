/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#pragma once

#include "CustomMatchers.hpp"
#include "TestKernels.hpp"

#include <catch2/catch_test_macros.hpp>

#include <common/Utilities.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>

namespace ArithmeticTest
{
    using namespace rocRoller;

    struct ArithmeticOp
    {
        std::string name;

        using CreateExpressionFunc
            = std::function<Expression::ExpressionPtr(std::vector<Expression::ExpressionPtr>)>;
        CreateExpressionFunc createExpression;

        using Predicate     = std::function<bool(std::vector<CommandArgumentValue>)>;
        Predicate predicate = [](std::vector<CommandArgumentValue>) { return true; };
    };


    class ArithmeticTestKernel : public AssemblyTestKernel
    {
    public:
        ArithmeticTestKernel(rocRoller::ContextPtr context, int arity)
            : AssemblyTestKernel(context)
            , m_arity(arity)
            , m_storeType(DataType::None)
        {
        }

        void setRegisterTypes(Register::Type resultRegType, std::vector<Register::Type> regTypes)
        {
            m_resultRegType = resultRegType;
            AssertFatal(regTypes.size() == m_arity,
                        "Invalid number of register types",
                        ShowValue(regTypes.size()),
                        ShowValue(m_arity));
            m_regTypes = regTypes;
        }

        void setDataTypes(DataType resultType, std::vector<DataType> dataTypes)
        {
            m_resultType = resultType;
            AssertFatal(dataTypes.size() == m_arity,
                        "Invalid number of data types",
                        ShowValue(dataTypes.size()),
                        ShowValue(m_arity));
            m_dataTypes = dataTypes;
        }

        void setStoreType(DataType storeType)
        {
            m_storeType = storeType;
        }

        void addOp(ArithmeticOp op)
        {
            m_ops.push_back(op);
        }

        size_t numOps() const
        {
            return m_ops.size();
        }

        std::vector<ArithmeticOp> const& ops() const
        {
            return m_ops;
        }

        void generate() override
        {
            if(m_storeType == DataType::None)
            {
                m_storeType = m_resultType;
            }

            auto k = m_context->kernel();

            k->addArgument(
                {"result_", {m_storeType, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            for(size_t i = 0; i < m_arity; ++i)
            {
                k->addArgument({fmt::format("a_{}", i), m_dataTypes[i]});
            }

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                std::vector<Register::ValuePtr> operands(m_arity);
                Register::ValuePtr              resultPtr;
                co_yield m_context->argLoader()->getValue("result_", resultPtr);
                for(size_t i = 0; i < m_arity; ++i)
                {
                    co_yield m_context->argLoader()->getValue(fmt::format("a_{}", i), operands[i]);
                }

                auto resultReg
                    = Register::Value::Placeholder(m_context, m_resultRegType, m_resultType, 1);

                for(size_t i = 0; i < m_arity; ++i)
                {
                    co_yield m_context->copier()->ensureType(
                        operands[i], operands[i], m_regTypes[i]);
                }
                co_yield m_context->copier()->ensureType(resultPtr, resultPtr, m_resultRegType);

                auto storeInfo = DataTypeInfo::Get(m_storeType);

                std::vector<Expression::ExpressionPtr> operandExpressions(m_arity);
                for(size_t j = 0; j < m_arity; ++j)
                {
                    operandExpressions[j] = operands[j]->expression();
                }

                for(size_t i = 0; i < m_ops.size(); ++i)
                {
                    auto expr = m_ops[i].createExpression(operandExpressions);
                    co_yield Instruction::Comment("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=");
                    co_yield Instruction::Comment(fmt::format(
                        "Op {}: {} -> {}", i, m_ops[i].name, Expression::toString(expr)));

                    co_yield Expression::generate(resultReg, expr, m_context);
                    co_yield m_context->mem()->store(
                        m_resultRegType == Register::Type::Scalar
                            ? MemoryInstructions::MemoryKind::Scalar
                            : MemoryInstructions::MemoryKind::Global,
                        resultPtr,
                        resultReg,
                        Register::Value::Literal(i * storeInfo.elementBytes),
                        storeInfo.elementBytes);
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
        void runAndValidate_impl(std::vector<CommandArgumentValue> operands)
        {
            using ResultValueType
                = std::conditional_t<std::is_same_v<ResultType, bool>, uint32_t, ResultType>;
            auto resultTypeEnum = TypeInfo<ResultValueType>::Var.dataType;

            auto d_results = make_shared_device<ResultValueType>(numOps());

            CAPTURE(operands);

            KernelArguments kargs;
            kargs.append<void*>("result", d_results.get());
            for(size_t i = 0; i < m_arity; ++i)
            {
                kargs.append(fmt::format("a_{}", i), operands[i]);
            }

            (*this)({}, kargs);

            std::vector<ResultValueType> h_results(numOps());
            REQUIRE_THAT(hipMemcpy(h_results.data(),
                                   d_results.get(),
                                   h_results.size() * sizeof(ResultValueType),
                                   hipMemcpyDeviceToHost),
                         HasHipSuccess());

            std::vector<Expression::ExpressionPtr> operandExpressions(m_arity);
            for(size_t i = 0; i < m_arity; ++i)
            {
                operandExpressions[i] = Expression::literal(operands[i]);
            }

            for(size_t i = 0; i < numOps(); ++i)
            {
                CAPTURE(i, ops()[i].name);
                if(ops()[i].predicate(operands))
                {
                    auto expr
                        = convert(resultTypeEnum, ops()[i].createExpression(operandExpressions));

                    CAPTURE(Expression::toString(expr));
                    CHECK(Expression::evaluate(expr) == CommandArgumentValue(h_results[i]));
                }
            }
        }

        template <size_t Index = 0>
        void runAndValidate_idx(std::vector<CommandArgumentValue> operands)
        {
            using IdxType = std::variant_alternative_t<Index, CommandArgumentValue>;

            // {
            //     if(m_resultType == DataType::Bool64)
            //     {
            //         runAndValidate_impl<uint64_t>(operands);
            //         return;
            //     }
            //     else if(m_resultType == DataType::Bool32)
            //     {
            //         runAndValidate_impl<uint32_t>(operands);
            //         return;
            //     }
            // }

            if constexpr(CHasTypeInfo<IdxType> && CArithmeticType<IdxType>)
            {
                if(TypeInfo<IdxType>::Var.dataType == m_storeType)
                {
                    runAndValidate_impl<IdxType>(operands);
                    return;
                }
            }

            if constexpr(Index + 1 < std::variant_size_v<CommandArgumentValue>)
            {
                runAndValidate_idx<Index + 1>(operands);
            }
            else
            {
                FAIL("Invalid DataType for store type: " << m_storeType);
            }
        }

        void runAndValidate(std::vector<CommandArgumentValue> operands)
        {
            runAndValidate_idx(operands);
        }

    protected:
        std::vector<ArithmeticOp> m_ops;

        int                         m_arity;
        Register::Type              m_resultRegType;
        DataType                    m_resultType;
        DataType                    m_storeType;
        std::vector<DataType>       m_dataTypes;
        std::vector<Register::Type> m_regTypes;
    };

}