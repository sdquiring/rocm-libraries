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

#pragma once

#include "../GenericContextFixture.hpp"
#include <rocRoller/Expression.hpp>

namespace KernelGraphTest
{
    class KernelGraphTest : public GenericContextFixture
    {
    public:
        rocRoller::Expression::FastArithmetic fastArith{m_context};

        void SetUp() override
        {
            GenericContextFixture::SetUp();
            fastArith = rocRoller::Expression::FastArithmetic(m_context);
        }
    };

    // TODO: Add transforms and tests for VectorAdd with: 1. ForLoop
    // and 2. ForLoop+Unroll.
    //
    // The tests should also make sure the lowering fails if the
    // WorkitemCount is missing.
    //
    // See KernelGraphTestGPULoopSize :: MissingWorkitemCount
    //     KernelGraphTestGPULoopSize :: TestForLoop
}
