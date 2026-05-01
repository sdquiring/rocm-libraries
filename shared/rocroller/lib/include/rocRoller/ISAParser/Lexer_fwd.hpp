// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

namespace rocRoller::ISAParser
{
    class Lexer;

    using LexerPtr = std::shared_ptr<Lexer>;

} // namespace rocRoller::ISAParser
