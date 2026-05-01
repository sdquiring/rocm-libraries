// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

namespace rocRoller::ISAParser
{
    class Parser;

    using ParserPtr = std::shared_ptr<Parser>;

    struct ParsedInstruction;
    struct ParsedFile;

} // namespace rocRoller::ISAParser
