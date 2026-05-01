// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <rocRoller/ISAParser/Lexer_fwd.hpp>
#include <rocRoller/ISAParser/Token.hpp>

namespace rocRoller::ISAParser
{
    /**
     * @brief Lexer for GCN ISA assembly files
     *
     * Tokenizes assembly source code into a stream of tokens for parsing.
     * Handles opcodes, registers, literals, directives, comments, and labels.
     */
    class Lexer
    {
    public:
        /**
         * @brief Construct a new Lexer
         * @param source The assembly source code to tokenize
         */
        explicit Lexer(std::string_view source);

        /**
         * @brief Tokenize the entire source into a vector of tokens
         * @return std::vector<Token> The tokenized source
         */
        std::vector<Token> tokenize();

    private:
        std::string_view m_source;
        size_t           m_position;
        int              m_line;
        int              m_column;

        // Token scanning methods
        Token scanNext();
        Token scanComment();
        Token scanDirective();
        Token scanYAMLMetadata(int startLine, int startColumn);
        Token scanNumber();
        Token scanIdentifier();
        Token scanRegister();
        Token scanString();

        // Helper methods
        bool isOpcode(std::string_view text) const;
        bool isModifier(std::string_view text) const;
        bool isSpecialRegister(std::string_view text) const;
        void skipWhitespace();
        void skipToEndOfLine();

        // Character inspection
        char peek(int offset = 0) const;
        char advance();
        bool isAtEnd() const;
        bool match(char expected);

        // Known opcode prefixes for classification
        static const std::unordered_set<std::string>& getKnownOpcodes();
        static const std::unordered_set<std::string>& getKnownModifiers();
        static const std::unordered_set<std::string>& getSpecialRegisters();
    };

} // namespace rocRoller::ISAParser
