// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/ISAParser/Parser_fwd.hpp>
#include <rocRoller/ISAParser/Token.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>

// Forward declare YAML::Node to avoid pulling in yaml-cpp header
namespace YAML
{
    class Node;
}

namespace rocRoller::ISAParser
{
    /**
     * @brief Represents a single parsed instruction from assembly
     */
    struct ParsedInstruction
    {
        std::string                     opcode;
        std::vector<Register::ValuePtr> dstOperands;
        std::vector<Register::ValuePtr> srcOperands;
        std::vector<std::string>        modifiers;
        std::string                     comment;
        int                             lineNumber;
        std::optional<std::string>      label; // Label on this line (e.g., "loop_start:")
        std::optional<std::string>      branchTarget; // Target for branch instructions

        ParsedInstruction()
            : lineNumber(0)
        {
        }
    };

    /**
     * @brief Represents a complete parsed assembly file
     */
    struct ParsedFile
    {
        std::string                          targetArch; // e.g., "gfx90a:sramecc+"
        std::string                          kernelName;
        std::vector<ParsedInstruction>       instructions;
        std::unordered_map<std::string, int> labelMap; // label -> instruction index
        std::shared_ptr<YAML::Node>          amdgpuMetadata; // Parsed YAML metadata

        struct KernelMetadata
        {
            int sgprCount     = 0;
            int vgprCount     = 0;
            int accvgprCount  = 0;
            int ldsSize       = 0; // In bytes
            int wavefrontSize = 64; // 32 or 64
        } metadata;
    };

    /**
     * @brief Parser for GCN ISA assembly - converts tokens to structured instructions
     */
    class Parser
    {
    public:
        /**
         * @brief Construct a new Parser
         * @param tokens The token stream from Lexer
         * @param context rocRoller context for creating register objects
         */
        Parser(std::vector<Token> const& tokens, ContextPtr context);

        /**
         * @brief Parse the token stream into a structured file representation
         * @return ParsedFile The parsed assembly file
         */
        ParsedFile parse();

    private:
        std::vector<Token> m_tokens;
        ContextPtr         m_context;
        size_t             m_position;

        // Parsing helpers
        Token const& current() const;
        Token const& peek(int offset = 1) const;
        bool         isAtEnd() const;
        Token const& advance();
        bool         check(TokenType type) const;
        bool         match(TokenType type);

        // High-level parsing
        void parseDirectives(ParsedFile& file);
        void parseInstruction(ParsedFile& file);
        void parseYAMLMetadata(ParsedFile& file, Token const& yamlToken);

        // Operand parsing
        Register::ValuePtr              parseOperand();
        Register::ValuePtr              parseRegister(Token const& tok);
        Register::ValuePtr              parseLiteral(Token const& tok);
        std::vector<Register::ValuePtr> parseRegisterRange(std::string const& rangeText);

        // Modifier parsing
        std::vector<std::string> parseModifiers();

        // Helper methods
        bool isRegisterToken(TokenType type) const;
        bool isLiteralToken(TokenType type) const;
    };

} // namespace rocRoller::ISAParser
