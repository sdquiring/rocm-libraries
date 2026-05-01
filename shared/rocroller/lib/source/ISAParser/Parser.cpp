// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/ISAParser/Parser.hpp>

#include <rocRoller/Context.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include <yaml-cpp/yaml.h>

#include <cctype>
#include <sstream>

namespace rocRoller::ISAParser
{
    Parser::Parser(std::vector<Token> const& tokens, ContextPtr context)
        : m_tokens(tokens)
        , m_context(context)
        , m_position(0)
    {
    }

    ParsedFile Parser::parse()
    {
        ParsedFile file;

        while(!isAtEnd())
        {
            Token const& tok = current();

            if(tok.type == TokenType::Directive)
            {
                parseDirectives(file);
            }
            else if(tok.type == TokenType::YAMLMetadata)
            {
                parseYAMLMetadata(file, tok);
                advance();
            }
            else if(tok.type == TokenType::Label)
            {
                // Store label for next instruction
                std::string labelText = tok.text;
                // Remove trailing colon if present
                if(!labelText.empty() && labelText.back() == ':')
                {
                    labelText = labelText.substr(0, labelText.size() - 1);
                }

                file.labelMap[labelText] = static_cast<int>(file.instructions.size());
                advance();
            }
            else if(tok.type == TokenType::Opcode)
            {
                parseInstruction(file);
            }
            else if(tok.type == TokenType::Comment || tok.type == TokenType::Newline)
            {
                // Skip standalone comments and newlines
                advance();
            }
            else
            {
                // Skip unknown tokens
                advance();
            }
        }

        return file;
    }

    void Parser::parseDirectives(ParsedFile& file)
    {
        Token const& tok = current();

        if(tok.text == ".amdgcn_target")
        {
            advance();
            // Collect all tokens until newline to form the target string
            std::string targetText;
            while(!isAtEnd() && current().type != TokenType::Newline)
            {
                targetText += current().text;
                advance();
            }

            // Remove surrounding quotes if present
            file.targetArch = targetText;
            if(file.targetArch.size() >= 2 && file.targetArch.front() == '"'
               && file.targetArch.back() == '"')
            {
                file.targetArch = file.targetArch.substr(1, file.targetArch.size() - 2);
            }
        }
        else if(tok.text == ".globl")
        {
            advance();
            // Next token is the kernel name
            if(!isAtEnd() && current().type != TokenType::Newline)
            {
                file.kernelName = current().text;
                advance();
            }
        }
        else
        {
            // Skip other directives for now
            advance();
        }
    }

    void Parser::parseInstruction(ParsedFile& file)
    {
        ParsedInstruction inst;
        inst.lineNumber = current().line;
        inst.opcode     = current().text;

        advance(); // Move past opcode

        // Parse operands until we hit a newline, comment, or EOF
        while(!isAtEnd() && current().type != TokenType::Newline
              && current().type != TokenType::Comment)
        {
            if(current().type == TokenType::Comma)
            {
                advance();
                continue;
            }

            if(isRegisterToken(current().type) || isLiteralToken(current().type))
            {
                auto operand = parseOperand();
                if(operand)
                {
                    // TODO: Classify as dst or src based on opcode
                    // For now, first operand is dst, rest are src
                    if(inst.dstOperands.empty())
                        inst.dstOperands.push_back(operand);
                    else
                        inst.srcOperands.push_back(operand);
                }
            }
            else if(current().type == TokenType::Modifier)
            {
                inst.modifiers.push_back(current().text);
                advance();
            }
            else if(current().type == TokenType::LabelRef)
            {
                // Branch target
                inst.branchTarget = current().text;
                advance();
            }
            else
            {
                // Unknown token, skip
                advance();
            }
        }

        // Capture inline comment if present (Lexer already strips "//" prefix)
        if(!isAtEnd() && current().type == TokenType::Comment)
        {
            inst.comment = current().text;
            advance();
        }

        file.instructions.push_back(inst);

        // Skip newline
        if(!isAtEnd() && current().type == TokenType::Newline)
            advance();
    }

    void Parser::parseYAMLMetadata(ParsedFile& file, Token const& yamlToken)
    {
        try
        {
            file.amdgpuMetadata = std::make_shared<YAML::Node>(YAML::Load(yamlToken.text));
        }
        catch(const YAML::Exception&)
        {
            // If YAML parsing fails, just skip it
            file.amdgpuMetadata = nullptr;
        }
    }

    Register::ValuePtr Parser::parseOperand()
    {
        Token const& tok = current();

        if(isRegisterToken(tok.type))
        {
            return parseRegister(tok);
        }
        else if(isLiteralToken(tok.type))
        {
            return parseLiteral(tok);
        }

        return nullptr;
    }

    Register::ValuePtr Parser::parseRegister(Token const& tok)
    {
        advance();

        // Check if it's a register range like s[2:5]
        if(tok.text.find('[') != std::string::npos)
        {
            // Parse register range
            return parseRegisterRange(tok.text).empty() ? nullptr
                                                        : parseRegisterRange(tok.text)[0];
        }

        // Parse simple register like s0, v5, a2
        Register::Type regType = Register::Type::Literal;

        if(tok.type == TokenType::SGPR)
            regType = Register::Type::Scalar;
        else if(tok.type == TokenType::VGPR)
            regType = Register::Type::Vector;
        else if(tok.type == TokenType::ACCVGPR)
            regType = Register::Type::Accumulator;
        else if(tok.type == TokenType::SpecialReg)
        {
            // Handle special registers - these don't have indices
            if(tok.text == "vcc")
                regType = Register::Type::VCC;
            else if(tok.text == "vcc_lo")
                regType = Register::Type::VCC_LO;
            else if(tok.text == "vcc_hi")
                regType = Register::Type::VCC_HI;
            else if(tok.text == "exec")
                regType = Register::Type::EXEC;
            else if(tok.text == "exec_lo")
                regType = Register::Type::EXEC_LO;
            else if(tok.text == "exec_hi")
                regType = Register::Type::EXEC_HI;
            else if(tok.text == "m0")
                regType = Register::Type::M0;
            else if(tok.text == "scc")
                regType = Register::Type::SCC;

            // Special registers - create with VariableType(DataType::Raw32)
            return std::make_shared<Register::Value>(
                m_context, regType, VariableType(DataType::Raw32), std::vector<int>{0});
        }

        // Extract register index
        int regIndex = 0;
        if(tok.type == TokenType::SGPR || tok.type == TokenType::VGPR
           || tok.type == TokenType::ACCVGPR)
        {
            // Extract number from "s5", "v10", etc.
            std::string numStr = tok.text.substr(1); // Skip prefix
            try
            {
                regIndex = std::stoi(numStr);
            }
            catch(...)
            {
                regIndex = 0;
            }
        }

        // Create Value with explicit register index
        // This represents an already-allocated hardware register
        return std::make_shared<Register::Value>(
            m_context, regType, VariableType(DataType::Raw32), std::vector<int>{regIndex});
    }

    Register::ValuePtr Parser::parseLiteral(Token const& tok)
    {
        advance();

        // Parse literal value based on type
        if(tok.type == TokenType::IntegerLiteral)
        {
            try
            {
                int64_t value = std::stoll(tok.text);
                return Register::Value::Literal(value);
            }
            catch(...)
            {
                return Register::Value::Literal(0);
            }
        }
        else if(tok.type == TokenType::HexLiteral)
        {
            try
            {
                // Remove 0x prefix if present
                std::string hexStr = tok.text;
                if(hexStr.size() >= 2 && hexStr[0] == '0' && (hexStr[1] == 'x' || hexStr[1] == 'X'))
                {
                    hexStr = hexStr.substr(2);
                }
                uint64_t value = std::stoull(hexStr, nullptr, 16);
                return Register::Value::Literal(static_cast<int64_t>(value));
            }
            catch(...)
            {
                return Register::Value::Literal(0);
            }
        }
        else if(tok.type == TokenType::FloatLiteral)
        {
            try
            {
                double value = std::stod(tok.text);
                return Register::Value::Literal(value);
            }
            catch(...)
            {
                return Register::Value::Literal(0.0);
            }
        }

        return nullptr;
    }

    std::vector<Register::ValuePtr> Parser::parseRegisterRange(std::string const& rangeText)
    {
        // Parse register range like s[2:5] or v[0:3]
        std::vector<Register::ValuePtr> registers;

        // Find the register type prefix
        Register::Type regType = Register::Type::Literal;
        if(rangeText[0] == 's')
            regType = Register::Type::Scalar;
        else if(rangeText[0] == 'v')
            regType = Register::Type::Vector;
        else if(rangeText[0] == 'a')
            regType = Register::Type::Accumulator;
        else
            return {}; // Unknown register type

        // Find the bracket positions
        size_t leftBracket  = rangeText.find('[');
        size_t colon        = rangeText.find(':');
        size_t rightBracket = rangeText.find(']');

        if(leftBracket == std::string::npos || rightBracket == std::string::npos)
            return {};

        try
        {
            if(colon != std::string::npos && colon > leftBracket && colon < rightBracket)
            {
                // Range like s[2:5]
                int startIdx
                    = std::stoi(rangeText.substr(leftBracket + 1, colon - leftBracket - 1));
                int endIdx = std::stoi(rangeText.substr(colon + 1, rightBracket - colon - 1));

                for(int i = startIdx; i <= endIdx; ++i)
                {
                    registers.push_back(std::make_shared<Register::Value>(
                        m_context, regType, VariableType(DataType::Raw32), std::vector<int>{i}));
                }
            }
            else
            {
                // Single register in brackets like s[5]
                int idx = std::stoi(
                    rangeText.substr(leftBracket + 1, rightBracket - leftBracket - 1));
                registers.push_back(std::make_shared<Register::Value>(
                    m_context, regType, VariableType(DataType::Raw32), std::vector<int>{idx}));
            }
        }
        catch(...)
        {
            return {};
        }

        return registers;
    }

    std::vector<std::string> Parser::parseModifiers()
    {
        std::vector<std::string> modifiers;

        while(!isAtEnd() && current().type == TokenType::Modifier)
        {
            modifiers.push_back(current().text);
            advance();
        }

        return modifiers;
    }

    Token const& Parser::current() const
    {
        if(m_position >= m_tokens.size())
            return m_tokens.back(); // Return EOF

        return m_tokens[m_position];
    }

    Token const& Parser::peek(int offset) const
    {
        size_t pos = m_position + offset;
        if(pos >= m_tokens.size())
            return m_tokens.back(); // Return EOF

        return m_tokens[pos];
    }

    bool Parser::isAtEnd() const
    {
        return m_position >= m_tokens.size() || current().type == TokenType::EndOfFile;
    }

    Token const& Parser::advance()
    {
        if(!isAtEnd())
            m_position++;

        return current();
    }

    bool Parser::check(TokenType type) const
    {
        return !isAtEnd() && current().type == type;
    }

    bool Parser::match(TokenType type)
    {
        if(check(type))
        {
            advance();
            return true;
        }
        return false;
    }

    bool Parser::isRegisterToken(TokenType type) const
    {
        return type == TokenType::SGPR || type == TokenType::VGPR || type == TokenType::ACCVGPR
               || type == TokenType::SpecialReg || type == TokenType::TTMP;
    }

    bool Parser::isLiteralToken(TokenType type) const
    {
        return type == TokenType::IntegerLiteral || type == TokenType::HexLiteral
               || type == TokenType::FloatLiteral;
    }

} // namespace rocRoller::ISAParser
