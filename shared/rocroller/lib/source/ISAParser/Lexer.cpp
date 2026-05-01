// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/ISAParser/Lexer.hpp>

#include <cctype>
#include <unordered_set>

namespace rocRoller::ISAParser
{
    Lexer::Lexer(std::string_view source)
        : m_source(source)
        , m_position(0)
        , m_line(1)
        , m_column(1)
    {
    }

    std::vector<Token> Lexer::tokenize()
    {
        std::vector<Token> tokens;

        while(!isAtEnd())
        {
            Token tok = scanNext();
            if(tok.type != TokenType::Invalid)
            {
                tokens.push_back(tok);
            }
        }

        tokens.push_back(Token(TokenType::EndOfFile, "", m_line, m_column));
        return tokens;
    }

    Token Lexer::scanNext()
    {
        skipWhitespace();

        if(isAtEnd())
            return Token(TokenType::Invalid, "", m_line, m_column);

        int  startLine   = m_line;
        int  startColumn = m_column;
        char c           = peek();

        // Newline
        if(c == '\n')
        {
            advance();
            return Token(TokenType::Newline, "\n", startLine, startColumn);
        }

        // Comment
        if(c == '/' && peek(1) == '/')
        {
            return scanComment();
        }

        // Directive
        if(c == '.')
        {
            return scanDirective();
        }

        // Punctuation
        if(c == ',')
        {
            advance();
            return Token(TokenType::Comma, ",", startLine, startColumn);
        }
        if(c == ':')
        {
            advance();
            return Token(TokenType::Colon, ":", startLine, startColumn);
        }
        if(c == '[')
        {
            advance();
            return Token(TokenType::LeftBracket, "[", startLine, startColumn);
        }
        if(c == ']')
        {
            advance();
            return Token(TokenType::RightBracket, "]", startLine, startColumn);
        }

        // Numbers
        if(std::isdigit(c) || (c == '-' && std::isdigit(peek(1))))
        {
            return scanNumber();
        }

        // Hex literals
        if(c == '0' && (peek(1) == 'x' || peek(1) == 'X'))
        {
            return scanNumber();
        }

        // Identifiers (opcodes, registers, labels, modifiers)
        if(std::isalpha(c) || c == '_' || c == '$')
        {
            return scanIdentifier();
        }

        // Unknown character - skip it
        advance();
        return Token(TokenType::Invalid, std::string(1, c), startLine, startColumn);
    }

    Token Lexer::scanComment()
    {
        int startLine   = m_line;
        int startColumn = m_column;

        // Skip //
        advance();
        advance();

        size_t start = m_position;
        skipToEndOfLine();

        std::string_view commentText = m_source.substr(start, m_position - start);
        return Token(TokenType::Comment, commentText, startLine, startColumn);
    }

    Token Lexer::scanDirective()
    {
        int    startLine   = m_line;
        int    startColumn = m_column;
        size_t start       = m_position;

        advance(); // Skip '.'

        while(!isAtEnd() && (std::isalnum(peek()) || peek() == '_'))
        {
            advance();
        }

        std::string_view directiveText = m_source.substr(start, m_position - start);

        // Check if this is the start of YAML metadata
        if(directiveText == ".amdgpu_metadata")
        {
            return scanYAMLMetadata(startLine, startColumn);
        }

        // Check if this is a label (directive followed by colon)
        if(!isAtEnd() && peek() == ':')
        {
            advance(); // Include the colon in the label token
            std::string labelText(m_source.substr(start, m_position - start));
            return Token(TokenType::Label, labelText, startLine, startColumn);
        }

        return Token(TokenType::Directive, directiveText, startLine, startColumn);
    }

    Token Lexer::scanYAMLMetadata(int startLine, int startColumn)
    {
        // We're positioned right after ".amdgpu_metadata"
        // Capture everything until ".end_amdgpu_metadata"

        size_t contentStart = m_position;

        // Skip to end of current line (the .amdgpu_metadata line)
        while(!isAtEnd() && peek() != '\n')
        {
            advance();
        }
        if(peek() == '\n')
            advance();

        // Now capture until we find ".end_amdgpu_metadata"
        while(!isAtEnd())
        {
            // Check if we're at the start of a line with ".end_amdgpu_metadata"
            if(peek() == '.')
            {
                // Peek ahead to check if it's ".end_amdgpu_metadata"
                size_t savedPos    = m_position;
                int    savedLine   = m_line;
                int    savedColumn = m_column;

                advance(); // Skip '.'
                std::string check;
                while(!isAtEnd() && (std::isalnum(peek()) || peek() == '_'))
                {
                    check += peek();
                    advance();
                }

                if(check == "end_amdgpu_metadata")
                {
                    // Found the end marker, capture content before it
                    size_t contentEnd = savedPos;

                    // Restore position to before the dot
                    m_position = savedPos;
                    m_line     = savedLine;
                    m_column   = savedColumn;

                    // Extract YAML content (excluding the start and end directives)
                    std::string_view yamlContent
                        = m_source.substr(contentStart, contentEnd - contentStart);

                    // Now skip past ".end_amdgpu_metadata" for next token
                    advance(); // Skip '.'
                    while(!isAtEnd() && (std::isalnum(peek()) || peek() == '_'))
                    {
                        advance();
                    }

                    return Token(TokenType::YAMLMetadata, yamlContent, startLine, startColumn);
                }
                else
                {
                    // Not the end marker, restore position and continue
                    m_position = savedPos;
                    m_line     = savedLine;
                    m_column   = savedColumn;
                }
            }

            advance();
        }

        // Reached EOF without finding end marker
        // Return what we have as YAML content (malformed but captured)
        std::string_view yamlContent = m_source.substr(contentStart, m_position - contentStart);
        return Token(TokenType::YAMLMetadata, yamlContent, startLine, startColumn);
    }

    Token Lexer::scanNumber()
    {
        int    startLine   = m_line;
        int    startColumn = m_column;
        size_t start       = m_position;

        // Handle negative sign
        if(peek() == '-')
            advance();

        // Hex literal
        if(peek() == '0' && (peek(1) == 'x' || peek(1) == 'X'))
        {
            advance(); // 0
            advance(); // x
            while(!isAtEnd() && std::isxdigit(peek()))
            {
                advance();
            }
            std::string_view hexText = m_source.substr(start, m_position - start);
            return Token(TokenType::HexLiteral, hexText, startLine, startColumn);
        }

        // Decimal number
        bool hasDecimalPoint = false;
        while(!isAtEnd() && (std::isdigit(peek()) || peek() == '.'))
        {
            if(peek() == '.')
            {
                if(hasDecimalPoint)
                    break; // Second decimal point, stop
                hasDecimalPoint = true;
            }
            advance();
        }

        // Scientific notation (e.g., 1.0e-5)
        if(!isAtEnd() && (peek() == 'e' || peek() == 'E'))
        {
            advance();
            if(peek() == '+' || peek() == '-')
                advance();
            while(!isAtEnd() && std::isdigit(peek()))
            {
                advance();
            }
            hasDecimalPoint = true;
        }

        std::string_view numberText = m_source.substr(start, m_position - start);
        TokenType type = hasDecimalPoint ? TokenType::FloatLiteral : TokenType::IntegerLiteral;
        return Token(type, numberText, startLine, startColumn);
    }

    Token Lexer::scanIdentifier()
    {
        int    startLine   = m_line;
        int    startColumn = m_column;
        size_t start       = m_position;

        while(!isAtEnd() && (std::isalnum(peek()) || peek() == '_' || peek() == '$'))
        {
            advance();
        }

        std::string_view identText = m_source.substr(start, m_position - start);

        // Check for register range: s[2:5]
        if(identText.length() >= 1 && peek() == '[')
        {
            // This is a register with range
            return scanRegister();
        }

        // Classify identifier
        TokenType type = TokenType::Invalid;

        // Check for registers
        if(identText.length() >= 2)
        {
            char prefix = identText[0];
            if(prefix == 's' && std::isdigit(identText[1]))
            {
                type = TokenType::SGPR;
            }
            else if(prefix == 'v' && std::isdigit(identText[1]))
            {
                type = TokenType::VGPR;
            }
            else if(prefix == 'a' && std::isdigit(identText[1]))
            {
                type = TokenType::ACCVGPR;
            }
            else if(identText.substr(0, 4) == "ttmp")
            {
                type = TokenType::TTMP;
            }
            else if(isSpecialRegister(identText))
            {
                type = TokenType::SpecialReg;
            }
            else if(isOpcode(identText))
            {
                type = TokenType::Opcode;
            }
            else if(isModifier(identText))
            {
                type = TokenType::Modifier;
            }
            else
            {
                // Could be a label reference or unknown identifier
                type = TokenType::LabelRef;
            }
        }
        else if(identText.length() == 1)
        {
            type = TokenType::LabelRef; // Single char, likely label or unknown
        }
        else
        {
            if(isOpcode(identText))
            {
                type = TokenType::Opcode;
            }
            else if(isModifier(identText))
            {
                type = TokenType::Modifier;
            }
            else if(isSpecialRegister(identText))
            {
                type = TokenType::SpecialReg;
            }
            else
            {
                type = TokenType::LabelRef;
            }
        }

        // Check if this identifier is followed by a colon (making it a label definition)
        if(!isAtEnd() && peek() == ':')
        {
            advance(); // Include the colon in the label token
            std::string labelText(m_source.substr(start, m_position - start));
            return Token(TokenType::Label, labelText, startLine, startColumn);
        }

        return Token(type, identText, startLine, startColumn);
    }

    Token Lexer::scanRegister()
    {
        int    startLine   = m_line;
        int    startColumn = m_column;
        size_t start       = m_position - 1; // Include the register prefix

        // We're positioned after the register prefix (s, v, a)
        // Now scan the range: [start:end]
        if(peek() == '[')
        {
            advance(); // [
            while(!isAtEnd() && peek() != ']')
            {
                advance();
            }
            if(peek() == ']')
                advance(); // ]
        }

        std::string_view registerText = m_source.substr(start, m_position - start);

        // Determine register type from prefix
        char      prefix = m_source[start];
        TokenType type   = TokenType::Invalid;
        if(prefix == 's')
            type = TokenType::SGPR;
        else if(prefix == 'v')
            type = TokenType::VGPR;
        else if(prefix == 'a')
            type = TokenType::ACCVGPR;

        return Token(type, registerText, startLine, startColumn);
    }

    bool Lexer::isOpcode(std::string_view text) const
    {
        const auto& opcodes = getKnownOpcodes();

        // Exact match
        if(opcodes.count(std::string(text)) > 0)
            return true;

        // Prefix match for common instruction families
        if(text.length() >= 2)
        {
            if(text[0] == 's' && text[1] == '_')
                return true; // Scalar instruction
            if(text[0] == 'v' && text[1] == '_')
                return true; // Vector instruction
        }
        if(text.length() >= 6 && text.substr(0, 6) == "buffer")
            return true; // Buffer instruction
        if(text.length() >= 6 && text.substr(0, 6) == "global")
            return true; // Global instruction
        if(text.length() >= 3 && text.substr(0, 3) == "ds_")
            return true; // LDS instruction
        if(text.length() >= 5 && text.substr(0, 5) == "flat_")
            return true; // Flat instruction

        return false;
    }

    bool Lexer::isModifier(std::string_view text) const
    {
        const auto& modifiers = getKnownModifiers();
        return modifiers.count(std::string(text)) > 0;
    }

    bool Lexer::isSpecialRegister(std::string_view text) const
    {
        const auto& specials = getSpecialRegisters();
        return specials.count(std::string(text)) > 0;
    }

    void Lexer::skipWhitespace()
    {
        while(!isAtEnd())
        {
            char c = peek();
            if(c == ' ' || c == '\t' || c == '\r')
            {
                advance();
            }
            else
            {
                break;
            }
        }
    }

    void Lexer::skipToEndOfLine()
    {
        while(!isAtEnd() && peek() != '\n')
        {
            advance();
        }
    }

    char Lexer::peek(int offset) const
    {
        size_t pos = m_position + offset;
        if(pos >= m_source.length())
            return '\0';
        return m_source[pos];
    }

    char Lexer::advance()
    {
        if(isAtEnd())
            return '\0';

        char c = m_source[m_position++];
        if(c == '\n')
        {
            m_line++;
            m_column = 1;
        }
        else
        {
            m_column++;
        }
        return c;
    }

    bool Lexer::isAtEnd() const
    {
        return m_position >= m_source.length();
    }

    bool Lexer::match(char expected)
    {
        if(isAtEnd() || peek() != expected)
            return false;
        advance();
        return true;
    }

    const std::unordered_set<std::string>& Lexer::getKnownOpcodes()
    {
        static const std::unordered_set<std::string> opcodes = {
            // Scalar ALU
            "s_mov_b32",
            "s_mov_b64",
            "s_add_u32",
            "s_add_i32",
            "s_sub_u32",
            "s_mul_i32",
            "s_and_b32",
            "s_or_b32",
            "s_xor_b32",
            "s_not_b32",
            "s_lshl_b32",
            "s_lshr_b32",
            "s_ashr_i32",
            "s_cmp_eq_u32",
            "s_cmp_lt_u32",
            "s_cmp_gt_u32",
            // Scalar memory
            "s_load_dword",
            "s_load_dwordx2",
            "s_load_dwordx4",
            "s_load_dwordx8",
            "s_load_b32",
            "s_load_b64",
            "s_load_b128",
            "s_load_b256",
            "s_load_b512",
            "s_store_dword",
            "s_store_dwordx2",
            "s_store_dwordx4",
            // Scalar control flow
            "s_branch",
            "s_cbranch_scc0",
            "s_cbranch_scc1",
            "s_cbranch_vccz",
            "s_cbranch_vccnz",
            "s_endpgm",
            "s_setpc_b64",
            "s_swappc_b64",
            // Scalar special
            "s_nop",
            "s_waitcnt",
            "s_wait_kmcnt",
            "s_barrier",
            "s_set_vgpr_msb",
            // Vector ALU
            "v_mov_b32",
            "v_add_f32",
            "v_add_u32",
            "v_add_co_u32",
            "v_add_nc_u32",
            "v_sub_f32",
            "v_sub_u32",
            "v_mul_f32",
            "v_mul_lo_u32",
            "v_mul_hi_u32",
            "v_fma_f32",
            "v_mad_u32",
            "v_mad_i32",
            "v_and_b32",
            "v_or_b32",
            "v_xor_b32",
            "v_lshlrev_b32",
            "v_lshrrev_b32",
            "v_cmp_eq_f32",
            "v_cmp_lt_f32",
            "v_cmp_gt_f32",
            // Vector memory
            "global_load_dword",
            "global_load_dwordx2",
            "global_load_dwordx4",
            "global_load_b32",
            "global_load_b64",
            "global_load_b128",
            "global_store_dword",
            "global_store_dwordx2",
            "global_store_dwordx4",
            "global_store_b32",
            "global_store_b64",
            "global_store_b128",
            "buffer_load_dword",
            "buffer_load_dwordx2",
            "buffer_load_dwordx4",
            "buffer_load_b32",
            "buffer_load_b64",
            "buffer_load_b128",
            "buffer_store_dword",
            "buffer_store_dwordx2",
            "buffer_store_dwordx4",
            "buffer_store_b32",
            "buffer_store_b64",
            "buffer_store_b128",
            // LDS
            "ds_read_b32",
            "ds_read_b64",
            "ds_write_b32",
            "ds_write_b64",
            // MFMA
            "v_mfma_f32_16x16x4f32",
            "v_mfma_f32_16x16x16f16",
            "v_mfma_f32_32x32x8f16",
            "v_mfma_f32_4x4x4f16",
            "v_mfma_f64_16x16x4f64",
        };
        return opcodes;
    }

    const std::unordered_set<std::string>& Lexer::getKnownModifiers()
    {
        static const std::unordered_set<std::string> modifiers = {
            "off",
            "glc",
            "slc",
            "lds",
            "offen",
            "idxen",
            "addr64",
            "tfe",
            "dlc",
            "scc0",
            "scc1",
            "vccz",
            "vccnz",
            "execz",
            "execnz",
        };
        return modifiers;
    }

    const std::unordered_set<std::string>& Lexer::getSpecialRegisters()
    {
        static const std::unordered_set<std::string> specials = {
            "vcc",
            "vcc_lo",
            "vcc_hi",
            "exec",
            "exec_lo",
            "exec_hi",
            "m0",
            "scc",
            "ttmp7",
            "ttmp9",
            "null",
        };
        return specials;
    }

} // namespace rocRoller::ISAParser
