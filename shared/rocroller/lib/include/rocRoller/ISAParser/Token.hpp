// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace rocRoller::ISAParser
{
    enum class TokenType
    {
        // Instructions and operands
        Opcode, // v_add_f32, s_load_dword, buffer_load_dwordx4
        SGPR, // s0, s[2:5]
        VGPR, // v0, v[10:13]
        ACCVGPR, // a0, a[4:7]
        SpecialReg, // vcc, vcc_lo, vcc_hi, exec, exec_lo, exec_hi, m0, scc
        TTMP, // ttmp7, ttmp9

        // Literals
        IntegerLiteral, // 42, 0
        HexLiteral, // 0x3f800000
        FloatLiteral, // 1.0, -0.5

        // Instruction modifiers
        Modifier, // off, glc, slc, lds, offen, idxen

        // Labels and control flow
        Label, // label_123:
        LabelRef, // label_123 (used in branches)

        // Assembly directives
        Directive, // .text, .globl, .amdgcn_target, .set

        // YAML metadata (entire .amdgpu_metadata block)
        YAMLMetadata, // YAML content between .amdgpu_metadata and .end_amdgpu_metadata

        // Comments (preserved for metadata extraction)
        Comment, // // This is a comment

        // Punctuation
        Comma, // ,
        Colon, // :
        LeftBracket, // [
        RightBracket, // ]

        // Special
        Newline,
        EndOfFile,
        Invalid
    };

    struct Token
    {
        TokenType   type;
        std::string text;
        int         line;
        int         column;

        Token()
            : type(TokenType::Invalid)
            , line(0)
            , column(0)
        {
        }

        Token(TokenType t, std::string_view txt, int ln, int col)
            : type(t)
            , text(txt)
            , line(ln)
            , column(col)
        {
        }
    };

    inline std::string toString(TokenType type)
    {
        switch(type)
        {
        case TokenType::Opcode:
            return "Opcode";
        case TokenType::SGPR:
            return "SGPR";
        case TokenType::VGPR:
            return "VGPR";
        case TokenType::ACCVGPR:
            return "ACCVGPR";
        case TokenType::SpecialReg:
            return "SpecialReg";
        case TokenType::TTMP:
            return "TTMP";
        case TokenType::IntegerLiteral:
            return "IntegerLiteral";
        case TokenType::HexLiteral:
            return "HexLiteral";
        case TokenType::FloatLiteral:
            return "FloatLiteral";
        case TokenType::Modifier:
            return "Modifier";
        case TokenType::Label:
            return "Label";
        case TokenType::LabelRef:
            return "LabelRef";
        case TokenType::Directive:
            return "Directive";
        case TokenType::YAMLMetadata:
            return "YAMLMetadata";
        case TokenType::Comment:
            return "Comment";
        case TokenType::Comma:
            return "Comma";
        case TokenType::Colon:
            return "Colon";
        case TokenType::LeftBracket:
            return "LeftBracket";
        case TokenType::RightBracket:
            return "RightBracket";
        case TokenType::Newline:
            return "Newline";
        case TokenType::EndOfFile:
            return "EndOfFile";
        case TokenType::Invalid:
            return "Invalid";
        default:
            return "Unknown";
        }
    }

} // namespace rocRoller::ISAParser
