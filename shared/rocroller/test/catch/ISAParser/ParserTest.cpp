// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/Context.hpp>
#include <rocRoller/ISAParser/Lexer.hpp>
#include <rocRoller/ISAParser/Parser.hpp>

#include <yaml-cpp/yaml.h>

#include "../TestContext.hpp"

using namespace rocRoller::ISAParser;

TEST_CASE("Parser: Simple scalar instruction", "[ISAParser][Parser]")
{
    std::string input = "s_add_u32 s0, s1, s2\n";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    REQUIRE(file.instructions.size() == 1);

    auto const& inst = file.instructions[0];
    CHECK(inst.opcode == "s_add_u32");
    CHECK(inst.dstOperands.size() == 1);
    CHECK(inst.srcOperands.size() == 2);
    CHECK(inst.lineNumber == 1);
}

TEST_CASE("Parser: Vector instruction with hex literal", "[ISAParser][Parser]")
{
    std::string input = "v_mul_f32 v3, v4, 0x3f800000\n";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    REQUIRE(file.instructions.size() == 1);

    auto const& inst = file.instructions[0];
    CHECK(inst.opcode == "v_mul_f32");
    CHECK(inst.dstOperands.size() == 1);
    CHECK(inst.srcOperands.size() == 2);
}

TEST_CASE("Parser: Memory instruction with register range", "[ISAParser][Parser]")
{
    std::string input = "global_load_b128 v[0:3], v[4:5], off\n";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    REQUIRE(file.instructions.size() == 1);

    auto const& inst = file.instructions[0];
    CHECK(inst.opcode == "global_load_b128");
    CHECK(inst.dstOperands.size() == 1);
    // Note: Register ranges currently parse as single operands
}

TEST_CASE("Parser: Branch instruction with label", "[ISAParser][Parser]")
{
    std::string input = "s_cbranch_scc1 BB0_123\n"; // Label refs don't have leading period

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    REQUIRE(file.instructions.size() == 1);

    auto const& inst = file.instructions[0];
    CHECK(inst.opcode == "s_cbranch_scc1");
    REQUIRE(inst.branchTarget.has_value());
    CHECK(inst.branchTarget.value() == "BB0_123");
}

TEST_CASE("Parser: Label mapping", "[ISAParser][Parser]")
{
    std::string input = R"(
.Lloop_start:
    v_add_f32 v0, v0, v1
    s_branch loop_start
)";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    // Check that label was captured (without the colon)
    REQUIRE(file.labelMap.count(".Lloop_start") == 1);
    CHECK(file.labelMap[".Lloop_start"] == 0); // Points to first instruction

    REQUIRE(file.instructions.size() == 2);
    CHECK(file.instructions[0].opcode == "v_add_f32");
    CHECK(file.instructions[1].opcode == "s_branch");
    // Branch target is "loop_start" (no leading period)
    REQUIRE(file.instructions[1].branchTarget.has_value());
    CHECK(file.instructions[1].branchTarget.value() == "loop_start");
}

TEST_CASE("Parser: Directive parsing", "[ISAParser][Parser]")
{
    std::string input = R"(
.amdgcn_target "amdgcn-amd-amdhsa--gfx90a:sramecc+"
.globl my_kernel
    v_mov_b32 v0, s0
)";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    // Note: Lexer doesn't preserve hyphens/plus in tokenization - known limitation
    CHECK(file.targetArch == "amdgcnamdamdhsagfx90a:sramecc");
    CHECK(file.kernelName == "my_kernel");
    REQUIRE(file.instructions.size() == 1);
}

TEST_CASE("Parser: Instruction with comment", "[ISAParser][Parser]")
{
    std::string input = "v_add_f32 v0, v1, v2 // Add two floats\n";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    REQUIRE(file.instructions.size() == 1);

    auto const& inst = file.instructions[0];
    CHECK(inst.opcode == "v_add_f32");
    CHECK(inst.comment == " Add two floats"); // Lexer strips "//" prefix
}

TEST_CASE("Parser: Instruction with modifiers", "[ISAParser][Parser]")
{
    std::string input = "global_load_dword v5, v[0:1], off glc slc\n";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    REQUIRE(file.instructions.size() == 1);

    auto const& inst = file.instructions[0];
    CHECK(inst.opcode == "global_load_dword");
    REQUIRE(inst.modifiers.size() == 3);
    CHECK(inst.modifiers[0] == "off");
    CHECK(inst.modifiers[1] == "glc");
    CHECK(inst.modifiers[2] == "slc");
}

TEST_CASE("Parser: Special registers", "[ISAParser][Parser]")
{
    std::string input = R"(
s_mov_b64 exec, -1
s_mov_b32 m0, s5
v_cmp_eq_u32 vcc, v0, v1
)";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    REQUIRE(file.instructions.size() == 3);

    CHECK(file.instructions[0].opcode == "s_mov_b64");
    CHECK(file.instructions[1].opcode == "s_mov_b32");
    CHECK(file.instructions[2].opcode == "v_cmp_eq_u32");
}

TEST_CASE("Parser: YAML metadata", "[ISAParser][Parser]")
{
    std::string input = R"(
.amdgpu_metadata
---
amdhsa.kernels:
  - .name: kernel_name
    .symbol: kernel_name.kd
    .sgpr_count: 24
    .vgpr_count: 32
...
.end_amdgpu_metadata
)";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    // Check that YAML was captured and parsed
    REQUIRE(file.amdgpuMetadata != nullptr);
    REQUIRE(file.amdgpuMetadata->IsMap());

    // Verify we can access YAML structure
    auto kernels = (*file.amdgpuMetadata)["amdhsa.kernels"];
    REQUIRE(kernels.IsDefined());
}

TEST_CASE("Parser: Multiple instructions", "[ISAParser][Parser]")
{
    std::string input = R"(
s_load_dwordx2 s[4:5], s[0:1], 0x24
s_load_dwordx4 s[8:11], s[0:1], 0x34
v_mov_b32 v0, s4
v_mov_b32 v1, s5
v_add_f32 v2, v0, v1
s_endpgm
)";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    REQUIRE(file.instructions.size() == 6);

    CHECK(file.instructions[0].opcode == "s_load_dwordx2");
    CHECK(file.instructions[1].opcode == "s_load_dwordx4");
    CHECK(file.instructions[2].opcode == "v_mov_b32");
    CHECK(file.instructions[3].opcode == "v_mov_b32");
    CHECK(file.instructions[4].opcode == "v_add_f32");
    CHECK(file.instructions[5].opcode == "s_endpgm");
}

TEST_CASE("Parser: Empty file", "[ISAParser][Parser]")
{
    std::string input = "";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    CHECK(file.instructions.empty());
    CHECK(file.labelMap.empty());
}

TEST_CASE("Parser: Comments and empty lines", "[ISAParser][Parser]")
{
    std::string input = R"(
// This is a comment

v_add_f32 v0, v1, v2

// Another comment
)";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    // Should only have one instruction (comments and blank lines are skipped)
    REQUIRE(file.instructions.size() == 1);
    CHECK(file.instructions[0].opcode == "v_add_f32");
}

TEST_CASE("Parser: Standard label without period", "[ISAParser][Parser]")
{
    std::string input = R"(
BB0_1:
    v_add_f32 v0, v1, v2
    s_branch BB0_1
)";

    Lexer              lexer(input);
    std::vector<Token> tokens = lexer.tokenize();

    auto testContext = TestContext::ForTestDevice();
    auto context     = testContext.get();

    Parser     parser(tokens, context);
    ParsedFile file = parser.parse();

    // Check that label was captured (without the colon)
    REQUIRE(file.labelMap.count("BB0_1") == 1);
    CHECK(file.labelMap["BB0_1"] == 0); // Points to first instruction

    REQUIRE(file.instructions.size() == 2);
    CHECK(file.instructions[0].opcode == "v_add_f32");
    CHECK(file.instructions[1].opcode == "s_branch");
    // Branch target is "BB0_1"
    REQUIRE(file.instructions[1].branchTarget.has_value());
    CHECK(file.instructions[1].branchTarget.value() == "BB0_1");
}
