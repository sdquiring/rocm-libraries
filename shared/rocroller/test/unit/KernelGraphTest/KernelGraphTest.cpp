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

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include "../GenericContextFixture.hpp"
#include "../SourceMatcher.hpp"
#include "../Utilities.hpp"

#include "KernelGraphTestFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;

namespace KernelGraphTest
{
    TEST_F(KernelGraphTest, Basic)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        // Control Graph
        int kernel_index = kgraph.control.addElement(Kernel());
        int loadA_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int loadB_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int body1_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadA_index});
        int body2_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadB_index});

        int op1_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(5)});
        int sequence1_index = kgraph.control.addElement(Sequence(), {loadA_index}, {op1_index});
        int sequence2_index = kgraph.control.addElement(Sequence(), {loadB_index}, {op1_index});

        int op2_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(7)});
        int sequence3_index = kgraph.control.addElement(Sequence(), {op1_index}, {op2_index});

        int op3_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(9)});
        int sequence4_index = kgraph.control.addElement(Sequence(), {op1_index}, {op3_index});
        int sequence5_index = kgraph.control.addElement(Sequence(), {op2_index}, {op3_index});

        int storeC_index    = kgraph.control.addElement(StoreLinear());
        int sequence6_index = kgraph.control.addElement(Sequence(), {op3_index}, {storeC_index});

        // Coordinate Graph
        int u1_index       = kgraph.coordinates.addElement(User());
        int sd1_index      = kgraph.coordinates.addElement(SubDimension());
        int split1_index   = kgraph.coordinates.addElement(Split(), {u1_index}, {sd1_index});
        int linear1_index  = kgraph.coordinates.addElement(Linear());
        int flatten1_index = kgraph.coordinates.addElement(Flatten(), {sd1_index}, {linear1_index});
        int dataflow1_index
            = kgraph.coordinates.addElement(DataFlow(), {u1_index}, {linear1_index});
        int buffer1_index = kgraph.coordinates.addElement(
            rocRoller::KernelGraph::CoordinateGraph::Buffer(), {u1_index}, {linear1_index});

        int u2_index       = kgraph.coordinates.addElement(User());
        int sd2_index      = kgraph.coordinates.addElement(SubDimension());
        int split2_index   = kgraph.coordinates.addElement(Split(), {u2_index}, {sd2_index});
        int linear2_index  = kgraph.coordinates.addElement(Linear());
        int flatten2_index = kgraph.coordinates.addElement(Flatten(), {sd2_index}, {linear2_index});
        int dataflow2_index
            = kgraph.coordinates.addElement(DataFlow(), {u2_index}, {linear2_index});

        int linear3_index   = kgraph.coordinates.addElement(Linear());
        int dataflow3_index = kgraph.coordinates.addElement(
            DataFlow(), {linear1_index, linear2_index}, {linear3_index});
        int linear4_index = kgraph.coordinates.addElement(Linear());
        int dataflow4_index
            = kgraph.coordinates.addElement(DataFlow(), {linear3_index}, {linear4_index});
        int linear5i_index  = kgraph.coordinates.addElement(Linear());
        int dataflow5_index = kgraph.coordinates.addElement(
            DataFlow(), {linear3_index, linear4_index}, {linear5i_index});

        int linear5o_index = kgraph.coordinates.addElement(Linear());
        int makeoutput1_index
            = kgraph.coordinates.addElement(MakeOutput(), {linear5i_index}, {linear5o_index});
        int sd5o_index   = kgraph.coordinates.addElement(SubDimension(0));
        int split3_index = kgraph.coordinates.addElement(Split(), {linear5o_index}, {sd5o_index});
        int u5o_index    = kgraph.coordinates.addElement(User({}, ""));
        int join1_index  = kgraph.coordinates.addElement(Join(), {sd5o_index}, {u5o_index});
        int dataflow6_index
            = kgraph.coordinates.addElement(DataFlow(), {linear5i_index}, {u5o_index});

        auto yamlData = toYAML(kgraph);
        auto graph2   = rocRoller::KernelGraph::fromYAML(yamlData);
        auto yaml2    = toYAML(graph2);
        EXPECT_EQ(yamlData, yaml2);

        std::string expected = R".(
        digraph {
        "coord1"[label="User{NA}(1)"];
        "coord2"[label="SubDimension{0, NA}(2)"];
        "coord3"[label="Split(3)",shape=box];
        "coord4"[label="Linear{NA}(4)"];
        "coord5"[label="Flatten(5)",shape=box];
        "coord6"[label="DataFlow(6)",shape=box];
        "coord7"[label="Buffer(7)",shape=box];
        "coord8"[label="User{NA}(8)"];
        "coord9"[label="SubDimension{0, NA}(9)"];
        "coord10"[label="Split(10)",shape=box];
        "coord11"[label="Linear{NA}(11)"];
        "coord12"[label="Flatten(12)",shape=box];
        "coord13"[label="DataFlow(13)",shape=box];
        "coord14"[label="Linear{NA}(14)"];
        "coord15"[label="DataFlow(15)",shape=box];
        "coord16"[label="Linear{NA}(16)"];
        "coord17"[label="DataFlow(17)",shape=box];
        "coord18"[label="Linear{NA}(18)"];
        "coord19"[label="DataFlow(19)",shape=box];
        "coord20"[label="Linear{NA}(20)"];
        "coord21"[label="MakeOutput(21)",shape=box];
        "coord22"[label="SubDimension{0, NA}(22)"];
        "coord23"[label="Split(23)",shape=box];
        "coord24"[label="User{NA}(24)"];
        "coord25"[label="Join(25)",shape=box];
        "coord26"[label="DataFlow(26)",shape=box];
        "coord1" -> "coord3"
        "coord1" -> "coord6"
        "coord1" -> "coord7"
        "coord2" -> "coord5"
        "coord3" -> "coord2"
        "coord4" -> "coord15"
        "coord5" -> "coord4"
        "coord6" -> "coord4"
        "coord7" -> "coord4"
        "coord8" -> "coord10"
        "coord8" -> "coord13"
        "coord9" -> "coord12"
        "coord10" -> "coord9"
        "coord11" -> "coord15"
        "coord12" -> "coord11"
        "coord13" -> "coord11"
        "coord14" -> "coord17"
        "coord14" -> "coord19"
        "coord15" -> "coord14"
        "coord16" -> "coord19"
        "coord17" -> "coord16"
        "coord18" -> "coord21"
        "coord18" -> "coord26"
        "coord19" -> "coord18"
        "coord20" -> "coord23"
        "coord21" -> "coord20"
        "coord22" -> "coord25"
        "coord23" -> "coord22"
        "coord25" -> "coord24"
        "coord26" -> "coord24"
        {
            rank=same
            "coord4"->"coord11"[style=invis]
            rankdir=LR
        }
        {
            rank=same
            "coord14"->"coord16"[style=invis]
            rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadLinear Value: Float(2)"];
        "cntrl3"[label="LoadLinear Value: Float(3)"];
        "cntrl4"[label="Body(4)",shape=box];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="Assign VGPR 5:I(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="Assign VGPR 7:I(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl11"[label="Assign VGPR 9:I(11)"];
        "cntrl12"[label="Sequence(12)",shape=box];
        "cntrl13"[label="Sequence(13)",shape=box];
        "cntrl14"[label="StoreLinear(14)"];
        "cntrl15"[label="Sequence(15)",shape=box];
        "cntrl1" -> "cntrl4"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl3" -> "cntrl8"
        "cntrl4" -> "cntrl2"
        "cntrl5" -> "cntrl3"
        "cntrl6" -> "cntrl10"
        "cntrl6" -> "cntrl12"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl6"
        "cntrl9" -> "cntrl13"
        "cntrl10" -> "cntrl9"
        "cntrl11" -> "cntrl15"
        "cntrl12" -> "cntrl11"
        "cntrl13" -> "cntrl11"
        "cntrl15" -> "cntrl14"
        }
            }
        ).";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(kgraph.toDOT()));
    }

    TEST_F(KernelGraphTest, UpdateParamsTMul)
    {
        auto command = std::make_shared<Command>();

        auto tagTensorA
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // A
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // B
        auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagStoreD
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B

        auto tagTensorD
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // D
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

        auto kgraph0 = translate(command);

        std::string expected0 = R".(
        digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
        "coord3"[label="SubDimension{1, CommandArgument(Tensor_0_size_1)I64}(3)"];
        "coord4"[label="MacroTile{NA}(2/None/None){}-()(4)"];
        "coord5"[label="Split(5)",shape=box];
        "coord6"[label="ConstructMacroTile(6)",shape=box];
        "coord7"[label="DataFlow(7)",shape=box];
        "coord8"[label="User{CommandArgument(Tensor_2_extent)I64}(8)"];
        "coord9"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="SubDimension{1, CommandArgument(Tensor_2_size_1)I64}(10)"];
        "coord11"[label="MacroTile{NA}(2/None/None){}-()(11)"];
        "coord12"[label="Split(12)",shape=box];
        "coord13"[label="ConstructMacroTile(13)",shape=box];
        "coord14"[label="DataFlow(14)",shape=box];
        "coord15"[label="MacroTile{NA}(0/None/None){}-()(15)"];
        "coord16"[label="DataFlow(16)",shape=box];
        "coord17"[label="SubDimension{0, NA}(17)"];
        "coord18"[label="SubDimension{1, NA}(18)"];
        "coord19"[label="User{CommandArgument(Tensor_5_extent)I64}(19)"];
        "coord20"[label="DestructMacroTile(20)",shape=box];
        "coord21"[label="Join(21)",shape=box];
        "coord22"[label="DataFlow(22)",shape=box];
        "coord1" -> "coord5"
        "coord1" -> "coord7"
        "coord2" -> "coord6"
        "coord3" -> "coord6"
        "coord4" -> "coord16"
        "coord5" -> "coord2"
        "coord5" -> "coord3"
        "coord6" -> "coord4"
        "coord7" -> "coord4"
        "coord8" -> "coord12"
        "coord8" -> "coord14"
        "coord9" -> "coord13"
        "coord10" -> "coord13"
        "coord11" -> "coord16"
        "coord12" -> "coord9"
        "coord12" -> "coord10"
        "coord13" -> "coord11"
        "coord14" -> "coord11"
        "coord15" -> "coord20"
        "coord15" -> "coord22"
        "coord16" -> "coord15"
        "coord17" -> "coord21"
        "coord18" -> "coord21"
        "coord20" -> "coord17"
        "coord20" -> "coord18"
        "coord21" -> "coord19"
        "coord22" -> "coord19"
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord4"->"coord11"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadTiled Value: Float(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadTiled Value: Float(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="TensorContraction(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="StoreTiled Value: Float(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl8"
        "cntrl5" -> "cntrl4"
        "cntrl6" -> "cntrl10"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl6"
        "cntrl10" -> "cntrl9"
        }
        }
    ).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT()));

        // macro tile sizes
        int mac_m = 64;
        int mac_n = 64;
        int mac_k = 64;

        auto macTileA = MacroTile({mac_m, mac_k}, MemoryType::VGPR); // A
        auto macTileB = MacroTile({mac_k, mac_n}, MemoryType::VGPR); // B

        auto params = std::make_shared<CommandParameters>();

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);

        kgraph0 = kgraph0.transform(updateParametersTransform);

        std::string expected1 = R".(
        digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
        "coord3"[label="SubDimension{1, CommandArgument(Tensor_0_size_1)I64}(3)"];
        "coord4"[label="MacroTile{NA}(2/VGPR/None){64,64}-()(4)"];
        "coord5"[label="Split(5)",shape=box];
        "coord6"[label="ConstructMacroTile(6)",shape=box];
        "coord7"[label="DataFlow(7)",shape=box];
        "coord8"[label="User{CommandArgument(Tensor_2_extent)I64}(8)"];
        "coord9"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="SubDimension{1, CommandArgument(Tensor_2_size_1)I64}(10)"];
        "coord11"[label="MacroTile{NA}(2/VGPR/None){64,64}-()(11)"];
        "coord12"[label="Split(12)",shape=box];
        "coord13"[label="ConstructMacroTile(13)",shape=box];
        "coord14"[label="DataFlow(14)",shape=box];
        "coord15"[label="MacroTile{NA}(2/WAVE/MATRIX_ACCUMULATOR){64,64}-()(15)"];
        "coord16"[label="DataFlow(16)",shape=box];
        "coord17"[label="SubDimension{0, NA}(17)"];
        "coord18"[label="SubDimension{1, NA}(18)"];
        "coord19"[label="User{CommandArgument(Tensor_5_extent)I64}(19)"];
        "coord20"[label="DestructMacroTile(20)",shape=box];
        "coord21"[label="Join(21)",shape=box];
        "coord22"[label="DataFlow(22)",shape=box];
        "coord1" -> "coord5"
        "coord1" -> "coord7"
        "coord2" -> "coord6"
        "coord3" -> "coord6"
        "coord4" -> "coord16"
        "coord5" -> "coord2"
        "coord5" -> "coord3"
        "coord6" -> "coord4"
        "coord7" -> "coord4"
        "coord8" -> "coord12"
        "coord8" -> "coord14"
        "coord9" -> "coord13"
        "coord10" -> "coord13"
        "coord11" -> "coord16"
        "coord12" -> "coord9"
        "coord12" -> "coord10"
        "coord13" -> "coord11"
        "coord14" -> "coord11"
        "coord15" -> "coord20"
        "coord15" -> "coord22"
        "coord16" -> "coord15"
        "coord17" -> "coord21"
        "coord18" -> "coord21"
        "coord20" -> "coord17"
        "coord20" -> "coord18"
        "coord21" -> "coord19"
        "coord22" -> "coord19"
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord4"->"coord11"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadTiled Value: Float(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadTiled Value: Float(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="TensorContraction(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="StoreTiled Value: Float(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl8"
        "cntrl5" -> "cntrl4"
        "cntrl6" -> "cntrl10"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl6"
        "cntrl10" -> "cntrl9"
        }
        }
    ).";

        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph0.toDOT()));
    }

    TEST_F(KernelGraphTest, WaitZero)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto kernel = kgraph.control.addElement(Kernel());
        auto wait   = kgraph.control.addElement(WaitZero());
        kgraph.control.addElement(Body(), {kernel}, {wait});

        kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        EXPECT_THAT(output(), testing::HasSubstr("s_waitcnt"));

        EXPECT_THAT(output(), testing::HasSubstr("vmcnt(0)"));
        EXPECT_THAT(output(), testing::HasSubstr("lgkmcnt(0)"));
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasExpcnt))
        {
            EXPECT_THAT(output(), testing::HasSubstr("expcnt(0)"));
        }
    }

}
