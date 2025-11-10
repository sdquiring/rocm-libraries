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
#include <rocRoller/KernelGraph/Reindexer.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <common/CommonGraphs.hpp>

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
    TEST_F(KernelGraphTest, BasicTranslateLinear)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto kgraph0 = example.getKernelGraph();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        auto visitor = rocRoller::KernelGraph::BaseGraphVisitor(m_context);
        auto kgraphC = rewrite(kgraph0, visitor);

        std::string expectedC = R".(
                digraph {
                "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
                "coord2"[label="User{CommandArgument(Tensor_2_extent)I64}(2)"];
                "coord3"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(3)"];
                "coord4"[label="Split(4)",shape=box];
                "coord5"[label="Linear{CommandArgument(Tensor_0_size_0)I64}(5)"];
                "coord6"[label="Flatten(6)",shape=box];
                "coord7"[label="DataFlow(7)",shape=box];
                "coord8"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(8)"];
                "coord9"[label="Split(9)",shape=box];
                "coord10"[label="Linear{CommandArgument(Tensor_2_size_0)I64}(10)"];
                "coord11"[label="Flatten(11)",shape=box];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord13"[label="Linear{NA}(13)"];
                "coord14"[label="DataFlow(14)",shape=box];
                "coord15"[label="Linear{NA}(15)"];
                "coord16"[label="DataFlow(16)",shape=box];
                "coord17"[label="Linear{NA}(17)"];
                "coord18"[label="DataFlow(18)",shape=box];
                "coord19"[label="SubDimension{0, NA}(19)"];
                "coord20"[label="Split(20)",shape=box];
                "coord21"[label="User{CommandArgument(Tensor_8_extent)I64}(21)"];
                "coord22"[label="Join(22)",shape=box];
                "coord23"[label="DataFlow(23)",shape=box];
                "coord1" -> "coord4"
                "coord1" -> "coord7"
                "coord2" -> "coord9"
                "coord2" -> "coord12"
                "coord3" -> "coord6"
                "coord4" -> "coord3"
                "coord5" -> "coord14"
                "coord6" -> "coord5"
                "coord7" -> "coord5"
                "coord8" -> "coord11"
                "coord9" -> "coord8"
                "coord10" -> "coord14"
                "coord11" -> "coord10"
                "coord12" -> "coord10"
                "coord13" -> "coord16"
                "coord13" -> "coord18"
                "coord14" -> "coord13"
                "coord15" -> "coord18"
                "coord16" -> "coord15"
                "coord17" -> "coord20"
                "coord17" -> "coord23"
                "coord18" -> "coord17"
                "coord19" -> "coord22"
                "coord20" -> "coord19"
                "coord22" -> "coord21"
                "coord23" -> "coord21"
                {
                rank=same
                "coord5"->"coord10"[style=invis]
                rankdir=LR
                }
                {
                rank=same
                "coord13"->"coord15"[style=invis]
                rankdir=LR
                }
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadLinear Value: Int32(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadLinear Value: Int32(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(5)NA, DataFlowTag(10)NA)NA(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(13)NA)NA(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(13)NA, DataFlowTag(15)NA)NA(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl14"[label="StoreLinear(14)"];
                "cntrl15"[label="Sequence(15)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
                "cntrl2" -> "cntrl7"
                "cntrl3" -> "cntrl2"
                "cntrl4" -> "cntrl8"
                "cntrl5" -> "cntrl4"
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
                "coord1" -> "to_cntrl_1_2"
                "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_1"
                "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
                "coord5" -> "to_cntrl_5_2"
                "to_cntrl_5_2"[label="2->5: Linear: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_5"
                "to_coord_2_5"[label="2->5: Linear: (0)", shape=cds]
                "coord2" -> "to_cntrl_2_4"
                "to_cntrl_2_4"[label="4->2: User: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_2"
                "to_coord_4_2"[label="4->2: User: (0)", shape=cds]
                "coord10" -> "to_cntrl_10_4"
                "to_cntrl_10_4"[label="4->10: Linear: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_10"
                "to_coord_4_10"[label="4->10: Linear: (0)", shape=cds]
                "coord13" -> "to_cntrl_13_6"
                "to_cntrl_13_6"[label="6->13: DEST", shape=cds]
                "cntrl6" -> "to_coord_6_13"
                "to_coord_6_13"[label="6->13: DEST", shape=cds]
                "coord15" -> "to_cntrl_15_9"
                "to_cntrl_15_9"[label="9->15: DEST", shape=cds]
                "cntrl9" -> "to_coord_9_15"
                "to_coord_9_15"[label="9->15: DEST", shape=cds]
                "coord17" -> "to_cntrl_17_11"
                "to_cntrl_17_11"[label="11->17: DEST", shape=cds]
                "cntrl11" -> "to_coord_11_17"
                "to_coord_11_17"[label="11->17: DEST", shape=cds]
                "coord17" -> "to_cntrl_17_14"
                "to_cntrl_17_14"[label="14->17: Linear: (0)", shape=cds]
                "cntrl14" -> "to_coord_14_17"
                "to_coord_14_17"[label="14->17: Linear: (0)", shape=cds]
                "coord21" -> "to_cntrl_21_14"
                "to_cntrl_21_14"[label="14->21: User: (0)", shape=cds]
                "cntrl14" -> "to_coord_14_21"
                "to_coord_14_21"[label="14->21: User: (0)", shape=cds]
                }).";

        EXPECT_EQ(NormalizedSource(expectedC), NormalizedSource(kgraphC.toDOT(true)));

        std::string expected0 = R".(
                digraph {
                "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
                "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
                "coord3"[label="Split(3)",shape=box];
                "coord4"[label="Linear{CommandArgument(Tensor_0_size_0)I64}(4)"];
                "coord5"[label="Flatten(5)",shape=box];
                "coord6"[label="DataFlow(6)",shape=box];
                "coord7"[label="User{CommandArgument(Tensor_2_extent)I64}(7)"];
                "coord8"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(8)"];
                "coord9"[label="Split(9)",shape=box];
                "coord10"[label="Linear{CommandArgument(Tensor_2_size_0)I64}(10)"];
                "coord11"[label="Flatten(11)",shape=box];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord13"[label="Linear{NA}(13)"];
                "coord14"[label="DataFlow(14)",shape=box];
                "coord15"[label="Linear{NA}(15)"];
                "coord16"[label="DataFlow(16)",shape=box];
                "coord17"[label="Linear{NA}(17)"];
                "coord18"[label="DataFlow(18)",shape=box];
                "coord19"[label="SubDimension{0, NA}(19)"];
                "coord20"[label="User{CommandArgument(Tensor_8_extent)I64}(20)"];
                "coord21"[label="Split(21)",shape=box];
                "coord22"[label="Join(22)",shape=box];
                "coord23"[label="DataFlow(23)",shape=box];
                "coord1" -> "coord3"
                "coord1" -> "coord6"
                "coord2" -> "coord5"
                "coord3" -> "coord2"
                "coord4" -> "coord14"
                "coord5" -> "coord4"
                "coord6" -> "coord4"
                "coord7" -> "coord9"
                "coord7" -> "coord12"
                "coord8" -> "coord11"
                "coord9" -> "coord8"
                "coord10" -> "coord14"
                "coord11" -> "coord10"
                "coord12" -> "coord10"
                "coord13" -> "coord16"
                "coord13" -> "coord18"
                "coord14" -> "coord13"
                "coord15" -> "coord18"
                "coord16" -> "coord15"
                "coord17" -> "coord21"
                "coord17" -> "coord23"
                "coord18" -> "coord17"
                "coord19" -> "coord22"
                "coord21" -> "coord19"
                "coord22" -> "coord20"
                "coord23" -> "coord20"
                {
                rank=same
                "coord4"->"coord10"[style=invis]
                rankdir=LR
                }
                {
                rank=same
                "coord13"->"coord15"[style=invis]
                rankdir=LR
                }
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadLinear Value: Int32(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadLinear Value: Int32(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(4)NA, DataFlowTag(10)NA)NA(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(13)NA)NA(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(13)NA, DataFlowTag(15)NA)NA(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl14"[label="StoreLinear(14)"];
                "cntrl15"[label="Sequence(15)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
                "cntrl2" -> "cntrl7"
                "cntrl3" -> "cntrl2"
                "cntrl4" -> "cntrl8"
                "cntrl5" -> "cntrl4"
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
                "coord1" -> "to_cntrl_1_2"
                "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_1"
                "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
                "coord4" -> "to_cntrl_4_2"
                "to_cntrl_4_2"[label="2->4: Linear: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_4"
                "to_coord_2_4"[label="2->4: Linear: (0)", shape=cds]
                "coord7" -> "to_cntrl_7_4"
                "to_cntrl_7_4"[label="4->7: User: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_7"
                "to_coord_4_7"[label="4->7: User: (0)", shape=cds]
                "coord10" -> "to_cntrl_10_4"
                "to_cntrl_10_4"[label="4->10: Linear: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_10"
                "to_coord_4_10"[label="4->10: Linear: (0)", shape=cds]
                "coord13" -> "to_cntrl_13_6"
                "to_cntrl_13_6"[label="6->13: DEST", shape=cds]
                "cntrl6" -> "to_coord_6_13"
                "to_coord_6_13"[label="6->13: DEST", shape=cds]
                "coord15" -> "to_cntrl_15_9"
                "to_cntrl_15_9"[label="9->15: DEST", shape=cds]
                "cntrl9" -> "to_coord_9_15"
                "to_coord_9_15"[label="9->15: DEST", shape=cds]
                "coord17" -> "to_cntrl_17_11"
                "to_cntrl_17_11"[label="11->17: DEST", shape=cds]
                "cntrl11" -> "to_coord_11_17"
                "to_coord_11_17"[label="11->17: DEST", shape=cds]
                "coord17" -> "to_cntrl_17_14"
                "to_cntrl_17_14"[label="14->17: Linear: (0)", shape=cds]
                "cntrl14" -> "to_coord_14_17"
                "to_coord_14_17"[label="14->17: Linear: (0)", shape=cds]
                "coord20" -> "to_cntrl_20_14"
                "to_cntrl_20_14"[label="14->20: User: (0)", shape=cds]
                "cntrl14" -> "to_coord_14_20"
                "to_coord_14_20"[label="14->20: User: (0)", shape=cds]
         }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));

        std::string expected1 = R".(
            digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="User{CommandArgument(Tensor_2_extent)I64}(2)"];
        "coord3"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(3)"];
        "coord4"[label="Split(4)",shape=box];
        "coord5"[label="Linear{CommandArgument(Tensor_0_size_0)I64}(5)"];
        "coord6"[label="Flatten(6)",shape=box];
        "coord7"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(7)"];
        "coord8"[label="Split(8)",shape=box];
        "coord9"[label="Linear{CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="Flatten(10)",shape=box];
        "coord11"[label="Linear{NA}(11)"];
        "coord12"[label="SubDimension{0, NA}(12)"];
        "coord13"[label="Split(13)",shape=box];
        "coord14"[label="User{CommandArgument(Tensor_8_extent)I64}(14)"];
        "coord15"[label="Join(15)",shape=box];
        "coord16"[label="VGPR{NA}(16)"];
        "coord17"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_0_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(17)"];
        "coord18"[label="Workitem{0, 64:U32}(18)"];
        "coord19"[label="Tile(19)",shape=box];
        "coord20"[label="Forget(20)",shape=box];
        "coord21"[label="DataFlow(21)",shape=box];
        "coord22"[label="VGPR{NA}(22)"];
        "coord23"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_2_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(23)"];
        "coord24"[label="Workitem{0, 64:U32}(24)"];
        "coord25"[label="Tile(25)",shape=box];
        "coord26"[label="Forget(26)",shape=box];
        "coord27"[label="DataFlow(27)",shape=box];
        "coord28"[label="VGPR{NA}(28)"];
        "coord29"[label="DataFlow(29)",shape=box];
        "coord30"[label="VGPR{NA}(30)"];
        "coord31"[label="DataFlow(31)",shape=box];
        "coord32"[label="VGPR{NA}(32)"];
        "coord33"[label="DataFlow(33)",shape=box];
        "coord34"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_8_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(34)"];
        "coord35"[label="Workitem{0, 64:U32}(35)"];
        "coord36"[label="Inherit(36)",shape=box];
        "coord37"[label="Flatten(37)",shape=box];
        "coord38"[label="DataFlow(38)",shape=box];
        "coord1" -> "coord4"
        "coord1" -> "coord21"
        "coord2" -> "coord8"
        "coord2" -> "coord27"
        "coord3" -> "coord6"
        "coord4" -> "coord3"
        "coord5" -> "coord19"
        "coord6" -> "coord5"
        "coord7" -> "coord10"
        "coord8" -> "coord7"
        "coord9" -> "coord25"
        "coord10" -> "coord9"
        "coord11" -> "coord13"
        "coord12" -> "coord15"
        "coord13" -> "coord12"
        "coord15" -> "coord14"
        "coord16" -> "coord29"
        "coord17" -> "coord20"
        "coord18" -> "coord20"
        "coord19" -> "coord17"
        "coord19" -> "coord18"
        "coord20" -> "coord16"
        "coord21" -> "coord16"
        "coord22" -> "coord29"
        "coord23" -> "coord26"
        "coord24" -> "coord26"
        "coord25" -> "coord23"
        "coord25" -> "coord24"
        "coord26" -> "coord22"
        "coord27" -> "coord22"
        "coord28" -> "coord31"
        "coord28" -> "coord33"
        "coord29" -> "coord28"
        "coord30" -> "coord33"
        "coord31" -> "coord30"
        "coord32" -> "coord36"
        "coord32" -> "coord38"
        "coord33" -> "coord32"
        "coord34" -> "coord37"
        "coord35" -> "coord37"
        "coord36" -> "coord34"
        "coord36" -> "coord35"
        "coord37" -> "coord11"
        "coord38" -> "coord14"
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
        {
        rank=same
        "coord23"->"coord24"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord23"->"coord24"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord16"->"coord22"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord28"->"coord30"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord34"->"coord35"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord34"->"coord35"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadVGPR Value: Int32(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadVGPR Value: Int32(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="Assign VGPR Add(DataFlowTag(16)NA, DataFlowTag(22)NA)NA(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="Assign VGPR Negate(DataFlowTag(28)NA)NA(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(28)NA, DataFlowTag(30)NA)NA(11)"];
        "cntrl12"[label="Sequence(12)",shape=box];
        "cntrl13"[label="Sequence(13)",shape=box];
        "cntrl14"[label="StoreVGPR(14)"];
        "cntrl15"[label="Sequence(15)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl8"
        "cntrl5" -> "cntrl4"
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
        "coord1" -> "to_cntrl_1_2"
        "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_1"
        "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
        "coord16" -> "to_cntrl_16_2"
        "to_cntrl_16_2"[label="2->16: VGPR: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_16"
        "to_coord_2_16"[label="2->16: VGPR: (0)", shape=cds]
        "coord2" -> "to_cntrl_2_4"
        "to_cntrl_2_4"[label="4->2: User: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_2"
        "to_coord_4_2"[label="4->2: User: (0)", shape=cds]
        "coord22" -> "to_cntrl_22_4"
        "to_cntrl_22_4"[label="4->22: VGPR: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_22"
        "to_coord_4_22"[label="4->22: VGPR: (0)", shape=cds]
        "coord28" -> "to_cntrl_28_6"
        "to_cntrl_28_6"[label="6->28: DEST", shape=cds]
        "cntrl6" -> "to_coord_6_28"
        "to_coord_6_28"[label="6->28: DEST", shape=cds]
        "coord30" -> "to_cntrl_30_9"
        "to_cntrl_30_9"[label="9->30: DEST", shape=cds]
        "cntrl9" -> "to_coord_9_30"
        "to_coord_9_30"[label="9->30: DEST", shape=cds]
        "coord32" -> "to_cntrl_32_11"
        "to_cntrl_32_11"[label="11->32: DEST", shape=cds]
        "cntrl11" -> "to_coord_11_32"
        "to_coord_11_32"[label="11->32: DEST", shape=cds]
        "coord14" -> "to_cntrl_14_14"
        "to_cntrl_14_14"[label="14->14: User: (0)", shape=cds]
        "cntrl14" -> "to_coord_14_14"
        "to_coord_14_14"[label="14->14: User: (0)", shape=cds]
        "coord32" -> "to_cntrl_32_14"
        "to_cntrl_32_14"[label="14->32: VGPR: (0)", shape=cds]
        "cntrl14" -> "to_coord_14_32"
        "to_coord_14_32"[label="14->32: VGPR: (0)", shape=cds]
        }).";

        auto one = Expression::literal(1u);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto lowerLinearTransform = std::make_shared<LowerLinear>(m_context);

        auto kgraph1 = kgraph0.transform(lowerLinearTransform);
        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT(true)));

        std::string expected2 = R".(
        digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="User{CommandArgument(Tensor_2_extent)I64}(2)"];
        "coord3"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(3)"];
        "coord4"[label="Split(4)",shape=box];
        "coord5"[label="Linear{CommandArgument(Tensor_0_size_0)I64}(5)"];
        "coord6"[label="Flatten(6)",shape=box];
        "coord7"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_0_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(7)"];
        "coord8"[label="Workitem{0, 64:U32}(8)"];
        "coord9"[label="Tile(9)",shape=box];
        "coord10"[label="Linear{16:I}(10)"];
        "coord11"[label="ForLoop{16:I}(11)"];
        "coord12"[label="DataFlow(12)",shape=box];
        "coord13"[label="VGPR{NA}(13)"];
        "coord14"[label="Forget(14)",shape=box];
        "coord15"[label="DataFlow(15)",shape=box];
        "coord16"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(16)"];
        "coord17"[label="Split(17)",shape=box];
        "coord18"[label="Linear{CommandArgument(Tensor_2_size_0)I64}(18)"];
        "coord19"[label="Flatten(19)",shape=box];
        "coord20"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_2_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(20)"];
        "coord21"[label="Workitem{0, 64:U32}(21)"];
        "coord22"[label="Tile(22)",shape=box];
        "coord23"[label="ForLoop{16:I}(23)"];
        "coord24"[label="DataFlow(24)",shape=box];
        "coord25"[label="VGPR{NA}(25)"];
        "coord26"[label="Forget(26)",shape=box];
        "coord27"[label="DataFlow(27)",shape=box];
        "coord28"[label="VGPR{NA}(28)"];
        "coord29"[label="DataFlow(29)",shape=box];
        "coord30"[label="VGPR{NA}(30)"];
        "coord31"[label="DataFlow(31)",shape=box];
        "coord32"[label="VGPR{NA}(32)"];
        "coord33"[label="DataFlow(33)",shape=box];
        "coord34"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_8_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(34)"];
        "coord35"[label="Workitem{0, 64:U32}(35)"];
        "coord36"[label="Inherit(36)",shape=box];
        "coord37"[label="ForLoop{16:I}(37)"];
        "coord38"[label="DataFlow(38)",shape=box];
        "coord39"[label="Linear{NA}(39)"];
        "coord40"[label="Flatten(40)",shape=box];
        "coord41"[label="SubDimension{0, NA}(41)"];
        "coord42"[label="Split(42)",shape=box];
        "coord43"[label="User{CommandArgument(Tensor_8_extent)I64}(43)"];
        "coord44"[label="Join(44)",shape=box];
        "coord45"[label="DataFlow(45)",shape=box];
        "coord1" -> "coord4"
        "coord1" -> "coord15"
        "coord2" -> "coord17"
        "coord2" -> "coord27"
        "coord3" -> "coord6"
        "coord4" -> "coord3"
        "coord5" -> "coord9"
        "coord6" -> "coord5"
        "coord7" -> "coord14"
        "coord8" -> "coord14"
        "coord9" -> "coord11"
        "coord9" -> "coord7"
        "coord9" -> "coord8"
        "coord10" -> "coord12"
        "coord10" -> "coord24"
        "coord10" -> "coord38"
        "coord11" -> "coord14"
        "coord12" -> "coord11"
        "coord13" -> "coord29"
        "coord14" -> "coord13"
        "coord15" -> "coord13"
        "coord16" -> "coord19"
        "coord17" -> "coord16"
        "coord18" -> "coord22"
        "coord19" -> "coord18"
        "coord20" -> "coord26"
        "coord21" -> "coord26"
        "coord22" -> "coord23"
        "coord22" -> "coord20"
        "coord22" -> "coord21"
        "coord23" -> "coord26"
        "coord24" -> "coord23"
        "coord25" -> "coord29"
        "coord26" -> "coord25"
        "coord27" -> "coord25"
        "coord28" -> "coord31"
        "coord28" -> "coord33"
        "coord29" -> "coord28"
        "coord30" -> "coord33"
        "coord31" -> "coord30"
        "coord32" -> "coord36"
        "coord32" -> "coord45"
        "coord33" -> "coord32"
        "coord34" -> "coord40"
        "coord35" -> "coord40"
        "coord36" -> "coord37"
        "coord36" -> "coord34"
        "coord36" -> "coord35"
        "coord37" -> "coord40"
        "coord38" -> "coord37"
        "coord39" -> "coord42"
        "coord40" -> "coord39"
        "coord41" -> "coord44"
        "coord42" -> "coord41"
        "coord44" -> "coord43"
        "coord45" -> "coord43"
        {
        rank=same
        "coord11"->"coord7"->"coord8"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord11"->"coord7"->"coord8"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord23"->"coord20"->"coord21"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord23"->"coord20"->"coord21"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord13"->"coord25"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord28"->"coord30"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord37"->"coord34"->"coord35"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord37"->"coord34"->"coord35"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="ForLoopOp : LessThan(DataFlowTag(10)I, 16:I)BL(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="Assign SGPR 0:I(4)"];
        "cntrl5"[label="Initialize(5)",shape=box];
        "cntrl6"[label="Assign SGPR Add(DataFlowTag(10)I, 1:I)I(6)"];
        "cntrl7"[label="ForLoopIncrement(7)",shape=box];
        "cntrl8"[label="LoadVGPR Value: Int32(8)"];
        "cntrl9"[label="Body(9)",shape=box];
        "cntrl10"[label="LoadVGPR Value: Int32(10)"];
        "cntrl11"[label="Body(11)",shape=box];
        "cntrl12"[label="Assign VGPR Add(DataFlowTag(13)NA, DataFlowTag(25)NA)NA(12)"];
        "cntrl13"[label="Sequence(13)",shape=box];
        "cntrl14"[label="Sequence(14)",shape=box];
        "cntrl15"[label="Assign VGPR Negate(DataFlowTag(28)NA)NA(15)"];
        "cntrl16"[label="Sequence(16)",shape=box];
        "cntrl17"[label="Assign VGPR Multiply(DataFlowTag(28)NA, DataFlowTag(30)NA)NA(17)"];
        "cntrl18"[label="Sequence(18)",shape=box];
        "cntrl19"[label="Sequence(19)",shape=box];
        "cntrl20"[label="StoreVGPR(20)"];
        "cntrl21"[label="Sequence(21)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl2" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl2" -> "cntrl9"
        "cntrl2" -> "cntrl11"
        "cntrl3" -> "cntrl2"
        "cntrl5" -> "cntrl4"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl13"
        "cntrl9" -> "cntrl8"
        "cntrl10" -> "cntrl14"
        "cntrl11" -> "cntrl10"
        "cntrl12" -> "cntrl16"
        "cntrl12" -> "cntrl18"
        "cntrl13" -> "cntrl12"
        "cntrl14" -> "cntrl12"
        "cntrl15" -> "cntrl19"
        "cntrl16" -> "cntrl15"
        "cntrl17" -> "cntrl21"
        "cntrl18" -> "cntrl17"
        "cntrl19" -> "cntrl17"
        "cntrl21" -> "cntrl20"
        }
        "coord10" -> "to_cntrl_10_2"
        "to_cntrl_10_2"[label="2->10: ForLoop: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_10"
        "to_coord_2_10"[label="2->10: ForLoop: (0)", shape=cds]
        "coord10" -> "to_cntrl_10_4"
        "to_cntrl_10_4"[label="4->10: DEST", shape=cds]
        "cntrl4" -> "to_coord_4_10"
        "to_coord_4_10"[label="4->10: DEST", shape=cds]
        "coord10" -> "to_cntrl_10_6"
        "to_cntrl_10_6"[label="6->10: DEST", shape=cds]
        "cntrl6" -> "to_coord_6_10"
        "to_coord_6_10"[label="6->10: DEST", shape=cds]
        "coord1" -> "to_cntrl_1_8"
        "to_cntrl_1_8"[label="8->1: User: (0)", shape=cds]
        "cntrl8" -> "to_coord_8_1"
        "to_coord_8_1"[label="8->1: User: (0)", shape=cds]
        "coord13" -> "to_cntrl_13_8"
        "to_cntrl_13_8"[label="8->13: VGPR: (0)", shape=cds]
        "cntrl8" -> "to_coord_8_13"
        "to_coord_8_13"[label="8->13: VGPR: (0)", shape=cds]
        "coord2" -> "to_cntrl_2_10"
        "to_cntrl_2_10"[label="10->2: User: (0)", shape=cds]
        "cntrl10" -> "to_coord_10_2"
        "to_coord_10_2"[label="10->2: User: (0)", shape=cds]
        "coord25" -> "to_cntrl_25_10"
        "to_cntrl_25_10"[label="10->25: VGPR: (0)", shape=cds]
        "cntrl10" -> "to_coord_10_25"
        "to_coord_10_25"[label="10->25: VGPR: (0)", shape=cds]
        "coord28" -> "to_cntrl_28_12"
        "to_cntrl_28_12"[label="12->28: DEST", shape=cds]
        "cntrl12" -> "to_coord_12_28"
        "to_coord_12_28"[label="12->28: DEST", shape=cds]
        "coord30" -> "to_cntrl_30_15"
        "to_cntrl_30_15"[label="15->30: DEST", shape=cds]
        "cntrl15" -> "to_coord_15_30"
        "to_coord_15_30"[label="15->30: DEST", shape=cds]
        "coord32" -> "to_cntrl_32_17"
        "to_cntrl_32_17"[label="17->32: DEST", shape=cds]
        "cntrl17" -> "to_coord_17_32"
        "to_coord_17_32"[label="17->32: DEST", shape=cds]
        "coord32" -> "to_cntrl_32_20"
        "to_cntrl_32_20"[label="20->32: VGPR: (0)", shape=cds]
        "cntrl20" -> "to_coord_20_32"
        "to_coord_20_32"[label="20->32: VGPR: (0)", shape=cds]
        "coord43" -> "to_cntrl_43_20"
        "to_cntrl_43_20"[label="20->43: User: (0)", shape=cds]
        "cntrl20" -> "to_coord_20_43"
        "to_coord_20_43"[label="20->43: User: (0)", shape=cds]
        }).";

        int  loopSize     = 16;
        auto loopSizeExpr = Expression::literal(loopSize);

        auto lowerLinerLoopTransform = std::make_shared<LowerLinearLoop>(loopSizeExpr, m_context);

        auto kgraph2 = kgraph1.transform(lowerLinerLoopTransform);
        EXPECT_EQ(NormalizedSource(expected2), NormalizedSource(kgraph2.toDOT(true)));
    }

    TEST_F(KernelGraphTest, BasicTranslateScalar)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>(true);
        auto kgraph0 = example.getKernelGraph();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        std::string expected0 = R".(
                digraph {
                "coord1"[label="User{NA}(1)"];
                "coord2"[label="VGPR{NA}(2)"];
                "coord3"[label="DataFlow(3)",shape=box];
                "coord4"[label="User{NA}(4)"];
                "coord5"[label="VGPR{NA}(5)"];
                "coord6"[label="DataFlow(6)",shape=box];
                "coord7"[label="VGPR{NA}(7)"];
                "coord8"[label="DataFlow(8)",shape=box];
                "coord9"[label="VGPR{NA}(9)"];
                "coord10"[label="DataFlow(10)",shape=box];
                "coord11"[label="VGPR{NA}(11)"];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord1" -> "coord3"
                "coord2" -> "coord8"
                "coord3" -> "coord2"
                "coord4" -> "coord6"
                "coord5" -> "coord8"
                "coord6" -> "coord5"
                "coord7" -> "coord10"
                "coord7" -> "coord12"
                "coord8" -> "coord7"
                "coord9" -> "coord12"
                "coord10" -> "coord9"
                "coord12" -> "coord11"
                {
                rank=same
                "coord2"->"coord5"[style=invis]
                rankdir=LR
                }
                {
                rank=same
                "coord7"->"coord9"[style=invis]
                rankdir=LR
                }
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadVGPR Value: Int32(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadVGPR Value: Int32(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(2)NA, DataFlowTag(5)NA)NA(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(7)NA)NA(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(7)NA, DataFlowTag(9)NA)NA(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
                "cntrl2" -> "cntrl7"
                "cntrl3" -> "cntrl2"
                "cntrl4" -> "cntrl8"
                "cntrl5" -> "cntrl4"
                "cntrl6" -> "cntrl10"
                "cntrl6" -> "cntrl12"
                "cntrl7" -> "cntrl6"
                "cntrl8" -> "cntrl6"
                "cntrl9" -> "cntrl13"
                "cntrl10" -> "cntrl9"
                "cntrl12" -> "cntrl11"
                "cntrl13" -> "cntrl11"
                }
                "coord1" -> "to_cntrl_1_2"
                "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_1"
                "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
                "coord2" -> "to_cntrl_2_2"
                "to_cntrl_2_2"[label="2->2: VGPR: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_2"
                "to_coord_2_2"[label="2->2: VGPR: (0)", shape=cds]
                "coord4" -> "to_cntrl_4_4"
                "to_cntrl_4_4"[label="4->4: User: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_4"
                "to_coord_4_4"[label="4->4: User: (0)", shape=cds]
                "coord5" -> "to_cntrl_5_4"
                "to_cntrl_5_4"[label="4->5: VGPR: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_5"
                "to_coord_4_5"[label="4->5: VGPR: (0)", shape=cds]
                "coord7" -> "to_cntrl_7_6"
                "to_cntrl_7_6"[label="6->7: DEST", shape=cds]
                "cntrl6" -> "to_coord_6_7"
                "to_coord_6_7"[label="6->7: DEST", shape=cds]
                "coord9" -> "to_cntrl_9_9"
                "to_cntrl_9_9"[label="9->9: DEST", shape=cds]
                "cntrl9" -> "to_coord_9_9"
                "to_coord_9_9"[label="9->9: DEST", shape=cds]
                "coord11" -> "to_cntrl_11_11"
                "to_cntrl_11_11"[label="11->11: DEST", shape=cds]
                "cntrl11" -> "to_coord_11_11"
                "to_coord_11_11"[label="11->11: DEST", shape=cds]
             }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));
    }

    TEST_F(KernelGraphTest, TranslateMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::Int32);
        auto kgraph0 = example.getKernelGraph();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

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
        "cntrl2"[label="LoadTiled Value: Int32(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadTiled Value: Int32(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="TensorContraction(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="StoreTiled Value: Int32(9)"];
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
        "coord1" -> "to_cntrl_1_2"
        "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_1"
        "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
        "coord4" -> "to_cntrl_4_2"
        "to_cntrl_4_2"[label="2->4: MacroTile: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_4"
        "to_coord_2_4"[label="2->4: MacroTile: (0)", shape=cds]
        "coord8" -> "to_cntrl_8_4"
        "to_cntrl_8_4"[label="4->8: User: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_8"
        "to_coord_4_8"[label="4->8: User: (0)", shape=cds]
        "coord11" -> "to_cntrl_11_4"
        "to_cntrl_11_4"[label="4->11: MacroTile: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_11"
        "to_coord_4_11"[label="4->11: MacroTile: (0)", shape=cds]
        "coord4" -> "to_cntrl_4_6"
        "to_cntrl_4_6"[label="6->4: LHS", shape=cds]
        "cntrl6" -> "to_coord_6_4"
        "to_coord_6_4"[label="6->4: LHS", shape=cds]
        "coord11" -> "to_cntrl_11_6"
        "to_cntrl_11_6"[label="6->11: RHS", shape=cds]
        "cntrl6" -> "to_coord_6_11"
        "to_coord_6_11"[label="6->11: RHS", shape=cds]
        "coord15" -> "to_cntrl_15_6"
        "to_cntrl_15_6"[label="6->15: DEST", shape=cds]
        "cntrl6" -> "to_coord_6_15"
        "to_coord_6_15"[label="6->15: DEST", shape=cds]
        "coord15" -> "to_cntrl_15_9"
        "to_cntrl_15_9"[label="9->15: MacroTile: (0)", shape=cds]
        "cntrl9" -> "to_coord_9_15"
        "to_coord_9_15"[label="9->15: MacroTile: (0)", shape=cds]
        "coord19" -> "to_cntrl_19_9"
        "to_cntrl_19_9"[label="9->19: User: (0)", shape=cds]
        "cntrl9" -> "to_coord_9_19"
        "to_coord_9_19"[label="9->19: User: (0)", shape=cds]
        }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));
    }

    TEST_F(KernelGraphTest, TranslateUnscaledMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::FP8,
                                                             DataType::FP6,
                                                             DataType::Float,
                                                             Operations::ScaleMode::None,
                                                             Operations::ScaleMode::None);

        auto expectedCommand = R".(
        Tensor.FP8.d2 0, (base=&0, lim=&8, sizes={&16 &24 }, strides={&32 &40 })
        T_LOAD_TILED 1 Source 0
        Tensor.FP6.d2 2, (base=&48, lim=&56, sizes={&64 &72 }, strides={&80 &88 })
        T_LOAD_TILED 3 Source 2
        T_Mul 1 3 Value: Float
        Tensor.Float.d2 5, (base=&96, lim=&104, sizes={&112 &120 }, strides={&128 &136 })
        T_STORE_TILED 6 Source 4 Dest 5
        ).";

        EXPECT_EQ(NormalizedSource(expectedCommand),
                  NormalizedSource(example.getCommand()->toString()));

        auto kgraph0 = example.getKernelGraph();

        auto roots = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(roots.size(), 2);
        for(auto const& id : roots)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto leaves = kgraph0.coordinates.leaves().to<std::vector>();
        ASSERT_EQ(leaves.size(), 1);
        for(auto const& id : leaves)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto reachableFromLeaf
            = kgraph0.coordinates.depthFirstVisit(leaves[0], Graph::Direction::Upstream)
                  .to<std::set>();

        for(auto id : roots)
        {
            EXPECT_TRUE(reachableFromLeaf.contains(id)) << ShowValue(id) << reachableFromLeaf;
        }

        // There should be 4 LoadTiled nodes with direct sequence edges to the single TensorContraction node.
        // These should be all of the LoadTiled nodes in the graph.
        auto is_tc = kgraph0.control.isElemType<TensorContraction>();
        auto tc_id = only(kgraph0.control.findElements(is_tc));
        ASSERT_TRUE(tc_id.has_value());

        auto is_loadTiled  = kgraph0.control.isElemType<LoadTiled>();
        auto loadTiled_ids = kgraph0.control.findElements(is_loadTiled).to<std::set>();

        auto tc_parent_ids = kgraph0.control.parentNodes(*tc_id).to<std::set>();
        EXPECT_EQ(loadTiled_ids, tc_parent_ids);
        EXPECT_EQ(2, loadTiled_ids.size());
    }

    TEST_F(KernelGraphTest, TranslateScaledMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::FP8,
                                                             DataType::FP6,
                                                             DataType::Float,
                                                             Operations::ScaleMode::Separate,
                                                             Operations::ScaleMode::Separate);

        auto expectedCommand = R".(
        Tensor.FP8.d2 0, (base=&0, lim=&8, sizes={&16 &24 }, strides={&32 &40 })
        T_LOAD_TILED 1 Source 0
        Tensor.E8M0.d2 2, (base=&48, lim=&56, sizes={&64 &72 }, strides={&80 &88 })
        T_LOAD_TILED 3 Source 2
        BlockScale(Separate, {1, 32}): Data: 1, Scale: 3
        Tensor.FP6.d2 5, (base=&96, lim=&104, sizes={&112 &120 }, strides={&128 &136 })
        T_LOAD_TILED 6 Source 5
        Tensor.E8M0.d2 7, (base=&144, lim=&152, sizes={&160 &168 }, strides={&176 &184 })
        T_LOAD_TILED 8 Source 7
        BlockScale(Separate, {32, 1}): Data: 6, Scale: 8
        T_Mul 4 9 Value: Float
        Tensor.Float.d2 11, (base=&192, lim=&200, sizes={&208 &216 }, strides={&224 &232 })
        T_STORE_TILED 12 Source 10 Dest 11
        ).";

        EXPECT_EQ(NormalizedSource(expectedCommand),
                  NormalizedSource(example.getCommand()->toString()));

        auto kgraph0 = example.getKernelGraph();

        auto roots = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(roots.size(), 4);
        for(auto const& id : roots)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto leaves = kgraph0.coordinates.leaves().to<std::vector>();
        ASSERT_EQ(leaves.size(), 1);
        for(auto const& id : leaves)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto reachableFromLeaf
            = kgraph0.coordinates.depthFirstVisit(leaves[0], Graph::Direction::Upstream)
                  .to<std::set>();

        for(auto id : roots)
        {
            EXPECT_TRUE(reachableFromLeaf.contains(id)) << ShowValue(id) << reachableFromLeaf;
        }

        // There should be 4 LoadTiled nodes with direct sequence edges to the single TensorContraction node.
        // These should be all of the LoadTiled nodes in the graph.
        auto is_tc = kgraph0.control.isElemType<TensorContraction>();
        auto tc_id = only(kgraph0.control.findElements(is_tc));
        ASSERT_TRUE(tc_id.has_value());

        auto is_loadTiled  = kgraph0.control.isElemType<LoadTiled>();
        auto loadTiled_ids = kgraph0.control.findElements(is_loadTiled).to<std::set>();

        auto tc_parent_ids = kgraph0.control.parentNodes(*tc_id).to<std::set>();
        EXPECT_EQ(loadTiled_ids, tc_parent_ids);
        EXPECT_EQ(4, loadTiled_ids.size());
    }

    TEST_F(KernelGraphTest, Translate02)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto command = example.getCommand();

        auto one = Expression::literal(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto lowerLinearTransform = std::make_shared<LowerLinear>(m_context);

        auto kgraph0 = translate(command);
        auto kgraph1 = kgraph0.transform(lowerLinearTransform);

        auto user0   = 1;
        auto subdim0 = 3;
        auto block0  = 17;
        auto thread0 = 18;

        // given block id and thread id, compute regular (user) index for first (0) dataflow array
        auto block_id  = Expression::literal(2);
        auto thread_id = Expression::literal(33);

        auto exprs = kgraph1.coordinates.reverse({block_id, thread_id}, {user0}, {block0, thread0});
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(2:I, 64:U32)U32, 33:I)U32}, "
                  "CommandArgument(Tensor_0_stride_0)I64)I64}");

        auto stride = getStride(kgraph1.coordinates.getNode(subdim0));
        auto fa     = fastArith(stride);

        exprs = kgraph1.coordinates.reverse({block_id, thread_id}, {user0}, {block0, thread0});
        sexpr = Expression::toString(fastArith(exprs[0]));
        EXPECT_EQ(sexpr, "{Split: Multiply(161:U32, Tensor_0_stride_0_0:I64)I64}");
    }
}
