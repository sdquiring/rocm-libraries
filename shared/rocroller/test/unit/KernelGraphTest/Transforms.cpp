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
#include <rocRoller/KernelGraph/Utils.hpp>
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
    TEST_F(KernelGraphTest, LowerTensor)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        int macK  = 16;
        int waveK = 8;

        example.setTileSize(128, 256, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, false, false);

        auto kgraph0 = example.getKernelGraph();
        auto params  = example.getCommandParameters();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);
        auto addLDSTransform           = std::make_shared<AddLDS>(params, m_context);
        auto lowerTileTransform        = std::make_shared<LowerTile>(params, m_context);
        auto lowerTensorContractionTransform
            = std::make_shared<LowerTensorContraction>(params, m_context);
        auto unrollLoopsTransform      = std::make_shared<UnrollLoops>(params, m_context);
        auto fuseLoopsTransform        = std::make_shared<FuseLoops>();
        auto removeDuplicatesTransform = std::make_shared<RemoveDuplicates>();

        auto cleanLoopsTransform      = std::make_shared<CleanLoops>();
        auto addComputeIndexTransform = std::make_shared<AddComputeIndex>();

        kgraph0      = kgraph0.transform(updateParametersTransform);
        auto kgraph1 = kgraph0.transform(addLDSTransform);
        kgraph1      = kgraph1.transform(lowerTileTransform);
        kgraph1      = kgraph1.transform(lowerTensorContractionTransform);

        // Verify the number of Multiply nodes in the graph after lowerTile
        auto multiplyNodes = kgraph1.control.getNodes<Multiply>().to<std::vector>();
        EXPECT_EQ(multiplyNodes.size(), macK / waveK);

        // Verify number of loads
        auto loads = kgraph0.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(loads.size(), 3); // A, B, C

        loads = kgraph1.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(loads.size(), 4); // 1 for A, 2 for B (no LDS), 1 for C

        loads = kgraph1.control.getNodes<LoadLDSTile>().to<std::vector>();
        EXPECT_EQ(loads.size(), 2); // 2 for A

        auto forLoops = kgraph1.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(forLoops.size(), 5); // main: X, Y, K; epilogue: X, Y

        auto kgraphUnrolled = kgraph1.transform(unrollLoopsTransform);

        // Verify that loops have been unrolled
        auto unrolledForLoops = kgraphUnrolled.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(unrolledForLoops.size(), 14);

        auto kgraphFused = kgraphUnrolled.transform(fuseLoopsTransform);
        kgraphFused      = kgraphFused.transform(removeDuplicatesTransform);

        // Verify that loops have been fused
        auto fusedForLoops = kgraphFused.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(fusedForLoops.size(), 5);

        auto fusedLoads = kgraphFused.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(fusedLoads.size(), 17);

        // Verify that single iteration loops have been removed.
        auto kgraphClean     = kgraphFused.transform(cleanLoopsTransform);
        auto cleanedForLoops = kgraphClean.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(cleanedForLoops.size(), 1);

        // Verify that there is only a single StoreLDSTile node per K loop
        auto unrolled_kgraph_lds = kgraphUnrolled.transform(addLDSTransform);
        auto unrolledStoreLDS
            = unrolled_kgraph_lds.control.getNodes<StoreLDSTile>().to<std::vector>();
        auto kloops = kgraphUnrolled.control.getNodes<ForLoopOp>()
                          .filter([&](int node) {
                              auto loop = kgraphUnrolled.control.getNode<ForLoopOp>(node);
                              return loop.loopName == KLOOP;
                          })
                          .to<std::vector>();
        EXPECT_EQ(unrolledStoreLDS.size(), kloops.size());

        // Verify number of ComputeIndexes: A loads; A LDS loads; B loads; C load; D
        // store: 3 + (2+2) + 3 + 3 + 3 = 12
        kgraph1             = kgraph1.transform(addComputeIndexTransform);
        auto computeIndexes = kgraph1.control.getNodes<ComputeIndex>().to<std::vector>();
        EXPECT_EQ(computeIndexes.size(), 16);

        // Verify number of deallocated dimensions.  They may be merged into fewer deallocate nodes.
        auto addDeallocate = std::make_shared<AddDeallocateDataFlow>();
        auto kgraph2       = kgraph1.transform(addDeallocate);
        {
            std::set<int> deallocatedDims;
            auto          deallocates = kgraph2.control.getNodes<Deallocate>();
            for(auto deallocate : deallocates)
            {
                auto connections = kgraph2.mapper.getConnections(deallocate);
                for(auto const& c : connections)
                {
                    EXPECT_THAT(deallocatedDims, ::testing::Not(::testing::Contains(c.coordinate)));
                    deallocatedDims.insert(c.coordinate);
                }
            }
            EXPECT_EQ(deallocatedDims.size(), 48);
        }

        auto storeLDS = kgraphUnrolled.control.getNodes<StoreLDSTile>().to<std::vector>();
        EXPECT_EQ(storeLDS.size(), 8);

        auto fusedStoreLDS = kgraphFused.control.getNodes<StoreLDSTile>().to<std::vector>();
        EXPECT_EQ(fusedStoreLDS.size(), 1);

        // Verify number of ComputeIndexes after unroll/fuse/lds
        unrolled_kgraph_lds = unrolled_kgraph_lds.transform(addComputeIndexTransform);
        computeIndexes = unrolled_kgraph_lds.control.getNodes<ComputeIndex>().to<std::vector>();
        EXPECT_EQ(computeIndexes.size(), 112);

        // Verify number of deallocated dimensions.  They may be merged into fewer deallocate nodes.
        unrolled_kgraph_lds = unrolled_kgraph_lds.transform(addDeallocate);
        {
            std::set<int> deallocatedDims;
            auto          deallocates = unrolled_kgraph_lds.control.getNodes<Deallocate>();
            for(auto deallocate : deallocates)
            {
                auto connections = unrolled_kgraph_lds.mapper.getConnections(deallocate);
                for(auto const& c : connections)
                {
                    EXPECT_THAT(deallocatedDims, ::testing::Not(::testing::Contains(c.coordinate)));
                    deallocatedDims.insert(c.coordinate);
                }
            }
            EXPECT_EQ(deallocatedDims.size(), 290);
        }
    }

    TEST_F(KernelGraphTest, InlineIncrement)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        example.setTileSize(128, 256, 8);
        example.setMFMA(32, 32, 2, 1);
        example.setUseLDS(true, true, true);

        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);
        auto addLDSTransform           = std::make_shared<AddLDS>(params, m_context);
        auto lowerLinearTransform      = std::make_shared<LowerLinear>(m_context);
        auto lowerTileTransform        = std::make_shared<LowerTile>(params, m_context);
        auto lowerTensorContractionTransform
            = std::make_shared<LowerTensorContraction>(params, m_context);
        auto unrollLoopsTransform        = std::make_shared<UnrollLoops>(params, m_context);
        auto cleanLoopsTransform         = std::make_shared<CleanLoops>();
        auto addComputeIndexTransform    = std::make_shared<AddComputeIndex>();
        auto inlineInrecrementsTransform = std::make_shared<InlineIncrements>();

        kgraph = kgraph.transform(updateParametersTransform);
        kgraph = kgraph.transform(addLDSTransform);
        kgraph = kgraph.transform(lowerLinearTransform);
        kgraph = kgraph.transform(lowerTileTransform);
        kgraph = kgraph.transform(lowerTensorContractionTransform);

        // Usual lowering, should be able to inline everything.
        auto kgraph1 = kgraph.transform(unrollLoopsTransform);
        kgraph1      = kgraph1.transform(cleanLoopsTransform);
        kgraph1      = kgraph1.transform(addComputeIndexTransform);

        auto pre1  = kgraph1.control.getEdges<ForLoopIncrement>().to<std::vector>();
        kgraph1    = kgraph1.transform(inlineInrecrementsTransform);
        auto post1 = kgraph1.control.getEdges<ForLoopIncrement>().to<std::vector>();

        EXPECT_TRUE(pre1.size() > 0);
        EXPECT_TRUE(post1.empty());
    }

    TEST_F(KernelGraphTest, TileAdd)
    {
        auto example = rocRollerTest::Graphs::TileDoubleAdd<int>();

        example.setTileSize(16, 8);
        example.setSubTileSize(4, 2);

        auto params  = example.getCommandParameters(512, 512);
        auto kgraph0 = example.getKernelGraph();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);

        kgraph0 = kgraph0.transform(updateParametersTransform);

        std::string expected0 = R".(
            digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
        "coord3"[label="SubDimension{1, CommandArgument(Tensor_0_size_1)I64}(3)"];
        "coord4"[label="MacroTile{NA}(2/LDS/None){16,8}-(4,2)(4)"];
        "coord5"[label="Split(5)",shape=box];
        "coord6"[label="ConstructMacroTile(6)",shape=box];
        "coord7"[label="DataFlow(7)",shape=box];
        "coord8"[label="User{CommandArgument(Tensor_2_extent)I64}(8)"];
        "coord9"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="SubDimension{1, CommandArgument(Tensor_2_size_1)I64}(10)"];
        "coord11"[label="MacroTile{NA}(2/VGPR/None){16,8}-(4,2)(11)"];
        "coord12"[label="Split(12)",shape=box];
        "coord13"[label="ConstructMacroTile(13)",shape=box];
        "coord14"[label="DataFlow(14)",shape=box];
        "coord15"[label="MacroTile{NA}(2/VGPR/None){16,8}-(4,2)(15)"];
        "coord16"[label="DataFlow(16)",shape=box];
        "coord17"[label="MacroTile{NA}(2/VGPR/None){16,8}-(4,2)(17)"];
        "coord18"[label="DataFlow(18)",shape=box];
        "coord19"[label="MacroTile{NA}(2/VGPR/None){16,8}-(4,2)(19)"];
        "coord20"[label="DataFlow(20)",shape=box];
        "coord21"[label="SubDimension{0, NA}(21)"];
        "coord22"[label="SubDimension{1, NA}(22)"];
        "coord23"[label="User{CommandArgument(Tensor_8_extent)I64}(23)"];
        "coord24"[label="DestructMacroTile(24)",shape=box];
        "coord25"[label="Join(25)",shape=box];
        "coord26"[label="DataFlow(26)",shape=box];
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
        "coord11" -> "coord18"
        "coord12" -> "coord9"
        "coord12" -> "coord10"
        "coord13" -> "coord11"
        "coord14" -> "coord11"
        "coord15" -> "coord20"
        "coord16" -> "coord15"
        "coord17" -> "coord20"
        "coord18" -> "coord17"
        "coord19" -> "coord24"
        "coord19" -> "coord26"
        "coord20" -> "coord19"
        "coord21" -> "coord25"
        "coord22" -> "coord25"
        "coord24" -> "coord21"
        "coord24" -> "coord22"
        "coord25" -> "coord23"
        "coord26" -> "coord23"
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
        "coord15"->"coord17"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord21"->"coord22"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord21"->"coord22"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadTiled Value: Int32(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadTiled Value: Int32(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="Assign VGPR Add(DataFlowTag(4)NA, DataFlowTag(4)NA)NA(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="Assign VGPR Add(DataFlowTag(11)NA, DataFlowTag(11)NA)NA(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl11"[label="Sequence(11)",shape=box];
        "cntrl12"[label="Assign VGPR Add(DataFlowTag(15)NA, DataFlowTag(17)NA)NA(12)"];
        "cntrl13"[label="Sequence(13)",shape=box];
        "cntrl14"[label="Sequence(14)",shape=box];
        "cntrl15"[label="StoreTiled Value: Int32(15)"];
        "cntrl16"[label="Sequence(16)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl2" -> "cntrl8"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl10"
        "cntrl4" -> "cntrl11"
        "cntrl5" -> "cntrl4"
        "cntrl6" -> "cntrl13"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl6"
        "cntrl9" -> "cntrl14"
        "cntrl10" -> "cntrl9"
        "cntrl11" -> "cntrl9"
        "cntrl12" -> "cntrl16"
        "cntrl13" -> "cntrl12"
        "cntrl14" -> "cntrl12"
        "cntrl16" -> "cntrl15"
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
        "coord15" -> "to_cntrl_15_6"
        "to_cntrl_15_6"[label="6->15: DEST", shape=cds]
        "cntrl6" -> "to_coord_6_15"
        "to_coord_6_15"[label="6->15: DEST", shape=cds]
        "coord17" -> "to_cntrl_17_9"
        "to_cntrl_17_9"[label="9->17: DEST", shape=cds]
        "cntrl9" -> "to_coord_9_17"
        "to_coord_9_17"[label="9->17: DEST", shape=cds]
        "coord19" -> "to_cntrl_19_12"
        "to_cntrl_19_12"[label="12->19: DEST", shape=cds]
        "cntrl12" -> "to_coord_12_19"
        "to_coord_12_19"[label="12->19: DEST", shape=cds]
        "coord19" -> "to_cntrl_19_15"
        "to_cntrl_19_15"[label="15->19: MacroTile: (0)", shape=cds]
        "cntrl15" -> "to_coord_15_19"
        "to_coord_15_19"[label="15->19: MacroTile: (0)", shape=cds]
        "coord23" -> "to_cntrl_23_15"
        "to_cntrl_23_15"[label="15->23: User: (0)", shape=cds]
        "cntrl15" -> "to_coord_15_23"
        "to_coord_15_23"[label="15->23: User: (0)", shape=cds]
        }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));

        auto addLDSTransform          = std::make_shared<AddLDS>(params, m_context);
        auto lowerTileTransform       = std::make_shared<LowerTile>(params, m_context);
        auto addComputeIndexTransform = std::make_shared<AddComputeIndex>();

        auto kgraph1 = kgraph0.transform(addLDSTransform);
        kgraph1      = kgraph1.transform(lowerTileTransform);
        kgraph1      = kgraph1.transform(addComputeIndexTransform);

        namespace CG = rocRoller::KernelGraph::ControlGraph;
        ASSERT_EQ(kgraph1.control.getNodes<CG::LoadTiled>().to<std::vector>().size(), 2);
        ASSERT_EQ(kgraph1.control.getNodes<CG::LoadLDSTile>().to<std::vector>().size(), 1);
        ASSERT_EQ(kgraph1.control.getNodes<CG::StoreLDSTile>().to<std::vector>().size(), 1);
    }

    TEST_F(KernelGraphTest, Transformer)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        int macK  = 16;
        int waveK = 8;

        example.setTileSize(128, 256, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, false, false);

        auto kgraph0 = example.getKernelGraph();
        auto params  = example.getCommandParameters();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);
        auto addLDSTransform           = std::make_shared<AddLDS>(params, m_context);
        auto lowerTileTransform        = std::make_shared<LowerTile>(params, m_context);
        auto lowerTensorContractionTransform
            = std::make_shared<LowerTensorContraction>(params, m_context);
        auto unrollLoopsTransform      = std::make_shared<UnrollLoops>(params, m_context);
        auto fuseLoopsTransform        = std::make_shared<FuseLoops>();
        auto removeDuplicatesTransform = std::make_shared<RemoveDuplicates>();

        auto cleanLoopsTransform      = std::make_shared<CleanLoops>();
        auto addComputeIndexTransform = std::make_shared<AddComputeIndex>();

        kgraph0      = kgraph0.transform(updateParametersTransform);
        auto kgraph1 = kgraph0.transform(addLDSTransform);
        kgraph1      = kgraph1.transform(lowerTileTransform);
        kgraph1      = kgraph1.transform(lowerTensorContractionTransform);

        //
        // Build transformer one by one
        //
        std::unordered_map<int, Transformer> transformers;
        for(auto op : kgraph1.control.getNodes())
            transformers.emplace(op, kgraph1.buildTransformer(op));

        //
        // Build all transformers at once
        //
        kgraph1.buildAllTransformers();

        //
        // The resulting transformers should be identical
        //
        for(auto op : kgraph1.control.getNodes())
            EXPECT_EQ(transformers.at(op).getIndexes(), kgraph1.buildTransformer(op).getIndexes());
    }

    TEST_F(KernelGraphTest, RemoveSetCoordinate)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        int kernel = kgraph.control.addElement(Kernel());

        int nop1 = kgraph.control.addElement(NOP());
        int nop2 = kgraph.control.addElement(NOP());
        int nop3 = kgraph.control.addElement(NOP());
        int nop4 = kgraph.control.addElement(NOP());
        int nop5 = kgraph.control.addElement(NOP());
        int nop6 = kgraph.control.addElement(NOP());

        auto one = Expression::literal(1u);
        int  sc1 = kgraph.control.addElement(SetCoordinate(one));
        int  sc2 = kgraph.control.addElement(SetCoordinate(one));
        int  sc3 = kgraph.control.addElement(SetCoordinate(one));
        int  sc4 = kgraph.control.addElement(SetCoordinate(one));
        int  sc5 = kgraph.control.addElement(SetCoordinate(one));
        int  sc6 = kgraph.control.addElement(SetCoordinate(one));

        int dim = kgraph.coordinates.addElement(Adhoc());
        kgraph.mapper.connect<Adhoc>(sc1, dim);
        kgraph.mapper.connect<Adhoc>(sc2, dim);
        kgraph.mapper.connect<Adhoc>(sc3, dim);
        kgraph.mapper.connect<Adhoc>(sc4, dim);
        kgraph.mapper.connect<Adhoc>(sc5, dim);
        kgraph.mapper.connect<Adhoc>(sc6, dim);

        //  Original:
        //
        //          Kernel
        //            |
        //            |[body]
        //            v
        //           nop1
        //            |
        //            |[seq]
        //            v
        //           nop2 --------------
        //            |                |
        //            |[body]          |[seq]
        //            v                v
        //           sc1              sc2  -------------
        //            |                |               |
        //            |[seq]           |[body]         |[body]
        //            v                v               v
        //           sc3              nop3            sc4 ------------------
        //            |------------                    |                   |
        //            |           |                    |[seq]              |[seq]
        //            |[seq]      |[seq]               v                   v
        //            v           v                   nop4                nop5
        //           sc5         sc6
        //                        |
        //                        |[body]
        //                        v
        //                       nop6
        //

        kgraph.control.addElement(Body(), {kernel}, {nop1});
        kgraph.control.addElement(Sequence(), {nop1}, {nop2});
        kgraph.control.addElement(Body(), {nop2}, {sc1});
        kgraph.control.addElement(Sequence(), {sc1}, {sc3});
        kgraph.control.addElement(Sequence(), {sc3}, {sc5});
        kgraph.control.addElement(Sequence(), {sc3}, {sc6});
        kgraph.control.addElement(Body(), {sc6}, {nop6});

        kgraph.control.addElement(Sequence(), {nop2}, {sc2});
        kgraph.control.addElement(Body(), {sc2}, {nop3});
        kgraph.control.addElement(Body(), {sc2}, {sc4});
        kgraph.control.addElement(Sequence(), {sc4}, {nop4});
        kgraph.control.addElement(Sequence(), {sc4}, {nop5});

        auto removeSetCoordinate = std::make_shared<RemoveSetCoordinate>();
        auto kg2                 = kgraph.transform(removeSetCoordinate);

        //  After:
        //
        //          Kernel
        //            |
        //            |[body]
        //            v
        //           nop1
        //            |
        //            |[seq]
        //            v
        //           nop2 -------------------------------------
        //            |                |           |          |
        //            |[body]          |[seq]      |[seq]     |[seq]
        //            v                v           v          v
        //           nop6              nop3        nop4       nop5
        //

        std::string expected = R".(
               digraph {
               "1"[label="Kernel(1)"];
               "2"[label="NOP(2)"];
               "3"[label="NOP(3)"];
               "4"[label="NOP(4)"];
               "5"[label="NOP(5)"];
               "6"[label="NOP(6)"];
               "7"[label="NOP(7)"];
               "14"[label="Body(14)",shape=box];
               "15"[label="Sequence(15)",shape=box];
               "30"[label="Sequence(30)",shape=box];
               "31"[label="Sequence(31)",shape=box];
               "32"[label="Sequence(32)",shape=box];
               "33"[label="Body(33)",shape=box];
               "1" -> "14"
               "2" -> "15"
               "3" -> "30"
               "3" -> "31"
               "3" -> "32"
               "3" -> "33"
               "14" -> "2"
               "15" -> "3"
               "30" -> "4"
               "31" -> "5"
               "32" -> "6"
               "33" -> "7"
               }).";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(kg2.control.toDOT()));
    }

    TEST_F(KernelGraphTest, StreamKTwoTileDPFirst)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        uint numTileM = 57;
        uint numTileN = 57;
        uint numTileK = 57;
        uint numWGs   = 128;

        auto kernel = kgraph.control.addElement(Kernel());
        auto [forKCoord, forKOp]
            = rangeFor(kgraph, Expression::literal(numTileK), rocRoller::KLOOP);

        auto user = kgraph.coordinates.addElement(User({}, "result"));

        auto tileM = kgraph.coordinates.addElement(
            MacroTileNumber(0, Expression::literal(numTileM), nullptr));
        auto tileN = kgraph.coordinates.addElement(
            MacroTileNumber(1, Expression::literal(numTileN), nullptr));
        auto tileK = kgraph.coordinates.addElement(
            MacroTileNumber(-1, Expression::literal(numTileK), nullptr));

        kgraph.coordinates.addElement(PassThrough(), {forKCoord}, {tileK});
        kgraph.coordinates.addElement(Flatten(), {tileM, tileN, tileK}, {user});

        auto dstVGPR = kgraph.coordinates.addElement(VGPR());
        auto wgDim   = kgraph.coordinates.addElement(Workgroup(0));
        auto wgExpr  = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{wgDim, Register::Type::Vector, DataType::UInt32});
        auto assignWGNumber = kgraph.control.addElement(Assign{Register::Type::Vector, wgExpr});
        kgraph.mapper.connect(assignWGNumber, dstVGPR, NaryArgument::DEST);

        kgraph.coordinates.addElement(PassThrough(), {user}, {dstVGPR});

        auto storeOp = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeOp, user);
        kgraph.mapper.connect<VGPR>(storeOp, dstVGPR);

        auto preWaitOp  = kgraph.control.addElement(WaitZero());
        auto loopWaitOp = kgraph.control.addElement(WaitZero());

        kgraph.control.addElement(Body(), {kernel}, {preWaitOp});
        kgraph.control.addElement(Sequence(), {preWaitOp}, {forKOp});
        kgraph.control.addElement(Body(), {forKOp}, {assignWGNumber});
        kgraph.control.addElement(Sequence(), {assignWGNumber}, {storeOp});
        kgraph.control.addElement(Sequence(), {storeOp}, {loopWaitOp});

        CommandParametersPtr params           = std::make_shared<CommandParameters>();
        params->loopOverOutputTilesDimensions = {0, 1};

        // Helper to check loop order in a given kernel graph
        auto checkStreamKLoopOrder = [&](rocRoller::KernelGraph::KernelGraph& kg,
                                         const std::string&                   firstLoop,
                                         const std::string&                   secondLoop) {
            int body = -1, sequence = -1;
            for(auto const scope :
                filter(kg.control.isElemType<Scope>(),
                       kg.control.depthFirstVisit(kg.control.roots().only().value())))
            {
                auto bodies = kg.control.getOutputNodeIndices<Body>(scope).to<std::unordered_set>();
                auto sequences
                    = kg.control.getOutputNodeIndices<Sequence>(scope).to<std::unordered_set>();

                if(bodies.size() + sequences.size() == 1)
                    continue;

                AssertFatal(
                    bodies.size() == 1 && sequences.size() == 1,
                    "Expected one body and one sequence in scope, found {} bodies and {} sequences",
                    bodies.size(),
                    sequences.size());

                body     = *bodies.begin();
                sequence = *sequences.begin();
                break;
            }

            // Check for first loop through the Body edge
            for(auto const loop :
                filter(kg.control.isElemType<ForLoopOp>(), kg.control.depthFirstVisit(body)))
            {
                auto forloop = kg.control.get<ForLoopOp>(loop).value();
                EXPECT_EQ(forloop.loopName, firstLoop);
                break;
            }

            // Check for second loop through the Sequence edge
            for(auto const loop :
                filter(kg.control.isElemType<ForLoopOp>(), kg.control.depthFirstVisit(sequence)))
            {
                auto forloop = kg.control.get<ForLoopOp>(loop).value();
                EXPECT_EQ(forloop.loopName, secondLoop);
                break;
            }
        };

        // For streamKTwoTile, the SK loop is first
        params->streamK = StreamKMode::TwoTile;

        auto kgraphTwoTile = kgraph.transform(std::make_shared<AddStreamK>(
            m_context, params, rocRoller::KLOOP, rocRoller::KLOOP, Expression::literal(numWGs)));

        if(m_context->kernelOptions()->removeSetCoordinate)
            kgraphTwoTile = kgraphTwoTile.transform(std::make_shared<RemoveSetCoordinate>());

        checkStreamKLoopOrder(kgraphTwoTile, "SKStreamTileLoop", "DPStreamTileLoop");

        m_context->kernel()->resetArguments();

        // For streamKTwoTileDPFirst, the DP loop is first
        params->streamK = StreamKMode::TwoTileDPFirst;

        auto kgraphTwoTileDPFirst = kgraph.transform(std::make_shared<AddStreamK>(
            m_context, params, rocRoller::KLOOP, rocRoller::KLOOP, Expression::literal(numWGs)));

        if(m_context->kernelOptions()->removeSetCoordinate)
            kgraphTwoTileDPFirst
                = kgraphTwoTileDPFirst.transform(std::make_shared<RemoveSetCoordinate>());

        checkStreamKLoopOrder(kgraphTwoTileDPFirst, "DPStreamTileLoop", "SKStreamTileLoop");
    }
}
