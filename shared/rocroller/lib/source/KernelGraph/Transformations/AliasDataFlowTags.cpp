/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/AliasDataFlowTags.hpp>
#include <rocRoller/KernelGraph/Transforms/AliasDataFlowTags_detail.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <rocRoller/Graph/GraphUtilities.hpp>

#define debug critical

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * 1. For every `MacroTile` dimension in the coordinate graph:
         *     1. We use a `ControlFlowRWTracer` to identify every control node
         *        that uses that particular MacroTile.
         *     2. We construct a graph that orders the records from the tracer.
         *         - We create the graph with every ordering we can get from
         *           the control graph.
         *         - Then we simplify that graph so it has no redundant edges.
         *     3. We use the graph to identify:
         *         - The lifetime of that `MacroTile`'s registers
         *           (roots -> leaves of the graph)
         *         - Any gaps in the liveness of those registers
         *            A `WRITE` record where all the incoming nodes are `READ`
         *            means there is a gap from those `READ` nodes to the
         *            `WRITE` node.
         *     4. We bin these records by MemoryType, LayoutType, DataType, and
         *        MacroTile size.
         * 2. For each bin:
         *   - For each pair of records:
         *     If the lifetime of record A fits within a gap in record B, then
         *     we can let A use B's register allocation.
         */
        namespace AliasDataFlowTagsDetail
        {
            bool GraphExtent::isWithin(KernelGraph const& kgraph, GraphExtent const& gap) const
            {
                AssertFatal(!gap.begin.empty());
                AssertFatal(!gap.end.empty());
                AssertFatal(!begin.empty());
                AssertFatal(!end.empty());

                for(int gapBegin : gap.begin)
                {
                    for(int inBegin : begin)
                    {
                        if(inBegin != gapBegin)
                        {
                            auto order = kgraph.control.compareNodes(
                                rocRoller::UpdateCache, gapBegin, inBegin);
                            if(order != ControlGraph::NodeOrdering::LeftFirst)
                                return false;
                        }
                    }
                }

                for(int gapEnd : gap.end)
                {
                    for(int inEnd : end)
                    {
                        if(inEnd != gapEnd)
                        {
                            auto order = kgraph.control.compareNodes(
                                rocRoller::UpdateCache, inEnd, gapEnd);
                            if(order != ControlGraph::NodeOrdering::LeftFirst)
                                return false;
                        }
                    }
                }

                return true;
            }

            std::string GraphExtent::toString() const
            {
                return fmt::format("[{}..{}]", begin, end);
            }

            std::ostream& operator<<(std::ostream& stream, GraphExtent const& extent)
            {
                return stream << extent.toString();
            }

            TagExtent::CategoryKey TagExtent::typeKey() const
            {
                int size = 1;
                for(int s : sizes)
                    size *= s;

                return std::make_tuple(memoryType, layoutType, dataType, size);
            }

            std::string TagExtent::toString() const
            {
                std::ostringstream msg;

                msg << "Tag " << tags << " (" << dataType << " ";
                rocRoller::streamJoin(msg, sizes, "x");
                msg << "): " << extent << std::endl;

                for(auto const& gap : gaps)
                {
                    msg << " - gap: " << gap << std::endl;
                }
                return msg.str();
            }

            std::string TagExtent::orderInfo(KernelGraph const& kgraph) const
            {
                std::set<int> allNodes_ = extent.begin;
                allNodes_.insert(extent.end.begin(), extent.end.end());
                for(auto const& gap : gaps)
                {
                    allNodes_.insert(gap.begin.begin(), gap.begin.end());
                    allNodes_.insert(gap.end.begin(), gap.end.end());
                }

                auto topoCompare = [&](int a, int b) {
                    return kgraph.control.compareNodes(rocRoller::UpdateCache, a, b)
                           == ControlGraph::NodeOrdering::LeftFirst;
                };

                std::vector<int> allNodes(allNodes_.begin(), allNodes_.end());
                std::sort(allNodes.begin(), allNodes.end(), topoCompare);

                std::string nodeInfo;
                {
                    bool first = true;
                    for(auto node : allNodes)
                    {
                        auto stack = controlStack(node, kgraph);

                        auto isInLoop = std::ranges::any_of(
                            stack, kgraph.control.isElemType<ControlGraph::ForLoopOp>());

                        if(!first)
                            nodeInfo += ", ";

                        nodeInfo += std::to_string(node);
                        if(isInLoop)
                            nodeInfo += " (l)";

                        first = false;
                    }
                }

                std::string splitInfo;
                {
                    std::ostringstream msg;
                    if(!validSplits.empty())
                        msg << "Splits: ";
                    for(auto split : validSplits)
                    {
                        msg << "{";
                        streamJoin(msg, split.begin, ", ");
                        msg << "} -> {";
                        streamJoin(msg, split.end, ", ");
                        msg << "}\n";
                    }
                    splitInfo = msg.str();
                }

                auto table = kgraph.control.nodeOrderTableString(allNodes_);

                return fmt::format("Nodes {{{}}}\n{}\n{}", nodeInfo, table, splitInfo);
            }

            std::set<int> TagExtent::allNodes() const
            {
                std::set<int> rv;
                rv.insert(extent.begin.begin(), extent.begin.end());
                rv.insert(extent.end.begin(), extent.end.end());
                for(auto const& gap : gaps)
                {
                    rv.insert(gap.begin.begin(), gap.begin.end());
                    rv.insert(gap.end.begin(), gap.end.end());
                }

                return rv;
            }

            void TagExtent::validate(KernelGraph const& kgraph) const
            {
                auto assertSetsOrdered = [&](auto const& setA, auto const& setB) {
                    for(int one : setA)
                    {
                        for(int two : setB)
                        {
                            if(one != two)
                            {
                                if(kgraph.control.compareNodes(UpdateCache, one, two)
                                   != ControlGraph::NodeOrdering::LeftFirst)
                                {
                                    auto nodes      = allNodes();
                                    auto orderTable = kgraph.control.nodeOrderTableString(nodes);
                                    AssertFatal(kgraph.control.compareNodes(UpdateCache, one, two)
                                                    == ControlGraph::NodeOrdering::LeftFirst,
                                                ShowValue(toString()),
                                                ShowValue(one),
                                                ShowValue(two),
                                                ShowValue(kgraph.control.compareNodes(
                                                    UpdateCache, one, two)),
                                                ShowValue(orderTable));
                                }
                            }
                        }
                    }
                };

                std::set<int> before = extent.begin;

                for(auto const& gap : gaps)
                {
                    assertSetsOrdered(before, gap.begin);

                    before.insert(gap.begin.begin(), gap.begin.end());

                    assertSetsOrdered(before, gap.end);

                    before.insert(gap.end.begin(), gap.end.end());
                }

                assertSetsOrdered(before, extent.end);
            }

            bool TagExtent::empty() const
            {
                if(extent.begin.empty() || extent.end.empty())
                {
                    AssertFatal(extent.begin.empty());
                    AssertFatal(extent.end.empty());
                    AssertFatal(gaps.empty());

                    return true;
                }

                return false;
            }

            void TagExtent::merge(KernelGraph const& kgraph, TagExtent const& inner)
            {
                bool found = false;

                AssertFatal(
                    typeKey() == inner.typeKey(), ShowValue(typeKey()), ShowValue(inner.typeKey()));
                AssertFatal(dataType != DataType::None, ShowValue(dataType));

                auto itFits
                    = [&](GraphExtent const& gap) { return inner.extent.isWithin(kgraph, gap); };

                auto whichGap = std::find_if(gaps.begin(), gaps.end(), itFits);

                AssertFatal(whichGap != gaps.end());

                GraphExtent before{std::move(whichGap->begin), inner.extent.begin};
                GraphExtent after{inner.extent.end, std::move(whichGap->end)};

                std::vector<GraphExtent> newGaps;
                newGaps.push_back(before);
                newGaps.insert(newGaps.end(), inner.gaps.begin(), inner.gaps.end());
                newGaps.push_back(after);

                auto iter = gaps.erase(whichGap);
                gaps.insert(iter, newGaps.begin(), newGaps.end());

                tags.insert(inner.tags.begin(), inner.tags.end());
            }

            TagRWGraph getOrdering(KernelGraph const& kgraph, std::vector<Record> const& records)
            {
                TagRWGraph ordering;

                std::vector<int> graphIndices;

                for(auto const& rec : records)
                {
                    graphIndices.push_back(ordering.addElement(rec));
                }

                for(int a = 0; a < records.size(); a++)
                {
                    for(int b = a + 1; b < records.size(); b++)
                    {
                        auto const& aRec = records[a];
                        auto const& bRec = records[b];
                        auto        aIdx = graphIndices[a];
                        auto        bIdx = graphIndices[b];

                        // This can happen with Exchange nodes
                        // when scaleSkipPermlane=True, because
                        // src and dst of each Exchange point
                        // to the same macrotile (register allocation),
                        // and therefore recorded twice in the
                        // ControlFlowRWTracer.
                        if(aRec.control == bRec.control)
                            continue;

                        auto order = kgraph.control.compareNodes(
                            rocRoller::UpdateCache, aRec.control, bRec.control);

                        switch(order)
                        {
                            namespace CG = ControlGraph;
                        case CG::NodeOrdering::LeftFirst:
                            ordering.addElement(CG::Sequence{}, {aIdx}, {bIdx});
                            break;
                        case CG::NodeOrdering::LeftInBodyOfRight:
                            Throw<FatalError>("Nodes that use MacroTiles should not have bodies");
                            break;
                        case CG::NodeOrdering::Undefined:
                            break;
                        case CG::NodeOrdering::RightInBodyOfLeft:
                            Throw<FatalError>("Nodes that use MacroTiles should not have bodies");
                            break;
                        case CG::NodeOrdering::RightFirst:
                            ordering.addElement(CG::Sequence{}, {bIdx}, {aIdx});
                            break;
                        case CG::NodeOrdering::Count:
                            Throw<FatalError>("Should not get here!");
                        }
                    }
                }

                {
                    auto truePred = [](int x) { return true; };
                    Graph::removeRedundantEdges(ordering, truePred);
                }

                return ordering;
            }

            TagExtent getInfo(KernelGraph const& kgraph, std::vector<Record> const& records)
            {
                TagExtent rv;

                for(auto const& rec : records)
                {
                    if(rv.baseTag == -1)
                        rv.baseTag = rec.coordinate;
                    rv.tags.insert(rec.coordinate);

                    if(rv.dataType == DataType::None)
                        rv.dataType
                            = ControlGraph::getDataType(kgraph.control.getNode(rec.control));

                    auto node = kgraph.coordinates.getNode(rec.coordinate);

                    auto visitor = rocRoller::overloaded{
                        [&](CoordinateGraph::MacroTile const& mt) {
                            if(rv.sizes.empty())
                                rv.sizes = mt.sizes;

                            if(rv.memoryType == MemoryType::None)
                                rv.memoryType = mt.memoryType;

                            if(rv.layoutType == LayoutType::None)
                                rv.layoutType = mt.layoutType;
                        },
                        [&](CoordinateGraph::LDS const& lds) {
                            rv.sizes      = {1};
                            rv.memoryType = MemoryType::LDS;
                        },
                        [&](auto const& node) {
                            Throw<FatalError>("Unexpected Dimension: {}", toString(node));
                        }};

                    std::visit(visitor, node);
                }

                return rv;
            }

            std::vector<GraphExtent> getLivenessGaps(KernelGraph const& kgraph,
                                                     TagRWGraph const&  ordering)
            {
                std::vector<GraphExtent> rv;

                using Tracer = ControlFlowRWTracer;

                auto isRead  = [&](int idx) { return ordering.getNode(idx).rw == Tracer::READ; };
                auto isWrite = [&](int idx) { return ordering.getNode(idx).rw == Tracer::WRITE; };

                auto writes = ordering.getNodes().filter(isWrite).to<std::set>();

                for(auto writeIter = writes.begin(); writeIter != writes.end(); writeIter++)
                {
                    auto incoming = ordering.getInputNodeIndices<Edge>(*writeIter).to<std::set>();

                    if(!incoming.empty() && std::all_of(incoming.begin(), incoming.end(), isRead))
                    {
                        std::set<int> outgoing;

                        for(int inc : incoming)
                        {
                            auto incOutgoing = ordering.getOutputNodeIndices<Edge>(inc);
                            outgoing.insert(incOutgoing.begin(), incOutgoing.end());
                        }

                        AssertFatal(
                            outgoing.size() == 1 && outgoing.contains(*writeIter),
                            "It doesn't make sense to have two unordered nodes that both write to "
                            "a register.",
                            ShowValue(incoming),
                            ShowValue(outgoing));

                        if(std::all_of(outgoing.begin(), outgoing.end(), isWrite))
                        {
                            std::set<int> incControl, outControl;
                            for(int inc : incoming)
                                incControl.insert(ordering.getNode(inc).control);

                            for(int out : outgoing)
                                outControl.insert(ordering.getNode(out).control);

                            rv.emplace_back(std::move(incControl), std::move(outControl));
                        }
                    }
                }

                auto topoCompareExtents = [&](GraphExtent const& a, GraphExtent const& b) {
                    AssertFatal(!a.begin.empty());
                    AssertFatal(!b.begin.empty());

                    return kgraph.control.compareNodes(
                               rocRoller::UpdateCache, *a.begin.begin(), *b.begin.begin())
                           == ControlGraph::NodeOrdering::LeftFirst;
                };

                std::sort(rv.begin(), rv.end(), topoCompareExtents);

                return rv;
            }

            TagExtent getExtent(KernelGraph const& kgraph, std::vector<Record> const& records)
            {
                using Tracer = ControlFlowRWTracer;

                TagExtent rv;

                if(records.empty())
                {
                    return rv;
                }

                rv = getInfo(kgraph, records);

                auto ordering = getOrdering(kgraph, records);
                rv.graph      = ordering;

                {
                    std::ofstream file(fmt::format("order_graph_{}.dot", rv.baseTag));
                    file << ordering.toDOT();
                }

                auto logger = Log::getLogger();
                if(logger->should_log(LogLevel::Trace))
                    logger->trace(ordering.toDOT());

                auto getNode = [&](int idx) {
                    auto rec = ordering.getNode(idx);
                    return rec.control;
                };

                rv.extent.begin = ordering.roots().map(getNode).to<std::set>();
                rv.extent.end   = ordering.leaves().map(getNode).to<std::set>();

                auto isRead  = [&](int idx) { return ordering.getNode(idx).rw == Tracer::READ; };
                auto isWrite = [&](int idx) { return ordering.getNode(idx).rw == Tracer::WRITE; };

                auto allGaps = getLivenessGaps(kgraph, ordering);

                // Find gaps that can contain an alias.
                for(auto const& gap : allGaps)
                {
                    auto forLoop = findContainingOperation<ControlGraph::ForLoopOp>(
                        *gap.begin.begin(), kgraph);

                    auto sameForLoop = [&](int node) {
                        return findContainingOperation<ControlGraph::ForLoopOp>(node, kgraph)
                               == forLoop;
                    };

                    if(std::ranges::all_of(gap.begin, sameForLoop)
                       && std::ranges::all_of(gap.end, sameForLoop))
                    {
                        rv.gaps.push_back(gap);
                    }
                }

                auto inForLoop = [&](int node) {
                    return findContainingOperation<ControlGraph::ForLoopOp>(node, kgraph)
                           != std::nullopt;
                };

                // Find if we can split this tag from the beginning.

                // Beginning of tag must be outside a loop.
                if(std::ranges::none_of(rv.extent.begin, inForLoop))
                {
                    GraphExtent splitA{rv.extent.begin, {}};
                    GraphExtent splitB{{}, rv.extent.end};
                    for(auto const& gap : allGaps)
                    {
                        AssertFatal(!gap.begin.empty());
                        AssertFatal(!gap.end.empty());

                        if(std::ranges::none_of(gap.begin, inForLoop)
                           && std::ranges::all_of(gap.end, inForLoop))
                        {
                            splitA.end   = gap.begin;
                            splitB.begin = gap.end;
                            break;
                        }

                        if(std::ranges::any_of(gap.begin, inForLoop)
                           || std::ranges::any_of(gap.end, inForLoop))
                            break;
                    }

                    if(!splitA.end.empty())
                    {
                        rv.validSplits.emplace_back(std::move(splitA));
                        rv.validSplits.emplace_back(std::move(splitB));
                    }
                }

                // auto writes = ordering.getNodes().filter(isWrite).to<std::set>();

                // for(auto writeIter = writes.begin(); writeIter != writes.end(); writeIter++)
                // {
                //     auto incoming = ordering.getInputNodeIndices<Edge>(*writeIter).to<std::set>();

                //     if(!incoming.empty() && std::all_of(incoming.begin(), incoming.end(), isRead))
                //     {
                //         std::set<int> outgoing;

                //         for(int inc : incoming)
                //         {
                //             auto incOutgoing = ordering.getOutputNodeIndices<Edge>(inc);
                //             outgoing.insert(incOutgoing.begin(), incOutgoing.end());
                //         }

                //         AssertFatal(
                //             outgoing.size() == 1 && outgoing.contains(*writeIter),
                //             "It doesn't make sense to have two unordered nodes that both write to "
                //             "a register.",
                //             ShowValue(rv.toString()),
                //             ShowValue(incoming),
                //             ShowValue(outgoing));

                //         if(std::all_of(outgoing.begin(), outgoing.end(), isWrite))
                //         {
                //             std::set<int> incControl, outControl;
                //             for(int inc : incoming)
                //                 incControl.insert(ordering.getNode(inc).control);

                //             for(int out : outgoing)
                //                 outControl.insert(ordering.getNode(out).control);

                //             auto forLoop = findContainingOperation<ControlGraph::ForLoopOp>(
                //                 *incControl.begin(), kgraph);

                //             auto sameForLoop = [&](int node) {
                //                 return findContainingOperation<ControlGraph::ForLoopOp>(node,
                //                                                                         kgraph)
                //                        == forLoop;
                //             };

                //             if(std::all_of(incControl.begin(), incControl.end(), sameForLoop)
                //                && std::all_of(outControl.begin(), outControl.end(), sameForLoop))
                //             {
                //                 rv.gaps.emplace_back(std::move(incControl), std::move(outControl));
                //             }
                //         }
                //     }
                // }

                return rv;
            }

            bool TagExtent::fitsWithin(KernelGraph const& kgraph, TagExtent const& outer) const
            {
                for(auto const& gap : outer.gaps)
                    if(extent.isWithin(kgraph, gap))
                        return true;

                return false;
            }

            std::map<TagExtent::CategoryKey, std::list<TagExtent>>
                getGroupedTagExtents(KernelGraph const& kgraph)
            {
                std::map<TagExtent::CategoryKey, std::list<TagExtent>> groupedExtents;

                ControlFlowRWTracer tracer(kgraph);

                for(auto lds : kgraph.coordinates.getNodes<CoordinateGraph::LDS>())
                {
                    auto records = tracer.coordinatesReadWrite(lds);

                    auto extent = getExtent(kgraph, records);

                    if(!extent.empty() && extent.dataType != DataType::None
                       && extent.layoutType != LayoutType::MATRIX_ACCUMULATOR)
                    {
                        groupedExtents[extent.typeKey()].push_back(std::move(extent));
                    }
                }

                for(auto mt : kgraph.coordinates.getNodes<CoordinateGraph::MacroTile>())
                {
                    auto isView = [&](auto const& edge) {
                        auto const* dfe = std::get_if<CoordinateGraph::DataFlowEdge>(&edge);
                        return dfe && std::holds_alternative<CoordinateGraph::View>(*dfe);
                    };

                    auto outViews
                        = kgraph.coordinates
                              .getConnectedNodeIndices<Graph::Direction::Downstream>(mt, isView)
                              .to<std::vector>();
                    auto inViews
                        = kgraph.coordinates
                              .getConnectedNodeIndices<Graph::Direction::Upstream>(mt, isView)
                              .to<std::vector>();

                    if(inViews.size() > 0)
                    {
                        AssertFatal(outViews.empty(),
                                    "Both in and out view edges are not supported.");
                    }

                    auto records = tracer.coordinatesReadWrite(mt);

                    // TODO: Push this into the ControlFlowRWTracer.
                    for(auto mtView : outViews)
                    {
                        auto viewRecs = tracer.coordinatesReadWrite(mtView);
                        records.insert(records.end(), viewRecs.begin(), viewRecs.end());
                    }

                    auto extent = getExtent(kgraph, records);

                    if(!extent.empty() && extent.dataType != DataType::None
                       && extent.layoutType != LayoutType::MATRIX_ACCUMULATOR)
                    {
                        groupedExtents[extent.typeKey()].push_back(std::move(extent));
                    }
                }

                return groupedExtents;
            }

            std::map<int, int> findAliasCandidatesForExtents(KernelGraph const&   kgraph,
                                                             std::list<TagExtent> extents)
            {
                std::map<int, int> aliases;

                bool foundAny = false;
                do
                {
                    auto e   = extents.size();
                    foundAny = false;
                    for(auto outer = extents.begin(); outer != extents.end(); outer++)
                    {
                        for(auto inner = extents.begin(); inner != extents.end();)
                        {
                            if(outer != inner && inner->fitsWithin(kgraph, *outer))
                            {
                                foundAny = true;
                                AssertFatal(!aliases.contains(inner->baseTag));
                                aliases[inner->baseTag] = outer->baseTag;
                                Log::debug("{} -> {}", inner->baseTag, outer->baseTag);

                                inner->validate(kgraph);

                                outer->merge(kgraph, *inner);

                                outer->validate(kgraph);

                                Log::debug("merged {}", outer->toString());
                                inner = extents.erase(inner);
                            }
                            else
                            {
                                inner++;
                            }
                        }
                    }
                    Log::debug("{} aliases so far.", aliases.size());
                } while(foundAny);

                for(auto ext : extents)
                {
                    Log::debug("{}\n{}", ext.toString(), ext.orderInfo(kgraph));
                    ext.validate(kgraph);
                }

                {
                    std::set<int> nodes;

                    for(auto const& extent: extents)
                    {
                        auto myNodes = extent.allNodes();
                        nodes.insert(myNodes.begin(), myNodes.end());
                    }

                    // Log::debug(rocRoller::toString(nodes));

                    Log::debug("\n{}", kgraph.control.nodeOrderTableString(nodes));

                }


                return aliases;
            }

            std::map<int, int> findAliasCandidates(KernelGraph const& kgraph)
            {
                // Use a list so we can erase without invalidating any other iterators.
                auto groupedExtents = getGroupedTagExtents(kgraph);

                std::map<int, int> aliases;

                for(auto& [typeKey, extents] : groupedExtents)
                {
                    auto logger = Log::getLogger();
                    if(logger->should_log(LogLevel::Critical))
                    {
                        std::ostringstream msg;
                        streamJoinTuple(msg, ", ", typeKey);

                        logger->debug("Aliases for {{{}}} tags:", msg.str());
                        logger->debug("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=");

                        for(auto const& ext : extents)
                            logger->debug(ext.toString());
                    }

                    auto theseAliases = findAliasCandidatesForExtents(kgraph, extents);

                    aliases.insert(theseAliases.begin(), theseAliases.end());
                }

                auto logger = Log::getLogger();
                if(logger->should_log(LogLevel::Debug))
                {
                    for(auto const& [a, b] : aliases)
                    {
                        logger->debug("{} -> {}", a, b);
                    }
                }

                return aliases;
            }

            std::map<int, std::vector<GraphExtent>> findSplitCandidates(KernelGraph const& kgraph)
            {
                auto groupedExtents = getGroupedTagExtents(kgraph);

                std::map<int, std::vector<GraphExtent>> rv;

                for(auto const& [key, extents] : groupedExtents)
                {
                    for(auto const& extent : extents)
                    {
                        if(!extent.validSplits.empty())
                        {
                            AssertFatal(extent.validSplits.size() >= 2,
                                        ShowValue(extent.validSplits.size()));
                            AssertFatal(!rv.contains(extent.baseTag), ShowValue(extent.baseTag));

                            rv[extent.baseTag] = extent.validSplits;
                        }
                    }
                }

                return rv;
            }

            bool GraphExtent::contains(KernelGraph const& kgraph, int opTag) const
            {
                if(begin.contains(opTag) || end.contains(opTag))
                    return true;

                auto afterOp = [&](int otherOp) {
                    return kgraph.control.compareNodes(UpdateCache, opTag, otherOp)
                           == ControlGraph::NodeOrdering::LeftFirst;
                };

                auto beforeOp = [&](int otherOp) {
                    return kgraph.control.compareNodes(UpdateCache, opTag, otherOp)
                           == ControlGraph::NodeOrdering::RightFirst;
                };

                return std::ranges::all_of(begin, beforeOp) && std::ranges::all_of(end, afterOp);
            }

            void moveMapperConnections(KernelGraph& kgraph, int opTag, int newDim, int oldDim)
            {
                auto connections = kgraph.mapper.getConnections(opTag);
                for(auto& connection : connections)
                {
                    if(connection.coordinate == oldDim)
                        connection.coordinate = newDim;
                }

                kgraph.mapper.purge(opTag);

                for(auto const& connection : connections)
                {
                    kgraph.mapper.connect(
                        connection.control, connection.coordinate, connection.connection);
                }
            }

            void applySplits(KernelGraph&                                   kgraph,
                             std::map<int, std::vector<GraphExtent>> const& splits)
            {
                ControlFlowRWTracer tracer(kgraph);

                for(auto const& [dimension, dimSplits] : splits)
                {
                    std::ostringstream msg;
                    msg << "Splitting " << dimension << ": ";

                    auto          dimRWs = tracer.coordinatesReadWrite(dimension);
                    std::set<int> dimOps;
                    for(auto const& record : dimRWs)
                        dimOps.insert(record.control);

                    auto splitIter = dimSplits.begin();
                    AssertFatal(splitIter != dimSplits.end());
                    ++splitIter;
                    AssertFatal(splitIter != dimSplits.end());

                    for(; splitIter != dimSplits.end(); ++splitIter)
                    {
                        auto newDim = kgraph.coordinates.addElement(
                            kgraph.coordinates.getElement(dimension));
                        kgraph.coordinates.addElement(
                            CoordinateGraph::Duplicate{}, {newDim}, {dimension});
                        msg << " - Adding " << newDim << ": {";

                        bool first = true;

                        for(auto opIter = dimOps.begin(); opIter != dimOps.end();)
                        {
                            if(splitIter->contains(kgraph, *opIter))
                            {
                                if(!first)
                                    msg << ", ";
                                msg << *opIter;
                                first = false;

                                moveMapperConnections(kgraph, *opIter, newDim, dimension);
                                opIter = dimOps.erase(opIter);
                            }
                            else
                            {
                                ++opIter;
                            }
                        }

                        msg << "}" << std::endl;
                    }

                    msg << " - Remaining at " << dimension << ": ";
                    streamJoin(msg, dimOps, ", ");
                    msg << std::endl;

                    Log::debug(msg.str());
                }
            }
        }

        KernelGraph AliasDataFlowTags::apply(KernelGraph const& original)
        {
            auto rv = original;

            {
                auto splits = AliasDataFlowTagsDetail::findSplitCandidates(rv);

                AliasDataFlowTagsDetail::applySplits(rv, splits);
            }

            {
                auto aliases = AliasDataFlowTagsDetail::findAliasCandidates(rv);

                for(auto const& [inner, outer] : aliases)
                {
                    rv.coordinates.addElement(CoordinateGraph::Alias{}, {inner}, {outer});
                }
            }

            return rv;
        }
    }
}
