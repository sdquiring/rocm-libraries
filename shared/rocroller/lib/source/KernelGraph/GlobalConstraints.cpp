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

#include <rocRoller/Graph/GraphUtilities.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/TopoVisitor.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        ConstraintStatus NoDanglingMappings(const KernelGraph& k)
        {
            TIMER(t, "Constraint::NoDanglingMappings");
            ConstraintStatus retval;
            for(auto control : k.mapper.getControls())
            {
                if(!k.control.exists(control))
                {
                    retval.combine(false,
                                   concatenate("Dangling Mapping: Control node ",
                                               control,
                                               " does not exist."));
                }
                for(auto& connection : k.mapper.getConnections(control))
                {
                    if(!k.coordinates.exists(connection.coordinate))
                    {
                        retval.combine(false,
                                       concatenate("Dangling Mapping: Control node ",
                                                   control,
                                                   " maps to coordinate node ",
                                                   connection.coordinate,
                                                   ", which doesn't exist."));
                    }
                }
            }
            return retval;
        }

        ConstraintStatus SingleControlRoot(const KernelGraph& k)
        {
            TIMER(t, "Constraint::SingleControlRoot");
            ConstraintStatus retval;

            auto controlRoots = k.control.roots().to<std::vector>();

            if(controlRoots.size() != 1)
            {
                std::ostringstream msg;
                msg << "Single Control Root: Control graph must have exactly one root node, not "
                    << controlRoots.size() << ". Nodes: (";
                streamJoin(msg, controlRoots, ", ");
                msg << ")";

                retval.combine(false, msg.str());
            }

            return retval;
        }

        ConstraintStatus NoRedundantSetCoordinates(const KernelGraph& k)
        {
            TIMER(t, "Constraint::NoRedundantSetCoordinates");
            using namespace ControlGraph;
            using GD = rocRoller::Graph::Direction;
            ConstraintStatus retval;

            for(const auto& op : k.control.leaves())
            {
                std::set<std::pair<int, int>> existingSetCoordData;
                int                           tag = op;

                while(true)
                {
                    auto parent = only(k.control.getInputNodeIndices<Body>(tag));
                    if(!parent)
                        break;

                    tag           = parent.value();
                    auto setCoord = k.control.get<SetCoordinate>(tag);
                    if(!setCoord)
                        break;

                    auto valueExpr = setCoord.value().value;
                    AssertFatal(evaluationTimes(valueExpr)[Expression::EvaluationTime::Translate],
                                "SetCoordinate::value should be a literal.");

                    auto value = getUnsignedInt(evaluate(valueExpr));
                    for(auto const& dst : k.mapper.getConnections(tag))
                    {
                        auto insertResult = existingSetCoordData.insert({dst.coordinate, value});
                        if(!insertResult.second)
                        {
                            auto setCoordData = insertResult.first;
                            retval.combine(false,
                                           concatenate("Redundant SetCoordinate for node ",
                                                       op,
                                                       ": SetCoordinate ",
                                                       tag,
                                                       " with target coordinate ",
                                                       setCoordData->first,
                                                       " and value ",
                                                       setCoordData->second));
                        }
                    }
                }
            }

            return retval;
        }

        struct WalkableControlGraphVisitor
            : public TopoControlGraphVisitor<WalkableControlGraphVisitor>
        {
            using TopoControlGraphVisitor<WalkableControlGraphVisitor>::TopoControlGraphVisitor;

            ConstraintStatus status;
            std::set<int>    visitedNodes;

            void operator()(int nodeIdx, auto const& node)
            {
                visitedNodes.insert(nodeIdx);
            }

            virtual void errorCondition(std::string const& message) override
            {
                status.combine(false, message);
            }
        };

        ConstraintStatus WalkableControlGraph(KernelGraph const& k)
        {
            TIMER(t, "Constraint::WalkableControlGraph");
            WalkableControlGraphVisitor visitor(k);
            visitor.walk();

            auto allNodes = k.control.getNodes().to<std::set>();

            if(visitor.visitedNodes != allNodes)
            {
                std::set<int> nonVisitedNodes;
                std::set_difference(allNodes.begin(),
                                    allNodes.end(),
                                    visitor.visitedNodes.begin(),
                                    visitor.visitedNodes.end(),
                                    std::inserter(nonVisitedNodes, nonVisitedNodes.end()));

                std::ostringstream msg;
                msg << "Not all nodes were visited! Missing: ";
                streamJoin(msg, nonVisitedNodes, ", ");
                msg << "\n All nodes: ";
                streamJoin(msg, allNodes, ", ");
                msg << "\n Visited nodes: ";
                streamJoin(msg, visitor.visitedNodes, ", ");

                visitor.status.combine(false, msg.str());
            }

            return visitor.status;
        }

        ConstraintStatus NeededParallelism(KernelGraph const& k)
        {
            TIMER(t, "Constraint::NeededParallelism");

            auto multiplyNodes = k.control.getNodes<ControlGraph::Multiply>().to<std::vector>();
            std::ranges::sort(multiplyNodes, [&](int a, int b) {
                auto order = k.control.compareNodes(UpdateCache, a, b);
                return order == ControlGraph::NodeOrdering::LeftFirst
                       || order == ControlGraph::NodeOrdering::RightInBodyOfLeft;
            });

            AssertFatal(!multiplyNodes.empty());

            auto rep = *multiplyNodes.begin();

            const int forLoop = [&]() {
                auto repStack = controlStack(rep, k);

                auto reversed = std::ranges::reverse_view(repStack);
                auto loop     = std::ranges::find_if(reversed, [&](int node) {
                    return k.control.getNode(node).index()
                           == variantIndex<ControlGraph::Operation, ControlGraph::ForLoopOp>();
                });

                AssertFatal(loop != reversed.end());

                for(auto node : multiplyNodes)
                {
                    auto nodeStack = controlStack(node, k);
                    AssertFatal(std::ranges::find(nodeStack, *loop) != nodeStack.end());
                }

                return *loop;
            }();

            ConstraintStatus status;

            auto isInLoop = [&](int node) -> bool {
                return k.control.compareNodes(UpdateCache, forLoop, node)
                       == ControlGraph::NodeOrdering::RightInBodyOfLeft;
            };

            auto d2lInLoop = k.control.getNodes<ControlGraph::LoadTileDirect2LDS>()
                                 .filter(isInLoop)
                                 .to<std::set>();

            auto loadTiledInLoop
                = k.control.getNodes<ControlGraph::LoadTiled>().filter(isInLoop).to<std::set>();

            auto loadLDSTileInLoop
                = k.control.getNodes<ControlGraph::LoadLDSTile>().filter(isInLoop).to<std::set>();

            Log::critical("{}{}{}",
                          ShowValue(multiplyNodes),
                          ShowValue(loadTiledInLoop),
                          ShowValue(loadLDSTileInLoop));

            using OrderCategories = std::map<ControlGraph::NodeOrdering, std::set<int>>;

            auto checkIt =
                [&]<bool Print>(int multiply, std::set<int> const& nodes, std::string const& name) {
                    OrderCategories cats;
                    for(int i = 0; i < static_cast<int>(ControlGraph::NodeOrdering::Count); i++)
                    {
                        cats[static_cast<ControlGraph::NodeOrdering>(i)];
                    }

                    for(auto node : nodes)
                    {
                        auto order = k.control.compareNodes(UpdateCache, multiply, node);
                        cats[order].insert(node);
                    }

                    if(Print)
                    {
                        Log::critical("");
                        Log::critical("Multiply {}/{}", multiply, name);
                        for(auto const& [order, loads] : cats)
                            Log::critical(
                                "{}: {}: ({})", toString(order), loads.size(), concatenate(loads));
                    }

                    return cats;
                };

            auto nodeDesc = [&](int node) { return toString(k.control.getNode(node)); };

            std::set<int> allNodes(multiplyNodes.begin(), multiplyNodes.end());
            allNodes.insert(d2lInLoop.begin(), d2lInLoop.end());
            allNodes.insert(loadTiledInLoop.begin(), loadTiledInLoop.end());
            allNodes.insert(loadLDSTileInLoop.begin(), loadLDSTileInLoop.end());

            // {
            //     auto someNodes = k.control.getNodes<ControlGraph::Assign>().filter(isInLoop);
            //     allNodes.insert(someNodes.begin(), someNodes.end());
            // }

            // {
            //     auto someNodes = k.control.getNodes<ControlGraph::Exchange>().filter(isInLoop);
            //     allNodes.insert(someNodes.begin(), someNodes.end());
            // }

            Graph::Hypergraph<std::string, std::nullopt_t, false> g;

            for(auto node : allNodes)
            {
                g.setElement(node, nodeDesc(node));
            }

            for(auto iterA = allNodes.begin(); iterA != allNodes.end(); ++iterA)
            {
                for(auto iterB = std::next(iterA); iterB != allNodes.end(); ++iterB)
                {
                    auto order = k.control.compareNodes(UpdateCache, *iterA, *iterB);

                    if(order == ControlGraph::NodeOrdering::LeftFirst)
                        g.addElement(std::nullopt, {*iterA}, {*iterB});
                    else if(order == ControlGraph::NodeOrdering::RightFirst)
                        g.addElement(std::nullopt, {*iterB}, {*iterA});
                    else
                        AssertFatal(order == ControlGraph::NodeOrdering::Undefined);
                }
            }

            auto truePred = [](auto x) { return true; };

            Graph::removeRedundantEdges(g, truePred);

            std::ofstream file("ordering.dot");

            auto subgraph = [&](int node) {
                auto text = g.getNode(node);
                return text.find("Assign") == std::string::npos
                       && text.find("Exchange") == std::string::npos;
            };

            file << "digraph {" << std::endl;

            // file << "subgraph clusterCF {label=\"Multiplies\";" << std::endl;
            for(auto node : g.getNodes())
            {
                if(subgraph(node))
                    file << "node" << node << "[label=\"" << node << " " << g.getNode(node)
                         << "\"];" << std::endl;
            }
            // file << "}" << std::endl;

            for(auto node : g.getNodes())
            {
                if(!subgraph(node))
                    file << "node" << node << "[label=\"" << node << " " << g.getNode(node)
                         << "\"];" << std::endl;
            }

            for(auto edge : g.getEdges())
            {
                auto loc = g.getLocation(edge);
                AssertFatal(loc.incoming.size() == 1);
                AssertFatal(loc.outgoing.size() == 1);

                file << "node" << loc.incoming.front() << " -> "
                     << "node" << loc.outgoing.front() << std::endl;
            }

            file << "}" << std::endl;

            // for(auto multiply : multiplyNodes)
            // {
            //     checkIt.template operator()<true>(multiply, d2lInLoop, "Direct2LDS");
            //     checkIt.template operator()<true>(multiply, loadTiledInLoop, "LoadTiled");
            //     checkIt.template operator()<true>(multiply, loadLDSTileInLoop, "LoadLDSTile");

            //     // OrderCategories globalCats;
            //     // for(auto load : loadTiledInLoop)
            //     // {
            //     //     auto order = k.control.compareNodes(UpdateCache, multiply, load);
            //     //     globalCats[order].insert(load);
            //     // }

            //     // OrderCategories ldsCats;
            //     // for(auto load : loadLDSTileInLoop)
            //     // {
            //     //     auto order = k.control.compareNodes(UpdateCache, multiply, load);
            //     //     ldsCats[order].insert(load);
            //     // }

            //     // Log::critical("Multiply {}", multiply);
            //     // Log::critical("  Global", multiply);
            //     // for(auto const& [order, loads] : globalCats)
            //     //     Log::critical(
            //     //         "{}: {}: ({})", toString(order), loads.size(), concatenate(loads));
            //     // if(!globalCats.contains(ControlGraph::NodeOrdering::Undefined))
            //     //     Log::critical("NO Global overlap!");

            //     // Log::critical("\n  LDS", multiply);
            //     // for(auto const& [order, loads] : ldsCats)
            //     //     Log::critical(
            //     //         "{}: {}: ({})", toString(order), loads.size(), concatenate(loads));
            //     // if(!ldsCats.contains(ControlGraph::NodeOrdering::Undefined))
            //     //     Log::critical("NO LDS overlap!");
            //     Log::critical("-----------");
            // }

            return status;
        }
    }
}
