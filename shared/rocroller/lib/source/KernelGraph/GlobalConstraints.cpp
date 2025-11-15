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

        std::string findEscapingSequenceEdges(KernelGraph const& k)
        {
            std::string rv;

            for(auto edge : k.control.getEdges<ControlGraph::Sequence>())
            {
                auto loc = k.control.getLocation(edge);

                AssertFatal(loc.incoming.size() == 1,
                            "Sequence edge should have exactly one incoming edge");
                AssertFatal(loc.outgoing.size() == 1,
                            "Sequence edge should have exactly one outgoing edge");

                // auto bodyParentIn  = bodyParents(loc.incoming[0], k).take(1).only().value();
                // auto bodyParentOut = bodyParents(loc.outgoing[0], k).take(1).only().value();

                auto notThisEdge = [edge](int x) { return x != edge; };

                auto roots = k.control.roots().to<std::set>();

                auto getBodyParent
                    = [&](int startNode) -> std::tuple<int, ControlGraph::ControlEdge> {
                    auto prevNode = startNode;

                    for(auto node : k.control.depthFirstVisit(
                            prevNode, notThisEdge, Graph::Direction::Upstream))
                    {
                        if(node == startNode)
                            continue;

                        auto lastEdgeIdx = k.control.findEdge(node, prevNode);

                        AssertFatal(lastEdgeIdx.has_value(),
                                    "Graph walking error: No edge found between ",
                                    prevNode,
                                    " and ",
                                    node);

                        auto lastEdge = k.control.getEdge(*lastEdgeIdx);

                        if(!std::holds_alternative<ControlGraph::Sequence>(lastEdge))
                        {
                            return {node, lastEdge};
                        }

                        prevNode = node;

                        AssertFatal(!roots.contains(node),
                                    "Got to a root node with only Sequence edges");
                    }

                    return {startNode, ControlGraph::Sequence()};
                };

                auto [bodyParentIn, bodyEdgeIn]   = getBodyParent(loc.incoming[0]);
                auto [bodyParentOut, bodyEdgeOut] = getBodyParent(loc.outgoing[0]);

                if(bodyParentIn != bodyParentOut || bodyEdgeIn.index() != bodyEdgeOut.index())
                {
                    rv += fmt::format(
                        "Sequence edge {} ({} -> {}) escapes from {} {} ({}) to {} {} ({})\n",
                        edge,
                        loc.incoming[0],
                        loc.outgoing[0],
                        bodyParentIn,
                        toString(k.control.getNode(bodyParentIn)),
                        toString(bodyEdgeIn),
                        bodyParentOut,
                        toString(k.control.getNode(bodyParentOut)),
                        toString(bodyEdgeOut));
                }
            }

            return rv;
        }

        std::string findCycles(KernelGraph const& k,
                               int                node,
                               std::set<int>&     visitedNodes,
                               std::vector<int>   path)
        {
            if(visitedNodes.contains(node))
            {
                auto iter = std::ranges::find(path, node);
                if(iter != path.end())
                {
                    path.push_back(node);
                    iter = std::ranges::find(path, node);
                    return fmt::format("Cycle found: {}\n", fmt::join(iter, path.end(), " -> "));
                }
                return "";
            }

            visitedNodes.insert(node);
            path.push_back(node);

            std::string rv;

            // rv += fmt::format("Visiting node {} from {}\n", node, fmt::join(path, " -> "));

            for(auto outgoingEdge : k.control.getNeighbours<Graph::Direction::Downstream>(node))
            {
                for(auto outgoingNode :
                    k.control.getNeighbours<Graph::Direction::Downstream>(outgoingEdge))
                {
                    // rv += fmt::format("    : {} -> {}\n", outgoingEdge, outgoingNode);
                    rv += findCycles(k, outgoingNode, visitedNodes, path);
                }
            }

            return rv;
        }

        std::string findControlGraphCycles(KernelGraph const& k)
        {
            std::string rv;

            std::set<int> visitedNodes;

            for(auto node : k.control.roots())
            {
                rv += findCycles(k, node, visitedNodes, {node});
            }

            return rv;
        }

        ConstraintStatus WalkableControlGraph(KernelGraph const& k)
        {
            TIMER(t, "Constraint::WalkableControlGraph");
            WalkableControlGraphVisitor visitor(k);
            visitor.walk();

            auto allNodes = k.control.getNodes().to<std::set>();

            // if(visitor.visitedNodes != allNodes)
            if(!visitor.status.satisfied)
            {
                // Log::critical("Here!");
                std::set<int> nonVisitedNodes;
                std::set_difference(allNodes.begin(),
                                    allNodes.end(),
                                    visitor.visitedNodes.begin(),
                                    visitor.visitedNodes.end(),
                                    std::inserter(nonVisitedNodes, nonVisitedNodes.end()));

                std::ostringstream msg;

                msg << "Not all nodes were visited! ";

                auto escapingSequenceMsg = findEscapingSequenceEdges(k);
                auto controlGraphCycles  = findControlGraphCycles(k);

                if(!escapingSequenceMsg.empty())
                {
                    msg << "Escaping sequence edges:\n" << escapingSequenceMsg;
                }

                if(!controlGraphCycles.empty())
                {
                    msg << "Control graph cycles:\n" << controlGraphCycles;
                }

                if(escapingSequenceMsg.empty() && controlGraphCycles.empty())
                {
                    msg << "Missing nodes: ";
                    streamJoin(msg, nonVisitedNodes, ", ");
                    msg << "\n All nodes: ";
                    streamJoin(msg, allNodes, ", ");
                    msg << "\n Visited nodes: ";
                    streamJoin(msg, visitor.visitedNodes, ", ");
                }

                visitor.status.combine(false, msg.str());
            }

            return visitor.status;
        }
    }
}
