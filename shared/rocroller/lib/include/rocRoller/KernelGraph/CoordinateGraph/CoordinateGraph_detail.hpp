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

#pragma once

#include <vector>

#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>

namespace rocRoller
{

    namespace KernelGraph::CoordinateGraph
    {
        template <typename Mapping, typename Visitor>
        std::optional<Expression::ExpressionPtr> traverseSingle(CoordinateGraph const& graph,
                                                                int                    dimToGet,
                                                                Mapping&               dimsWeHave,
                                                                Visitor&               visitor,
                                                                int                    depth = 0)
        {
            constexpr Graph::Direction Dir = Visitor::Direction;
            bool constexpr forward         = Dir == Graph::Direction::Downstream;
            auto constexpr OppositeDir     = opposite(Dir);

            std::string tab;
            for(int i = 0; i < depth; i++)
                tab += "  ";

            {
                auto iter = dimsWeHave.find(dimToGet);
                if(iter != dimsWeHave.end())
                {
                    Log::critical(
                        "{}traverseSingle({}) -> {}", tab, dimToGet, toString(iter->second));
                    return iter->second;
                }
            }

            Log::critical("{}traverseSingle({})", tab, dimToGet);

            // The real implementation, wrapped in a lambda so we can early return.
            auto rv = [&]() -> std::optional<Expression::ExpressionPtr> {
                auto isCoordinateEdge = [&](int edgeTag) -> bool {
                    auto e = graph.getEdge(edgeTag);
                    return std::holds_alternative<CoordinateTransformEdge>(e);
                };

                auto getDim = [&](int dimTag) { return graph.getNode(dimTag); };

                auto allNeighbours = graph.getNeighbours<OppositeDir>(dimToGet);
                auto ctNeighbours  = std::ranges::views::filter(allNeighbours, isCoordinateEdge);

                for(int ctEdgeTag : ctNeighbours)
                {
                    auto loc = graph.getLocation(ctEdgeTag);

                    auto const& srcTags = forward ? loc.incoming : loc.outgoing;
                    auto const& dstTags = forward ? loc.outgoing : loc.incoming;

                    std::vector<Expression::ExpressionPtr> indexes;
                    indexes.reserve(srcTags.size());
                    for(auto srcTag : srcTags)
                    {
                        auto exp = traverseSingle(graph, srcTag, dimsWeHave, visitor, depth + 1);
                        if(!exp.has_value())
                            break;

                        indexes.push_back(*exp);
                    }
                    if(indexes.size() != srcTags.size())
                        continue;

                    auto incomingNodes = map(getDim, loc.incoming).template to<std::vector>();
                    auto outgoingNodes = map(getDim, loc.outgoing).template to<std::vector>();

                    visitor.setLocation(std::move(indexes),
                                        std::move(incomingNodes),
                                        std::move(outgoingNodes),
                                        loc.incoming,
                                        loc.outgoing);

                    auto exps    = visitor.call(std::get<Edge>(loc.element));
                    auto locName = toString(loc.element);
                    AssertFatal(exps.size() == dstTags.size(),
                                ShowValue(dstTags.size()),
                                ShowValue(exps.size()),
                                ShowValue(locName),
                                ShowValue(Visitor::Name));
                    for(int idx = 0; idx < exps.size(); idx++)
                    {
                        auto tag = dstTags[idx];
                        // auto iter = dimsWeHave.find(tag);
                        // AssertFatal(iter == dimsWeHave.end(),
                        //             ShowValue(ctEdgeTag),
                        //             ShowValue(tag),
                        //             ShowValue(iter->second),
                        //             ShowValue(exps[idx]));
                        if(!dimsWeHave.contains(tag))
                            dimsWeHave[tag] = exps[idx];
                    }

                    // AssertFatal(exps.size() == 1, ShowValue(exps.size()));
                    return dimsWeHave.at(dimToGet);
                }

                return std::nullopt;
            }();

            // if(rv != nullptr)
            //     dimsWeHave[dimToGet] = rv;

            if(rv.has_value())
                Log::critical("{}traverseSingle({}) -> {}", tab, dimToGet, toString(*rv));
            else
                Log::critical("{}traverseSingle({}) -> nullopt", tab, dimToGet);

            return rv;
        }

        template <typename Mapping, typename Visitor>
        std::vector<Expression::ExpressionPtr> traverseLazy(CoordinateGraph const& graph,
                                                            std::vector<int>       dimsToGet,
                                                            Mapping                dimsWeHave,
                                                            Visitor&               visitor)
        {
            {
                std::ofstream f("thistest.dot");
                f << graph.toDOT();
            }

            Log::critical("traverseLazy<{}>({})", toString(Visitor::Direction), dimsToGet.at(0));
            for(auto const& [tag, expr] : dimsWeHave)
            {
                Log::critical("{}: {}", tag, toString(expr));
            }

            std::vector<Expression::ExpressionPtr> rv;
            rv.reserve(dimsToGet.size());

            for(auto& dim : dimsToGet)
            {
                auto expr = traverseSingle(graph, dim, dimsWeHave, visitor);
                AssertFatal(expr.has_value(), ShowValue(dim));
                rv.push_back(*expr);
            }

            return rv;
        }

    }
}
