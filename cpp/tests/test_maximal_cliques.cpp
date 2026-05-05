#include <gtest/gtest.h>
#include "graph.hpp"
#include "maximal_cliques.hpp"

TEST(test_maximal_cliques, empty_graph) {
    AdjList empty_adj;
    ColorList empty_colors;
    Graph test_graph(empty_adj, empty_colors);

    EXPECT_TRUE(find_maximal_cliques(test_graph) == CliqueList{});
}