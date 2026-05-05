#include <vector>
#include <algorithm>
#include "graph.hpp"
#include "data_types.hpp"
using std::remove_if;
using std::vector;


Graph::Graph(AdjList& adj, ColorList& colors) : adj_list(std::move(adj)), colors(std::move(colors)) {}

void Graph::remove_monochromatic_edges() {
    for (int i = 0; i < adj_list.size(); i++) {
        adj_list[i].erase(remove_if(adj_list[i].begin(), adj_list[i].end(), [i, this](int j) {
            return are_same_color(i, j);
        }), adj_list[i].end());
    }
}

bool Graph::are_same_color(int i, int j) {
    return colors[i] == colors[j];
}

const vector<VertexId>& Graph::get_neighbors(int i) const {
    return adj_list[i];
}