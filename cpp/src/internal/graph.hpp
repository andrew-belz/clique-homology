#pragma once
#include <vector>
#include "data_types.hpp"
using std::vector;

class Graph {
private:
    AdjList adj_list;
    const ColorList colors;

    void remove_monochromatic_edges();
    bool are_same_color(VertexId i, VertexId j);
   
public: 
    Graph(AdjList &adj, ColorList &colors);
    const vector<VertexId>& get_neighbors(VertexId i) const;
};