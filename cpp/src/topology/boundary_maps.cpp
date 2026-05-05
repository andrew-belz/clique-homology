#include <vector>
#include "sparse_matrix.hpp"
#include <map>
#include <algorithm>

using std::vector, std::map, std::pair;


vector<vector<unsigned int>> get_faces(const vector<unsigned int>& clique) {
    // get a vector of the faces (n-1)-dimensional cliques for a given clique
    vector<vector<unsigned int>> faces(clique.size());
    // loop through the clique and leave one vertex out: populate the faces
    for (int i = 0; i < clique.size(); i++) {
        for (int j = 0; j < faces.size(); j++) {
            if (i != j) {faces[j].push_back(clique[i]);}
        }
    }

    return faces;
}

// size_map is vector of 
//  vector of same size cliques
//      a clique is a vector of integers
// assume that it is all sorted in ascending order for binary search
vector<SparseBinaryMatrix> construct_boundary_maps(const vector<vector<vector<unsigned int>>>& size_map) {
    vector<SparseBinaryMatrix> boundary_maps(size_map.size()-1);
    // for each clique size k
    for (int k = 0; k < size_map.size()-1; k++) {

        // get our lists of cliques of different sizes
        const auto& k_cliques = size_map[k];
        const auto& kplus1_cliques = size_map[k+1];

        // initalize boundary matrix to zeros of appropriate size
        boundary_maps[k] = SparseBinaryMatrix(k_cliques.size(), kplus1_cliques.size());

        // loop through the k+1 cliques and get the faces
        // keep track of which position we assign it to in the binary matrix
        unsigned int j = 0;
        for (const auto& kplus1_clique : kplus1_cliques) {
            vector<vector<unsigned int>> faces = get_faces(kplus1_clique);

            // do a binary search of the faces since it is sorted
            for (const auto& face : faces) {
                // Perform a binary search
                auto it = std::lower_bound(k_cliques.begin(), k_cliques.end(), face);
        
                // Calculate the exact index, which will correspond to the position in the matrix
                unsigned int i = std::distance(k_cliques.begin(), it);

                // get the k-th boundary map and insert a 1 in the appropriate spot
                // one method sets (i, j)th entry to 1
                boundary_maps[k].one(i, j);

            }
            // increment to the next clique
            j++;
        }

        // optionally row reduce the matrix at this step, but no need
        // auto [rank, nullity] = boundar_maps[k].row_reduce();
    }

    // vector of binary matrices
    return boundary_maps;
}