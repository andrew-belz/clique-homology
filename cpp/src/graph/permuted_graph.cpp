#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
# include "betti.hpp"
using namespace std;


vector <vector <int>> generate_random_colorings(const std::vector<int>& original_coloring, int N) {
    // Initialize random colorings vector and allocate memory for it.
    vector <vector <int>> random_colorings;
    random_colorings.reserve(N);

    // Get random seed and create random number generator
    random_device rd;
    mt19937 gen(rd());

    vector <int> current_coloring = original_coloring;

    for (int i = 0; i < N; i++) {
        // Randomly shuffle the current coloring
        shuffle(current_coloring.begin(), current_coloring.end(), gen);
        random_colorings.push_back(current_coloring);
    }

    return random_colorings;
}




// unordered_set <vector <int>> generate_permuted_graph(const vector<vector<int>>& graph, const vector<int>& original_coloring, int N) {
    
//     unordered_set <vector <int>> null_dist;
//     null_dist.reserve(N);

//     // Get random colorings matrix
//     vector <vector <int>> random_colorings = generate_random_colorings(original_coloring, N);

//     for (int i = 0; i < N; i++) {
//         // Generate permuted graph and add it to the null distribution. Using
//         //    emplace for performance.
//         null_dist.emplace(generate_betti_vector(graph, random_colorings[i]));
//     }

//     return null_dist;    
// }