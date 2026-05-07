import pytest
import numpy as np
import networkx as nx
import networkit as nk
from random import seed

# functions to test
from clique_homology.stats_engine.betti_numbers import betti_numbers, boundary_maps

# --- Test Environment Setup ---
# Zero multi-threading errors to worry about
nk.setNumberOfThreads(1)
seed(122)

# --- Support Functions ---
def generate_edge_case_graphs():
    """
    Generates test data for Betti numbers.
    Returns a list of tuples: (networkit_graph, colors_list, expected_array)
    """
    # convert a networkx graph to a networkit graph
    convert = nk.nxadapter.nx2nk 

    # Case 0: empty graph
    G0 = nk.Graph()
    c0 = []
    exp0 = np.array([])

    # Case 1: 5 nodes, 0 edges, 1 color
    G1 = convert(nx.Graph([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])) # ensuring nodes exist without edges
    G1 = nk.Graph(5) # Simpler way to initialize 5 disconnected nodes in networkit
    c1 = ["red"] * 5
    exp1 = np.array([5])

    # Case 2: 1 node, 0 edges, 1 color
    G2 = nk.Graph(1)
    c2 = ["red"]
    exp2 = np.array([1])

    # Case 3: 2 nodes, 1 edge, 1 color
    G3 = convert(nx.Graph([(0, 1)]))
    c3 = ["red"] * 2
    exp3 = np.array([1, 0])

    # Case 4: 2 nodes, 1 edge, 2 colors
    G4 = convert(nx.Graph([(0, 1)]))
    c4 = ["red", "blue"]
    exp4 = np.array([2, 0])

    # Case 5: 3 nodes, 3 edges, 1 color
    G5 = convert(nx.complete_graph(3))
    c5 = ["red"] * 3
    exp5 = np.array([1, 0, 0])

    # Case 6: petersen graph, 1 color
    G6 = convert(nx.petersen_graph())
    c6 = ["red"] * 10
    exp6 = np.array([1, 6])

    # Case 7: petersen graph, 2 colors
    G7 = convert(nx.petersen_graph())
    c7 = ["red"] * 5 + ["blue"] * 5
    exp7 = np.array([2, 2])

    # Case 8: octahedron: a hollow 2-sphere comprised of triangles
    G8 = convert(nx.octahedral_graph()) 
    c8 = ["red"] * 6
    exp8 = np.array([1, 0, 1])

    return [
        (G0, c0, exp0), (G1, c1, exp1),
        (G2, c2, exp2), (G3, c3, exp3),
        (G4, c4, exp4), (G5, c5, exp5),
        (G6, c6, exp6), (G7, c7, exp7),
        (G8, c8, exp8)
    ]

# --- Test Cases ---

@pytest.mark.parametrize("graph, colors, expected", generate_edge_case_graphs())
def test_edge_case_graphs(graph, colors, expected):
    """Test standard graph configurations using method='clique'."""
    observed = betti_numbers(graph, colors)
    
    # Use numpy's built-in testing assert for clearer diffs on failure
    np.testing.assert_array_equal(observed, expected)


def test_betti_numbers_handles_non_contiguous_node_ids() -> None:
    # Build a connected path on node IDs [0, 2, 3, 4].
    graph = nk.Graph(5, weighted=False, directed=False)
    graph.addEdge(0, 2)
    graph.addEdge(2, 3)
    graph.addEdge(3, 4)
    graph.removeNode(1)

    colors = ["red", "red", "red", "red"]
    observed = betti_numbers(graph, colors, method="clique")

    np.testing.assert_array_equal(observed, np.array([1, 0]))


def test_subgraph_method_empty_graph_returns_matrix_shape() -> None:
    graph = nk.Graph()
    observed = betti_numbers(graph, [], method="subgraph")

    assert observed.ndim == 2
    assert observed.shape == (0, 0)


def test_boundary_maps_accept_unsorted_complete_cliques() -> None:
    maps = boundary_maps([(0, 1), (0,), (1,)])

    assert len(maps) == 1
    assert maps[0].shape == (2, 1)
    assert maps[0].sum() == 2