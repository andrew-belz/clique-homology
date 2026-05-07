# for generating test cases
import networkx as nx # type: ignore
import networkit as nk # type: ignore
from random import choice, seed, sample
import pytest
import time

# zero multi-threading errors to worry about
nk.setNumberOfThreads(1)

# functions to test
from clique_homology.stats_engine.betti_numbers import *

# to visually verify the test case
import matplotlib.pyplot as plt

seed(122)

# --- Support Functions ---
@pytest.fixture
def generate_edge_case_graphs():

    # convert a networkx graph to a networkit graph
    convert = lambda G: nk.nxadapter.nx2nk(G)

    """
    use method='clique'.
    G[i]: nk.Graph object.
    c[i]: associated node coloring.
    exp[i]: expected output of the betti_numbers function.
    """

    # empty graph
    G0 = nk.Graph()
    c0 = []
    exp0 = np.array([])

    # 5 nodes, 0 edges, 1 color
    G1 = nx.Graph()
    G1.add_nodes_from([0, 1, 2, 3, 4])
    G1 = convert(G1)
    c1 = ["red"] * 5
    exp1 = np.array([5])

    # 1 node, 0 edges, 1 color
    G2 = nx.Graph()
    G2.add_node(0)
    G2 = convert(G2)
    c2 = ["red"]
    exp2 = np.array([1])

    # 2 nodes, 1 edge, 1 color
    G3 = nx.Graph([(0, 1)])
    G3 = convert(G3)
    c3 = ["red"] * 2
    exp3 = np.array([1, 0])

    # 2 nodes, 1 edge, 2 colors
    G4 = nx.Graph([(0, 1)])
    G4 = convert(G4)
    c4 = ["red", "blue"]
    exp4 = np.array([2, 0])

    # 3 nodes, 3 edges, 1 color
    G5 = nx.complete_graph(3)
    G5 = convert(G5)
    c5 = ["red"] * 3
    exp5 = np.array([1, 0, 0])

    # petersen graph, 1 color
    G6 = nx.petersen_graph()
    G6 = convert(G6)
    c6 = ["red"] * 10
    exp6 = np.array([1, 6])
    # use this code to see what this graph looks like:
    # nx.draw(nx.petersen_graph(), node_color = c5)
    # plt.show()
    # plt.clf()

    # petersen graph, 2 colors
    G7 = G6
    c7 = ["red"] * 5 + ["blue"] * 5
    exp7 = np.array([2, 2])

    # octohedreon: a hollow 2-sphere comprised of triangles
    G8 = nx.octahedral_graph() 
    G8 = convert(G8)
    c8 = ["red"] * 6
    exp8 = np.array([1, 0, 1])

    # nx.draw(nx.octahedral_graph(), node_color = c8)
    # plt.show()
    # plt.clf()

    cases = [
        (G0, c0, exp0), (G1, c1, exp1),
        (G2, c2, exp2), (G3, c3, exp3),
        (G4, c4, exp4), (G5, c5, exp5),
        (G6, c6, exp6), (G7, c7, exp7),
        (G8, c8, exp8)
    ]

    return cases

# --- Main ---

def test_edge_case_graphs():
    cases = generate_edge_case_graphs()
    print("Start")

    for i, (G, c, exp) in enumerate(cases):
        print(f"Case {i}:")
        obs = betti_numbers(G, c)
        print(obs)
        assert np.array_equal(obs, exp)
        print("passed")


def test_betti_numbers_handles_non_contiguous_node_ids() -> None:
    # Build a connected path on node IDs [0, 2, 3, 4].
    graph = nk.Graph(5, weighted=False, directed=False)
    graph.addEdge(0, 2)
    graph.addEdge(2, 3)
    graph.addEdge(3, 4)
    graph.removeNode(1)

    colors = ["red", "red", "red", "red"]
    observed = betti_numbers(graph, colors, method="clique")

    assert np.array_equal(observed, np.array([1, 0]))


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
# -------------------------------------------------
if __name__ == "__main__":
    print("START")
    test_edge_case_graphs()
