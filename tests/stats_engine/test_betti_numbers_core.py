import networkit as nk
import networkx as nx
import numpy as np
import pytest

from clique_homology.stats_engine.betti_numbers import (
    betti_numbers,
    boundary_maps,
    get_cliques,
    get_colored_subgraphs,
    get_max_clique_size,
    ranks_and_nullities,
    _validate_colors
)


def nx_to_nk(graph: nx.Graph) -> nk.Graph:
    return nk.nxadapter.nx2nk(graph)


def test_get_max_clique_size_for_empty_graph() -> None:
    assert get_max_clique_size(nk.Graph()) == 0


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        (nx.path_graph(4), 2),
        (nx.complete_graph(4), 4),
    ],
)
def test_get_max_clique_size_known_graphs(
    graph: nx.Graph, expected: int
    ) -> None:
    assert get_max_clique_size(nx_to_nk(graph)) == expected


def test_get_cliques_triangle_is_complete_and_unique() -> None:
    cliques = list(get_cliques(nx_to_nk(nx.complete_graph(3))))
    assert set(cliques) == {
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    }


def test_get_colored_subgraphs_partitions_nodes_by_color() -> None:
    graph = nx_to_nk(nx.path_graph(3))
    subgraphs = list(get_colored_subgraphs(graph, ["red", "blue", "red"]))

    assert len(subgraphs) == 2
    assert subgraphs[0].numberOfNodes() == 2
    assert subgraphs[0].numberOfEdges() == 0
    assert subgraphs[1].numberOfNodes() == 1
    assert subgraphs[1].numberOfEdges() == 0


def test_boundary_maps_empty_input_returns_no_maps() -> None:
    assert boundary_maps([]) == []


def test_boundary_maps_creates_correct_shape() -> None:
    maps = boundary_maps([(0,), (1,), (0, 1)])
    assert len(maps) == 1
    assert maps[0].shape == (2, 1)
    assert maps[0].sum() == 2


@pytest.mark.parametrize(
    ("matrix", "expected_rank", "expected_nullity"),
    [
        (np.zeros((2, 3), dtype=int), 0, 3),
        (np.eye(3, dtype=int), 3, 0),
        (np.array([[1, 1], [1, 1]], dtype=int), 1, 1),
        (np.array([[1, 0, 0], [0, 1, 0]], dtype=int), 2, 1),
    ],
)
def test_ranks_and_nullities_known_cases(
    matrix: np.ndarray,
    expected_rank: int,
    expected_nullity: int,
) -> None:
    rank, nullity = ranks_and_nullities(matrix)
    assert rank == expected_rank
    assert nullity == expected_nullity
    assert rank + nullity == matrix.shape[1]


def test_betti_numbers_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="Invalid method"):
        betti_numbers(nx_to_nk(nx.path_graph(3)), 
                      ["red"] * 3, method="bad")


def test_betti_numbers_rejects_color_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must match number of graph nodes"):
        betti_numbers(nx_to_nk(nx.path_graph(3)), ["red", "blue"])


def test_betti_numbers_clique_equals_summed_subgraph_rows() -> None:
    graph = nx_to_nk(nx.path_graph(4))
    colors = ["red", "red", "blue", "blue"]
    clique = betti_numbers(graph, colors, method="clique")
    subgraph = betti_numbers(graph, colors, method="subgraph")

    assert np.array_equal(clique, subgraph.sum(axis=0))


def test_betti_numbers_path_graph_single_color() -> None:
    observed = betti_numbers(nx_to_nk(nx.path_graph(4)), 
                             ["red"] * 4, method="clique")
    assert np.array_equal(observed, np.array([1, 0]))

def test_validate_colors() -> None:
    colors = [0, 1, 2]

    graph = nk.Graph(5, weighted=False, directed=False)
    graph.addEdge(0, 2)
    graph.addEdge(2, 3)
    graph.addEdge(3, 4)
    graph.removeNode(1)
    with pytest.raises(ValueError, match="must match number of graph nodes"):
        _validate_colors(graph, colors)

    with pytest.raises(TypeError, match="must be strings"):
        _validate_colors(nk.Graph(3), colors)