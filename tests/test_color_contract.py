import networkx as nx
import networkit as nk
import pytest

from clique_homology import betti_numbers
from clique_homology.stats_engine.random_coloring import random_coloring


def _line_graph_nk() -> nk.Graph:
    return nk.nxadapter.nx2nk(nx.path_graph(3))


def test_betti_numbers_rejects_non_string_colors() -> None:
    graph = _line_graph_nk()
    with pytest.raises(TypeError, match="must be strings"):
        betti_numbers(graph, ["RED", 1, "BLUE"])  # type: ignore[list-item]


def test_betti_numbers_rejects_color_outside_allowed_palette() -> None:
    graph = _line_graph_nk()
    with pytest.raises(ValueError, match="outside the allowed palette"):
        betti_numbers(graph, ["RED", "BLUE", "GREEN"], 
                      allowed_colors=["RED", "BLUE"])


def test_betti_numbers_uses_case_sensitive_palette_membership() -> None:
    graph = _line_graph_nk()
    with pytest.raises(ValueError, match="outside the allowed palette"):
        betti_numbers(graph, ["RED", "red", "RED"], allowed_colors=["RED"])


def test_random_coloring_stays_within_allowed_palette() -> None:
    recolored = random_coloring(["RED", "BLUE", "GREEN"], 
                                allowed_colors=["RED", "BLUE"])
    assert set(recolored).issubset({"RED", "BLUE"})
    assert len(recolored) == 3
    

def test_random_coloring_rejects_non_string_inputs() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        random_coloring(["RED", 1, "BLUE"])  # type: ignore[list-item]
