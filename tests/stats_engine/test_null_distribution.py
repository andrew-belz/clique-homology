import sys
import numpy as np
import pytest

from clique_homology.stats_engine.null_distribution import null_distribution


def test_null_distribution_returns_one_result_per_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = object()
    coloring = ["red", "blue"]

    # Mock for random_coloring, which is called with proportional=True
    def fake_random_coloring(colors: list[str], proportional: bool) -> list[str]:
        assert colors == ["red", "blue"]
        assert proportional is True
        return ["blue", "red"]

    # Mock for betti_numbers
    def fake_betti_numbers(_graph: object, new_coloring: list[str]) -> np.ndarray:
        assert new_coloring == ["blue", "red"]
        return np.array([1, 0], dtype=int)

    # Patch dependencies in the module where they are used
    module = sys.modules["clique_homology.stats_engine.null_distribution"]
    monkeypatch.setattr(module, "random_coloring", fake_random_coloring)
    monkeypatch.setattr(module, "betti_numbers", fake_betti_numbers)

    observed = null_distribution(graph, coloring, iterations=4)

    assert len(observed) == 4
    assert all(np.array_equal(value, np.array([1, 0])) for value in observed)


def test_null_distribution_zero_iterations_returns_empty_list() -> None:
    observed = null_distribution(object(), ["red"], iterations=0)
    assert observed == []


def test_null_distribution_rejects_negative_iterations() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        null_distribution(object(), ["red"], iterations=-1)


def test_null_distribution_propagates_downstream_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_random_coloring(colors: list[str], proportional: bool) -> list[str]:
        return ["red"]

    def fake_betti_numbers(_graph: object, _colors: list[str]) -> np.ndarray:
        raise RuntimeError("boom")

    module = sys.modules["clique_homology.stats_engine.null_distribution"]
    monkeypatch.setattr(module, "random_coloring", fake_random_coloring)
    monkeypatch.setattr(module, "betti_numbers", fake_betti_numbers)

    with pytest.raises(RuntimeError, match="boom"):
        null_distribution(object(), ["red"], iterations=1)
