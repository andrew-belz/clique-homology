import pytest
from clique_homology.stats_engine.stats_engine import StatsEngine
from networkit import Graph, nxadapter
import networkx as nx
import numpy as np


def test_stats_engine():
    nxPG = nx.petersen_graph()
    colors = np.random.choice(["red", "green"], nxPG.number_of_nodes(), replace=True).tolist()

    engine = StatsEngine(G=nxadapter.nx2nk(nxPG), labels=colors)
    p, obs, dist = engine.pval(iters=1000)

    # make sure outputs make sense
    assert (0 <= p <= 1)
    assert (obs >= 0)
    assert (len(dist) == 1000)
    assert np.all(dist >= 0)
    
    # assert p-value is being computed accurately
    assert (np.sum(dist >= obs) == int(p*1000))

    # test caching behavior
    p2, obs2, dist2 = engine.pval()
    assert p == p2
    assert obs == obs2
    assert np.array_equal(dist, dist2)
