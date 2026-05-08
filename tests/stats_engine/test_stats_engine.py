import pytest
from clique_homology.stats_engine.stats_engine import stats_engine
from networkit import Graph, nxadapter
import networkx as nx
import numpy as np


def test_stats_engine():
    nxPG = nx.petersen_graph()
    colors = np.random.choice(["red", "green"], nxPG.number_of_nodes(), replace=True)

    p, obs, dist = stats_engine(G=nxadapter.nx2nk(nxPG), colors=colors, iters=1000)

    # make sure outputs make sense
    assert (0 <= p <= 1)
    assert (obs >= 0)
    assert (len(dist) == 1000)
    assert np.all(dist >= 0)
    
    # assert p-value is being computed accurately
    assert (np.sum(dist >= obs) == int(p*1000))
