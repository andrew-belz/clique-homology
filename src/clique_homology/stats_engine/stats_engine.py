from clique_homology.stats_engine.betti_numbers import betti_numbers
from clique_homology.stats_engine.null_distribution import null_distribution
from clique_homology.stats_engine.p_values import calculate_p_vector
from networkit import Graph
import numpy as np

def stats_engine(
        G:Graph, colors:list[str], iters:int=100
        ) -> tuple[float, float, np.ndarray]:
    # out: list of numpy arrays
    null_dist = np.array(null_distribution(graph=G, 
                                           coloring=colors, 
                                           iterations=iters))
    obs_betti = betti_numbers(G=G, colors=colors)
    pval, obs, dist = calculate_p_vector(obs_betti, null_dist)

    return pval, obs, dist

    