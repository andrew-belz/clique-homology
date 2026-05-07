from clique_homology.stats_engine.betti_numbers import betti_numbers
from clique_homology.stats_engine.null_distribution import null_distribution
# change this back to p_values instead
from clique_homology.stats_engine.p_values import calculate_p_vector
from networkit import Graph, nxadapter
import numpy as np

def stats_engine(
        G:Graph, colors:list[str], iters:int=100, 
        allowed_colors: list[str] | None = None
        ) -> tuple[float, float, np.ndarray]:
    # out: list of numpy arrays
    null_dist = np.array(null_distribution(graph=G, 
                                           coloring=colors, 
                                           iterations=iters, 
                                           allowed_colors=allowed_colors))
    obs_betti = betti_numbers(G=G, colors=colors, allowed_colors=allowed_colors)
    pval, obs, dist = calculate_p_vector(obs_betti, null_dist)

    return pval, obs, dist

    