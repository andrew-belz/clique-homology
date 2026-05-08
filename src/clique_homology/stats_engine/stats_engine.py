from clique_homology.stats_engine.betti_numbers import betti_numbers
from clique_homology.stats_engine.null_distribution import null_distribution
from clique_homology.stats_engine.p_values import calculate_p_value
from networkit import Graph
import numpy as np

def stats_engine(
        G:Graph, colors:list[str], iters:int=100,
        ) -> tuple[float, float, np.ndarray]:
    """Compute the P-value, distance from the mean of the observed betti vector,
        and the distance from the mean of each vector of betti numbers in the 
        null distribution.

    Args:
        G (Graph): A graph
        colors (list[str]): A list of colors
        iters (int, optional): A desired number of iterations; this determines
            the size of the null distribution. Defaults to 100.

    Returns:
        tuple[float, float, np.ndarray]: A tuple containing three things, in
            this order:
                1. a float representing the P-value of the hypothesis test, or
                    the likelihood that the observed betti vector could have
                    been produced by random chance, given a provided null
                    distribution of random betti vector permutations.
                2. a float representing the distance of the observed betti
                    vector from the mean in standard deviations.
                3. a numpy array containing the distance from the mean of every
                    vector of betti numbers in the null distribution, measured
                    in standard deviations.
    """

    null_dist = np.array(null_distribution(graph=G, 
                                           coloring=colors, 
                                           iterations=iters))
    obs_betti = betti_numbers(G=G, colors=colors)
    pval, obs, dist = calculate_p_value(obs_betti, null_dist)

    return pval, obs, dist