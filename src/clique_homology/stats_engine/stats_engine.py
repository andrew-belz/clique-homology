from clique_homology.stats_engine.betti_numbers import betti_numbers
from clique_homology.stats_engine.null_distribution import null_distribution
from clique_homology.stats_engine.p_values import calculate_p_value
from networkit import Graph
import numpy as np

class StatsEngine:

    def __init__(self, G:Graph, labels:list[str]):
        self.G = G
        self.labels = labels


    def observed_betti(self) -> np.ndarray:
        return betti_numbers(G=self.G, colors=self.labels)

    def null_dist(self, iters:int=2000):
        return np.array(null_distribution(graph=self.G, 
                                           coloring=self.labels, 
                                           iterations=iters))

    def pval(self, iters:int=2000):
        obs = self.observed_betti()
        null_dist = self.null_dist(iters=iters)

        p, _ , _ = calculate_p_value(obs, null_dist)

        return p


