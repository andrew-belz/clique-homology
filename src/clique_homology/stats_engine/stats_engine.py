from clique_homology.stats_engine.betti_numbers import betti_numbers
from clique_homology.stats_engine.null_distribution import null_distribution
from clique_homology.stats_engine.p_values import calculate_p_value
from networkit import Graph
import numpy as np

class StatsEngine:

    def __init__(self, G:Graph, labels:list[str]):
        self.G = G
        self.labels = labels
        self.obs_betti = None
        self.null_betti = None
        self.p = None
        self.obsD2 = None
        self.nullD2 = None


    def observed_betti(self) -> np.ndarray:
        self.obs_betti = betti_numbers(G=self.G, colors=self.labels)
        return self.obs_betti

    def null_dist(self, iters:int=2000):
        self.null_betti = np.array(null_distribution(graph=self.G, 
                                           coloring=self.labels, 
                                           iterations=iters))
        
        return self.null_betti

    def pval(self, iters:int=2000):
            
        if self.p is None:
            obs = self.observed_betti() if self.obs_betti is None else self.obs_betti
            null_dist = self.null_dist(iters=iters) if self.null_betti is None else self.null_betti

            p, obsD2 , nullD2 = calculate_p_value(obs, null_dist)

            self.p = p
            self.obsD2 = obsD2
            self.nullD2 = nullD2

            return p, obsD2, nullD2
        
        else:
            return self.p, self.obsD2, self.nullD2
