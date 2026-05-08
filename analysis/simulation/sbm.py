import numpy as np
from networkit import Graph, nxadapter
import networkx as nx

class SBM():

    def __init__(self, 
                 p_similar:float, 
                 p_diff:float, 
                 num_nodes:int, 
                 num_groups:int,
                 props:list[float]=None,
                 sigma:float=None
                 ):
        
        self.validate_inputs(p_similar, p_diff, 
                             num_nodes, num_groups, 
                             props, sigma)

        self.p_similar = p_similar # probability of homogeneous edge
        self.p_diff = p_diff # probabilility of hetergeneous edge
        self.num_nodes = num_nodes # number of nodes in the graph
        self.num_groups = num_groups # how many groups to include

        if not props:

            # uniform group sizes
            self.groups = list(np.random.choice(
                np.arange(self.num_groups),    
                num_nodes,
                replace=True))
        else:
            # respect given proportions
            g = 0
            remainder = num_nodes
            self.groups = []
            for i in range(num_groups-1):
                # the number of nodes to have the current group
                num_in_curr = int(props[i]*num_nodes) 
                # append the group label for each node in curr group
                self.groups = self.groups + [g] * num_in_curr 

                # keep track of remaining nodes to assign
                remainder -= num_in_curr 
                # move to next group label
                g += 1 

            # add the remainder as the last group
            self.groups = self.groups + [g] * remainder
            # randomize group assignments
            np.random.shuffle(self.groups)

        
        self.nodes = np.arange(num_nodes)
        self.generate_probability_matrix(sigma=sigma)
        self.generate_graph()

    def validate_inputs(self, p_similar, p_diff, 
                        num_nodes, num_groups, 
                        props, sigma):
        if not (0 <= p_similar <= 1): raise ValueError("p_similar must be in [0, 1]")
        if not (0 <= p_diff <= 1): raise ValueError("p_diff must be in [0, 1]")
        if num_nodes <= 0: raise ValueError("num_nodes must be positive")
        if num_groups <= 0: raise ValueError("num_groups must be positive")
        if props:
            if not all(0 <= p <= 1 for p in props): raise ValueError("All props must be in [0, 1]")
            if not np.isclose(sum(props), 1.0): raise ValueError("props must sum to 1.0")
        if sigma is not None and sigma < 0: raise ValueError("sigma must be non-negative")


    def generate_probability_matrix(self, sigma:float=None):
        # initialize probability matrix
        np_groups = np.array(self.groups)
        P_mask = np_groups[: , None] == np_groups[None, :]
        self.P = np.where(P_mask, self.p_similar, self.p_diff)

        if sigma:
            # Add symmetric noise matrix
            epsilon = np.random.normal(0, sigma, self.P.shape)
            epsilon = np.triu(epsilon, 1)
            epsilon = epsilon + epsilon.T
            self.P = self.P + epsilon
            # Clip to guarantee valid probabilities, 
            # avoid altering un-noised probabilities
            self.P = np.clip(self.P, 0.0, 1.0)
            
        np.fill_diagonal(self.P, 0) # no self loops

    def generate_graph(self):
        # Evaluate only the upper triangle to ensure an undirected graph
        A_upper = np.triu(
            (np.random.random(self.P.shape) < self.P).astype(int), 
            1)
        self.A = A_upper + A_upper.T
        self.num_edges = int(np.sum(A_upper))
        G = nx.from_numpy_array(self.A)
        self.G = nxadapter.nx2nk(G)
