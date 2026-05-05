import networkx as nx
import networkit as nk # for network generation and for use in the betti_numbers function eventually
import numpy as np # matrix calculations and data generation
from sklearn.model_selection import ParameterSampler # sampling a parameter grid for simulation
# import time # for time benchmarking in simulation
import pandas as pd # analzying and visualizing the results
from clique_homology.stats_engine.stats_engine import stats_engine

np.random.seed(122)

# script parameters #
n_validations = 1 # num of times to run one set of parameters
param_sample_size = 10 # number of parameter sets to sample
null_dist_iters = 10 # how many observations to compute in the null distribution
alpha = 0.05 # significance level

# ---

def generate_SBM_growth(P:np.ndarray) -> nx.Graph:
    """
    Generate iterative SBM of size n given a set of vertex edge pair probabilities
    
    :param P: n*n edge probability matrix
    """
    n = P.shape[0]
    
    # add the first node
    G = nx.Graph()
    G.add_node(0)

    for i in range(1, n):

        # add the node to the graph, then check it against the existing nodes to see if we will add an edge or not
        G.add_node(i)

        for j in range(0, i):
            # choose rand num U(0, 1)
            runif = np.random.random()

            # we ensure we only look at the upper triangular portion of P since it is upper triangular
            if runif <= P[j, i]:
                # there is an edge
                G.add_edge(i, j)

    return G

def generate_SBM_static(P:np.ndarray):
    """
    P is an upper-triangular square probability matrix.
    """
    # generate the adjacency matrix based on edge probabilities in P
    A = (np.random.random(P.shape) < P).astype(int)
    # make the matrix symmetric
    full_A = np.maximum(A, A.T)
    # convert to a networkx graph
    return nx.Graph(full_A)

def generate_prob_matrix(n:int, c:int, x_sim:float, x_diff:float, sigma_eps:float):

    # possible color choices
    color_choices = np.array(range(c))
    # assign nodes a color uniformly at random
    color_arr = np.random.choice(color_choices, size=n, replace=True)

    # compute the error terms
    Epsilon = np.random.normal(0, sigma_eps, (n, n))

    # compute the fixed similarities
    #X = np.array([[x_sim if color_arr[i]==color_arr[j] else x_diff for i in range(n)] for j in range(n)])
    color_mask = (color_arr[:, None] == color_arr[None, :])
    X = np.where(color_mask, x_sim, x_diff)

    # use the sigmoid function to transform the result into a valid probability matrix
    sigmoid = lambda t: 1 / (1 + np.exp(-t))
    return np.triu(sigmoid(X + Epsilon), 1), list(color_arr)

def simulate(x_sim, x_diff, n, c, sigma_eps, generation_method="static"):
    # uncomment this when we actually pass in our true function
    P, colors = generate_prob_matrix(n, c, x_sim, x_diff, sigma_eps)
    if generation_method == "growth":
        G = generate_SBM_growth(P)
    elif generation_method == "static":
        G = generate_SBM_static(P)
    else: raise ValueError(f"{generation_method} is invalid generation method.")

    # something
    pval, _, _ = stats_engine(nk.nxadapter.nx2nk(G), [str(c) for c in colors], iters=null_dist_iters)
    return pval

if __name__ == "__main__":
    param = {
        'x_sim': range(0, 1), # we want to test when x_sim > x_diff as well as x_sim <= x_diff
        'x_diff': range(-1, 0),
        'n': range(10, 300, 20),
        'c': range(2, 10),
        'sigma_eps': range(0, 0.1), # standard deviation of the random noise
        'generation_method': ['static']
        }

    grid = ParameterSampler(param, n_iter=param_sample_size, random_state=42)

    results = {}

    print("START")
    for params in grid:
        print("Parameters:", params)
        param_key = tuple(sorted(params.items()))

        # we will store the results of the tests themselves for each iteration, 
        # and then calculate the proportion that were correct
        results[param_key] = [0, []]
        
        
        for i in range(n_validations):
            pval = simulate(**params)
            print(f"Validation{i}: p={pval}")
            results[param_key][1].append(pval)
        
        results[param_key][0] = sum([elem < alpha for elem in results[param_key][1]]) / n_validations

    # --- Process results for output ---
    output_data = []
    print("Processing Results...")
    for param_tuple, (prop, pvals) in results.items():
        row = dict(param_tuple)
        # Rename keys for clarity in the output file
        row['color_num'] = row.pop('c')
        row['node_num'] = row.pop('n')
        row['true_positive_prop'] = prop
        for i, pval in enumerate(pvals):
            row[f"pval_iter{i}"] = pval
        output_data.append(row)

    df = pd.DataFrame(output_data)
    # Ensure consistent column order
    column_order = ['color_num', 'generation_method', 'node_num', 'sigma_eps', 'x_diff', 'x_sim', 'true_positive_prop'] + [f"pval_iter{i}" for i in range(n_validations)]
    df = df[column_order]
    df.to_csv("statistical_validation_results.csv", index=False)
    print("Done.")
