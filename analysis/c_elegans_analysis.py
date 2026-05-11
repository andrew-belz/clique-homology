from clique_homology.stats_engine.stats_engine import stats_engine
from c_elegans_colors import c_elegans_colors
from c_elegans_edges import c_elegans_edges

from pandas import DataFrame
import networkit as nk


def main():
    """Perform a test to determine the statistical significance of the
        topological structure (measured using simplicial homology) of the
        connectome of the C. Elegans roundworm, partitioned by neuron function.

    See c_elegans_colors.py and c_elegans_edges.py for more details about how we
        prepared our data for this analysis.
    """

    # Initialize the graph
    edges = c_elegans_edges()
    neurons, indices, colors = c_elegans_colors()
    num_nodes = len(colors)

    # Initialize networkit graph object
    g = nk.Graph(num_nodes, weighted=False, directed=False)

    # Add edges efficiently
    for u, v in edges:
        g.addEdge(u, v)
    g.removeSelfLoops()

    # save to a data frame
    p, obs, dist = stats_engine(g, [str(c) for c in colors], iters=2000)
    df = DataFrame(dist)
    df.to_csv("null_distribution_celegans.csv", header=False, index=False)

    # Print desired measurements
    print("P-value:", p)
    print("Observed test statistic:", obs)
    print("Number of nodes:", num_nodes)
    print("Number of edges:", len(edges))

if __name__ == "__main__":
    main()