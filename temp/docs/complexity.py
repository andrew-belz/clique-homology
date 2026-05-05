import time
import networkit as nk # type: ignore
import networkx as nx # type: ignore
import matplotlib.pyplot as plt

from clique_homology.stats_engine.betti_numbers import betti_numbers

def test_runtime(n, m, k):
    """
    What kind of runtime can we expect for the C. elegans data set?

    Generates monochromatic G(n, m) random graphs with 302 nodes and 5000 edges.
    Run the betti numbers algorithm and time each performance.
    Resulting plot is stored in docs:
    docs\time_to_compute_gnm_rand_302nodes_5000edges.png
    """

    times = []
    for _ in range(k):

        G = nx.gnm_random_graph(n, m)
        G = nk.nxadapter.nx2nk(G)
        c2 = ["red" for _ in range(n)]


        start = time.perf_counter()
        print(f"n: {n}; output: {betti_numbers(G, c2)}.")
        stop = time.perf_counter()
        times.append(stop - start)

    return times


if __name__ == "__main__":
    n, m = 302, 5000
    times = test_runtime(302, 5000, 10)
    plt.hist(times)
    plt.savefig(f"time_to_compute_gnm_rand_{n}nodes_{m}edges.png")