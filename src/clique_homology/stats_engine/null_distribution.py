from clique_homology import random_coloring, betti_numbers
from networkit.graph import Graph
import numpy as np

def null_distribution(
    graph: Graph,
    coloring: list[str],
    iterations: int = 100,
) -> list[np.ndarray]:
    """Partitions a graph according to a random coloring, computes the vector
        of betti numbers for that colored subgraph, appends it to a list, and
        repeats this process (iterations) times. The result is a list of vectors
        that we can use as a null distribution against which we can test an
        observed coloring for statistical significance.

    Args:
        graph (Graph): A graph.
        coloring (list[str]): A coloring of the graph.
        iterations (int, optional): Desired size of null distribution. Defaults
            to 100.

    Raises:
        ValueError: rased if iterations is a negative number.

    Returns:
        list[np.ndarray]: a list of betti vectors to act as a null distribution.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")

    distribution: list[np.ndarray] = []
    
    # Make a new coloring based proportionally on the provided coloring, then
    #   generate a betti number vector on the graph partitioned according to the
    #   new coloring and add it to the distribution
    for _ in range(iterations):
        new_coloring = random_coloring(coloring, proportional=True)
        distribution.append(betti_numbers(graph, new_coloring))
    
    return distribution