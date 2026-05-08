import networkit as nk
import numpy as np
import itertools

# ----------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------
def get_max_clique_size(G:nk.Graph):
    """Return the size of the largest clique of a graph G.

    Args:
        G (nk.Graph): a graph

    Returns:
        int: size of largest clique in the graph
    """

    # Get maximal cliques, store them in cliques
    finder = nk.clique.MaximalCliques(G, maximumOnly=True)
    finder.run()
    cliques = finder.getCliques()
    
    if not cliques:
        return 0
    
    return len(cliques[0])


def get_cliques(G:nk.Graph):
    """Return a generator for cliques of a graph G.

    Args:
        G (nk.Graph): A colored graph.

    Yields:
        tuple: A tuple of vertices representing a single clique, ordered by
            length.
    """

    # this part seems like it isn't the most efficient 
    #   since we check so many duplicate cliques. 
    # Maybe have to look at different methods for this:
    #   write a custom bottom-up approach instead
    all_cliques = set()
    def collect_subcliques(C):
        for r in range(1, len(C)+1):
            for subset in itertools.combinations(C, r):
                all_cliques.add(tuple(sorted(subset)))

    # Find maximal cliques:
    #   the callback function runs when every maximal
    #   clique is found.
    clique_finder = nk.clique.MaximalCliques(G,
         maximumOnly=False, callback=collect_subcliques)
    clique_finder.run()

    all_cliques = sorted(list(all_cliques), key=len)

    # generate all the cliques
    for clique in all_cliques:
        yield clique

# ----------------------------------------------------------------------------------------------------------------


    """
    Return a generator for colored subgraphs of a graph G.
    
    :param G: A colored graph.
    :type G: nk.Graph
    :param node_attr: A dictionary mapping node IDs to their attribute (color).
    :type node_attr: list
    """

def get_colored_subgraphs(G:nk.Graph, node_colors:list[str]):
    """Return a generator for colored subgraphs of a graph G. 

    Args:
        G (nk.Graph): A colored graph.
        node_colors (list[str]): A list mapping node IDs (list index)
            to their attribute (color).

    Yields:
        nk.Graph: a subgraph of G containing all the nodes which have a specific
            color and the edges between those nodes.
    """

    # Group nodes by their attribute value, forming color:[nodes] key-value
    #   pairs inside node_subsets
    node_subsets: dict[str, list[int]] = {}
    for node, color in zip(G.iterNodes(), node_colors):
        if color not in node_subsets:
            node_subsets[color] = [node]
        else:
            node_subsets[color].append(node)
        
    # turn a list of nodes for a color into an nk.Graph and yield it
    for color in node_subsets.keys():
        yield nk.graphtools.subgraphFromNodes(G, node_subsets[color])

# ----------------------------------------------------------------------------------------------------------------

def boundary_maps(cliques:list) -> list:
    """Construct the boundary maps D_k given a complete list of cliques
        (simplicies).

    Args:
        cliques (list): A complete list of cliques for a graph. Cliques should
            be lists of vertices.

    Returns:
        list: a list of numpy arrays. These are the boundary maps D_k for each
            k.
    """

    def clique_order(cliques:list) -> list:
        """Define an ordering for each clique with respect to the other cliques
            of their given size. This will be used to construct the boundary
            maps.

        Args:
            cliques (list): a list of cliques (tuples of vertices)

        Returns:
            list: A list of dictionaries, one for each size of 
                clique: list(dict(tuple:int))
        """

        if not cliques:
            return []

        # Find the size of the largest clique, initialize the ordering
        max_clique_size = len(cliques[-1])
        ordering = [{} for _ in range(max_clique_size)]

        # track the current dictionary in result
        i = 0
        # track the positions we are assigning for the current dictionary
        j = 0
        # track the current size clique
        k = 1
        for clique in cliques:
            if len(clique) > k:
                k = len(clique)
                i += 1
                j = 0
            
            # assign the index j to the clique in the i-th dictionary in result
            #   and increment j, resulting an ordered indexing of the cliques
            #   of a given size. 
            ordering[i][clique] = j
            j += 1

        return ordering
    
    def build_map(position_dict1, position_dict2) -> np.ndarray:
        """Construct a boundary map from position_dict1 to position_dict2.

        Args:
            position_dict1 (dict): Dictionary of positions for (k-1)-cliques:
                dict(tuple:int)
            position_dict2 (dict): Dictionary of positions for (k)-cliques.
                dict(tuple:int)

        Returns:
            np.ndarray: a 2d numpy array representing the boundary map for 
                k-1 cliques to k cliques.
        """

        # We are mapping from the (k-1)-cliques (nrow) to the k-cliques (ncol) 
        M = np.zeros((len(position_dict1), len(position_dict2)), dtype=int)
        # get each simplex (clique) as k2, with its corresponding col index v2
        for k2, v2 in position_dict2.items():
            # Iterate over all faces of the simplex k2
            for i in range(len(k2)):
                # Create face by removing the i-th vertex w/ tuple slicing
                face = k2[:i] + k2[i+1:]
                # get index of that face in the position dict 1 and set its
                #   corresponding position in the boundary map to 1 to 
                #   indicate its presence
                M[position_dict1[face], v2] = 1

        return M

    # get the list of dicitonaries representing the clique ordering, then return
    #   a list of 2d numpy arrays representing the boundary maps
    positions = clique_order(cliques)
    return [build_map(positions[k-1], positions[k])
                        for k in range(1, len(positions))]

# ----------------------------------------------------------------------------------------------------------------

def ranks_and_nullities(M:np.array) -> tuple:
    """Return the rank and nullity of a matrix over Z_2 (contains only values of
        0 or 1).

    Args:
        M (np.array): a matrix M({0, 1}).

    Returns:
        tuple: a tuple containing the rank and nullity of M.
    """
    
    def rank_Z2(M:np.array) -> int:
        """Return the rank of a matrix M({0, 1}) (a matrix containing only 
            values of 0 or 1).

        Args:
            M (np.array): a matrix M({0, 1}).

        Returns:
            int: the rank of M.
        """
        M2 = M.copy()
        nrows, ncols = M2.shape
        rank = 0

        for j in range(ncols):
            if rank >= nrows:
                break

            # Find a pivot row for column j, looking only at rows >= rank
            # np.where returns a tuple, so we take [0] to get the array of indices
            pivot_candidates = np.where(M2[rank:, j] == 1)[0]

            if len(pivot_candidates) > 0:
                # Get the first available pivot (index is relative to the slice, so add rank)
                pivot_row = pivot_candidates[0] + rank

                # Swap the current row (rank) with the pivot row
                if pivot_row != rank:
                    M2[[rank, pivot_row]] = M2[[pivot_row, rank]]

                # Eliminate 1s in this column for all rows BELOW the pivot
                # We use (^) for addition modulo 2
                rows_to_eliminate = np.where(M2[rank+1:, j] == 1)[0] + (rank + 1)
                if len(rows_to_eliminate) > 0:
                    M2[rows_to_eliminate] ^= M2[rank]

                rank += 1
        
        return rank

    # utilize the rank-nullity theorem here.
    # returns (rank, nullity)
    rank = rank_Z2(M)
    nullity = M.shape[1] - rank
    return rank, nullity

# ----------------------------------------------------------------------------------------------------------------

def _validate_colors(
    G: nk.Graph,
    colors: list[str],
) -> None:
    """Ensure a coloring is valid for a graph G.

    Args:
        G (nk.Graph): a graph
        colors (list[str]): a list mapping each node (index) in G to a color.

    Raises:
        ValueError: length of coloring list must match num nodes in G.
        TypeError: colors values must be strings.
    """

    if len(colors) != G.numberOfNodes():
        raise ValueError(
            f"Coloring length ({len(colors)}) must match number of graph nodes ({G.numberOfNodes()})."
        )

    if not all(isinstance(color, str) for color in colors):
        raise TypeError("All node colors must be strings.")

   

def betti_numbers(
    G: nk.Graph,
    colors: list[str],
    method: str = "clique",
) -> np.ndarray:
    """Compute the Betti numbers of a colored graph. 

    If 'clique' method is specified (default), 
    build simplicial complex out of all monochromatic cliques for the entire graph.
    Returns a vector of Betti numbers.
    
    If 'subgraph' method is specified, 
    builds a distinct simplicial complex for each colored subgraph.
    Returns a matrix where each row vector gives the Betti numbers for a colored subgraph.

    If the resulting matrix of the 'subgraph' method is summed by rows, it will yield the result
    of the 'clique' method.
    
    Args:
        G (Union[nk.Graph, nx.Graph]): A colored graph.
        colors (list[str]): A list mapping each node in G (index) to a color
        method (str, optional): A string speficying which method to use.
            Defaults to "clique".

    Raises:
        ValueError: the method argument either needs to be "subgraph" or
            "clique."

    Returns:
        np.ndarray: _description_
    """

    if method not in ["subgraph", "clique"]:
        raise ValueError(f"Invalid method '{method}'. Expected 'subgraph', or 'clique'.")
    _validate_colors(G, colors)

    if G.numberOfNodes() == 0:
        return np.array([])

    max_len = get_max_clique_size(G)

    if method == "subgraph":
        # in this case, compute all the betti numbers separately
        betti_lists = []
        for subgraph in get_colored_subgraphs(G, colors):

            # get the maps for each subgraph
            cliques = [clique for clique in get_cliques(subgraph)]
            maps = boundary_maps(cliques)

            ranks, nullities = [], []
            for boundary_map in maps:
                # get the ranks and nullities for each map
                rank, nullity = ranks_and_nullities(boundary_map)
                ranks.append(rank)
                nullities.append(nullity)

            if maps:
                # prepend the number of nodes to nullities, append 
                #   zero to ranks
                nullities = [maps[0].shape[0]] + nullities
                ranks.append(0)
                # compute the betti numbers    
                betti = [nullities[k] - ranks[k] for k in range(
                                                       len(ranks))]
            else:
                betti = [len(cliques)]
                
            betti_lists.append(betti)
        
        # pad with zeros
        # a matrix of betti numbers
        padded_betti = [b + [0] * (max_len - len(b)) for b in
                                                       betti_lists]
        B = np.array(padded_betti)

        return B

    
    elif method == "clique":
        # the difference here is we compute the cliques, aggregate them, then compute the homology
        cliques = sorted([clique for H in get_colored_subgraphs(G, colors) for clique in get_cliques(H)], key=len)
        maps = boundary_maps(cliques)
        ranks, nullities = [], []

        for boundary_map in maps:
            # get the ranks and nullities for each map
            rank, nullity = ranks_and_nullities(boundary_map)
            ranks.append(rank)
            nullities.append(nullity)

        if maps:
            # prepend the number of nodes to nullities, append zero to ranks
            nullities = [maps[0].shape[0]] + nullities
            ranks.append(0)
            # compute the betti numbers    
            betti = [nullities[k] - ranks[k] for k in range(len(ranks))]
        else:
            betti = [len(cliques)]
        # pad with zeros to ensure consistency in size across permutations
        padded_betti = betti + [0] * (max_len - len(betti))
        # vector of betti numbers
        return np.array(padded_betti)
