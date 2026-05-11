import csv

def c_elegans_edges() -> list[tuple[int, int]]:
    """Return chemical synaptical connections of male C. elegans.

    Returns:
        list[tuple[int, int]]: A list of two-item tuples representing edges of a
            graph.
    """

    # Open file containing the connectome edges, convert it to a list of tuples,
    #   and return the list.
    with open(r"c_elegans_data\c_elegans_edges.csv", "r") as file:
        
        reader = csv.reader(file)

        edges = []
        for row in reader:
            # stored as node pairs
            edges.append((int(row[0]), int(row[1])))


        return edges

if __name__ == "__main__":
    print(c_elegans_edges())
