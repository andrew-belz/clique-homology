import csv

def c_elegans_edges():
    """
    Return chemical synaptical connections of male C. elegans.
    """

    with open(r"c_elegans_data\c_elegans_edges.csv", "r") as file:
        
        reader = csv.reader(file)

        edges = []
        for row in reader:
            # stored as node pairs
            edges.append((int(row[0]), int(row[1])))


        return edges

if __name__ == "__main__":
    print(c_elegans_edges())
