import pandas as pd
import json
import csv

"""
Data retrieved from wormwiring.org.
The data is a 576x576 adjacency matrix.
"""

if __name__ == "__main__":

    """
    Extract chemically connected neuron pairs.
    """
    df = pd.read_csv(r"c_elegans_data\maleChemicalAdjMatrix.csv")

    # index the rows by the neuron names
    matrix = df.set_index(df.columns[0])

    # extract the index/column pair where the entry is not na
    edges = matrix.stack().dropna()

    # save edges to list
    edge_list = edges.index.to_list()

    """
    Compile list of unique edges
    """

    # ensure no duplicate edges
    unique_edges = []
    for v1, v2 in edge_list:
        if (v1, v2) not in unique_edges and (v2, v1) not in unique_edges:
            unique_edges.append((v1, v2))

    # assign an integer index to each node
    neuron_indices = dict() # map neuron name: integer
    integer_edges = [] # compile list of integer-valued nodes

    i = 0 # initialize index to zero
    for v1, v2 in unique_edges:
        if v1 not in neuron_indices:
            neuron_indices[v1] = [i]
            i += 1
        
        if v2 not in neuron_indices:
            neuron_indices[v2] = [i]
            i += 1

        integer_edges.append((neuron_indices[v1][0], neuron_indices[v2][0]))

    """
    Record functional group for each neuron.
    """

    # the neuron and its respective group are recorded in this csv
    with open(r"c_elegans_data\maleChemicalFunctionGroupColoring.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            neuron = row[0]
            group = int(row[1])

            if neuron in neuron_indices:
                neuron_indices[neuron].append(group)

    """
    Store data
    """

    # store integer edges in a csv
    edge_df = pd.DataFrame(integer_edges)
    edge_df.to_csv(r"c_elegans_data\c_elegans_edges.csv")

    # store neuron-integer map
    with open(r"c_elegans_data\neuron_indices.json", "w") as json_file:
        json.dump(neuron_indices, json_file, indent=4)
