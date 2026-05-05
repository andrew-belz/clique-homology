#%%
import pandas as pd
import numpy as np

"""
Data retrieved from ...
The data is a 576x576 adjacency matrix.
"""
#%%
df = pd.read_csv(r"c_elegans_data\maleChemicalAdjMatrix.csv")

# index the rows by the neuron names
matrix = df.set_index(df.columns[0])

edges = matrix.stack().dropna()
edge_list = edges.index.to_numpy()

#%%

unique_edges = []
for v1, v2 in edge_list:
    if (v1, v2) not in unique_edges and 