import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_edge_to_graph(G, e1, e2, w):
    G.add_edge(e1, e2, weight=w)

G = nx.Graph()
file_name = '../data/demo-4.csv'
df = pd.read_csv(file_name)

for index, row in df.iterrows():
    print(row['lon'], row['lat'], row['length'])

#
# points = [(1, 10), (8, 10), (10, 8), (7, 4), (3, 1)]  # (x,y) points
# edges = [(0, 1, 10), (1, 2, 5), (2, 3, 25), (0, 3, 3), (3, 4, 8)]  # (v1,v2, weight)
# for i in range(len(edges)):
#     add_edge_to_graph(G, points[edges[i][0]], points[edges[i][1]], edges[i][2])
#
# pos = nx.spring_layout(G)
# nx.draw(G, pos=pos, node_color='k')
# nx.draw(G, pos=pos, node_size=1500)  # draw nodes and edges
# nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names
# # draw edge weights
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.axis()
# plt.show()