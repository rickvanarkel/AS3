import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_edge_to_graph(G, e1, e2, w):
    G.add_edge(e1, e2, weight=w)
def make_points_edges(df):
    road_dict = {}
    roads = df['road'].unique()
    point_num = 0
    total_length = 0

    for road_type in roads:
        temp_df =  df.loc[df['road'] == road_type]
        temp_points = []
        temp_edges = []
        df_length = temp_df.shape[0]

        for index, row in temp_df.iterrows():
            temp_points.append((row['lon'], row['lat']))
            if point_num != 0 and point_num != total_length:
                temp_edges.append((point_num-1, point_num, row['length']))
            point_num += 1
        road_dict[road_type] = (temp_points, temp_edges)
        total_length += df_length
    return road_dict

def make_networkx(road_dict):
    G = nx.Graph()
    edges_num = 0

    for roads in df['road'].unique():
        temp_list = road_dict[roads]

        points = temp_list[0]
        edges = temp_list[1]
        edges_num += len(edges)

        for i in range(len(edges)):
            add_edge_to_graph(G, points[edges[i][0]-edges[0][0]], points[edges[i][1]-edges[0][0]], edges[i][2])

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos=pos, node_color='k')
    nx.draw_networkx(G, pos=pos, node_size=800)  # draw nodes and edges
    #nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names

    # draw edge weights
    #labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.axis()
    plt.show()

file_name = '../data/demo-4.csv'
df = pd.read_csv(file_name)

road_dict = make_points_edges(df)
make_networkx(road_dict)
