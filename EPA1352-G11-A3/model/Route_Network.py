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

    for road_type in roads:
        temp_df =  df.loc[df['road'] == road_type]
        point_ids = []
        temp_points = []
        temp_edges = []
        temp_df.reset_index(inplace = True)
        temp_id = 0

        for index, row in temp_df.iterrows():
            temp_points.append((row['lon'], row['lat']))
            point_ids.append(row['id'])
            if index != 0:
                temp_edges.append((temp_id, row['id'], row['length']))
            temp_id = row['id']
        road_dict[road_type] = (temp_points, temp_edges, point_ids)
    return road_dict

def make_networkx(road_dict, df):
    G = nx.Graph()
    for roads in df['road'].unique():
        temp_list = road_dict[roads]
        points = temp_list[0]
        edges = temp_list[1]
        point_ids = temp_list[2]
        for i in range(len(point_ids)):
            G.add_node(point_ids[i], pos=points[i])
        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1], weight = edges[i][2])
    return G

def create_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx(G, pos=pos, node_color='k', with_labels=False)
    nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False) # draw nodes and edges
    #nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names

    # Get the nodes with more than 2 edges
    nodes_with_more_than_2_edges = [node for node, degree in dict(G.degree()).items() if degree > 2]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_more_than_2_edges, node_color='r', node_size=50)

    # draw edge weights
    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.axis()
    plt.show()



# file_name = '../data/demo_all_roads_compact_LB.csv'
# df = pd.read_csv(file_name)
#
# road_dict = make_points_edges(df)
# G = make_networkx(road_dict, df)
# create_graph(G)




