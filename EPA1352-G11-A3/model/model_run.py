import pandas as pd
import random
import model
from components import Source
import networkx as nx
from Route_Network import make_points_edges
from Route_Network import make_networkx
from Route_Network import create_graph
import csv

'''
This file is used to run the simulation model and get the corresponding results. 
'''

def run_iteration(nscenario, df_list, run_length, graph, seed):
    '''
    This run_iteration function can be used to run a iteration of the BangladeshModel
    it needs to be provided with a scenario number, a dataframe used to store the results
    and a run length. This function returns the df with the new data added.
    '''

    sim_model = model.BangladeshModel(seed=seed, scenario=nscenario, G=graph, filename=file_name)
    # Check if the seed is set
    print("SEED " + str(sim_model._seed))

    # One run with given steps
    for j in range(run_length):
        sim_model.step()
    average_time = sim_model.reporter['Time'].mean()
    df_list.loc[k] = [average_time]
    Source.truck_counter = 0

    return df_list


run_length = 5 * 24 * 60
nscenario = 5
niteration = 10
iteration = 0

exec(open("preparing_data.py").read())

seeds = []
with open('../data/seeds.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        seeds.append(int(row[0]))

file_name = '../data/demo_all_roads_compact_LB.csv'
df_file = pd.read_csv(file_name)
road_dict = make_points_edges(df_file)
G = make_networkx(road_dict, df_file)
create_graph(G)

for i in range(nscenario):
    df_list = pd.DataFrame(columns=["run", "avarage"])
    df_list.set_index("run", inplace=True)
    for k in range(niteration):
        print(f"This is iteration {k} of scenrio {i}")
        df_list = run_iteration(i, df_list, run_length, G, seeds[iteration])
        iteration += 1
    df_list.to_csv(f'../data/scenario{i}.csv')


