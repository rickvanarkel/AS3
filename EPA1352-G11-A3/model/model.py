from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Source, Sink, SourceSink, Bridge, Link, Intersection
import pandas as pd
from collections import defaultdict
import random
import networkx as nx

# ---------------------------------------------------------------
def set_lat_lon_bound(lat_min, lat_max, lon_min, lon_max, edge_ratio=0.02):
    """
    Set the HTML continuous space canvas bounding box (for visualization)
    give the min and max latitudes and Longitudes in Decimal Degrees (DD)

    Add white borders at edges (default 2%) of the bounding box
    """

    lat_edge = (lat_max - lat_min) * edge_ratio
    lon_edge = (lon_max - lon_min) * edge_ratio

    x_max = lon_max + lon_edge
    y_max = lat_min - lat_edge
    x_min = lon_min - lon_edge
    y_min = lat_max + lat_edge
    return y_min, y_max, x_min, x_max


# ---------------------------------------------------------------
class BangladeshModel(Model):
    """
    The main (top-level) simulation model

    One tick represents one minute; this can be changed
    but the distance calculation need to be adapted accordingly

    Class Attributes:
    -----------------
    step_time: int
        step_time = 1 # 1 step is 1 min

    path_ids_dict: defaultdict
        Key: (origin, destination)
        Value: the shortest path (Infra component IDs) from an origin to a destination

        Only straight paths in the Demo are added into the dict;
        when there is a more complex network layout, the paths need to be managed differently

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """

    step_time = 1

    def __init__(self, seed=None, x_max=500, y_max=500, x_min=0, y_min=0, scenario = 0, G= None, filename= None):

        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = defaultdict(lambda: pd.Series())
        self.space = None
        self.sources = []
        self.sinks = []
        self.scenario_dict = {}
        self.G = G
        self.file_name= filename

        self.reporter = pd.DataFrame(columns=["Name", "Time"])
        self.reporter.set_index("Name", inplace=True)

        '''
        Creates a parameter for setting the scenario number!
        '''
        self.scenario = scenario
        self.generate_model()

    def generate_model(self):
        """
        generate the simulation model according to the csv file component information

        Warning: the labels are the same as the csv column labels
        """

        df = pd.read_csv(self.file_name)

        # a list of names of roads to be generated based on the different unique values in the "Road" column of the df
        roads = df['road'].unique()

        """
        self.scenario_dict creates a dictionary containing dictionaries for all 
        the different scenario's this way we can easily and efficiently select 
        which parameters for the bridge quality should be used. 
        """

        # Import the scenario CSV file as a new dictionary
        scenario_df = pd.read_csv('../data/scenario_dict.csv', index_col='Scenario')
        for scenario, row in scenario_df.iterrows():
            self.scenario_dict[scenario] = row.to_dict()

        df_objects_all = []
        for road in roads:
        #     # Select all the objects on a particular road in the original order as in the cvs
             df_objects_on_road = df[df['road'] == road]
        #
             if not df_objects_on_road.empty:
                 df_objects_all.append(df_objects_on_road)


        # put back to df with selected roads so that min and max and be easily calculated
        df = pd.concat(df_objects_all)
        y_min, y_max, x_min, x_max = set_lat_lon_bound(
            df['lat'].min(),
            df['lat'].max(),
            df['lon'].min(),
            df['lon'].max(),
            0.05
        )

        # ContinuousSpace from the Mesa package;
        # not to be confused with the SimpleContinuousModule visualization
        self.space = ContinuousSpace(x_max, y_max, True, x_min, y_min)

        for df in df_objects_all:
            for _, row in df.iterrows():  # index, row in ...

                # create agents according to model_type
                model_type = row['model_type'].strip()
                agent = None

                name = row['name']
                if pd.isna(name):
                    name = ""
                else:
                    name = name.strip()

                if model_type == 'source':
                    agent = Source(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                elif model_type == 'sink':
                    agent = Sink(row['id'], self, row['length'], name, row['road'])
                    self.sinks.append(agent.unique_id)
                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)
                elif model_type == 'bridge':
                    '''
                    This part makes sure that bridge are created based on the data set. Each run has a
                    certain scenario, and each bridge has a condition. This results in a certain chance
                    of the bridge collapsing and causing a delay. The state on the bridge (either intact or broken)
                    is added as an attribute to the bridge.
                    '''
                    runscenario_dict = self.scenario_dict[self.scenario]
                    cat_probability = runscenario_dict[row['condition']]
                    if random.randint(0, 100) > cat_probability:
                        state = 'intact'
                    else:
                        state = 'broken'
                    agent = Bridge(row['id'], self, row['bridge_length'], row['name'], row['road'], state)
                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], name, row['road'])
                elif model_type == 'intersection':
                    if not row['id'] in self.schedule._agents:
                        agent = Intersection(row['id'], self, row['length'], name, row['road'])

                if agent:
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

    def get_random_route(self, source):
        """
        pick up a random route given an origin
        """
        while True:
            # different source and sink
            sink = self.random.choice(self.sinks)
            if sink is not source:
                break

        return sink

    def get_route(self, source):
        """
        Creates a route list based on the source and sink nodes with dijkstras algorithm
        from the previously created networkX model
        """
        sink = self.get_random_route(source)
        if (sink, source) not in self.path_ids_dict:
            route = nx.shortest_path(self.G, source=source, target=sink, method='dijkstra')
            self.path_ids_dict[(sink, source)] = route
        else:
            route = self.path_ids_dict[(sink, source)]
        return route

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

# EOF -----------------------------------------------------------
