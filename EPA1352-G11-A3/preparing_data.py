import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import geopandas as gpd
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

roads_link = './data/_roads3.csv'
bridges_link = './data/BMMS_overview.xlsx'

df_roads = pd.read_csv(roads_link)
df_bridges = pd.read_excel(bridges_link)

def filter_roads():
    """
    Adds columns needed for modelling to the dataframe
    Identifies all the roads available in the dataframe
    Filters the roads on the condition that the length is >25km
    Filters the roads based on the casus
    Calls prepare_data for all the separate roads
    """

    # Adds columns needed for modelling by calling the function add_columns
    add_columns(df_roads)

    # Makes a list of unique roads in the dataset
    unique_roads = df_roads.road.unique()

    print('All relevant roads are being identified based on length and the casus.')

    # Filters the roads based on the condition that the length needs to be >25km. Appends the relevant roads to a list
    long_roads = []
    for i in unique_roads:
        df_road_temp = df_roads[df_roads['road'] == i]
        if df_road_temp["chainage"].iloc[-1] >= 25:
            long_roads.append(i)

    # Filters the roads based on the casus. Appends the relevant roads to a list
    casus_roads = ['N1', 'N2']
    global relevant_roads
    relevant_roads = []
    for i in long_roads:
        df_road_temp = df_roads[df_roads['road'] == i]
        for j in casus_roads:
            if df_road_temp['road'].str.contains(j).any():
                relevant_roads.append(i)

    print(f'In total there are {len(relevant_roads)} relevant roads found, which are: {relevant_roads}.')
    print(f'The pre-processing of each road is done separately.')

    # Processes all roads individually
    for i in relevant_roads:
        print(f'The road that is pre-processed now, is: {i}.')
        df_road_temp = df_roads[df_roads['road'] == i]
        prepare_data(df_road_temp)

def add_columns(df_road):
    """
    Adds empty columns for the information needed for modeling
    Makes a list of all column names in the dataframe
    """
    # Add new columns
    df_road['model_type'] = ''
    df_road['length'] = np.nan
    df_road['id'] = ''
    df_road['id_jump'] = ''
    df_road['name'] = ''
    df_road['condition'] = np.nan
    df_road['road_name'] = ''
    df_road['bridge_length'] = np.nan

    # Makes a list of all column names
    global column_names
    column_names = []
    for i in df_road:
        column_names.append(i)

def change_model_type(df_road):
    """
    This function checks if the road object is a bridge and replaces it with 'bridge'
    Thereby replaces all other objects in 'link'
    The first and last object are given the model type 'sourcesink' or 'intersection', based on the network
    """
    # Recognises and sets model type to bridge
    bridge_types = ['Bridge', 'Culvert']
    for i in bridge_types:
        df_road.loc[df_road['type'].str.contains(i), 'model_type'] = 'bridge'

    # Recognises and sets model type to link
    df_road.loc[~df_road['model_type'].str.contains('bridge'), 'model_type'] = 'link'

    # Recognises and sets model type to sourcesink or intersection
    if (df_road['road'] == 'N1').any():
        df_road['model_type'].iloc[0] = 'sourcesink'
        df_road['model_type'].iloc[-1] = 'sourcesink'
    elif (df_road['road'] == 'N2').any():
        df_road['model_type'].iloc[0] = 'intersection'
        df_road['model_type'].iloc[-1] = 'sourcesink'
    elif (df_road['road'] == 'N105').any():
        df_road['model_type'].iloc[0] = 'intersection'
        df_road['model_type'].iloc[-1] = 'sourcesink'
    elif (df_road['road'] == 'N104').any():
        df_road['model_type'].iloc[0] = 'intersection'
        df_road['model_type'].iloc[-1] = 'sourcesink'
    elif (df_road['road'] == 'N106').any():
        df_road['model_type'].iloc[0] = 'intersection'
        df_road['model_type'].iloc[-1] = 'sourcesink'
    else:
        df_road['model_type'].iloc[0] = 'intersection'
        df_road['model_type'].iloc[-1] = 'intersection'

def complete_intersections(df_road):
    """
    Locates the potential points of intersection, where the road type indicates a 'SideRoad'
    Connects the intersections with the potential intersections with a nearest neighbor method
    Harmonizes the id's of the found intersection matches for different roads
    Updates the 'potential intersections' into 'intersection' or back to 'link'
    """
    # recognises and sets model type to potential intersection
    df_road.loc[df_road['type'].str.contains('SideRoad', na=False), 'model_type'] = 'potential intersection'

    # Makes a gdf from the df
    gdf_road = gpd.GeoDataFrame(df_road, geometry=gpd.points_from_xy(df_road['lon'], df_road['lat']))

    # Filters to only include points with model_type = 'intersection'
    intersection_points = gdf_road[gdf_road['model_type'] == 'intersection']
    intersection_points = intersection_points.drop(['id'], axis = 1)

    # Filters to only include points with model_type = 'potential intersection'
    potential_points = gdf_road[gdf_road['model_type'] == 'potential intersection']
    potential_points = potential_points.drop(['road_id'], axis=1)

    # Find the closest match for each intersection from the potential intersections
    gdf_match_intersection = ckdnearest(intersection_points, potential_points)

    # Integrating the outcomes from the match process into the df
    list_of_ids = gdf_match_intersection['id'].tolist()
    for i in list_of_ids:
        if (df_road['id'] == i).any():
            df_road.loc[df_road['id'] == i, 'model_type'] = 'intersection'
            df_road.loc[df_road['id'] == i, 'lon'] = gdf_match_intersection.loc[gdf_match_intersection['id'] == i, 'lon'].values[0]
            df_road.loc[df_road['id'] == i, 'lat'] = gdf_match_intersection.loc[gdf_match_intersection['id'] == i, 'lat'].values[0]

    list_of_roads = gdf_match_intersection['road_id'].tolist()
    road = 0
    for i in list_of_roads:
        if (df_road['road_id'] == i).any():
            df_road.loc[df_road['road_id'] == i, 'id'] = list_of_ids[road]
        road += 1

    # Puts the model type of roads without a match back to link
    df_road.loc[df_road['model_type'].str.contains('potential intersection', na=False), 'model_type'] = 'link'

    # Saves the match data
    gdf_match_intersection.to_csv('./data/check_location_intersections.csv')

def ckdnearest(gdA, gdB):
    """
    Performs a nearest neighbor search between two geodata sets
    Extracts the coordinates of the Point objects from gdA and gdB using a lambda function and the apply method
    Constructs a KD-tree (btree) from the coordinates in gdB using cKDTree
    Is used to find the nearest neighbor of each point in gdA.
    The k parameter is set to 1 to return the closest neighbor.
    Dist contains the distances between the nearest neighbors
    The function then concats gdA, the nearest neighbors in gdB (gdB_nearest), and the distance between them (dist).
    The loc method is used to select only certain columns from gdB (nearest_information)
    """

    nearest_information = ['id', 'model_type'] # 'road_id',

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)

    gdB_nearest = gdB.iloc[idx][nearest_information].reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)

    return gdf

def standardize_bridges(df_road):
    """
    Since some bridges have >1 LRPs, we only model for every bridge one delay,
    and we only model traffic one-way, we consider these bridges as duplicates
    Therefore, we changed them to 'link' and removed the condition
    """
    # Drop bridge end from the roads file
    df_road.loc[df_road['gap'].str.contains('BE', na=False), 'model_type'] = 'link'
    df_road.loc[df_road['gap'].str.contains('BE', na=False), 'condition'] = np.nan

def make_infra_id(df_road):
    # Make bridge_id and road_id based on road and LRP
    df_road['road_id'] = df_road['road'] + df_road['lrp']
    df_bridges['bridge_id'] = df_bridges['road'] + df_bridges['LRPName']

def connect_infra(bridges_file, df_road):
    """
    This function connects the bridges df with the road df, to obtain information about bridge condition and length
    """
    # find exact match between road+LRP
    for index, row in df_road.iterrows():
        if 'bridge' in row['model_type']:
            road_id = row['road_id'].strip()
            matching_bridge = bridges_file[bridges_file['bridge_id'].str.contains(road_id)]
            if not matching_bridge.empty:
                bridge_condition = matching_bridge.iloc[0]['condition']
                bridge_length = matching_bridge.iloc[0]['length']
                df_road.at[index, 'condition'] = bridge_condition
                df_road.at[index, 'bridge_length'] = bridge_length

    # Since there are inconsistencies between the two datasets, the procedure is ran again for less exact matches
    fill_in_infra(bridges_file, df_road)

def fill_in_infra(bridges_file, df_road):
    """
    This function connects the bridges df with the road df, to obtain information about bridge condition and length
    This is done making a less exact match due to inconsistencies between the roads and bridges data
    Iterates only over the columns with model type bridges and empty condition (NaN)
    """
    # Slice the road_id to obtain an eight number value, without the extension a-z
    df_road['road_id_sliced'] = df_road['road_id'].str.slice(stop=8)

    # find match between the reduced road+LRP id
    for index, row in df_road.loc[df_road['condition'].isna()].iterrows():
        if 'bridge' in row['model_type']:
            road_id = row['road_id_sliced']
            matching_bridge = bridges_file[bridges_file['bridge_id'].str.contains(road_id)]
            if not matching_bridge.empty:
                bridge_condition = matching_bridge.iloc[0]['condition']
                bridge_length = matching_bridge.iloc[0]['length']
                df_road.at[index, 'condition'] = bridge_condition
                df_road.at[index, 'bridge_length'] = bridge_length

def bridge_to_link(df_road):
    '''
    If no match is found between the id's of the bridges and roads,
    the model type of these bridges is set to link for modeling purposes.
    '''
    for index, row in df_road.loc[df_road['condition'].isna()].iterrows():
        if 'bridge' in row['model_type']:
            df_road.loc[index, 'model_type'] = 'link'
            df_road.loc[index, 'name'] = 'link'

def get_length(df_road):
    '''
    Fills in the length of each road part based on the chainage
    '''
    df_road['length'] = abs(df_road['chainage'].astype(float).diff()) * 1000
    df_road['length'][0] = 0

def get_name(df_road):
    '''
    Fills in the name of the road part, based on the model type
    '''

    sosicounter = 1
    for i, row in df_road.iterrows():
        if row['model_type'] != 'sourcesink':
            df_road.at[i, 'name'] = ''
        else:
            df_road.at[i, 'name'] = f'SoSi{sosicounter}'
            sosicounter += 1

def get_road_name(df_road):
    '''
    In components.py, a road name is asked. It is set as the standard value 'Unknown'
    '''
    df_road['road_name'] = 'Unknown'

def make_id_once(df_road):
    '''
    Generates a unique id for each road, with big jumps between two roads
    '''
    unique_id = 1000000
    for i in range(df_road.shape[0]):
        df_road.loc[i, 'id'] = unique_id
        unique_id += 1

list_all_roads = []
def collect_roads(df_road):
    '''
    Appends all df_roads as new rows
    '''
    list_all_roads.append(df_road)

    return list_all_roads

def prepare_data(df_road):
    '''
    Runs all procedures to obtain the right columns and information for modeling
    '''
    make_infra_id(df_road)
    change_model_type(df_road)
    standardize_bridges(df_road)
    connect_infra(df_bridges, df_road) # also calls for fill_in_infra() within the function
    bridge_to_link(df_road)
    get_length(df_road)
    get_road_name(df_road)
    collect_roads(df_road)

def make_figure(df):
    '''
    Makes a plot of the relevant roads,
    with different colors for the model type (source, link, bridge, sink),
    or for the different roads
    '''
    sns.lmplot(x='lon', y='lat', data=df, hue='road', fit_reg=False, scatter_kws={"s": 1})
    sns.lmplot(x='lon', y='lat', data=df, hue='model_type', fit_reg=False, scatter_kws={"s": 1})
    plt.show()

def combine_data():
    '''
    Combines all the separate dataframes from all the roads
    '''
    df_all_roads = pd.DataFrame(columns=column_names)  # initialize empty dataframe

    print('All roads are pre-processed. The seperate files are being combined, completed and presented (figure and csv exports).')

    for df in list_all_roads:
        df_all_roads = pd.concat([df_all_roads, df])  # append to df_all_roads in each iteration

    df_all_roads = df_all_roads.reset_index()

    make_id_once(df_all_roads)
    get_name(df_all_roads)
    complete_intersections(df_all_roads)
    make_figure(df_all_roads)
    save_data(df_all_roads)

    print('The data is pre-processed and available for the next step.')

model_columns = ['road', 'id', 'model_type', 'name', 'lat', 'lon', 'length', 'condition', 'bridge_length'] # 'road_name'
def save_data(df):
    '''
    Saves the files
    '''
    # Write the dataframe to csv
    df.to_csv('./data/demo_all_roads_LB.csv')

    # Make compact datafile and export to csv
    df_all_roads_compact = df.loc[:, model_columns]
    df_all_roads_compact.to_csv('./data/demo_all_roads_compact_LB.csv')

# Run the prepare data function
filter_roads() # calls prepare_data function
combine_data()