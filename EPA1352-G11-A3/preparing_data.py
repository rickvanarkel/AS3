import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

roads_link = './data/_roads3.csv'
bridges_link = './data/BMMS_overview.xlsx'

df_roads = pd.read_csv(roads_link)
df_bridges = pd.read_excel(bridges_link)

def filter_roads():
    '''
    Changes the column names in the right format,
    Determines all the roads available,
    Filters the roads to only behold roads >25km
    Calls prepare_data for all the separate roads
    '''
    change_column_names(df_roads)
    unique_roads = df_roads.road.unique()

    short_roads = []
    for i in unique_roads:
        df_road_temp = df_roads[df_roads['road'] == i]
        if df_road_temp["chainage"].iloc[-1] >= 25:
            short_roads.append(i)

    casus_roads = ['N1', 'N2']
    relevant_roads = []
    for i in short_roads:
        df_road_temp = df_roads[df_roads['road'] == i]
        for j in casus_roads:
            if df_road_temp['road'].str.contains(j).any():
                relevant_roads.append(i)

    print(relevant_roads)

    for i in relevant_roads:
        df_road_temp = df_roads[df_roads['road'] == i]
        prepare_data(df_road_temp)

def change_column_names(df_road):
    """
    The column names are updated and empty columns are generated for the information needed for modeling
    """
    df_road['model_type'] = ''
    df_road['length'] = np.nan
    df_road['id'] = ''
    df_road['name'] = ''
    df_road['condition'] = np.nan
    df_road['road_name'] = ''
    df_road['bridge_length'] = np.nan

    global column_names
    column_names = []
    for i in df_road:
        column_names.append(i)

def change_model_type(df_road):
    """
    This function checks if the road object is a bridge and replaces it with 'bridge'
    Thereby replaces all other objects in 'link'
    The first and last object are given the model type 'source' and 'sink'
    """
    bridge_types = ['Bridge', 'Culvert'] # in doubt over: CrossRoad, RailRoadCrossing

    for i in bridge_types:
        df_road.loc[df_road['type'].str.contains(i), 'model_type'] = 'bridge'

    df_road.loc[~df_road['model_type'].str.contains('bridge'), 'model_type'] = 'link'

    df_road['model_type'].iloc[0] = 'sourcesink'
    df_road['model_type'].iloc[-1] = 'sourcesink'

def standardize_bridges(df_road):
    """
    Since some bridges have >1 LRPs, we only model for every bridge one delay,
    and we only model traffic one-way, we consider these bridges as duplicates
    Therefore, we changed them to 'link' and removed the condition
    """
    # Drop bridge end from the roads file
    df_road.loc[df_road['gap'].str.contains('BE', na=False), 'model_type'] = 'link'
    df_road.loc[df_road['gap'].str.contains('BE', na=False), 'condition'] = np.nan

    '''
    A beginning to only keep one side of the road for connecting it to the bridges. We did not proceed with this.
    The differences are little, and now the first match is used. Sometimes where was no distincion between left and right
    There were also a lot of inconsistencies. The code does NOT work yet. 
    
    one_way_roads = ['(R)', 'Right', 'right', 'Right']
    for i in one_way_roads:
        df_bridgesN1.drop(df_bridgesN1[df_bridgesN1['name'].str.contains(i)].index, inplace=True)
    '''

def make_infra_id(df_road):
    # Make bridge_id and road_id based on road and LRP
    # WORDT MOEILIJKER: VOOR ELKE ROAD NIEUW STARTNUMMER
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
    df_road['name'] = df_road['model_type']

def get_road_name(df_road):
    '''
    In components.py, a road name is asked. It is set as the standard value 'Unknown'
    '''
    df_road['road_name'] = 'Unknown'

def make_id(df_road):
    '''
    Generates an unique id for each road
    '''
    unique_id = 1000000
    for i in range(len(df_road['id'])):
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
    get_name(df_road)
    get_road_name(df_road)
    make_id(df_road)
    collect_roads(df_road)

def make_figure(df):
    '''
    Makes a plot of the N1, with different colors for the model type (source, link, bridge, sink)
    '''
    # CHANGE FOR TOTAL NETWORK
    sns.lmplot(x='lon', y='lat', data=df, hue='model_type', fit_reg=False, scatter_kws={"s": 1})
    plt.show()
def combine_data():
    #df_all_roads_temp = pd.DataFrame(columns=column_names)
    #print(df_all_roads_temp)

    df_all_roads = pd.DataFrame(columns=column_names)  # initialize empty dataframe outside of function

    for df in list_all_roads:
        df_all_roads = pd.concat([df_all_roads, df])  # append to df_all_roads in each iteration

    make_figure(df_all_roads)
    save_data(df_all_roads)
    #make_upperbound(df_all_roads)

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

def make_upperbound(df):
    '''
    Sorts the bridges df by the highest condition, and makes a new match between the bridges and roads.
    '''
    # sort the bridges file by highest condition
    df_bridges_sorted = df_bridges.sort_values(by='condition', ascending=False)

    # refill the column by connecting the two infra files
    connect_infra(df_bridges_sorted, df)

    # save the files
    df.to_csv('./data/demo_all_roads_UB.csv')
    df_all_roads_compact = df.loc[:, model_columns]
    df_all_roads_compact.to_csv('./data/demo_all_roads_compact_UB.csv')

# Run the prepare data function
filter_roads() # calls prepare_data function
combine_data()

def validate_bridges():
    '''
    Generates dataframes to check the statistics of the BMMS file with the two different sorting methods
    '''
    df_BMMS_LB = df_bridges.drop_duplicates(subset='bridge_id', keep='first')
    df_BMMS_UB = df_bridges.drop_duplicates(subset='bridge_id', keep='last')

    df_BMMS_LB.to_excel('./data/BMMS_LB.xlsx')
    df_BMMS_UB.to_excel('./data/BMMS_UB.xlsx')

    df_bridgesN1 = df_bridges[df_bridges['road'] == 'N1']

    # sns.lmplot(x='lon', y='lat', data=df_bridgesN1, fit_reg=False, scatter_kws={"s": 1})
    # plt.show()