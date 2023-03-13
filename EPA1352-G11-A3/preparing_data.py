# Filter on road N1
df_roadN1 = df_roads[df_roads['road'] == 'N1']

'''
Conceptualisation for automating this file for multiple roads

# obtain all unique roads
all_roads = df_roads['road'].unique()

# make a dataframe for each road
road_dfs = {}
for i in all_roads:
    roadname = f'df_road_{i}'
    road_dfs[roadname] = df_roads[df_roads['road'] == i]

And then this file can be adjusted as follows: 
* the functions below need a road dataframe as input, within ()
* the output dfs (right format) need to be dynamically named 
* this whole file runs through a for loop, looping over all the unique roads (the list all_roads)
* after all runs, one big datafile needs to be made where all the roads are put together with the right data for modeling (pd.concat)
* the file of all the roads needs to be exported to csv for modeling 
'''

def change_column_names():
    """
    The column names are updated and empty columns are generated for the information needed for modeling
    """
    df_roadN1['model_type'] = ''
    df_roadN1['length'] = np.nan
    df_roadN1['id'] = ''
    df_roadN1['name'] = ''
    df_roadN1['condition'] = np.nan
    df_roadN1['road_name'] = ''
    df_roadN1['bridge_length'] = np.nan

def change_model_type():
    """
    This function checks if the road object is a bridge and replaces it with 'bridge'
    Thereby replaces all other objects in 'link'
    The first and last object are given the model type 'source' and 'sink'
    """
    bridge_types = ['Bridge', 'Culvert'] # in doubt over: CrossRoad, RailRoadCrossing

    for i in bridge_types:
        df_roadN1.loc[df_roadN1['type'].str.contains(i), 'model_type'] = 'bridge'

    df_roadN1.loc[~df_roadN1['model_type'].str.contains('bridge'), 'model_type'] = 'link'

    df_roadN1['model_type'].iloc[0] = 'source'
    df_roadN1['model_type'].iloc[-1] = 'sink'

def standardize_bridges():
    """
    Since some bridges have >1 LRPs, we only model for every bridge one delay,
    and we only model traffic one-way, we consider these bridges as duplicates
    Therefore, we changed them to 'link' and removed the condition
    """
    # Drop bridge end from the roads file
    df_roadN1.loc[df_roadN1['gap'].str.contains('BE', na=False), 'model_type'] = 'link'
    df_roadN1.loc[df_roadN1['gap'].str.contains('BE', na=False), 'condition'] = np.nan

    '''
    A beginning to only keep one side of the road for connecting it to the bridges. We did not proceed with this.
    The differences are little, and now the first match is used. Sometimes where was no distincion between left and right
    There were also a lot of inconsistencies. The code does NOT work yet. 
    
    one_way_roads = ['(R)', 'Right', 'right', 'Right']
    for i in one_way_roads:
        df_bridgesN1.drop(df_bridgesN1[df_bridgesN1['name'].str.contains(i)].index, inplace=True)
    '''

def make_infra_id():
    # Make bridge_id and road_id based on road and LRP
    df_roadN1['road_id'] = df_roadN1['road'] + df_roadN1['lrp']
    df_bridges['bridge_id'] = df_bridges['road'] + df_bridges['LRPName']

def connect_infra(bridges_file):
    """
    This function connects the bridges df with the road df, to obtain information about bridge condition and length
    """
    # find exact match between road+LRP
    for index, row in df_roadN1.iterrows():
        if 'bridge' in row['model_type']:
            road_id = row['road_id'].strip()
            matching_bridge = bridges_file[bridges_file['bridge_id'].str.contains(road_id)]
            if not matching_bridge.empty:
                bridge_condition = matching_bridge.iloc[0]['condition']
                bridge_length = matching_bridge.iloc[0]['length']
                df_roadN1.at[index, 'condition'] = bridge_condition
                df_roadN1.at[index, 'bridge_length'] = bridge_length

    # Since there are inconsistencies between the two datasets, the procedure is ran again for less exact matches
    fill_in_infra(bridges_file)

def fill_in_infra(bridges_file):
    """
    This function connects the bridges df with the road df, to obtain information about bridge condition and length
    This is done making a less exact match due to inconsistencies between the roads and bridges data
    Iterates only over the columns with model type bridges and empty condition (NaN)
    """
    # Slice the road_id to obtain an eight number value, without the extension a-z
    df_roadN1['road_id_sliced'] = df_roadN1['road_id'].str.slice(stop=8)

    # find match between the reduced road+LRP id
    for index, row in df_roadN1.loc[df_roadN1['condition'].isna()].iterrows():
        if 'bridge' in row['model_type']:
            road_id = row['road_id_sliced']
            matching_bridge = bridges_file[bridges_file['bridge_id'].str.contains(road_id)]
            if not matching_bridge.empty:
                bridge_condition = matching_bridge.iloc[0]['condition']
                bridge_length = matching_bridge.iloc[0]['length']
                df_roadN1.at[index, 'condition'] = bridge_condition
                df_roadN1.at[index, 'bridge_length'] = bridge_length

def bridge_to_link():
    '''
    If no match is found between the id's of the bridges and roads,
    the model type of these bridges is set to link for modeling purposes.
    '''
    for index, row in df_roadN1.loc[df_roadN1['condition'].isna()].iterrows():
        if 'bridge' in row['model_type']:
            df_roadN1.loc[index, 'model_type'] = 'link'
            df_roadN1.loc[index, 'name'] = 'link'

def get_length():
    '''
    Fills in the length of each road part based on the chainage
    '''
    df_roadN1['length'] = abs(df_roadN1['chainage'].astype(float).diff()) * 1000
    df_roadN1['length'][0] = 0

def get_name():
    '''
    Fills in the name of the road part, based on the model type
    '''
    df_roadN1['name'] = df_roadN1['model_type']

def get_road_name():
    '''
    In components.py, a road name is asked. It is set as the standard value 'Unknown'
    '''
    df_roadN1['road_name'] = 'Unknown'

def make_id():
    '''
    Generates an unique id for each road
    '''
    unique_id = 1000000
    for i in range(len(df_roadN1['id'])):
        df_roadN1.loc[i, 'id'] = unique_id
        unique_id += 1

def make_figure():
    '''
    Makes a plot of the N1, with different colors for the model type (source, link, bridge, sink)
    '''
    sns.lmplot(x='lon', y='lat', data=df_roadN1, hue='model_type', fit_reg=False, scatter_kws={"s": 1})
    plt.show()

def prepare_data():
    '''
    Runs all procedures to obtain the right columns and information for modeling
    '''
    change_column_names()
    make_infra_id()
    change_model_type()
    standardize_bridges()
    connect_infra(df_bridges) # also calls for fill_in_infra() within the function
    bridge_to_link()
    get_length()
    get_name()
    get_road_name()
    make_id()
    make_figure()

model_columns = ['road', 'id', 'model_type', 'name', 'lat', 'lon', 'length', 'condition', 'bridge_length'] # 'road_name'
def save_data():
    '''
    Saves the files
    '''
    # Write the dataframe to csv
    df_roadN1.to_csv('./data/demo_N1_LB.csv')

    # Make compact datafile and export to csv
    df_N1_compact = df_roadN1.loc[:, model_columns]
    df_N1_compact.to_csv('./data/demo_N1_compact_LB.csv')

def make_upperbound():
    '''
    Sorts the bridges df by the highest condition, and makes a new match between the bridges and roads.
    '''
    # sort the bridges file by highest condition
    df_bridges_sorted = df_bridges.sort_values(by='condition', ascending=False)

    # refill the column by connecting the two infra files
    connect_infra(df_bridges_sorted)

    # save the files
    df_roadN1.to_csv('./data/demo_N1_UB.csv')
    df_N1_compact = df_roadN1.loc[:, model_columns]
    df_N1_compact.to_csv('./data/demo_N1_compact_UB.csv')

# Run the prepare data function
prepare_data()
save_data()
make_upperbound()

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