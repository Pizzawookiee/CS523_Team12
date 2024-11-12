#gufi, arrival_airport, actual_arrival_time, actual_departure_time, estimated_arrival_time, estimated_arrival_time_prediction_timestamp, estimated_departure_time, estimated_departure_time_prediction_timestamp, aircraft_type, is_arrival

#note: arrival times are probably the most necessary. departure times relate to flights *leaving* the airport. perhaps delayed departures mean runways are occupied and such more arrival delays?
#null vals if none
#a flight may have an actual time listed on one day but have estimated times on other days
#a flight may arrive earlier or later than expected
#discard rows without both an estimated or actual arrival time (applies to departure as well) at the end of the query
#discard rows without both airplane engine class and airplane type at the end of the query

#don't save the pandas dataframe as csv, instead try a format that doesn't convert everything to strings

#assume all times are in UTC

import pandas as pd
import os
import re
import datetime
#pip install fastparquet to handle saving as parquet

folder = "FUSER_train_KORD/KORD"

#variable to help with naming the final exported file
arrival_airport = folder[-4:]

#defining a class to filter through file names of FUSAR data and retrieve info is good.
class filenameParserFUSER:
    def __init__(self, filename):
        #filename is the string representation of the filename
        #files are in form <AIRPORT>_<DATE>.<DATASET_TYPE>.csv
        # Step 1: Remove the .csv extension
        
        self.filename = filename
        #print(self.filename)
        
        filename_without_extension = filename.replace(".csv", "")

        # Step 2: Split at the first underscore. first_split[0] gives airport as string
        
        first_split = filename_without_extension.split('_', 1)

        # Step 3: Split at the first period in the second part of the split. second_split[0] gives datetime, second_split[1] gives dataset type
        second_split = first_split[1].split('.', 1)
        
        self.airport = first_split[0]
        self.date = datetime.datetime.strptime(second_split[0], "%Y-%m-%d")
        self.dataset_type = second_split[1]
        
    @classmethod
    def from_components(airport, date, dataset_type):
        self.airport = airport #string
        self.date = date #datetime object
        self.dataset_type = dataset_type #string
        self.filename = airport + "_" + date.strftime("%Y-%m-%d") + "." + dataset_type + ".csv"
       
        
#reference for types of datasets
dataset_types = ['runways_data_set', 'MFS_data_set', 'TFM_track_data_set', 'ETD_data_set']    

two_dates_pattern = r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}" #only handle files that handle one day at a time
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv') and not re.search(two_dates_pattern, f)]
csv_files_objects = [filenameParserFUSER(c) for c in csv_files]


TFM_files_objects = [c for c in csv_files_objects if c.dataset_type == 'TFM_track_data_set']
MFS_files_objects = [c for c in csv_files_objects if c.dataset_type == 'MFS_data_set']
runways_files_objects = [c for c in csv_files_objects if c.dataset_type == 'runways_data_set']
ETD_files_objects = [c for c in csv_files_objects if c.dataset_type == 'ETD_data_set']

'''
#test block to limit size of datasets
TFM_files_objects = TFM_files_objects[:1]
MFS_files_objects = MFS_files_objects[:1]
runways_files_objects = runways_files_objects[:1]
ETD_files_objects = ETD_files_objects[:1]
'''

columns_to_convert = {'runways_data_set': ['arrival_runway_actual_time', 'departure_runway_actual_time'],
'TFM_track_data_set': ['arrival_runway_estimated_time', 'timestamp'],
'ETD_data_set': ['departure_runway_estimated_time', 'timestamp'],
'MFS_data_set': []}

#convert_columns_string_to_datetime(runways_dataset, ['arrival_runway_actual_time', 'departure_runway_actual_time'])
#convert_columns_string_to_datetime(TFM_dataset, ['arrival_runway_estimated_time', 'timestamp'])
#convert_columns_string_to_datetime(ETD_dataset, ['departure_runway_estimated_time', 'timestamp'])

def convert_columns_string_to_datetime(df, columns_to_convert):
    #df is a pandas dataframe
    #columns is a list of columns (i.e. a list of strings representing column names
    #df[columns_to_convert] = df[columns_to_convert].apply(lambda col: pd.to_datetime(col, format='%Y-%m-%d %H:%M:%S'))
    if columns_to_convert != []:
        df[columns_to_convert] = df[columns_to_convert].apply(lambda col: pd.to_datetime(col, errors = 'coerce')) #throws NaN if failed, test for NaN with pd.isnull()
        df[columns_to_convert] = df[columns_to_convert].apply(lambda col: col.dt.floor('7min')) #sample a little more than twice every 15 min
        df.drop_duplicates(subset=columns_to_convert, inplace=True)
    #some estimations are made after the actual time, what to do then?
    #some estimations are made more than 4 hrs away from actual time, what to do then?



def create_single_dataset_from_many(FUSER_objects):
    dfs = []

    for csv_object in FUSER_objects:
        file_path = os.path.join(folder, csv_object.filename)
        temp_df = pd.read_csv(file_path)
        temp_df['arrival_airport'] = csv_object.airport
        convert_columns_string_to_datetime(temp_df, columns_to_convert[csv_object.dataset_type])
        dfs += [temp_df]
        
    return pd.concat(dfs, ignore_index = True)
    
runways_dataset = create_single_dataset_from_many(runways_files_objects)
TFM_dataset = create_single_dataset_from_many(TFM_files_objects)
ETD_dataset = create_single_dataset_from_many(ETD_files_objects)
MFS_dataset = create_single_dataset_from_many(MFS_files_objects)
'''
#test_block to limit size of dataframes
runways_dataset = runways_dataset.iloc[:100]
TFM_dataset = TFM_dataset.iloc[:100]
ETD_dataset = ETD_dataset.iloc[:100]
MFS_dataset = MFS_dataset.iloc[:100]
'''


#drop unnecessary columns
#runways_dataset.drop(['arrival_runway_actual', 'departure_runway_actual', 'relevant_date'], axis=1, inplace=True)
#MFS_dataset.drop(['major_carrier', 'isarrival', 'isdeparture', 'arrival_stand_actual', 'arrival_stand_actual_time', 'arrival_runway_actual', 'arrival_runway_actual_time',
#'departure_stand_actual', 'departure_stand_actual_time', 'departure_runway_actual', 'departure_runway_actual_time', 'gufi_day', 'flight_type', 'arrival_aerodrome_icao_name',
#'aircraft_engine_class'], axis=1, inplace=True)

#arrival_airport is a variable added to the dataframe pulled from the csv, see create_single_dataset_from_many()
runways_dataset = runways_dataset[['gufi', 'arrival_airport', 'arrival_runway_actual_time', 'departure_runway_actual_time']]
MFS_dataset = MFS_dataset[['gufi', 'arrival_airport', 'aircraft_type']]


#note: engine type (aircraft_engine_class) is predominantly JET with few examples for non-JET and sometimes null value
#note: airplane type is quite sparse, keep for now but might not be useful
#note: arrival_aerodrome_icao_name is also very sparse, probably hard to find any trends


#drop unnecessary vals from gufi
#first part of gufi is plane identifier


#convert data types (i.e. turn strings into datetime objects)


#rename columns
TFM_dataset.rename(columns={'arrival_runway_estimated_time': 'estimated_arrival_time'}, inplace=True)
TFM_dataset.rename(columns={'timestamp': 'estimated_arrival_time_prediction_timestamp'}, inplace=True)
ETD_dataset.rename(columns={'departure_runway_estimated_time': 'estimated_departure_time'}, inplace=True)
ETD_dataset.rename(columns={'timestamp': 'estimated_departure_time_prediction_timestamp'}, inplace=True)
        
#add extra values
TFM_dataset['is_arrival'] = True
ETD_dataset['is_arrival'] = False





#perform joins, finalize data

result = pd.merge(TFM_dataset, ETD_dataset, on=['gufi', 'is_arrival', 'arrival_airport'], how='outer')
result = pd.merge(runways_dataset, result, on=['arrival_airport','gufi'], how='inner')
result = pd.merge(MFS_dataset, result, on=['arrival_airport', 'gufi'], how='inner')
#result['arrival_airport'] = arrival_airport

#display all columns
pd.set_option('display.max_columns', None)

# Display the DataFrame
print(result.head())

#save as file
result.to_parquet(arrival_airport + '_data_train.parquet', engine='fastparquet', compression='brotli')
#print(result.sample(n=10))
#result.to_csv(arrival_airport + '_data_train.csv', index=False)

