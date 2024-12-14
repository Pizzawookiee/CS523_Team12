import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import math
import numpy as np
from collections import OrderedDict
from fastparquet import write
import re
from tqdm import tqdm
import pickle

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from darts.dataprocessing.transformers import StaticCovariatesTransformer

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers.scaler import Scaler
from darts.metrics import mae, mse, mql
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.callbacks import TFMProgressBar
from sklearn.preprocessing import LabelEncoder #for one-hot encoding of categorical variables
#since we keep the ID column attached to each row we don't necessarily need to 'reverse'

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from darts.dataprocessing.transformers import Scaler


weather_vals = {
        'temperature':None,
        'visibility':None,
        'wind_speed':None,
        'cover_0_500':None,
        'CB_0_500':None,
        'TCU_0_500':None,
        'cover_500_2000':None,
        'CB_500_2000':None,
        'TCU_500_2000':None,
        'cover_2000_10000':None,
        'CB_2000_10000':None,
        'TCU_2000_10000':None,
        'cover_10000_20000':None,
        'CB_10000_20000':None,
        'TCU_10000_20000':None,
        'TCU_200_10000':None,
        'cover_10000_20000':None,
        'CB_10000_20000':None,
        'TCU_10000_20000':None,
        'cover_20000_35000':None,
        'CB_20000_35000':None,
        'TCU_20000_35000':None,   
    }
    
weather_covariate_list = list(weather_vals.keys())

train_data = pd.read_parquet('train_data_final.parquet')
test_data = pd.read_parquet('test_data_final.parquet')
train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).astype('float16')
print(train_data.columns)

'''
def extract_to_datetime(s):
    # Regex to match the format "yy-mm-dd_hh00_mm"
    match = re.search(r'\d{2}-\d{2}-\d{2}_\d{2}00_\d{2}', s)
    if match:
        date_time_list = match.group(0).split('_')
        return dt.strptime(date_time_list[0], '%y-%m-%d') + datetime.timedelta(hours = int(date_time_list[1][:2]), minutes = int(date_time_list[2]))
        
    else:
        return None  # Handle cases where no match is found
        
train_data['datetime'] = train_data['ID'].apply(extract_to_datetime)
test_data['datetime'] = test_data['ID'].apply(extract_to_datetime)
train_data['airport'] = train_data['ID'].apply(lambda x: x[:4])
test_data['airport'] = test_data['ID'].apply(lambda x: x[:4])
'''


df = train_data
df = df.sort_values(by='datetime')
# Function to split each group
split_idx = int(10 * 10 * 24 * 24 * 4)
train_data_splitted = df.iloc[:split_idx]
val_data_splitted = df.iloc[split_idx:]
    

'''

#STAGE 2 (TRANSFORM DATA TO CORRECT FORMAT)

'''

train_data = train_data_splitted
val_data = val_data_splitted
test_data = test_data

#normalization procedure:
#each airport has 2 scalers, one for covariates and one for target.

covariate_scalers = {}
target_scalers = {}


def scale_group(train):
    #assumes that train scalers are created first before attempting to scale val and test
    def scale_group(group):
        airport = group.iloc[0]['airport']
        
        covariates = weather_covariate_list
        target = ['Value']
        if train:
            scaler1 = MinMaxScaler(feature_range=(0, 1))
            scaler2 = MinMaxScaler(feature_range=(0, 1))
            group[covariates] = scaler1.fit_transform(group[covariates])
            group[target] = scaler2.fit_transform(group[target])
            covariate_scalers[airport] = scaler1
            target_scalers[airport] = scaler2
        else:
            scaler1 = covariate_scalers[airport]
            scaler2 = target_scalers[airport]
            group[covariates] = scaler1.transform(group[covariates])
            group[target] = scaler2.transform(group[target])
            
        
        
        return group
        
    return scale_group
    
    
    

#we need to sort by datetime, otherwise we run into errors when splitting data into rows
train_data = train_data.groupby('airport').apply(scale_group(train=True), include_groups=True).reset_index(drop=True).sort_values(by='datetime')
val_data = val_data.groupby('airport').apply(scale_group(train=False), include_groups=True).reset_index(drop=True).sort_values(by='datetime')
test_data = test_data.groupby('airport').apply(scale_group(train=False), include_groups=True).reset_index(drop=True)




train_series_increment = 24 * 24 * 4 * 10
val_series_increment = 24 * 24 * 4 * 10
#test_series_increment = 8 * 24 * 4



train_data['series'] = np.floor(np.arange(len(train_data)) / train_series_increment)
val_data['series'] = np.floor(np.arange(len(val_data)) / val_series_increment)
#test_data['series'] = np.floor(np.arange(len(test_data)) / test_series_increment)
#print(train_data.head())

train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).astype('float64')
val_data[val_data.select_dtypes(include=['number']).columns] = val_data.select_dtypes(include=['number']).astype('float64')
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).astype('float64')





#fix bug with testing data being asked to predict throughput for times which are outside the 8 day blocks specified in instructions
test_data_target_df = test_data.copy()
# Create a list to collect the rows to keep
past = []
future = []

# Iterate over the DataFrame in chunks of n
for i in range(0, len(test_data_target_df), 16):
    chunk_indices = test_data_target_df.index[i:i+16]
    test_data_target_df.loc[chunk_indices, 'test_period'] = i
    past.append(test_data_target_df.iloc[i:i+4])
    future.append(test_data_target_df.iloc[i:i+16])
    

# Concatenate the chunks back into a single DataFrame
test_data_past = pd.concat(past, ignore_index=True)
test_data_future = pd.concat(future, ignore_index=True)


#normalization procedure:
#each airport has 2 scalers, one for covariates and one for target.






train_data_covariates = []
train_data_target = []


# Process each group (airport)
for airport, group_data in train_data.groupby('airport'):
    # Convert group dataframe to TimeSeries
    group_covariate_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['series'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=weather_covariate_list,
        freq=pd.DateOffset(minutes=15),
    )
    
    group_target_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['series'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=['Value'],
        static_cols=['airport'],
        freq=pd.DateOffset(minutes=15),
    )
    
    train_data_covariates += group_covariate_ts

    
    train_data_target += group_target_ts
    
val_data_covariates = []
val_data_target = []

#sanity check line; we're in trouble if there's nans
#print([df.pd_dataframe()[df.pd_dataframe().isna().any(axis=1)] for df in train_data_target])

for airport, group_data in val_data.groupby('airport'):
    # Convert group dataframe to TimeSeries
    group_covariate_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['series'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=weather_covariate_list,
        freq=pd.DateOffset(minutes=15),
    )
    
    group_target_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['series'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=['Value'],
        static_cols=['airport'],
        freq=pd.DateOffset(minutes=15),
    )
    
    
    val_data_covariates += group_covariate_ts


    val_data_target += group_target_ts




test_data_future_covariates = []
test_data_past_covariates = []
test_data_target = []
test_data_target_airport = []

for airport, group_data in test_data_past.groupby('airport'):
    # Convert group dataframe to TimeSeries
    group_covariate_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['test_period'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=weather_covariate_list,
        freq=pd.DateOffset(minutes=15),
    )
    
    group_target_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['test_period'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=['Value'],
        static_cols=['airport'],
        freq=pd.DateOffset(minutes=15),
    )
    
    test_data_target_airport += [airport] * 3 * len(group_data)
    
    
    
    test_data_past_covariates += group_covariate_ts


    test_data_target += group_target_ts
#print(test_data_target_airport)

for airport, group_data in test_data_future.groupby('airport'):
    # Convert group dataframe to TimeSeries
    group_covariate_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['test_period'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=weather_covariate_list,
        freq=pd.DateOffset(minutes=15),
    )
    
    test_data_future_covariates += group_covariate_ts
    

with open('train_data_covariates.pkl', 'wb') as file:
    pickle.dump(train_data_covariates, file)

with open('train_data_target.pkl', 'wb') as file:
    pickle.dump(train_data_target, file)

with open('val_data_covariates.pkl', 'wb') as file:
    pickle.dump(val_data_covariates, file)

with open('val_data_target.pkl', 'wb') as file:
    pickle.dump(val_data_target, file)

with open('test_data_future_covariates.pkl', 'wb') as file:
    pickle.dump(test_data_future_covariates, file)

with open('test_data_past_covariates.pkl', 'wb') as file:
    pickle.dump(test_data_past_covariates, file)

with open('test_data_target.pkl', 'wb') as file:
    pickle.dump(test_data_target, file)
    
with open('test_data_target_airport.pkl', 'wb') as file:
    pickle.dump(test_data_target_airport, file)

with open('target_scalers.pkl', 'wb') as file:
    pickle.dump(target_scalers, file)