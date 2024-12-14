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


from sklearn.preprocessing import LabelEncoder #for one-hot encoding of categorical variables
#since we keep the ID column attached to each row we don't necessarily need to 'reverse'

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from gluonts.dataset.pandas import PandasDataset

airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']
    

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
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).astype('float16')

train_data['airport_categorical'] = pd.Categorical(train_data['airport'], categories=airports)
test_data['airport_categorical'] = pd.Categorical(test_data['airport'], categories=airports)


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


#gluonts has some issues here with 24*24*24*10 being the increment, since the second interval is not the same size as previous intervals for the validation dataset
#therefore, only include the first interval, which is fully 24*24*24*10 elements
val_data = val_data.head(len(val_data)//val_series_increment * val_series_increment)

train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).astype('float64')
val_data[val_data.select_dtypes(include=['number']).columns] = val_data.select_dtypes(include=['number']).astype('float64')
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).astype('float64')





#fix bug with testing data being asked to predict throughput for times which are outside the 8 day blocks specified in instructions
test_data_target_df = test_data.copy()
# Create a list to collect the rows to keep


# Iterate over the DataFrame in chunks of n
for i in range(0, len(test_data_target_df), 16):
    chunk_indices = test_data_target_df.index[i:i+16]
    test_data_target_df.loc[chunk_indices, 'test_period'] = i
    
    #past.append(test_data_target_df.iloc[i:i+4])
    #future.append(test_data_target_df.iloc[i:i+16])
    

# Concatenate the chunks back into a single DataFrame
#test_data_past = pd.concat(past, ignore_index=True)
#test_data_future = pd.concat(future, ignore_index=True)


#normalization procedure:
#each airport has 2 scalers, one for covariates and one for target.






train_data_timeseries=[]


# Process each group (airport)
for airport, group_data in train_data.groupby('airport'):
    # Convert group dataframe to TimeSeries
    
    
    group_ts = PandasDataset.from_long_dataframe(
        group_data, item_id="series", target="Value", timestamp="datetime", static_feature_columns=["airport_categorical"], feat_dynamic_real = weather_covariate_list, freq="15min"
    )
    
    
    train_data_timeseries += group_ts
    
val_data_timeseries = []

#sanity check line; we're in trouble if there's nans
#print([df.pd_dataframe()[df.pd_dataframe().isna().any(axis=1)] for df in train_data_target])

for airport, group_data in val_data.groupby('airport'):
    # Convert group dataframe to TimeSeries
    group_ts = PandasDataset.from_long_dataframe(
        group_data, item_id="series", target="Value", timestamp="datetime", static_feature_columns=["airport_categorical"], feat_dynamic_real = weather_covariate_list, freq="15min"
    )
    
    
    val_data_timeseries += group_ts



test_data_timeseries=[]

test_data_target_airport = []

for airport, group_data in test_data_target_df.groupby('airport'):
    # Convert group dataframe to TimeSeries
    group_ts = PandasDataset.from_long_dataframe(
        group_data, item_id="test_period", target="Value", timestamp="datetime", static_feature_columns=["airport_categorical"], feat_dynamic_real = weather_covariate_list,
        past_feat_dynamic_real = weather_covariate_list,
        future_length=12,
        freq="15min"
    )
    
    
    test_data_timeseries += group_ts
    
    test_data_target_airport += [airport] * 3 * len(group_data)



with open('train_data_timeseries_gluonts.pkl', 'wb') as file:
    pickle.dump(train_data_timeseries, file)


with open('val_data_timeseries_gluonts.pkl', 'wb') as file:
    pickle.dump(val_data_timeseries, file)

with open('test_data_timeseries_gluonts.pkl', 'wb') as file:
    pickle.dump(test_data_timeseries, file)
    
with open('test_data_target_airport.pkl', 'wb') as file:
    pickle.dump(test_data_target_airport, file)

with open('target_scalers_gluonts.pkl', 'wb') as file:
    pickle.dump(target_scalers, file)
    
