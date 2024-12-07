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

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers.scaler import Scaler
from darts.metrics import mae, mse, mql
from darts.models import TSMixerModel, TiDEModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.callbacks import TFMProgressBar
from sklearn.preprocessing import LabelEncoder #for one-hot encoding of categorical variables
#since we keep the ID column attached to each row we don't necessarily need to 'reverse'

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from darts.dataprocessing.transformers import Scaler


#REDO data: use METAR weather as past covariates, and TAF weather as future covariates. This requires making a new train dataset with TAF weather.


airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']

#STAGE 1 (TRAIN-VAL SPLIT; something in the last 15-20% of train data is wrong and causing all the validation data to NaN)
#no NaNs in train data

'''

train_data = pd.read_parquet('train_data_final.parquet')
test_data = pd.read_parquet('test_data_final.parquet')
train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).astype('float16')
print(train_data.columns)


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
train_data = train_data.drop(columns = ['hour', 'minute'])
test_data = test_data.drop(columns = ['hour', 'minute']) 


df = train_data
df = df.sort_values(by='datetime')
# Function to split each group
split_idx = int(10 * 10 * 24 * 24 * 4)
train_data_splitted = df.iloc[:split_idx]
val_data_splitted = df.iloc[split_idx:]
    



write('train_data_tsmixer.parquet', train_data_splitted, compression='brotli', write_index=False)

write('val_data_tsmixer.parquet', val_data_splitted, compression='brotli', write_index=False)

write('test_data_tsmixer.parquet', test_data, compression='brotli', write_index=False)

'''

#STAGE 2 (TRANSFORM DATA TO CORRECT FORMAT)



train_data = pd.read_parquet('train_data_tsmixer.parquet')
val_data = pd.read_parquet('val_data_tsmixer.parquet')
test_data = pd.read_parquet('test_data_tsmixer.parquet')

#normalization procedure:
#each airport has 2 scalers, one for covariates and one for target.

covariate_scalers = {}
target_scalers = {}


def scale_group(train):
    #assumes that train scalers are created first before attempting to scale val and test
    def scale_group(group):
        airport = group.iloc[0]['airport']
        
        covariates = ['temperature', 'wind_speed', 'visibility', 'cloud_severity']
        target = ['Value']
        if train:
            scaler1 = MinMaxScaler(feature_range=(-1, 1))
            scaler2 = MinMaxScaler(feature_range=(-1, 1))
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


'''
#operating on too large datasets, causing stalling

#the train data uses METAR and treats METAR as the 'past data' and the 'future data'
train_data_covariates = TimeSeries.from_group_dataframe(train_data, group_cols = ['airport', 'series'], time_col='datetime', value_cols=['temperature', 'wind_speed', 'visibility', 'cloud_severity'], freq = pd.DateOffset(minutes=15))
train_data_target = TimeSeries.from_group_dataframe(train_data, group_cols = ['airport', 'series'], time_col='datetime', value_cols=['Value'], freq = pd.DateOffset(minutes=15))

#the val data uses METAR and treats METAR as the 'past data' and the 'future data'
val_data_covariates = TimeSeries.from_group_dataframe(val_data, group_cols = ['airport', 'series'], time_col='datetime', value_cols=['temperature', 'wind_speed', 'visibility', 'cloud_severity'], freq = pd.DateOffset(minutes=15))
val_data_target = TimeSeries.from_group_dataframe(val_data, group_cols = ['airport', 'series'], time_col='datetime', value_cols=['Value'], freq = pd.DateOffset(minutes=15))

#the test data uses METAR for 1st hr of 4 hr block and TAF for next three hours
test_data_future_covariates = TimeSeries.from_group_dataframe(test_data_future, group_cols = ['airport', 'test_period'], time_col='datetime', value_cols=['temperature', 'wind_speed', 'visibility', 'cloud_severity'], freq = pd.DateOffset(minutes=15))
test_data_past_covariates = TimeSeries.from_group_dataframe(test_data_past, group_cols = ['airport', 'test_period'], time_col='datetime', value_cols=['temperature', 'wind_speed', 'visibility', 'cloud_severity'], freq = pd.DateOffset(minutes=15))
test_data_target = TimeSeries.from_group_dataframe(test_data_past, group_cols = ['airport', 'test_period'], time_col='datetime', value_cols=['Value'], freq = pd.DateOffset(minutes=15))
'''




train_data_covariates = []
train_data_target = []


# Process each group (airport)
for airport, group_data in train_data.groupby('airport'):
    # Convert group dataframe to TimeSeries
    group_covariate_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['series'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=['temperature', 'wind_speed', 'visibility', 'cloud_severity'],
        freq=pd.DateOffset(minutes=15),
    )
    
    group_target_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['series'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=['Value'],
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
        value_cols=['temperature', 'wind_speed', 'visibility', 'cloud_severity'],
        freq=pd.DateOffset(minutes=15),
    )
    
    group_target_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['series'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=['Value'],
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
        value_cols=['temperature', 'wind_speed', 'visibility', 'cloud_severity'],
        freq=pd.DateOffset(minutes=15),
    )
    
    group_target_ts = TimeSeries.from_group_dataframe(
        group_data,
        group_cols=['test_period'],  # Use the series column for sub-groups
        time_col='datetime',
        value_cols=['Value'],
        static_cols = ['airport'],
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
        value_cols=['temperature', 'wind_speed', 'visibility', 'cloud_severity'],
        freq=pd.DateOffset(minutes=15),
    )
    
    test_data_future_covariates += group_covariate_ts
    

    
#STAGE 3 (TRAIN MODEL)
#need to re-do scaling approach due to dimensionality errors


def create_params(
    input_chunk_length: int,
    output_chunk_length: int,
    full_training=True,
):
    # early stopping: this setting stops training once the the validation
    # loss has not decreased by more than 1e-5 for 10 epochs
    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-5,
        mode="min",
    )

    # PyTorch Lightning Trainer arguments (you can add any custom callback)
    if full_training:
        limit_train_batches = None
        limit_val_batches = None
        max_epochs = 200
        batch_size = 256
    else:
        limit_train_batches = 20
        limit_val_batches = 10
        max_epochs = 40
        batch_size = 64

    # only show the training and prediction progress bars
    progress_bar = TFMProgressBar(
        enable_sanity_check_bar=False, enable_validation_bar=True
    )
    
    
    
    pl_trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_epochs": max_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "accelerator": "auto",
        "callbacks": [early_stopper, progress_bar],
    }

    # optimizer setup, uses Adam by default
    optimizer_cls = torch.optim.AdamW
    optimizer_kwargs = {
        "lr": 1e-4,
    }

    # learning rate scheduler
    lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    lr_scheduler_kwargs = {"patience": 5}

    # for probabilistic models, we use quantile regression, and set `loss_fn` to `None`
    likelihood = None
    loss_fn = torch.nn.SmoothL1Loss()
    #likelihood = QuantileRegression()
    #loss_fn = None
    
    return {
        "input_chunk_length": input_chunk_length,  # lookback window
        "output_chunk_length": output_chunk_length,  # forecast/lookahead window
        "use_reversible_instance_norm": True,
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "likelihood": likelihood,  # use a `likelihood` for probabilistic forecasts
        "loss_fn": loss_fn,  # use a `loss_fn` for determinsitic model
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True, #this deletes all checkpoints when the script runs. MAKE BACKUPS!
        "batch_size": batch_size,
        "random_state": 42,
        "add_encoders": {
            "cyclic": {
                "future": ["hour", "dayofweek", "month"]
            }  # add cyclic time axis encodings as future covariates
        },
    }

input_chunk_length = 4
output_chunk_length = 12
use_static_covariates = False
full_training = True




def main():
    #return None
    def train():
    
        model_tsm = TSMixerModel(
            **create_params(
                input_chunk_length,
                output_chunk_length,
                full_training=full_training,
            ),
            use_static_covariates=use_static_covariates,
            model_name="tsm",
        )
        

        
        model_tsm.fit(
            series = train_data_target,
            past_covariates = train_data_covariates,
            future_covariates = train_data_covariates,
            val_series = val_data_target,
            val_past_covariates = val_data_covariates,
            val_future_covariates = val_data_covariates
        )
        
    def test():
        model_tsm = TSMixerModel.load_from_checkpoint(
            model_name="tsm", best=True
        )
        
        pred = model_tsm.predict(n = 12, series = test_data_target, past_covariates = test_data_past_covariates, future_covariates = test_data_future_covariates)
        pred_df = pd.concat([x.pd_dataframe() for x in pred])

        pred_df['airport'] = test_data_target_airport
        
        result = []
        
        for airport, group_data in pred_df.groupby('airport'):
        # Convert group dataframe to TimeSeries
            
            group_data['Value'] = target_scalers[airport].inverse_transform(pd.DataFrame(group_data['Value']))
            result.append(group_data)
            
        result = pd.concat(result)
        
        def reconstruct_ID_from_info(airport, datetime):
            formatted_components = (airport + '_' + datetime.strftime('%y%m%d_%H00_%M')).split('_')
            hour = int(formatted_components[2][:2])
            minute = int(formatted_components[3])
            day = int(formatted_components[1][-2:])
            
            
            if hour == 0:
                #handle wraparound to next day
                hour = 21
                minute = 180
                day = day - 1
            elif hour % 4 == 0 and minute == 00:
                #handle end of 4 hr period
                hour = hour - 3
                minute = 180
            else:
                diff = hour - (hour//4 * 4 + 1)
                hour = hour - diff
                minute = minute + diff * 60
            
            formatted_components[1] = formatted_components[1][:-2] + str(day)
            formatted_components[2] = str(hour).zfill(2) + '00'
            formatted_components[3] = str(minute)
            return "_".join(formatted_components)
            
        #datetime stored as index, so we retrieve it with row.name
            
        result['ID'] = result.apply(lambda row: reconstruct_ID_from_info(row['airport'], row.name), axis = 1)
            
        
        result = result.drop('airport', axis=1)
        result['Value'] = result['Value'].clip(lower=0).round()
        result = result[['ID', 'Value']]
        
        print(result)
        
        result.to_csv("test_tsmixer_output.csv", index=False)
        
        reference = pd.read_csv('submission_format.csv/submission_format.csv')
        reference.drop(['Value'], axis=1, inplace = True)
        submission = pd.read_csv('test_tsmixer_output.csv')
        submission['Value'] = submission['Value'].astype(int)

        result = pd.merge(reference, submission, on='ID', how='left') 

        result.to_csv('submission.csv', index=False)
    

    
    #train()
    test()
    
    
if __name__ == '__main__':
    main()
    
