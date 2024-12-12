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

'''

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
'''


    
#STAGE 3 (TRAIN MODEL)
transformer = StaticCovariatesTransformer() #encodes airport data properly now as numerical categorical static covariate


with open('train_data_covariates.pkl', 'rb') as file:
    train_data_covariates = pickle.load(file)

with open('train_data_target.pkl', 'rb') as file:
    train_data_target = transformer.fit_transform(pickle.load(file))

with open('val_data_covariates.pkl', 'rb') as file:
    val_data_covariates = pickle.load(file)

with open('val_data_target.pkl', 'rb') as file:
    val_data_target = transformer.transform(pickle.load(file))

with open('test_data_future_covariates.pkl', 'rb') as file:
    test_data_future_covariates = pickle.load(file)

with open('test_data_past_covariates.pkl', 'rb') as file:
    test_data_past_covariates = pickle.load(file)

with open('test_data_target.pkl', 'rb') as file:
    test_data_target = transformer.fit_transform(pickle.load(file))

with open('test_data_target_airport.pkl', 'rb') as file:
    test_data_target_airport = pickle.load(file)

with open('target_scalers.pkl', 'rb') as file:
    target_scalers = pickle.load(file)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))




use_static_covariates = True
full_training = True




def main():
    #return None
    
    
    
    def train():
    

        model_tft = TFTModel(
            input_chunk_length=4,
            output_chunk_length=12,
            hidden_size=64,
            lstm_layers=1,
            use_static_covariates = True,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=64,
            n_epochs=300,
            optimizer_cls = torch.optim.AdamW,
            optimizer_kwargs = {
                "lr": 1e-3,
            },

            # learning rate scheduler
            lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
            lr_scheduler_kwargs = {"patience": 5},
            add_relative_index=True,
            add_encoders = {
                "cyclic": {
                        "future": ["hour", "dayofweek", "month"],
                        "past": ["hour", "dayofweek", "month"]
                    },  # add cyclic time axis encodings as future covariates
                'position': {'past': ['relative'], 'future': ['relative']},
                'transformer': Scaler(),
            },
            likelihood=None,
            loss_fn=RMSELoss(),
            random_state=2024,
            save_checkpoints = True,  # checkpoint to retrieve the best performing model state,
            force_reset = True,
        )
    
        
        

        
        model_tft.fit(
            series = train_data_target,
            past_covariates = train_data_covariates,
            future_covariates = train_data_covariates,
            val_series = val_data_target,
            val_past_covariates = val_data_covariates,
            val_future_covariates = val_data_covariates,
            verbose=True,            
        )
        
        return model_tft
        
    def test(model_tft):
        #model_lgbm = LightGBMModel.load_from_checkpoint(
        #    model_name="lgbm", best=True
        #)
        
        pred = model_tft.predict(n = 12, series = test_data_target, past_covariates = test_data_past_covariates, future_covariates = test_data_future_covariates, verbose=True)
        pred_df = pd.concat([x.pd_dataframe() for x in pred])

        pred_df['airport'] = test_data_target_airport
        
        result = []
        
        for airport, group_data in pred_df.groupby('airport'):
        # Convert group dataframe to TimeSeries
            
            group_data['Value'] = target_scalers[airport].inverse_transform(pd.DataFrame(group_data['Value']))
            result.append(group_data)
            
        result = pd.concat(result)
        
        def reconstruct_ID_from_info(airport, datetime_obj):
        
            def format_hour(dt):
                hour_str = f"{dt.hour:02d}"  # Ensure two digits with leading zero if needed
                return f"{hour_str}00"  # Add two zeros at the end
        
            temp = datetime_obj - datetime.timedelta(hours=4)
                
            temp = temp.replace(minute=0, second=0, microsecond=0)
            
            while temp.hour % 4 != 0:
                # If not, increment by 1 hour
                temp = temp + datetime.timedelta(hours=1)            
            
            temp = temp + datetime.timedelta(hours=1)
            
            hr = format_hour(temp)
            mins = math.ceil((datetime_obj - temp).total_seconds() / 60 / 15) * 15
                
            id_val = airport + '_' + temp.strftime('%y%m%d') + '_' + hr + '_' + str(mins)
            
            
            return id_val
            
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
    

    
    model = train()
    model = TFTModel.load_from_checkpoint(
            model_name="tft", best=True
        )
    
    test(model)
    
    
if __name__ == '__main__':
    main()
    
