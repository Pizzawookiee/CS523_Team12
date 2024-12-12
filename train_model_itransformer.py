import pandas as pd
import numpy as np
from pathlib import Path
#import matplotlib.pyplot as plt
import datetime
import re
from datetime import datetime as dt
#from fastparquet import write
import pickle
import torch
import math

from pathlib import Path

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation import MultivariateEvaluator

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.i_transformer import ITransformerEstimator
from gluonts.model.predictor import Predictor

airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']
airports_dict = {index: item for index, item in enumerate(airports)}
#print(airports_dict)


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


# Load train_data_timeseries
with open('train_data_timeseries_gluonts.pkl', 'rb') as file:
    train_data_timeseries = ListDataset(pickle.load(file), freq="15min")

# Load val_data_timeseries
with open('val_data_timeseries_gluonts.pkl', 'rb') as file:
    val_data_timeseries = ListDataset(pickle.load(file), freq="15min")

# Load test_data_timeseries
with open('test_data_timeseries_gluonts.pkl', 'rb') as file:
    test_data_timeseries = ListDataset(pickle.load(file), freq="15min")

# Load test_data_target_airport
with open('test_data_target_airport.pkl', 'rb') as file:
    test_data_target_airport = pickle.load(file)

# Load target_scalers
with open('target_scalers.pkl', 'rb') as file:
    target_scalers = pickle.load(file)


        


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


start_times = [test_data_timeseries[i]['start'].start_time for i in range(len(test_data_timeseries))]
airport_rows = [airports_dict[test_data_timeseries[i]['feat_static_cat'].item()] for i in range(len(test_data_timeseries))]

ID_vals = []
airport_vals = []

for i in range(len(start_times)):
    pointer = start_times[i] + datetime.timedelta(minutes = 45)
    airport = airport_rows[i]
    for i in range(12):
        pointer = pointer + datetime.timedelta(minutes=15)
        ID_vals += [reconstruct_ID_from_info(airport, pointer)]
        airport_vals += [airport]


#print(test_data_timeseries)
grouper_train = MultivariateGrouper(max_target_dim=len(train_data_timeseries))
train_data_timeseries = grouper_train(train_data_timeseries)
grouper_val = MultivariateGrouper(max_target_dim=len(val_data_timeseries))
val_data_timeseries = grouper_val(val_data_timeseries)
grouper_test = MultivariateGrouper(max_target_dim=len(test_data_timeseries))
test_data_timeseries = grouper_test(test_data_timeseries)




class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))



estimator = ITransformerEstimator(
    prediction_length=12,
    context_length=4,
    scaling="std",
    nonnegative_pred_samples=False,
    trainer_kwargs=dict(max_epochs=25),
    
)

predictor = estimator.train(
    training_data = train_data_timeseries, validation_data = val_data_timeseries, cache_data=True, shuffle_buffer_length=1024
)



predictor.serialize(Path("/tmp/"))
#predictor_deserialized = Predictor.deserialize(Path("/tmp/"))


evaluator = MultivariateEvaluator(
    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum}
)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data_timeseries, predictor=predictor, num_samples=100
)

forecasts = np.mean(list(forecast_it)[0].samples, axis=0).flatten()

pred_df = pd.DataFrame({'ID': ID_vals, 'Value': forecasts, 'airport': airport_vals})

result = []
        
for airport, group_data in pred_df.groupby('airport'):
# Convert group dataframe to TimeSeries
    
    group_data['Value'] = target_scalers[airport].inverse_transform(pd.DataFrame(group_data['Value']))
    result.append(group_data)
    
result = pd.concat(result)

result = result[['ID', 'Value']]

reference = pd.read_csv('submission_format.csv/submission_format.csv')
reference.drop(['Value'], axis=1, inplace = True)
submission = pd.read_csv('test_tsmixer_output.csv')
submission['Value'] = submission['Value'].astype(int)

result = pd.merge(reference, submission, on='ID', how='left') 

result.to_csv('submission.csv', index=False)

