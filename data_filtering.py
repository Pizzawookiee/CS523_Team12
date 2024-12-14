import os
import re
from datetime import datetime, timedelta
from calendar import monthrange
import pandas as pd
from fastparquet import write


'''
STAGE 1: retrieve all TAF data relating to only arrival airports
'''
folder = 'METAR_parquet_batches_train/METAR_parquet_batches_train'
airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']

batch_filenames = [f for f in os.listdir(folder) if f.endswith('.parquet') and 'batch' in f]

result = pd.DataFrame()

for b in batch_filenames:
    df = pd.read_parquet(os.path.join(folder, b))
    result = pd.concat([result, df.loc[df['station_id'].isin(airports)]], ignore_index = True)
    del df
'''
result = pd.DataFrame()
airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']
df = pd.read_parquet('metar_test_data.parquet')
result = pd.concat([result, df.loc[df['station_id'].isin(airports)]], ignore_index = True)
del df
'''

write('METAR_train' + '_only_arrival_airports.parquet', result, compression='brotli', write_index=False)

print(result.head())
