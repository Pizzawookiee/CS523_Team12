import os
import re
from datetime import datetime, timedelta
from calendar import monthrange
import pandas as pd
from fastparquet import write


#script that demonstrates how we filtered METAR (and TAF) data to only use values from airports that are in FUSER data


folder = 'METAR_parquet_batches_train/METAR_parquet_batches_train'
airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']

batch_filenames = [f for f in os.listdir(folder) if f.endswith('.parquet') and 'batch' in f]

result = pd.DataFrame()

for b in batch_filenames:
    df = pd.read_parquet(os.path.join(folder, b))
    result = pd.concat([result, df.loc[df['station_id'].isin(airports)]], ignore_index = True)
    del df


write('METAR_train' + '_only_arrival_airports.parquet', result, compression='brotli', write_index=False)

print(result.head())
