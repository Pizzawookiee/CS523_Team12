import pandas as pd
import datetime
import math
from collections import OrderedDict
from fastparquet import write

pd.set_option('display.max_columns', None)

#adds is_arrival to FUSER data

actual = pd.DataFrame(columns = ['ID', 'Value'])
airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']

train_or_test = 'test'

for a in airports:
    
   
    v1 = pd.read_parquet(a+"_data_"+train_or_test+".parquet")
    v2 = pd.read_parquet(a+"_data_"+train_or_test+"_v2.parquet")
     
    v1_is_arrival = v1[['gufi', 'is_arrival']].drop_duplicates()
    v3 = pd.merge(v1_is_arrival, v2, on='gufi')
    write(a+"_data_"+train_or_test+"_v3.parquet", v3, compression='brotli', write_index=False)
    print("Finished processing v3 FUSER for " + a) 