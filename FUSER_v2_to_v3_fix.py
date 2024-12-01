import pandas as pd
import datetime
import math
from collections import OrderedDict
from fastparquet import write

pd.set_option('display.max_columns', None)

#adds is_arrival to FUSER data

actual = pd.DataFrame(columns = ['ID', 'Value'])
airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']

train_or_test = 'train'

for a in airports:
    
   
    
    v2 = pd.read_parquet(a+"_data_"+train_or_test+"_v2.parquet")
    
    
    v3 = v2
    v3['is_arrival'] = ~v3["arrival_runway_actual_time"].isna()
    write(a+"_data_"+train_or_test+"_v3.parquet", v3, compression='brotli', write_index=False)
    print("Finished processing v3 FUSER for " + a) 