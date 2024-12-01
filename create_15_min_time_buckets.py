import pandas as pd
import datetime
import math
from collections import OrderedDict
from fastparquet import write

pd.set_option('display.max_columns', None)

#collect actual plane data going with 3 hr timeframes. aggregate distribution of arrivals throughout 3 hr timeframes (histogram, perhaps)

actual = pd.DataFrame(columns = ['ID', 'Value'])
airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']

#O(n) solution): build dict of all ids first, then add to it looking at each line of database

results = OrderedDict({})
start_date = datetime.datetime(2022, 9, 1)

def format_hour(dt):
    hour_str = f"{dt.hour:02d}"  # Ensure two digits with leading zero if needed
    return f"{hour_str}00"  # Add two zeros at the end

for a in airports:
    date_pointer = start_date
    for i in range(11):
        #date_pointer = date_pointer + datetime.timedelta(days = 24)
        #print(date_pointer)
        for j in range(24):
            for k in range(6):
                date_pointer = date_pointer + datetime.timedelta(hours = 1)
                
                temp = date_pointer
                h = format_hour(temp)
                c = 0
                for l in range(12):
                    c += 15
                    #print(date_pointer, a)
                    delta = datetime.timedelta(minutes = 15)
                    #inclusive when calculating accuracy (i.e. both endpoints of 15 minute domain count)
                    id_val = a + '_' + temp.strftime('%y-%m-%d') + '_' + str(h) + '_' + str(c)
                    #value = test_df[(test_df['arrival_runway_actual_time'] >= date_pointer) & (test_df['arrival_runway_actual_time'] <= date_pointer + delta)]['gufi'].nunique()
                    #print(id_val)
                    results[id_val] = 0
                    #new_row = pd.DataFrame({'ID': [id_val], 'Value':[value]})
                    #actual = pd.concat([actual, new_row], ignore_index = True)
                    date_pointer = date_pointer + delta
                    #print(date_pointer)
        date_pointer = date_pointer + datetime.timedelta(days = 8)


#print(results)

explored = {}

hist_overall = {}
for i in [15,30,45,60,75,90,105,120,135,150,165,180]:
    hist_overall[i] = 0
    
'''
results of hist_overall on training data
{15: 27279, 30: 27696, 45: 27847, 60: 27749, 75: 27531, 90: 27379, 105: 27229, 120: 27002, 135: 26328, 150: 26095, 165: 26224, 180: 24846}
flights arrive uniformly throughout the three-hr period
'''    


for a in airports:
    
   
    test_df = pd.read_parquet(a+"_data_train.parquet")
    #print(len(test_df))
    for row in test_df.itertuples(index=False):
        
        if row.gufi not in explored and row.is_arrival:
            
            #why are all the rows full of NaN
            #create a key, see if it exists in results
            try:
                time = row.arrival_runway_actual_time
                #print(time)
               
                ref_hr = math.floor(time.hour / 4) * 4 + 1
                #print(time, time.hour, ref_hr)
                
                ref_time = time.replace(hour=ref_hr, minute=0, second=0, microsecond=0)
                #print(time, ref_time)
                hr = format_hour(ref_time)
                #print('time', type(time))
                #print('ref_time', type(ref_time))
                mins = math.ceil((time - ref_time).total_seconds() / 60 / 15) * 15
                hist_overall[mins] += 1
                id_val = a + '_' + time.strftime('%y-%m-%d') + '_' + hr + '_' + str(mins)
                print(row.gufi, time, id_val)
                #print(id_val)
                if id_val in results:
                    
                    results[id_val] += 1
                
                explored[row.gufi] = True
            except:
                #print("explored")
                pass

           
df = pd.DataFrame(list(results.items()), columns=['ID', 'Value'])
print(df.head())
#print(hist_overall)
write('airport_target_variable_train.parquet', df, compression='brotli', write_index=False)
'''

       
           
   

#this is O(n^2), very inefficient!
for a in airports:
    test_df = pd.read_parquet(a+"_data_test.parquet")
    start_date = datetime.datetime(2022, 9, 1)
    date_pointer = start_date
    for i in range(12):
        date_pointer = date_pointer + datetime.timedelta(days = 24)
        for j in range(8):
            for k in range(6):
                date_pointer = date_pointer + datetime.timedelta(hours = 1)
                def format_hour(dt):
                    hour_str = f"{dt.hour:02d}"  # Ensure two digits with leading zero if needed
                    return f"{hour_str}00"  # Add two zeros at the end
                temp = date_pointer
                h = format_hour(temp)
                c = 0
                for l in range(12):
                    c += 15
                    print(date_pointer, a)
                    delta = datetime.timedelta(minutes = 15)
                    #inclusive when calculating accuracy (i.e. both endpoints of 15 minute domain count)
                    id_val = a + '_' + temp.strftime('%y-%m-%d') + '_' + str(h) + '_' + str(c)
                    value = test_df[(test_df['arrival_runway_actual_time'] >= date_pointer) & (test_df['arrival_runway_actual_time'] <= date_pointer + delta)]['gufi'].nunique()
                    new_row = pd.DataFrame({'ID': [id_val], 'Value':[value]})
                    actual = pd.concat([actual, new_row], ignore_index = True)
                    date_pointer = date_pointer + delta
                    
                    
print(actual.head())
                
                   
'''