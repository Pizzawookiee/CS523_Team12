import pandas as pd
import datetime
from datetime import datetime as dt
import math
import numpy as np
from collections import OrderedDict
from fastparquet import write

#display all columns
pd.set_option('display.max_columns', None)

#load flight_vals from FUSER data

#add a new feature which is log10(actual - estimated arrival in minutes); probably not necessaru given lack of proper testing data
#unless we do something weird like predict estimated number of arrivals in 3 hrs from guessing parameters of dist of 'estimated' arrivals
#initialize estimated times then use weather data to predict residual leading to actual
#not really working since any delays shift the entire predicted distribution forward in time, which is not consistent with uniform distribution of flights seen in training data throughout 3-hr periods



#load METAR data, connect it to train data






#load TAF data, connect it to test data


def calc_cloud_val(row): #make sure to test with and without this feature if time allows
    #assume row is a pandas DataFrame row
    #aggregate all cloud cover info in the row and turn into metric: 0 for least severity, then approaching 1 for increasing severity
    #if None, return 0; assuming if no cloud information, then no clouds at all
    
    
    
    
    
    fields = {
        'cover_0_500':0,
        'CB_0_500':0,
        'TCU_0_500':0,
        'cover_500_2000':0,
        'CB_500_2000':0,
        'TCU_500_2000':0,
        'cover_2000_10000':0,
        'CB_2000_10000':0,
        'TCU_2000_10000':0,
        'cover_10000_20000':0,
        'CB_10000_20000':0,
        'TCU_10000_20000':0,
        'TCU_200_10000':0,
        'cover_10000_20000':0,
        'CB_10000_20000':0,
        'TCU_10000_20000':0,
        'cover_20000_35000':0,
        'CB_20000_35000':0,
        'TCU_20000_35000':0,   
    }
    
    cover_type_dict = {'SKC': 0, 'FEW': 2, 'SCT': 4, 'BKN': 7, 'OVC': 8}
    cloud_types = ['CB', 'TCU']
    boundaries = [0,500,2000,10000, 20000, 35000]
    
    
    
    
    
    def cover_type(entry):
        #uses https://www.weather.gov/media/mhx/TAF_Card.pdf as a source to encode cover_type as a number
        try:
            return cover_type_dict[entry]
        except Exception as e:
            return None
            
    
    def find_buckets(cover, cloud, altitude):

        start = 0
        end = 0

        
        if pd.isna(altitude): #stop early, since without an altitude we will not consider the cloud data
            return None
        
        for i in range(len(boundaries) - 1):
            if boundaries[i] < altitude <= boundaries[i + 1]:
                start = boundaries[i]
                end = boundaries[i+1]
        #print(start, end, altitude)
        
        
        for i in cloud_types:
            if cloud == i:
                fields[cloud + '_' + str(start) + '_' + str(end)] = 1 #toggle 1 if there is a CB or TCU cloud in relevant cell
                #print(cloud + '_' + str(start) + '_' + str(end))
        
        fields['cover_' + str(start) + '_' + str(end)] = cover_type(str(cover))
        
        return None
    
    find_buckets(row['cover_type_1'], row['cloud_type_1'], row['altitude_1'])
    find_buckets(row['cover_type_2'], row['cloud_type_2'], row['altitude_2'])
    find_buckets(row['cover_type_3'], row['cloud_type_3'], row['altitude_3']) 
    
    
    #print(fields)
    return fields
    
    









def get_predictions_given_query_time(sub_df, query_time):
    #given a sub_df and a query_time, returns a dict with populated weather vals
    #the sub_df is assumed to be already filtered for station_id
    #the sub_df may be from original, ammend or correct; further processing needs to be done
    
    taf_test_df = sub_df 
    
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

    
    if len(taf_test_df) == 0: #there might be an empty ammend/correction sub_df 
        return weather_vals
    
    validity_start_uniques = list(taf_test_df['validity_start'].unique())
    
    
    #print(validity_start_uniques, query_time)
    
    
    try:
        valid_rows = taf_test_df[taf_test_df['validity_start'] == max([t for t in validity_start_uniques if t <= query_time])]
    except Exception as e: #the ammend or correct sub_df may not have entries for all times
        return weather_vals
    
    #print(valid_rows)
    valid_rows = valid_rows.copy() #avoids "A value is trying to be set on a copy of a slice from a DataFrame." warning

    #valid_rows['cloud_severity'] = valid_rows.apply(lambda row: calc_cloud_val(row), axis=1) #old v1 line
    valid_rows = pd.concat([valid_rows, valid_rows.apply(calc_cloud_val, axis=1, result_type='expand')], axis=1)
    
    #travel up the rows, finding the first row with time < query_time (check from_time, tempo_start, becmg_start). retrieve vals from that row.
    #THEN continue until first row that HAS NULL for from_time, tempo_start, becmg_start. take vals from that row and populate any empty fields
    #make sure to handle nulls for temperature, visibility and wind speed

    #print(valid_rows['cloud_severity'])

    for idx in range(len(valid_rows) - 1, -1, -1):  # Loop from last row to first row
        row = valid_rows.iloc[idx]
        
        # Condition 1: Check if any of from_time, tempo_start, or becmg_start contain a non-null time less than the query time
        # We make the assumption that TEMPO, FROM, BECMG have same impact
        # (this is not quite true, if time allows program in more accurate behavior such as BECMG having gradual progression of weather, TEMPO having <1/2 appearance of conditions, etc.)
        #print(row)
        
        if ((pd.notna(row['from_time']) and row['from_time'] <= query_time) or
            (pd.notna(row['tempo_start']) and row['tempo_start'] <= query_time) or
            (pd.notna(row['becmg_start']) and row['becmg_start'] <= query_time)):
            #print(f"Row {idx} satisfies the first condition.")
            for key in weather_vals.keys():
                if pd.notna(row[key]):
                    weather_vals[key] = row[key]
            # Now, look for the next row where from_time, tempo_start, and becmg_start are all null
            # we want to populate any vals that didn't show up in the prior search with vals from the above lines which are 'general' predictions for the time period
            for idx2 in range(idx - 1, -1, -1):
                row2 = valid_rows.iloc[idx2]
                if pd.isna(row2['from_time']) and pd.isna(row2['tempo_start']) and pd.isna(row2['becmg_start']):
                    #print(f"Row {idx2} satisfies the second condition with all nulls.")
                    #print("Values from row with all nulls:", row2)
                    for key in weather_vals.keys():
                        if pd.notna(row2[key]) and weather_vals[key] == None: #we don't replace an existing weather val since we are going back in time
                            weather_vals[key] = row2[key]
                    break
            break

    #let's say we found nothing, so we do a 2nd search...
    for idx in range(len(valid_rows) - 1, -1, -1):
        row = valid_rows.iloc[idx]
        if pd.isna(row['from_time']) and pd.isna(row['tempo_start']) and pd.isna(row['becmg_start']):
            #print(f"Row {idx} satisfies the second condition with all nulls.")
            #print("Values from row with all nulls:", row2)
            for key in weather_vals.keys():
                if pd.notna(row[key]):
                    weather_vals[key] = row[key]
            break

    return(weather_vals)



#looping through taf_data:
#topmost level: airport
#next: day
#next: split into original, ammend, correct
#next: for each time bucket in the day, find vals from each
#next: return one row
#next: attach data to time bucket


#filter by day, airport. create a new db relating 15-min windows for the day at the airport to weather
#sort into three piles (original, ammend, correct)
#sort by validity_start, from_time, tempo_start, becmg_start
#--> first in DESCENDING order (i.e. we go from forward time to backward) of validity start, for all 15-min buckets from validity_start to pointer
#--> (pointer is last validity_start accessed (pointer variable), and equal to end of day at initialization
#--> populate the buckets
#--> first do from original, then correct, then ammend






def process_taf_data(taf_dataset):
    #assuming right now that all taf_data goes to test
    airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']
    start_date = datetime.datetime(2022, 9, 1)
    end_date = datetime.datetime(2023, 9, 1) + datetime.timedelta(hours = 20)
    results = OrderedDict({})
    
    #populate results dict
    
    def format_hour(dt):
        hour_str = f"{dt.hour:02d}"  # Ensure two digits with leading zero if needed
        return f"{hour_str}00"  # Add two zeros at the end

    for a in airports:
        date_pointer = start_date
        for i in range(12):
            date_pointer = date_pointer + datetime.timedelta(days = 24)
            #remake this so it includes info for ALL hours if possible to allow for sliding windows for ALL possible 4 hr blocks
            #which means, for each time, go back 4 hours (floor) 
            
            
            for j in range(8):
                for k in range(24):
                    #date_pointer = date_pointer + datetime.timedelta(hours = 1)
                    
                    temp = date_pointer
                    if temp <= end_date:  
                        h = format_hour(temp)
                        c = 0
                        
                        for l in range(4):
                            c += 15
                            
                            
                            id_val = a + '_' + temp.strftime('%y-%m-%d') + '_' + str(h) + '_' + str(c) 
                            
                            
                            
                            results[id_val] = None
                            
                    date_pointer = date_pointer + datetime.timedelta(hours = 1)
            
        #print(date_pointer)
    #print(results)
    
    
    current_airport = ''
    current_airport_subdf = taf_dataset
    
    for k in results.keys():
        parts = k.split('_')
       
    
        # Extract station_id and datetime
        station_id = parts[0]
        datetime_str = parts[1]
        hours_to_add = int(parts[2][:2])  #example: '0100' becomes 1 hr
        minutes_to_add = int(parts[3])

        # Convert the datetime_str to a datetime object
        datetime_obj = dt.strptime(datetime_str, "%y-%m-%d")
        
        # Add hours and minutes to the datetime
        query_datetime = datetime_obj + datetime.timedelta(hours=hours_to_add, minutes=minutes_to_add)
        
        if current_airport != station_id:
            #filter dataset by airport, replace current_subdf
            #we keep the current_subdf and current_airport in memory to avoid having to keep filtering the dataset
            current_airport = station_id
            current_airport_subdf = taf_dataset[taf_dataset['station_id'] == current_airport]
            
        day_of = query_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        day_after = day_of + datetime.timedelta(days=1)
        
        #print(day_of, day_after)
        
        #filter by day
        
        #current_airport_day_subdf = current_airport_subdf[(current_airport_subdf['validity_start'] < day_after) & (current_airport_subdf['validity_start'] >= day_of)] #failure points at 2300 hr 60 minutes AND from 0000 hr 15 min to 0500 hr 60 min
        current_airport_day_subdf = current_airport_subdf[(current_airport_subdf['validity_start'] <= day_after + datetime.timedelta(hours = 6)) & (current_airport_subdf['validity_start'] >= day_of - datetime.timedelta(hours = 6))] #possible failure point around the late July - early August point?
        

        
        #split into original, ammend, correct
        current_airport_day_subdf_original = current_airport_day_subdf [(current_airport_day_subdf['is_ammend'] == False) & (current_airport_day_subdf['is_correct'] == False)]
        current_airport_day_subdf_ammend = current_airport_day_subdf [current_airport_day_subdf['is_ammend'] == True]
        current_airport_day_subdf_correct = current_airport_day_subdf [current_airport_day_subdf['is_correct'] == True]
        
        
        
        #get candidate row_vals from all three
        original_row = get_predictions_given_query_time(current_airport_day_subdf_original, query_datetime)
        ammend_row = get_predictions_given_query_time(current_airport_day_subdf_ammend, query_datetime)
        correction_row = get_predictions_given_query_time(current_airport_day_subdf_correct, query_datetime)
        
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

        for i in [original_row, ammend_row, correction_row]:
            for m in list(weather_vals.keys()):
                if i[m] != None:
                    weather_vals[m] = np.float16(i[m]) #cast as float16
                    
        

        results[k] = weather_vals
        print(k, weather_vals)
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.rename(columns={'index': 'station_id'}, inplace=True)
    return df
            

'''
taf_test_df = pd.read_parquet('TAF_test_only_arrival_airports.parquet')
#print(taf_test_df.head())
taf_test_weather_data = process_taf_data(taf_test_df) #this returns a dataframe with columns 'station_id', 'temperature', 'visibility', 'wind_speed' and 'cloud_severity'

write('taf_test_weather_data.parquet', taf_test_weather_data, compression='brotli', write_index=False)
'''

#taf_test_weather_data = pd.read_parquet('taf_test_weather_data.parquet')
#print(taf_test_weather_data.head())

#need to handle NaN rows in TAF; maybe substitute with METAR for test since we are also generating METAR for test hrs as well (i.e. METAR for 1st hr, TAF for rest)
#(i.e. go back in time on the full METAR data, assuming that *all* METAR data before the 1st hr of 4 hr block is available)


def get_predictions_given_query_time_metar(sub_df, query_time): #if time allows, fix internal variable names to say 'metar' not 'taf'
    #given a sub_df and a query_time, returns a dict with populated weather vals
    #the sub_df is assumed to be already filtered for station_id
    #the sub_df may be from original, ammend or correct; further processing needs to be done
    
   
    
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

    if len(sub_df) == 0: #there might be an empty ammend/correction sub_df 
        return weather_vals
    
    valid_rows = sub_df.copy() #avoids "A value is trying to be set on a copy of a slice from a DataFrame." warning
    valid_rows = valid_rows[valid_rows['datetime'] <= query_time]
    #print(len(valid_rows))
    
    #valid_rows['cloud_severity'] = valid_rows.apply(lambda row: calc_cloud_val(row), axis=1)
    valid_rows = pd.concat([valid_rows, valid_rows.apply(calc_cloud_val, axis=1, result_type='expand')], axis=1)
    

    #METAR records visibility in meters, while TAF uses U.S. statute miles truncated in range 1-6
    #therefore, convert meters to U.S. statute miles and truncate vals in [7,10] to 6
    
   
    valid_rows['visibility'] = np.float16(np.where(valid_rows['visibility'] // 1609.344 >= 7, 6, valid_rows['visibility'] // 1609.344))

    #travel up the rows, get the most recent value
    for idx in range(len(valid_rows) - 1, -1, -1):
        row = valid_rows.iloc[idx]
        
        #print(row)
        for key in weather_vals.keys():
            if pd.notna(row[key]):
                #print(key, row[key])
                
                weather_vals[key] = row[key]
                
        break
        
    
    #print(weather_vals)
    return(weather_vals)





def process_metar_data_test(taf_dataset):
    #input variable is named taf_dataset bc too lazy to switch all variable names
    
    #assuming right now that all taf_data goes to test
    airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']
    start_date = datetime.datetime(2022, 9, 1)
    end_date = datetime.datetime(2023, 9, 1) + datetime.timedelta(hours = 20)
    results = OrderedDict({})
    
    #populate results dict
    
    
    def format_hour(dt):
        hour_str = f"{dt.hour:02d}"  # Ensure two digits with leading zero if needed
        return f"{hour_str}00"  # Add two zeros at the end

    for a in airports:
        date_pointer = start_date
        for i in range(12):
            date_pointer = date_pointer + datetime.timedelta(days = 24)
            #remake this so it includes info for ALL hours if possible to allow for sliding windows for ALL possible 4 hr blocks
            #which means, for each time, go back 4 hours (floor) 
            
            
            for j in range(8):
                for k in range(24):
                    #date_pointer = date_pointer + datetime.timedelta(hours = 1)
                    
                    temp = date_pointer
                    if temp <= end_date:  
                        h = format_hour(temp)
                        c = 0
                        
                        for l in range(4):
                            c += 15
                            
                            
                            id_val = a + '_' + temp.strftime('%y-%m-%d') + '_' + str(h) + '_' + str(c) 
                            
                            
                            
                            results[id_val] = None
                            
                    date_pointer = date_pointer + datetime.timedelta(hours = 1)
            
        #print(date_pointer)
    #print(results)
    
    
    current_airport = ''
    current_airport_subdf = taf_dataset
    
    for k in results.keys():
        parts = k.split('_')
       
    
        # Extract station_id and datetime
        station_id = parts[0]
        datetime_str = parts[1]
        hours_to_add = int(parts[2][:2])  #example: '0100' becomes 1 hr
        minutes_to_add = int(parts[3])

        # Convert the datetime_str to a datetime object
        datetime_obj = dt.strptime(datetime_str, "%y-%m-%d")
        
        # Add hours and minutes to the datetime
        query_datetime = datetime_obj + datetime.timedelta(hours=hours_to_add, minutes=minutes_to_add)
        
        if current_airport != station_id:
            #filter dataset by airport, replace current_subdf
            #we keep the current_subdf and current_airport in memory to avoid having to keep filtering the dataset
            current_airport = station_id
            current_airport_subdf = taf_dataset[taf_dataset['station_id'] == current_airport]
            
        
        current_airport_subtime_subdf = current_airport_subdf[(current_airport_subdf['datetime'] <= query_datetime + datetime.timedelta(hours = 6)) & (current_airport_subdf['datetime'] >= query_datetime - datetime.timedelta(hours = 6))]
        #print(len(current_airport_subtime_subdf))
        
        row = get_predictions_given_query_time_metar(current_airport_subtime_subdf, query_datetime)
        
        
        
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

        for m in list(weather_vals.keys()):
            if row[m] != None:
                weather_vals[m] = np.float16(row[m]) #cast as float16
                    
        

        results[k] = weather_vals
        print(k, weather_vals)
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.rename(columns={'index': 'station_id'}, inplace=True)
    return df
    
def process_metar_data_train(dataset):
    
    #assuming right now that all taf_data goes to test
    airports = ['KDEN', 'KMEM', 'KORD', 'KSEA', 'KDFW', 'KPHX', 'KCLT', 'KATL', 'KMIA', 'KJFK']
    start_date = datetime.datetime(2022, 9, 1)
    end_date = datetime.datetime(2023, 9, 1) + datetime.timedelta(hours = 23)
    results = OrderedDict({})
    
    #populate results dict so the keys match the 'ID' vals in submission
    
    
    def format_hour(dt):
        hour_str = f"{dt.hour:02d}"  # Ensure two digits with leading zero if needed
        return f"{hour_str}00"  # Add two zeros at the end

    for a in airports:
        date_pointer = start_date
        for i in range(12):
            
            
            for j in range(24):
                for k in range(24):
                    #date_pointer = date_pointer + datetime.timedelta(hours = 1)
                    
                    temp = date_pointer
                    if temp <= end_date:  
                        h = format_hour(temp)
                        c = 0
                        
                        for l in range(4):
                            c += 15
                            
                            
                            id_val = a + '_' + temp.strftime('%y-%m-%d') + '_' + str(h) + '_' + str(c) 
                            
                            
                            
                            results[id_val] = None
                            
                    date_pointer = date_pointer + datetime.timedelta(hours = 1)
            date_pointer = date_pointer + datetime.timedelta(days = 8)
            
        #print(date_pointer)
    #print(results)
    
    
    current_airport = ''
    current_airport_subdf = dataset
    
    for k in results.keys():
        parts = k.split('_')
       
    
        # Extract station_id and datetime
        station_id = parts[0]
        datetime_str = parts[1]
        hours_to_add = int(parts[2][:2])  #example: '0100' becomes 1 hr
        minutes_to_add = int(parts[3])

        # Convert the datetime_str to a datetime object
        datetime_obj = dt.strptime(datetime_str, "%y-%m-%d")
        
        # Add hours and minutes to the datetime
        query_datetime = datetime_obj + datetime.timedelta(hours=hours_to_add, minutes=minutes_to_add)
        
        if current_airport != station_id:
            #filter dataset by airport, replace current_subdf
            #we keep the current_subdf and current_airport in memory to avoid having to keep filtering the dataset
            current_airport = station_id
            current_airport_subdf = taf_dataset[taf_dataset['station_id'] == current_airport]
            
        
        current_airport_subtime_subdf = current_airport_subdf[(current_airport_subdf['datetime'] <= query_datetime + datetime.timedelta(hours = 6)) & (current_airport_subdf['datetime'] >= query_datetime - datetime.timedelta(hours = 6))]
        #print(len(current_airport_subtime_subdf))
        
        row = get_predictions_given_query_time_metar(current_airport_subtime_subdf, query_datetime)
        
        
        
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

        for m in list(weather_vals.keys()):
            if row[m] != None:
                weather_vals[m] = np.float16(row[m]) #cast as float16
                    
        

        results[k] = weather_vals
        print(k, weather_vals)
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.rename(columns={'index': 'station_id'}, inplace=True)
    return df



'''
metar_test_df = pd.read_parquet('Metar_Test.parquet')
#print(np.unique(metar_test_df[metar_test_df['station_id'] == 'KMEM']['wind_speed']))
#print(metar_test_df.head())
metar_test_weather_data = process_metar_data_test(metar_test_df) #this returns a dataframe with columns 'station_id', 'temperature', 'visibility', 'wind_speed' and 'cloud_severity'
write('metar_test_weather_data.parquet', metar_test_weather_data, compression='brotli', write_index=False)


metar_train_df = pd.read_parquet('Metar_Train.parquet')
#print(np.unique(metar_test_df[metar_test_df['station_id'] == 'KMEM']['wind_speed']))
#print(metar_test_df.head())
metar_train_weather_data = process_metar_data_train(metar_train_df) #this returns a dataframe with columns 'station_id', 'temperature', 'visibility', 'wind_speed' and 'cloud_severity'
write('metar_train_weather_data.parquet', metar_train_weather_data, compression='brotli', write_index=False)

'''



#pick either METAR or TAF data depending on what's available. preference for METAR in 1st hour, preference for TAF in hrs 2-4. If no TAF in hrs 2-4, use weather vals from 1st hour

metar_test_weather_data = pd.read_parquet('metar_test_weather_data.parquet')
metar_train_weather_data = pd.read_parquet('metar_train_weather_data.parquet')
taf_test_weather_data = pd.read_parquet('taf_test_weather_data.parquet')
airport_target_variable_train = pd.read_parquet('airport_target_variable_train.parquet')
airport_target_variable_test = pd.read_parquet('airport_target_variable_test.parquet')


def handle_nulls_test(A,B):
    block_size = 16

    for start in range(0, len(A), block_size):
        # Select the current 16-row block for both A and B
        A_16 = A.iloc[start:start + block_size]
        B_16 = B.iloc[start:start + block_size]

        # Rule 1: For the first 4 rows of B_16, replace null values with corresponding values from A_16
        B_16.iloc[:4] = B_16.iloc[:4].fillna(A_16.iloc[:4])

        # Rule 2: For the last 12 rows of B_16, replace null values with preceding values in B_16
        B_16.iloc[4:] = B_16.iloc[4:].ffill()

        # Rule 3: If any value is still null in the last 12 rows, replace with the 4th row of A_16
        B_16.iloc[4:] = B_16.iloc[4:].fillna(A_16.iloc[3])

        # Now, B_16 is updated. You can assign it back to B if needed
        B.iloc[start:start + block_size] = B_16
    
    return B
        
intermediate_test_weather_data = handle_nulls_test(metar_test_weather_data, taf_test_weather_data)
intermediate_test_weather_data.rename(columns={'station_id': 'ID'}, inplace=True)
intermediate_train_weather_data = metar_train_weather_data.ffill()
intermediate_train_weather_data.rename(columns={'station_id': 'ID'}, inplace=True)




del metar_test_weather_data
del taf_test_weather_data
del metar_train_weather_data

train_data = pd.merge(airport_target_variable_train, intermediate_train_weather_data, on='ID', how = 'left')
test_data = pd.merge(airport_target_variable_test, intermediate_test_weather_data, on='ID', how = 'left')

#fix null vals in test_data with a forward fill. a previous weather record in test dataframe either came from a prior TAF record (okay) or a METAR record from
#the 'first hour' of the four hour block (also okay). so all good
test_data = test_data.ffill()

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

write('train_data_final.parquet', train_data, compression='brotli', write_index=False)
write('test_data_final.parquet', test_data, compression='brotli', write_index=False)

#sanity check
print(len(airport_target_variable_train), len(train_data))
print(len(airport_target_variable_test), len(test_data))

