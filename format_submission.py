import pandas as pd
import datetime
from datetime import datetime as dt
import math

'''
reference = pd.read_csv('submission_format.csv/submission_format.csv')
reference.drop(['Value'], axis=1, inplace = True)
submission = pd.read_csv('test_tsmixer_output.csv')
submission['Value'] = submission['Value'].astype(int)

result = pd.merge(reference, submission, on='ID', how='left') 

result.to_csv('submission.csv', index=False)
'''
reference = pd.read_csv('submission_format.csv/submission_format.csv')
reference.drop(['Value'], axis=1, inplace = True)
submission = pd.read_csv('prediction.csv')
#submission['Value'] = submission['Value'].astype(int)


def reconstruct_ID_from_info_strings(airport, datetime_str):
    #assumes string in format %y-%m-%d_%H00_%M and %M in buckets of 15, 30, 45, 60 (need to handle 60 wraparound)
    helper = datetime_str[-2:]
    if helper == '60':
        datetime_obj = dt.strptime(datetime_str[:-2], "%y-%m-%d_%H00_") + datetime.timedelta(minutes=60)
    else:
        datetime_obj = dt.strptime(datetime_str, "%y-%m-%d_%H00_%M")
        
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
    datetime.strptime(datetime_str, "%y-%m-%d_%H%M_%S")
#datetime stored as index, so we retrieve it with row.name
    
submission['ID'] = submission.apply(lambda row: reconstruct_ID_from_info_strings(row['ID'][:4], row['ID'][5:]), axis = 1)
submission['Value'] = submission['Value'].clip(lower=0).round().astype(int)
result = pd.merge(reference, submission, on='ID', how='left') 
result = result.ffill() #a few missing values at the end?

result.to_csv('submission_2.csv', index=False)