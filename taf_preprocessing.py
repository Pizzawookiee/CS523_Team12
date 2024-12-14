#DATAFRAME(consistent across original, AMMEND and CORRECT)

#arrival_airport, forecast_timestamp, start_timestamp, end_timestamp, change_type, max_temperature (Celsius), max_temperature_timestamp,
#min_temperature (Celsius), min_temperature_timestamp, humidity...

#maybe temperature isn't crucial if categorical weather variables related to precipitation, cloud cover, weather, already incorporate?
#visbility could also manifest as '9999' which we cheat and set equivalent to 'P6SM' to 'convert' into US units; or, if None (no visibility available) NaN

#source: https://www.weather.gov/media/mhx/TAF_Card.pdf
#source: https://met.nps.edu/~bcreasey/mr3222/files/helpful/DecodeMETAR-TAF.html


#forecast_timestamp is time of forecast
#start_timestamp is start of valid period of forecast
#end_timestamp is end of valid period of forecast
#change_type is either None, FROM, TEMPO or BECMG
#--> None means N/A for this field
#--> FROM indicates significant changes beginning at the specified time
#--> TEMPO refers to conditions expected for < 1/2 of time interval of forecast
#--> PROB refers to conditions expected during interval at specified probability
#--> BECMG refers to change expected in time interval

'''
Question for ChatGPT: when a plane is landing, what parts of taf are most important?

When a plane is landing, the following parts of the Terminal Aerodrome Forecast (TAF) are most important for ensuring a safe and efficient landing:
1. Wind

    Direction and Speed:
        The wind direction (e.g., 33007KT) is critical for determining the appropriate runway for landing (aligned with the wind as much as possible).
        Wind speed, including gusts (e.g., 32012G20KT), affects the aircraft's handling and the approach stability. Strong crosswinds or tailwinds can make landing more challenging.

2. Visibility

    Prevailing Visibility (e.g., P6SM):
        Indicates how far the pilot can see forward. Limited visibility due to fog, mist, or heavy precipitation can affect the approach and landing.
        Visibility thresholds are critical for determining whether an approach can be conducted under instrument flight rules (IFR) or visual flight rules (VFR).

3. Ceiling (Cloud Base)

    Cloud Cover and Heights (e.g., BKN030, SCT015):
        Describes the altitude of cloud layers. A "ceiling" is the lowest layer of broken (BKN) or overcast (OVC) clouds, which affects visibility on approach.
        Low ceilings may require the use of instrument landing systems (ILS) or alternate airports if below the airport’s minimum landing requirements.

4. Weather Phenomena

    Significant Weather Conditions (e.g., -SHRA, VCTS):
        Precipitation (e.g., rain, snow, or hail) can affect braking action and visibility.
        Thunderstorms (TS or VCTS for "vicinity of airport") may cause turbulence or downdrafts on approach.
        Low-level wind shear or other adverse phenomena must be carefully monitored.

5. Runway Condition Impacts

    Weather affecting the runway surface:
        Rain, snow, or ice may reduce braking efficiency and make the runway slippery.
        Observations like TEMPO or BECMG that indicate changes near landing time are particularly critical.

6. Altimeter Setting (QNH)

    QNH (e.g., QNH2961INS):
        This indicates the barometric pressure setting pilots must use to ensure the altimeter reads correctly for the airport's elevation.
        Incorrect altimeter settings can lead to altitude misjudgments, which is critical during landing.

'''


'''
Question: What parts of TAF can cause a plane to delay arrival at airport?

Several parts of a Terminal Aerodrome Forecast (TAF) can indicate conditions that might delay a plane’s arrival at an airport. These conditions typically relate to safety concerns, airport operations, or air traffic management. Below are the critical TAF elements that could lead to arrival delays:
1. Low Visibility

    What to Look For:
        Visibility less than the airport’s operational minimums (e.g., below 1 statute mile for Category I ILS approaches).
        Examples: 1SM, 2000, or 6000 (visibility in meters in some regions).

    Impact:
        Pilots may not be able to see the runway in time for a safe landing.
        Airports might reduce the number of aircraft allowed to approach simultaneously (flow control).

2. Low Ceilings (Cloud Base)

    What to Look For:
        Broken or overcast cloud layers (e.g., BKN or OVC) with bases below the approach minimums.

    Impact:
        If clouds are too low, pilots may not see the runway in time, even with instrument approaches.
        Aircraft might hold until conditions improve or divert to another airport.

3. Strong Winds or Gusts

    What to Look For:
        High sustained winds or gusts (e.g., 26020G35KT).
        Crosswinds near or above aircraft or runway limits.

    Impact:
        Difficult or unsafe landing conditions, particularly with strong crosswinds or tailwinds.
        May require delays, holding patterns, or rerouting to an alternate airport.

4. Thunderstorms (TS)

    What to Look For:
        Convective activity near the airport, indicated by terms like TS, VCTS (thunderstorms in the vicinity), or CB (cumulonimbus clouds).

    Impact:
        Thunderstorms bring turbulence, wind shear, heavy rain, and lightning, all of which pose risks during approach and landing.
        Arrival delays are common during active thunderstorms.

5. Precipitation

    What to Look For:
        Intense or persistent precipitation, such as -RA, RA, SHRA, SN, or FZRA (freezing rain).

    Impact:
        Heavy rain can reduce visibility and runway friction.
        Snow or ice may delay landings due to slower runway operations or the need for deicing.

6. Wind Shear

    What to Look For:
        Indicated by WS in TAFs, wind shear refers to sudden changes in wind speed or direction.

    Impact:
        Wind shear during approach can destabilize an aircraft, leading to aborted landings.
        Airports may impose delays for safety.

7. Low-Level Turbulence

    What to Look For:
        Turbulence near the surface caused by strong winds, temperature gradients, or terrain effects, often implied in TS, SH, or gusty wind conditions.

    Impact:
        Can make approach and landing uncomfortable or unsafe.
        Pilots may delay arrival until conditions stabilize.

8. Alternate Airport Requirements

    What to Look For:
        Long periods of adverse conditions (e.g., a TEMPO or BECMG group showing extended low visibility or strong winds).

    Impact:
        Pilots may have to plan for diversions, which can lead to holding patterns or delays while waiting for conditions to improve.
'''


'''

Question: is temperature important (in TAF)

Yes, temperature is an important factor in aviation operations, including landings, but it is typically less critical than visibility, wind, and precipitation in causing immediate delays. Here's how temperature can affect operations and why it matters:
1. Impact on Aircraft Performance

    Cold Temperatures:
        In extremely cold weather, air becomes denser, which can improve aircraft engine performance and lift. However:
            Ground operations may be slowed due to the need for deicing aircraft and clearing snow/ice from runways.
            Hydraulic and fuel systems can be affected in extreme cold.
    Hot Temperatures:
        High temperatures reduce air density, leading to:
            Reduced engine performance: Engines generate less thrust.
            Reduced lift: Longer runways may be needed for takeoff or landing.
            Weight restrictions: Aircraft may need to carry less fuel, cargo, or passengers to compensate for reduced performance.

2. Frost and Freezing Conditions

    What to Look For in TAFs:
        Indicators of frost or freezing conditions, such as SN (snow), FZRA (freezing rain), or temperatures below 0°C.
    Impact:
        Frost on wings and control surfaces affects lift and handling.
        Freezing rain can lead to rapid icing, grounding aircraft for safety until conditions improve.

3. Runway Conditions

    Hot Temperatures:
        Heat can soften asphalt, especially in poorly maintained airports, potentially causing longer braking distances.

    Cold Temperatures:
        Ice and snow accumulation reduce braking efficiency and increase the risk of skidding on landing.

4. Passenger Comfort and Safety

    Extreme Heat:
        May require additional cooling systems during ground operations.
    Extreme Cold:
        Can create discomfort and safety concerns for passengers during extended delays or emergencies.

5. TAF Indicators for Temperature

    TAFs often include maximum (TX) and minimum (TN) temperatures during the forecast period.
        Example: TX24/2618Z TN14/2610Z:
            TX24/2618Z: Maximum temperature of 24°C expected at 1800 UTC on the 26th.
            TN14/2610Z: Minimum temperature of 14°C expected at 1000 UTC on the 26th.

These values are useful for:

    Anticipating density altitude issues.
    Monitoring potential frost, icing, or heat-related concerns.

Conclusion

While temperature itself rarely causes delays during landing, it can indirectly affect:

    Aircraft performance (especially at high or low extremes).
    Runway conditions (e.g., ice or softening asphalt).
    Operational adjustments like weight restrictions or deicing.

Pilots and air traffic controllers closely monitor temperature forecasts, but other factors (visibility, wind, precipitation) are usually the primary drivers of delays.


'''



#Encoding data notes:
#Times are in UTC
#Cloud cover amount: SKy Clear 0/8, FEW >0/8-2/8,vSCaTtered 3/8-4/8, BroKeN 5/8-7/8, OVerCast 8/8
#Cloud type: <cloud cover amount><three-digit height in hundreds of feet>
#Cloud type may optionally be (in event of Vertical Visibility at height 'number'): VV<number>
#max_temperature: TX<temperature in celsius>/<two-digit date><two-digit hour>
#min_temperature: TN<temperature in celsius>/<two-digit date><two-digit hour>
#Humidity: 
#Visibility: in form <number>SM, with SM being Statute Miles; if number above 6, reports either P6SM or 9999 (maybe take logarithm of visibility to combat heavy tailed distribution of observations?)


#Wind Shear: WS<three-digit height>/<3-digit direction><2-3 digit speed above indicated height><units, typically 'KT' and convert if not
#change_type:
#-->FM<2-digit date><2-digit hour><2-digit minute>
#-->TEMPO<2-digit date start><2-digit hour start>/<2-digit date end><2-digit hour end>
#-->PROB<2-digit percent> <2-digit date start><2-digit hour start>/<2-digit date end><2-digit hour end>
#-->BECMG<2-digit date start><2-digit hour start>/<2-digit date end><2-digit hour end>

#-----> 2-digit date refers to day in month; if value is *less* than date of report that symbolizes next month as well (+1 to month when recording date)



import os
import re
from datetime import datetime, timedelta
from calendar import monthrange
import pandas as pd
from fastparquet import write

# Data type definitions
dtype_dict = {
    "station_id": "category",
    "visibility": "float32",
    "altimeter_tenths_hpa": "Int32",  # Store altimeter in tenths of hPa as integer
    "temperature": "float32",
    "dew_point": "float32",
    "wind_direction": "float32",
    "wind_speed": "float32",
    "wind_gust": "float32",
    "cover_type_1": "category",
    "altitude_1": "float32",
    "cloud_type_1": "category",
    "cover_type_2": "category",
    "altitude_2": "float32",
    "cloud_type_2": "category",
    "cover_type_3": "category",
    "altitude_3": "float32",
    "cloud_type_3": "category"
}

def parse_taf_entry(entry, base_date):
    if entry == None:
        return []
        
    # Station ID
    station_id_search = re.search(r"\b[A-Z]{4}\b", entry)
    station_id = station_id_search.group() if station_id_search else None
    
    
    # Use datetime of TAF report issuance to handle validity period and time changes
    datetime_parsed = base_date
    _, days_in_month = monthrange(base_date.year, base_date.month)
    
    # Validity Period
    validity_start = None
    validity_end = None
    validity_match = re.search(r"\b(\d{4}/\d{4})\b", entry)
    if validity_match:
        try:
            validity_period = validity_match.group(1)
            start_day = int(validity_period[:2])
            start_hour = int(validity_period[2:4]) % 24
            end_day = int(validity_period[5:7])
            end_hour = int(validity_period[7:9]) % 24
           

            validity_start = base_date.replace(day=start_day, hour=start_hour, minute=0)
            # Get the number of days in the current month
            _, days_in_month = monthrange(base_date.year, base_date.month)

            # If the end day exceeds the days in the current month, move to the next month
            if end_day > days_in_month:
                # Increment the month and adjust the day if necessary
                if base_date.month == 12:
                    validity_end = base_date.replace(year=base_date.year + 1, month=1, day=end_day, hour=end_hour, minute=0)
                else:
                    validity_end = base_date.replace(month=base_date.month + 1, day=end_day, hour=end_hour, minute=0)
            else:
                # Otherwise, just set the tempo_end in the same month
                validity_end = base_date.replace(day=end_day, hour=end_hour, minute=0)
        except:
            pass


    def parse_line (station_id, base_date, days_in_month, validity_start, validity_end, entry):
        #for every line in the block, return a dict


        

        # Time Changes (FM)
        # if from_time field is present, this represents change in weather at the time in from_time
        # for populating estimation data, expect the most recent FM entry to have most updated conditions
        from_time = None
        fm_change = re.search(r"\bFM(\d{6})\b", entry)
        if fm_change:
            try: 
                # Extract the time part
                time = fm_change.group(1)  # The time matched by (\d{6})
                
                # Convert time into day, hour, and minute
                day = int(time[:2])  # The first two digits represent the day
                hour = int(time[2:4]) % 24 # The next two digits represent the hour
                minute = int(time[4:6])  # The last two digits represent the minute
                from_time = base_date.replace(day=day, hour=hour, minute=minute)
            except:
                pass

        
        # note: the first line is typically NOT TEMPO and NOT BECMG and NOT FM
        
     
        
        # thus, we always query by most recent time in mix of tempo_start, becmg_start, change_time and validity_start
        # where time of estimated arrival is not after the end of the time period for tempo, becmg, etc. where it applies
        
        
        
        # Tempo time
        # gives us a time range at which we should expect weather conditions
        # to appear *sporadically* or less than half of the time interval
        # for populating estimation data, consider the most recent start time where end time is still after time of predicted arrival
        tempo_start = None
        tempo_end = None
        tempo_changes = re.search(r"TEMPO\s(\d{4}/\d{4}) (.*?)", entry, re.DOTALL)
        if tempo_changes:
            try:
                period = tempo_changes.group(1)
                details = tempo_changes.group(2)

                # Convert TEMPO start and end times to datetime
                start_day, start_hour = int(period[:2]), int(period[2:4]) % 24
                end_day, end_hour = int(period[5:7]), int(period[7:9]) % 24
                tempo_start = base_date.replace(day=start_day, hour=start_hour, minute=0)
                # Get the number of days in the current month
                _, days_in_month = monthrange(base_date.year, base_date.month)

                # If the end day exceeds the days in the current month, move to the next month
                if end_day > days_in_month:
                    # Increment the month and adjust the day if necessary
                    if base_date.month == 12:
                        tempo_end = base_date.replace(year=base_date.year + 1, month=1, day=end_day, hour=end_hour, minute=0)
                    else:
                        tempo_end = base_date.replace(month=base_date.month + 1, day=end_day, hour=end_hour, minute=0)
                else:
                    # Otherwise, just set the tempo_end in the same month
                    tempo_end = base_date.replace(day=end_day, hour=end_hour, minute=0)
            except:
                pass
            
        # Becmg time
        # gives us a time range at which we should expect weather conditions
        # to change steadily over time
        # for populating estimation data, consider the most recent start time where end time is still after time of predicted arrival
        becmg_start = None
        becmg_end = None
        becmg_changes = re.search(r"BECMG\s(\d{4}/\d{4}) (.*?)", entry, re.DOTALL)
        if becmg_changes:
            try:
                period = becmg_changes.group(1)
                details = becmg_changes.group(2)

                # Convert BECMG start and end times to datetime
                start_day, start_hour = int(period[:2]), int(period[2:4]) % 24
                end_day, end_hour = int(period[5:7]), int(period[7:9]) % 24
                becmg_start = base_date.replace(day=start_day, hour=start_hour, minute=0)
                # Get the number of days in the current month
                

                # If the end day exceeds the days in the current month, move to the next month
                if end_day > days_in_month:
                    # Increment the month and adjust the day if necessary
                    if base_date.month == 12:
                        becmg_end = base_date.replace(year=base_date.year + 1, month=1, day=end_day, hour=end_hour, minute=0)
                    else:
                        becmg_end = base_date.replace(month=base_date.month + 1, day=end_day, hour=end_hour, minute=0)
                else:
                    # Otherwise, just set the tempo_end in the same month
                    becmg_end = base_date.replace(day=end_day, hour=end_hour, minute=0)
            except:
                pass
  
        
        
        # Visibility
        visibility_match = re.search(r"(9999|\d{1,2}SM)", entry)
        if visibility_match:
            if visibility_match.group(1) == "9999": #assumption: set equal to P6SM since 9999 should only appear in airports outside the US
                visibility = float(6)
            else:
                visibility = float(visibility_match.group(1).replace("SM", ""))
        else:
            visibility = None

        # Cloud Layers (individual columns)
        cloud_layer_pattern = r"(CLR|FEW\d{3}[A-Z]*|SCT\d{3}[A-Z]*|BKN\d{3}[A-Z]*|OVC\d{3}[A-Z]*)"
        cloud_layers = re.findall(cloud_layer_pattern, entry)
        cloud_data = {}
        for i, layer in enumerate(cloud_layers[:3], start=1):
            cover_type = layer[:3]
            altitude = int(layer[3:6]) * 100 if len(layer) > 3 else None
            cloud_type = layer[6:] if len(layer) > 6 else None
            cloud_data[f"cover_type_{i}"] = cover_type
            cloud_data[f"altitude_{i}"] = altitude
            cloud_data[f"cloud_type_{i}"] = cloud_type

        # Altimeter in tenths of hPa as integer
        '''
        METAR version (doesn't work for TAF data)
        
        altimeter_match = re.search(r"([AQ])(\d{4})", entry)
        altimeter_tenths_hpa = None
        if altimeter_match:
            prefix, value = altimeter_match.groups()
            if prefix == "A":  # inches of mercury to hPa (multiplied by 10)
                altimeter_tenths_hpa = int(round(float(value) * 33.8639))
            elif prefix == "Q":  # already in hPa
                altimeter_tenths_hpa = int(value) * 10
        '''
        altimeter_tenths_hpa = None
        inches_mercury_match = re.search(r"(QNH(\d{4})INS|A(\d{4}))", entry)
        if inches_mercury_match:
            try:
                value = inches_mercury_match.groups()[1]
                altimeter_tenths_hpa = int(round(float(value)/10 * 33.8639))
            except:
                pass
        
        inches_hpa_match = re.search(r"QNH(\d{4})HPA", entry)
        if inches_hpa_match:
            try:
                value = inches_hpa_match.groups()[1]
                altimeter_tenths_hpa = int(value) * 10
            except:
                pass
           
        
        # Temperature and Dew Point
        temp_dew_match = re.search(r"(\d{2}|M\d{2})/(\d{2}|M\d{2})", entry)
        temperature = float(temp_dew_match.group(1).replace("M", "-")) if temp_dew_match else None
        dew_point = float(temp_dew_match.group(2).replace("M", "-")) if temp_dew_match else None

        # Wind Information
        wind_match = re.search(r"(\d{3})(\d{2})(?:G(\d{2}))?KT", entry)
        wind_direction = float(wind_match.group(1)) if wind_match else None
        wind_speed = float(wind_match.group(2)) if wind_match else None
        wind_gust = float(wind_match.group(3)) if wind_match and wind_match.group(3) else None

        return {
            "station_id": station_id,
            "datetime": datetime_parsed,
            "validity_start": validity_start,
            "validity_end": validity_end,
            "from_time": from_time,
            "tempo_start": tempo_start,
            "tempo_end": tempo_end,
            "becmg_start": becmg_start,
            "becmg_end": becmg_end,            
            "visibility": visibility,
            "altimeter_tenths_hpa": altimeter_tenths_hpa,
            "temperature": temperature,
            "dew_point": dew_point,
            "wind_direction": wind_direction,
            "wind_speed": wind_speed,
            "wind_gust": wind_gust,
            **cloud_data  # Unpack cloud layers
        }
    returns = []
    for line in entry.split('\n'):
        returns.append(parse_line(station_id, base_date, days_in_month, validity_start, validity_end, line))
        
    return returns
    
        
    
'''
def process_metar_file(file_path, output_directory):
    # Open and read the METAR file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            entries = file.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            entries = file.readlines()

    # Parse entries and save as DataFrame
    parsed_data = [parse_metar_entry(entry) for entry in entries if entry.strip()]
    df = pd.DataFrame(parsed_data).astype(dtype_dict)

    # Save DataFrame as Parquet
    output_file = os.path.join(output_directory, f"{os.path.basename(file_path)}.parquet")
    write(output_file, df, compression='brotli', write_index=False)
    print(f"Processed and saved {file_path} as {output_file}")
'''

folder = 'TAF_train/TAF_train'

class filenameParserTAF:
    def __init__(self, filename):
        self.filename = filename
        temp = filename.split('.')
        self.taf = temp[0]
        self.day = temp[1]
        self.hour = temp[2]
        self.extension = temp[3]
        
        date_str = self.day + self.hour
        self.time = datetime.strptime(date_str, '%Y%m%d%HZ')
        
txt_filenames = [f for f in os.listdir(folder) if f.endswith('.txt')]

TAF_filename_objects = [filenameParserTAF(f) for f in txt_filenames]

def process_single_dataset(filename, datetime):
    #skip already processed files!!!
    
    if any(filename in name for name in os.listdir(".")):    
        return None

    with open(os.path.join(folder, filename), 'r') as file:
        content = file.read()
        
    df = pd.DataFrame(columns=['station_id', 'forecast_timestamp', 'Column3'])
        
    # Use a regular expression to split the text into blocks based on date pattern
    blocks = re.split(r'(?=\d{4}/\d{2}/\d{2} \d{2}:\d{2})', content.strip())
    
    # Remove any empty blocks and strip leading/trailing whitespace from each block
    blocks = [block.strip() for block in blocks if block.strip()]
    #temp = [b for b in blocks if len(b.splitlines()) < 2]
    #temp = [blocks[t:t+2] for t in range(len(blocks)) if len(blocks[t].splitlines()) < 2]
    #ids = [b.splitlines()[1].split()[0] for b in blocks]
    

    def clean_block(block):

        #remove leading and trailing whitespace
        lines = block.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        # Remove 'TAF' from the text
        processed_lines = [line.replace('TAF', '') for line in cleaned_lines]
        stripped_blocks =  '\n'.join(processed_lines)        #remove empty lines
        return re.sub(r'\n{2,}', '\n', stripped_blocks)
    
    processed_blocks = [clean_block(block) for block in blocks]
    
    #filter out ammendment and correction; be able to filter out all fields as necessary, however might be inherently less fields with values
    #note that it appears there is a consistent typo of 'ammendment' with two m's
    
    original_blocks = [block for block in processed_blocks if re.search(rf'\b{"Ammendment"}\b', block, re.IGNORECASE) is None and re.search(rf'\b{"Correction"}\b', block, re.IGNORECASE) is None]
    
    ammend_blocks = [block for block in processed_blocks if re.search(rf'\b{"Ammendment"}\b', block, re.IGNORECASE) is not None]
    
    correct_blocks = [block for block in processed_blocks if re.search(rf'\b{"Correction"}\b', block, re.IGNORECASE) is not None]
    '''
    #handle original data
    # Parse entries and save as DataFrame
    parsed_data = [parse_metar_entry(entry) for entry in entries if entry.strip()]
    df = pd.DataFrame(parsed_data).astype(dtype_dict)

    # Save DataFrame as Parquet
    output_file = os.path.join(output_directory, f"{os.path.basename(file_path)}.parquet")
    write(output_file, df, compression='brotli', write_index=False)
    print(f"Processed and saved {file_path} as {output_file}")
    '''
    
    
    
    def process_block (block, block_type):
        #block_type is either None, 'AMMEND' or 'CORRECT'

        if block_type == 'AMMEND':
            #remove all 'Ammendment' and 'AMD'
            #then remove all leading and trailing whitespaces
            #remove leading and trailing whitespace
            lines = block.split('\n')
            processed_lines = [line.replace('AMD', '') for line in lines]
            processed_lines = [line.replace('Ammendment', '') for line in processed_lines]
            cleaned_lines = [line.strip() for line in processed_lines]
            block =  '\n'.join(processed_lines)
            
            
        if block_type == 'CORRECT':
            #remove all 'Correction' and 'COR'
            #then remove all leading and trailing whitespaces
            lines = block.split('\n')
            processed_lines = [line.replace('Correction', '') for line in lines]
            processed_lines = [line.replace('COR', '') for line in processed_lines]
            cleaned_lines = [line.strip() for line in processed_lines]
            block =  '\n'.join(processed_lines)
        
        return block
    
    ammend_blocks = [process_block(block, 'AMMEND') for block in ammend_blocks]
    correct_blocks = [process_block(block, 'CORRECT') for block in correct_blocks]
    original_blocks = [process_block(block, None) for block in original_blocks]
    
    
    
    ammend_df = pd.DataFrame(sum([parse_taf_entry(block, datetime) for block in ammend_blocks], []))
    correct_df = pd.DataFrame(sum([parse_taf_entry(block, datetime) for block in correct_blocks], []))
    original_df = pd.DataFrame(sum([parse_taf_entry(block, datetime) for block in original_blocks], []))

    #ammendment needs to be added to stuff
    #correction needs to replace stuff
    
    #ammend_df.to_csv('TAF_test_ammend.csv', index=False)
    #correct_df.to_csv('TAF_test_correct.csv', index=False)
    #original_df.to_csv('TAF_test_original.csv', index=False)
    
    write('TAF_train_ammend' + filename + '.parquet', ammend_df, compression='brotli', write_index=False)
    write('TAF_train_correct' + filename + '.parquet', correct_df, compression='brotli', write_index=False)
    write('TAF_train_original' + filename + '.parquet', original_df, compression='brotli', write_index=False)

#process_single_dataset(TAF_filename_objects[3].filename)


for i, o in enumerate(TAF_filename_objects):
    process_single_dataset(o.filename, o.time)
    print (o.filename + ' processed. Progress is ' + str(i+1) + '/' + str(len(TAF_filename_objects)))