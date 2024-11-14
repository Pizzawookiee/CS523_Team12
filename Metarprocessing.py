import os
import re
import pandas as pd
import datetime
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

def parse_metar_entry(entry):
    # Station ID
    station_id_match = re.match(r"([A-Z0-9]{4})", entry)
    station_id = station_id_match.group(1) if station_id_match else None

    # DateTime with additional handling for extra characters
    datetime_match = re.search(r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2})", entry)
    datetime_parsed = None
    if datetime_match:
        date_str = datetime_match.group(1)
        try:
            datetime_parsed = datetime.datetime.strptime(date_str, "%Y/%m/%d %H:%M")
        except ValueError as e:
            print(f"Skipping datetime parsing due to error: {e} in entry: {entry}")

    # Visibility
    visibility_match = re.search(r"(\d{1,2})SM", entry)
    visibility = float(visibility_match.group(1)) if visibility_match else None

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
    altimeter_match = re.search(r"([AQ])(\d{4})", entry)
    altimeter_tenths_hpa = None
    if altimeter_match:
        prefix, value = altimeter_match.groups()
        if prefix == "A":  # inches of mercury to hPa (multiplied by 10)
            altimeter_tenths_hpa = int(round(float(value) * 33.8639))
        elif prefix == "Q":  # already in hPa
            altimeter_tenths_hpa = int(value) * 10

    # Temperature and Dew Point
    temp_dew_match = re.search(r"(\d{2}|M\d{2})/(\d{2}|M\d{2})", entry)
    temperature = float(temp_dew_match.group(1).replace("M", "-")) if temp_dew_match else None
    dew_point = float(temp_dew_match.group(2).replace("M", "-")) if temp_dew_match else None

    # Wind Information
    wind_match = re.search(r"(\d{3})(\d{2})KT(?:G(\d{2}))?", entry)
    wind_direction = float(wind_match.group(1)) if wind_match else None
    wind_speed = float(wind_match.group(2)) if wind_match else None
    wind_gust = float(wind_match.group(3)) if wind_match and wind_match.group(3) else None

    return {
        "station_id": station_id,
        "datetime": datetime_parsed,
        "visibility": visibility,
        "altimeter_tenths_hpa": altimeter_tenths_hpa,
        "temperature": temperature,
        "dew_point": dew_point,
        "wind_direction": wind_direction,
        "wind_speed": wind_speed,
        "wind_gust": wind_gust,
        **cloud_data  # Unpack cloud layers
    }

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

def process_all_metar_files(directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                process_metar_file(file_path, output_directory)

def consolidate_parquet_files(output_directory, final_output_path):
    all_dfs = [pd.read_parquet(os.path.join(output_directory, file)) for file in os.listdir(output_directory) if file.endswith('.parquet')]
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_parquet(final_output_path, compression='brotli', index=False)
    print(f"Consolidated all files into {final_output_path}")

# Define paths and process files
metar_directory = r'C:\Users\damol\Downloads\Competition\METAR_train_data'
output_directory = r'C:\Users\damol\Downloads\Competition\METAR_parquet_chunks'
final_output_path = r'C:\Users\damol\Downloads\Competition\final_aligned_metar_data.parquet'

# Process each file and consolidate
process_all_metar_files(metar_directory, output_directory)
consolidate_parquet_files(output_directory, final_output_path)