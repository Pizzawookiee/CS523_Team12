import os
import re
from datetime import datetime, timedelta
import pandas as pd
from fastparquet import write

# Data type definitions for METAR
dtype_dict = {
    "station_id": "category",
    "datetime": "datetime64[ns]",  # Combined date and report time
    "visibility": "float32",
    "altimeter_tenths_hpa": "Int32",
    "temperature": "float32",
    "dew_point": "float32",
    "wind_direction": "float32",
    "wind_speed": "float32",
    "wind_gust": "float32",
    "cover_type_1": "category",
    "altitude_1": "Int32",
    "cloud_type_1": "category",
    "cover_type_2": "category",
    "altitude_2": "Int32",
    "cloud_type_2": "category",
    "cover_type_3": "category",
    "altitude_3": "Int32",
    "cloud_type_3": "category"
}

# List of arrival airport station identifiers
station_ids_of_interest = [
    "KATL", "KCLT", "KDEN", "KDFW", "KJFK",
    "KMEM", "KMIA", "KORD", "KPHX", "KSEA"
]

def parse_metar_entry(entry, base_datetime):
    """Parse a single METAR entry into a structured dictionary."""
    entry_lines = entry.strip().split('\n')
    if len(entry_lines) < 2:
        return None

    # Extract the date line and the station data line
    datetime_line = entry_lines[0].strip()
    station_data_line = entry_lines[1].strip()

    # Extract Station ID (ICAO 4-character code)
    station_id_match = re.match(r"\b([A-Z]{4})\b", station_data_line)
    station_id = station_id_match.group(1) if station_id_match else None

    # Filter out stations not of interest
    if station_id not in station_ids_of_interest:
        return None

    # Extract Report issuance time (e.g., 010000Z)
    report_time_match = re.search(r"\b(\d{2})(\d{2})(\d{2})Z\b", station_data_line)
    report_datetime = None
    if report_time_match:
        report_day = int(report_time_match.group(1))
        report_hour = int(report_time_match.group(2))
        report_minute = int(report_time_match.group(3))

        # Use the base date to determine the correct report date and time
        try:
            report_datetime = base_datetime.replace(day=report_day, hour=report_hour, minute=report_minute)
        except ValueError:
            # Handle month rollover if day is not valid for the current month
            next_month = base_datetime.month % 12 + 1
            year_adjustment = base_datetime.year + (1 if next_month == 1 else 0)
            report_datetime = base_datetime.replace(year=year_adjustment, month=next_month, day=report_day, hour=report_hour, minute=report_minute)

    # Extract visibility (e.g., 9999 or 10SM)
    visibility_match = re.search(r"\b(\d{4})\b", station_data_line)
    visibility = float(visibility_match.group(1)) if visibility_match else None

    # Extract cloud layers (e.g., SCT019)
    cloud_layer_pattern = r"(CLR|FEW\d{3}[A-Z]*|SCT\d{3}[A-Z]*|BKN\d{3}[A-Z]*|OVC\d{3}[A-Z]*)"
    cloud_layers = re.findall(cloud_layer_pattern, station_data_line)
    cloud_data = {}
    for i, layer in enumerate(cloud_layers[:3], start=1):
        cover_type = layer[:3]
        altitude = int(layer[3:6]) * 100 if len(layer) > 3 else None
        cloud_type = layer[6:] if len(layer) > 6 else None
        cloud_data[f"cover_type_{i}"] = cover_type
        cloud_data[f"altitude_{i}"] = altitude
        cloud_data[f"cloud_type_{i}"] = cloud_type

    # Extract altimeter setting (e.g., Q1011)
    altimeter_match = re.search(r"Q(\d{4})", station_data_line)
    altimeter_tenths_hpa = int(altimeter_match.group(1)) * 10 if altimeter_match else None

    # Extract temperature and dew point (e.g., 31/25)
    temp_dew_match = re.search(r"(\d{2}|M\d{2})/(\d{2}|M\d{2})", station_data_line)
    temperature = float(temp_dew_match.group(1).replace("M", "-")) if temp_dew_match else None
    dew_point = float(temp_dew_match.group(2).replace("M", "-")) if temp_dew_match else None

    # Extract wind information (e.g., 09017KT)
    wind_match = re.search(r"(\d{3})(\d{2})KT(?:G(\d{2}))?", station_data_line)
    wind_direction = float(wind_match.group(1)) if wind_match else None
    wind_speed = float(wind_match.group(2)) if wind_match else None
    wind_gust = float(wind_match.group(3)) if wind_match and wind_match.group(3) else None

    # Create the dictionary to return
    return {
        "station_id": station_id,
        "datetime": report_datetime,
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
    """Process a METAR file into a structured DataFrame."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split content into blocks based on date pattern (each report entry starts with a date)
        blocks = re.split(r"(?=\d{4}/\d{2}/\d{2} \d{2}:\d{2})", content.strip())
        blocks = [block.strip() for block in blocks if block.strip()]  # Remove empty blocks

        # Parse each block using the base date from the first entry
        parsed_data = []
        for block in blocks:
            base_date_match = re.search(r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2})", block)
            if base_date_match:
                base_datetime = datetime.strptime(base_date_match.group(1), "%Y/%m/%d %H:%M")
                entry_data = parse_metar_entry(block, base_datetime)
                if entry_data:
                    parsed_data.append(entry_data)

        # Convert to DataFrame
        if parsed_data:
            df = pd.DataFrame(parsed_data).astype(dtype_dict)

            # Drop duplicate rows
            df.drop_duplicates(inplace=True)

            # Save DataFrame as Parquet
            output_file = os.path.join(output_directory, f"{os.path.basename(file_path)}.parquet")
            write(output_file, df, compression='brotli', write_index=False)
            print(f"Processed and saved {file_path} as {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_all_metar_files(directory, output_directory):
    """Process all METAR files in a directory."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                process_metar_file(file_path, output_directory)

def consolidate_parquet_files(output_directory, final_output_path):
    """Consolidate all Parquet files into one."""
    all_dfs = [pd.read_parquet(os.path.join(output_directory, file)) for file in os.listdir(output_directory) if file.endswith('.parquet')]
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_parquet(final_output_path, compression='brotli', index=False)
    print(f"Consolidated all files into {final_output_path}")

# Define paths and process files
metar_directory = r'C:\Users\damol\OneDrive\Desktop\Verify\METAR_test_data'
output_directory = r'C:\Users\damol\OneDrive\Desktop\Verify\Verified'
final_output_path = r'C:\Users\damol\OneDrive\Desktop\Verify\final_consolidated_metar_data_test.parquet'

# Process files
process_all_metar_files(metar_directory, output_directory)

# Consolidate processed files
consolidate_parquet_files(output_directory, final_output_path)
