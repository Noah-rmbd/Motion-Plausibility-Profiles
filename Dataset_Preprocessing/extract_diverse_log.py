import pandas as pd
import numpy as np
from pathlib import Path
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def convert_csv_to_log(csv_path, log_path):
    df = pd.read_csv(csv_path, encoding='latin1')

    # Extract user ID, date, time, lat, lon
    user_col = 'picture author id'
    date_col = 'date'
    time_col = 'time'
    lat_col = 'lat'
    lon_col = 'long'

    # Remove duplicates
    df = df.drop_duplicates(subset=[user_col, date_col, time_col])

    # Remove invalid dates and times
    df = df[~df[date_col].isna() & (df[date_col] != 'no date')]
    df = df[~df[time_col].isna() & (df[time_col] != 'no time')]

    # Convert date/time to datetime
    df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime'])

    # Sort
    df = df.sort_values(by=[user_col, 'datetime'])

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open('w', encoding='utf-8') as f:
        for user_id, user_df in df.groupby(user_col):
            user_df = user_df.reset_index(drop=True)
            min_date = user_df[date_col].min()
            max_date = user_df[date_col].max()

            f.write("============================\n")
            f.write(f"{user_id}, active from {min_date} to {max_date}\n")
            f.write("============================\n")

            current_date = None
            prev_point = None

            for idx, row in user_df.iterrows():
                obs_id = f"Diverse-{user_id}-{idx}"
                lat = row[lat_col]
                lon = row[lon_col]
                date_str = row[date_col]
                time_str = row[time_col]
                current_dt = row['datetime']

                if date_str != current_date:
                    f.write(f"{obs_id}, {lat}, {lon}, {date_str}, {time_str}, first entry of the day\n")
                    current_date = date_str
                    prev_point = (lat, lon, current_dt)
                else:
                    prev_lat, prev_lon, prev_dt = prev_point
                    elapsed_time_s = (current_dt - prev_dt).total_seconds()
                    distance_m = haversine_distance(prev_lat, prev_lon, lat, lon)

                    if elapsed_time_s > 0:
                        speed_kmh = (distance_m / elapsed_time_s) * 3.6
                    else:
                        speed_kmh = 0.0

                    f.write(f"{obs_id}, {lat}, {lon}, {date_str}, {time_str}, {elapsed_time_s:.1f}s, {distance_m:.1f}m, {speed_kmh:.1f}km/h\n")
                    prev_point = (lat, lon, current_dt)

if __name__ == '__main__':
    csv_input = 'data/raw/Diverse_Datasets/cp_data-plus-time-incl-duplicates-for-visualization.csv'
    log_output = 'data/raw/Diverse_Datasets/diverse_dataset.log'
    convert_csv_to_log(csv_input, log_output)
    print(f"Extraction complete. Log written to {log_output}")
