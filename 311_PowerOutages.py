import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import warnings
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*not successfully written.*')

# Variables for easy editing
INPUT_FILE = r'C:\Users\olive\One Architecture Dropbox\_NYC PROJECTS\P2415_CSC Year Two\05 GIS\02 Data\00 Sources & Packages\311_Requests\311_Service_Requests_from_2010_to_Present_20240920.csv'
OUTPUT_FOLDER = r'C:\Users\olive\One Architecture Dropbox\Oliver Atwood\P2415_CSC Year Two\05 GIS\02 Data\01 Vector\311_PowerOutages'
POWER_OUTAGE_FILE = '311_power_outage_requests.shp'
YEARS = 1  # Look-back period in years
POWER_OUTAGE_KEYWORDS = ['power outage', 'blackout', 'electric outage', 'power loss', 'con ed', 'conedison']
CHUNK_SIZE = 100000  # Adjust based on available memory

# Column names
DATE_COLUMN = 'Created Date'
DESCRIPTOR_COLUMN = 'Descriptor'
LONGITUDE_COLUMN = 'Longitude'
LATITUDE_COLUMN = 'Latitude'

def parse_date(date_string):
    date_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%d-%b-%Y %H:%M:%S',
        '%m/%d/%Y %I:%M:%S %p',
    ]
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_string, format=fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(date_string)
    except ValueError:
        print(f"Failed to parse date: {date_string}")
        return pd.NaT

def process_chunk(chunk, threshold_date):
    # Parse the date column
    chunk[DATE_COLUMN] = chunk[DATE_COLUMN].apply(parse_date)
    chunk = chunk.dropna(subset=[DATE_COLUMN])
    
    # Filter rows to only include requests within the last 'YEARS'
    recent_chunk = chunk[chunk[DATE_COLUMN] >= threshold_date]
    
    # Filter rows based on power outage-related keywords in the descriptor column
    pattern = '|'.join(POWER_OUTAGE_KEYWORDS)
    power_outage_requests = recent_chunk[recent_chunk[DESCRIPTOR_COLUMN].str.contains(pattern, case=False, na=False)]
    
    if not power_outage_requests.empty:
        geometry = [Point(xy) for xy in zip(power_outage_requests[LONGITUDE_COLUMN], power_outage_requests[LATITUDE_COLUMN])]
        gdf = gpd.GeoDataFrame(power_outage_requests, geometry=geometry, crs="EPSG:4326")
        return gdf
    return None

def prepare_for_shapefile(df):
    for column in df.columns:
        if df[column].dtype == 'int64':
            df[column] = df[column].astype('int32')
        elif df[column].dtype == 'float64':
            df[column] = df[column].astype('float32')
        elif df[column].dtype == 'object':
            df[column] = df[column].astype(str).str[:254]
    return df

# Calculate the threshold date: today's date minus the number of years specified
threshold_date = pd.Timestamp.now() - pd.DateOffset(years=YEARS)
print(f"Filtering 311 requests from {threshold_date.strftime('%Y-%m-%d')} to present...")

print("Processing CSV file in chunks...")
chunks = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)

all_power_outage_gdfs = []
total_rows = 0
power_outage_rows = 0

for chunk in tqdm(chunks, desc="Processing chunks"):
    total_rows += len(chunk)
    try:
        outage_gdf = process_chunk(chunk, threshold_date)
        if outage_gdf is not None:
            all_power_outage_gdfs.append(outage_gdf)
            power_outage_rows += len(outage_gdf)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        print("First few rows of problematic chunk:")
        print(chunk.head())
        continue

print("Combining processed data...")
if all_power_outage_gdfs:
    combined_outage_gdf = pd.concat(all_power_outage_gdfs, ignore_index=True)
    print("Preparing data for shapefile export...")
    combined_outage_gdf = prepare_for_shapefile(combined_outage_gdf)

    print("Creating output folder...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Saving power outage-related requests as shapefile...")
    outage_shapefile = os.path.join(OUTPUT_FOLDER, POWER_OUTAGE_FILE)
    combined_outage_gdf.to_file(outage_shapefile)

    print(f"Shapefile has been created in the '{OUTPUT_FOLDER}' folder.")
    print(f"Number of rows processed: {total_rows}")
    print(f"Number of power outage-related calls: {power_outage_rows}")
else:
    print("No power outage-related requests found in the provided data.")