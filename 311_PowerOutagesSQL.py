import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import warnings
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sqlite3

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*not successfully written.*')

# Variables for easy editing
INPUT_FILE = r'C:\Users\olive\One Architecture Dropbox\_NYC PROJECTS\P2415_CSC Year Two\05 GIS\02 Data\00 Sources & Packages\311_Requests\311_Service_Requests_from_2010_to_Present_20240920.csv'
OUTPUT_FOLDER = r'C:\Users\olive\One Architecture Dropbox\Oliver Atwood\P2415_CSC Year Two\05 GIS\02 Data\01 Vector\311_PowerOutages'
POWER_OUTAGE_FILE = '311_power_outage_requests.shp'
YEARS = 1  # Look-back period in years
POWER_OUTAGE_KEYWORDS = ['power outage', 'blackout', 'electric outage', 'power loss', 'con ed', 'conedison']
CHUNK_SIZE = 100000  # Adjust based on available memory

# Original CSV column names
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

def count_csv_lines(filepath):
    # Count lines in the CSV (excluding header) to estimate progress.
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        total = sum(1 for line in f) - 1
    return total

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define the path to the SQLite database (here we save it to disk)
db_path = os.path.join(OUTPUT_FOLDER, "311_requests.db")
db_exists = os.path.exists(db_path)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

if not db_exists:
    # Create the table with consistent lowercase column names for spatial data.
    create_table_query = """
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_date TEXT,
        descriptor TEXT,
        longitude REAL,
        latitude REAL
    )
    """
    cursor.execute(create_table_query)
    conn.commit()

    # Calculate the threshold date (only requests within the last 'YEARS' are needed)
    threshold_date = pd.Timestamp.now() - pd.DateOffset(years=YEARS)
    threshold_date_str = threshold_date.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Filtering 311 requests from {threshold_date_str} to present...")

    # Count the number of rows in the CSV to set the progress bar total
    total_rows = count_csv_lines(INPUT_FILE)
    total_chunks = (total_rows // CHUNK_SIZE) + 1
    print(f"Estimated total rows: {total_rows} in ~{total_chunks} chunks.")

    # Insert CSV data into SQLite in chunks
    print("Inserting CSV data into SQLite database...")
    chunks = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)
    rows_processed = 0

    for chunk in tqdm(chunks, desc="Processing chunks", total=total_chunks):
        rows_processed += len(chunk)
        # Parse the date column and drop rows that cannot be parsed
        chunk[DATE_COLUMN] = chunk[DATE_COLUMN].apply(parse_date)
        chunk = chunk.dropna(subset=[DATE_COLUMN])
        
        # Create a normalized ISO date string (which is lexically comparable)
        chunk['created_date'] = chunk[DATE_COLUMN].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Select only the columns needed for filtering and rename to match table schema
        chunk_to_insert = chunk[['created_date', DESCRIPTOR_COLUMN, LONGITUDE_COLUMN, LATITUDE_COLUMN]].copy()
        chunk_to_insert.columns = ['created_date', 'descriptor', 'longitude', 'latitude']
        
        # Insert the chunk into the SQLite table; using 'append' to build the table in pieces
        chunk_to_insert.to_sql('requests', conn, if_exists='append', index=False)

    conn.commit()
    print(f"Finished inserting {rows_processed} rows.")
else:
    print("Database already exists. Skipping CSV processing.")
    # If the DB exists, set the threshold date for subsequent query
    threshold_date = pd.Timestamp.now() - pd.DateOffset(years=YEARS)
    threshold_date_str = threshold_date.strftime("%Y-%m-%d %H:%M:%S")

# Build the SQL query that filters by created_date and power outage-related keywords.
# The descriptor filtering is done case-insensitively using lower() and the LIKE operator.
keyword_conditions = " OR ".join([f"lower(descriptor) LIKE '%{kw}%'" for kw in POWER_OUTAGE_KEYWORDS])
query = f"""
SELECT * FROM requests
WHERE created_date >= '{threshold_date_str}'
AND ({keyword_conditions})
"""

print("Running SQL query to filter power outage-related requests...")
filtered_df = pd.read_sql_query(query, conn)

print(f"Number of power outage-related calls found: {len(filtered_df)}")

# If filtered data exists, convert to a GeoDataFrame and export as a shapefile.
if not filtered_df.empty:
    # Use the lowercase column names defined in the SQLite schema
    geometry = [Point(xy) for xy in zip(filtered_df['longitude'], filtered_df['latitude'])]
    gdf = gpd.GeoDataFrame(filtered_df, geometry=geometry, crs="EPSG:4326")
    
    def prepare_for_shapefile(df):
        for column in df.columns:
            if df[column].dtype == 'int64':
                df[column] = df[column].astype('int32')
            elif df[column].dtype == 'float64':
                df[column] = df[column].astype('float32')
            elif df[column].dtype == 'object':
                df[column] = df[column].astype(str).str[:254]
        return df

    print("Preparing data for shapefile export...")
    gdf = prepare_for_shapefile(gdf)

    shapefile_path = os.path.join(OUTPUT_FOLDER, POWER_OUTAGE_FILE)
    gdf.to_file(shapefile_path)
    print(f"Shapefile has been created in the '{OUTPUT_FOLDER}' folder at:")
    print(shapefile_path)
else:
    print("No power outage-related requests found in the provided data.")

conn.close()
if not db_exists:
    print(f"Total rows processed from CSV: {rows_processed}")
else:
    print("Using pre-existing database.")