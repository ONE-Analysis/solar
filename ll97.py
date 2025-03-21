import os
import pandas as pd
import geopandas as gpd

# Specify file paths
csv_path = "./Input/LL97.csv"
geojson_path = "./Input/buildings.geojson"
output_dir = "./output/LL97_bldgs"
shapefile_path = os.path.join(output_dir, "LL97_bldgs.shp")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file
csv_df = pd.read_csv(csv_path)

# Build a set of all BINs from the 'Primary BIN' column.
# This will handle both single BIN values and comma-separated lists.
bin_set = set(
    x.strip()  # Remove any extra whitespace
    for entry in csv_df['Preliminary BIN'].dropna()  # Exclude missing values
    for x in str(entry).split(',')  # Split cell values if there are multiple BINs
)

# Load the GeoJSON file containing building polygons
gdf_buildings = gpd.read_file(geojson_path)

# Convert the 'bin' field to string to ensure the comparison works correctly
gdf_buildings['bin'] = gdf_buildings['bin'].astype(str)

# Filter the GeoDataFrame to include only rows where 'bin' is in our BIN set
filtered_buildings = gdf_buildings[gdf_buildings['bin'].isin(bin_set)]

# Write the filtered GeoDataFrame as a shapefile
filtered_buildings.to_file(shapefile_path, driver='ESRI Shapefile')