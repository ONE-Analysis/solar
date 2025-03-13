import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from pyproj import CRS
import time

def geocode_address(address, geolocator, delay=1):
    """
    Geocodes an address using the provided geolocator.
    Returns (latitude, longitude) if found, else (None, None).
    """
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Error geocoding '{address}': {e}")
    return None, None

def main():
    # Path to your CSV file
    csv_path = r"./input/ExistingCommunitySolarProject_NYC.csv"
    
    # Read CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Initialize the geolocator with a custom user agent
    geolocator = Nominatim(user_agent="my_geocoder")
    
    latitudes = []
    longitudes = []
    
    # Loop through each address in the 'Site Address' column and geocode it
    for address in df['Site Address']:
        lat, lon = geocode_address(address, geolocator)
        latitudes.append(lat)
        longitudes.append(lon)
        # Delay to avoid overwhelming the geocoding service
        time.sleep(1)
    
    # Add latitude and longitude columns to the DataFrame
    df['Latitude'] = latitudes
    df['Longitude'] = longitudes
    
    # Create a geometry column with Point objects (if geocoding was successful)
    geometry = [
        Point(lon, lat) if (lon is not None and lat is not None) else None 
        for lat, lon in zip(latitudes, longitudes)
    ]
    
    # Create a GeoDataFrame with the initial coordinate reference system (WGS84)
    crs_wgs84 = CRS.from_epsg(4326)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs_wgs84)
    
    # Reproject the GeoDataFrame to EPSG:6539
    gdf = gdf.to_crs(epsg=6539)
    
    # Export the GeoDataFrame to a GeoJSON file
    output_geojson = "./output/ExistingCommunitySolarProject_NYC.geojson"
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"GeoJSON exported successfully to {output_geojson}")

if __name__ == "__main__":
    main()