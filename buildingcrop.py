#!/usr/bin/env python
import geopandas as gpd
import sys

def main():
    # Define input and output file paths.
    nyc_buildings_fp = "./input/NYC_Buildings.geojson"
    coneyisland_fp = "./input/coneyisland.geojson"
    output_fp = "./input/buildings.geojson"
    
    try:
        # Read the NYC buildings GeoJSON.
        buildings = gpd.read_file(nyc_buildings_fp)
        print(f"Loaded {len(buildings)} building features from {nyc_buildings_fp}")
    except Exception as e:
        sys.exit(f"Error reading {nyc_buildings_fp}: {e}")

    try:
        # Read the Coney Island polygon(s) GeoJSON.
        coney = gpd.read_file(coneyisland_fp)
        print(f"Loaded {len(coney)} coney island features from {coneyisland_fp}")
    except Exception as e:
        sys.exit(f"Error reading {coneyisland_fp}: {e}")

    # Combine all Coney Island geometries into a single geometry (union).
    coney_union = coney.unary_union

    # Select buildings that are completely within the Coney Island polygon.
    selected_buildings = buildings[buildings.geometry.within(coney_union)]
    print(f"Selected {len(selected_buildings)} building features within Coney Island.")

    try:
        # Save the selected buildings to the output GeoJSON.
        selected_buildings.to_file(output_fp, driver='GeoJSON')
        print(f"Output saved to {output_fp}")
    except Exception as e:
        sys.exit(f"Error writing {output_fp}: {e}")

if __name__ == "__main__":
    main()