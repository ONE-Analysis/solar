import geopandas as gpd
import rasterio
import os
from pathlib import Path
import warnings
import multiprocessing as mp
import time
from datetime import timedelta
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Input folder path (relative to script location)
INPUT_FOLDER = Path(__file__).parent / 'input'

# Target CRS
TARGET_CRS = 'EPSG:6539'

# Minimum parking lot area (sq ft)
MIN_PARKING_AREA = 5000

# Function to print dataset information
def print_dataset_info(name, data):
    print(f"\n=== {name} Dataset ===")
    if isinstance(data, gpd.GeoDataFrame):
        print("\nColumns:")
        for col in data.columns:
            print(f"- {col}")
        print("\nCRS Information:")
        print(data.crs)
        print(f"Number of features: {len(data)}")
    elif isinstance(data, rasterio.DatasetReader):
        print("\nRaster Summary:")
        print(f"Width: {data.width}")
        print(f"Height: {data.height}")
        print(f"Bands: {data.count}")
        print(f"Bounds: {data.bounds}")
        print("\nCRS Information:")
        print(data.crs)
    else:
        print("No data loaded or invalid data format.")

# Function to load and reproject GeoJSON if needed
def load_geojson(file_path, columns=None):
    try:
        gdf = gpd.read_file(
            file_path,
            columns=columns,
            engine='pyogrio'
        )

        # Check if reprojection is needed
        if gdf.crs is None:
            print(f"Warning: {file_path.name} has no CRS defined. Setting to {TARGET_CRS}")
            gdf.set_crs(TARGET_CRS, inplace=True)
        elif gdf.crs.to_string() != TARGET_CRS:
            print(f"Reprojecting {file_path.name} from {gdf.crs} to {TARGET_CRS}")
            gdf = gdf.to_crs(TARGET_CRS)

        return gdf
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to load and reproject raster if needed
def load_raster(file_path):
    try:
        with rasterio.open(file_path) as src:
            # Check if reprojection is needed
            if src.crs.to_string() != TARGET_CRS:
                print(f"Reprojecting {file_path.name} from {src.crs} to {TARGET_CRS}")

                transform, width, height = calculate_default_transform(
                    src.crs, TARGET_CRS, src.width, src.height, *src.bounds)

                temp_path = file_path.parent / f"temp_{file_path.name}"

                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': TARGET_CRS,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                with rasterio.open(temp_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=TARGET_CRS,
                            resampling=Resampling.bilinear
                        )

                return rasterio.open(temp_path)
            else:
                return rasterio.open(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Dictionary of data files
data_files = {
    'nsi': {'path': INPUT_FOLDER / 'NYC_NSI.geojson', 'columns': None},
    'firehouses': {'path': INPUT_FOLDER / 'FDNY_Firehouse.geojson', 'columns': None},
    'lots': {'path': INPUT_FOLDER / 'MapPLUTO.geojson', 'columns': None},
    'pofw': {'path': INPUT_FOLDER / 'NYC_POFW.geojson', 'columns': None},
    'buildings': {'path': INPUT_FOLDER / 'NYC_Buildings.geojson', 'columns': None},
    'pois': {'path': INPUT_FOLDER / 'NYC_POIS.geojson', 'columns': None},
    'libraries': {'path': INPUT_FOLDER / 'NYC_Libraries.geojson', 'columns': None},
    'schools': {'path': INPUT_FOLDER / 'NYC_Schools.geojson', 'columns': None},
    'busdepots': {'path': INPUT_FOLDER / 'NYC_BusDepots.geojson', 'columns': None},
    'policestations': {'path': INPUT_FOLDER / 'NYC_PoliceStations.geojson', 'columns': None},
    'postoffices': {'path': INPUT_FOLDER / 'NYC_PostOffices.geojson', 'columns': None},
}

datasets = {}
temp_files = []

if not INPUT_FOLDER.exists():
    raise FileNotFoundError(f"Input folder not found at: {INPUT_FOLDER}")

n_cores = max(1, mp.cpu_count() - 1)

try:
    total_start_time = time.time()

    # Load all datasets
    for key, file_info in data_files.items():
        file_path = file_info['path']
        columns = file_info['columns']

        start_time = time.time()
        print(f"\nLoading {file_path.name}...")
        if file_path.suffix == '.geojson':
            datasets[key] = load_geojson(file_path, columns)
        elif file_path.suffix == '.tif':
            datasets[key] = load_raster(file_path)
            if datasets[key] is not None and datasets[key].name != str(file_path):
                temp_files.append(Path(datasets[key].name))
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Loading time for {key} ({file_path.name}): {timedelta(seconds=duration)}")

        # Print immediate summary after loading each dataset
        if datasets[key] is not None:
            if isinstance(datasets[key], gpd.GeoDataFrame):
                print(f"{key} loaded with {len(datasets[key])} features.")
            elif isinstance(datasets[key], rasterio.DatasetReader):
                print(f"{key} loaded as a raster with {datasets[key].count} bands.")
            else:
                print(f"{key} loaded but no recognizable format.")
        else:
            print(f"{key} dataset could not be loaded or is empty.")

    # Print information for all datasets
    print("\n=== DATASET SUMMARIES ===")
    for name, data in datasets.items():
        print_dataset_info(name, data)

    # Prepare POIs
    print("\nPrepare places of interest...")
    pois = datasets['pois']
    if pois is not None:
        print(f"Number of points in original POIs dataset: {len(pois)}")
        typologies = ['arts_centre', 'college', 'community_centre', 'shelter']
        pois = pois[pois['fclass'].isin(typologies)].copy()
        keep_cols = ['fclass', 'name', 'geometry']
        pois = pois[keep_cols]
        print(f"Number of points in filtered POIs dataset: {len(pois)}")
        datasets['pois'] = pois

    # POFW
    print("\nPrepare places of worship...")
    pofw = datasets['pofw']
    if pofw is not None:
        pofw['fclass'] = 'pofw_' + pofw['fclass'].astype(str)
        pofw['name'] = pofw['name'].fillna('Unknown')
        pofw = pofw[['fclass', 'name', 'geometry']]
        print(f"Number of points in filtered POFW dataset: {len(pofw)}")
        datasets['pofw'] = pofw

    # Firehouses
    print("\nPrepare firehouses...")
    firehouses = datasets['firehouses']
    if firehouses is not None:
        firehouses['fclass'] = "Firehouse"
        firehouses['name'] = firehouses['FacilityName']
        firehouses = firehouses[['fclass', 'name', 'geometry']]
        print(f"Number of firehouses in prepared dataset: {len(firehouses)}")
        datasets['firehouses'] = firehouses

    # Libraries
    print("\nPrepare libraries...")
    libraries = datasets['libraries']
    if libraries is not None:
        libraries['fclass'] = "Library"
        libraries['name'] = libraries['NAME']
        libraries = libraries[['fclass', 'name', 'geometry']]
        print(f"Number of libraries in prepared dataset: {len(libraries)}")
        datasets['libraries'] = libraries

    # Schools
    print("\nPrepare schools...")
    schools = datasets['schools']
    if schools is not None:
        schools['fclass'] = "School"
        schools['name'] = schools['Name']
        schools = schools[['fclass', 'name', 'geometry']]
        print(f"Number of schools in prepared dataset: {len(schools)}")
        datasets['schools'] = schools

    # Busdepots
    print("\nPrepare busdepots...")
    busdepots = datasets['busdepots']
    if busdepots is not None:
        busdepots['fclass'] = "Bus Depot"
        busdepots['name'] = busdepots['FACNAME']
        busdepots = busdepots[['fclass', 'name', 'geometry']]
        print(f"Number of busdepots in prepared dataset: {len(busdepots)}")
        datasets['busdepots'] = busdepots

    # Policestations
    print("\nPrepare policestations...")
    policestations = datasets['policestations']
    if policestations is not None:
        policestations['fclass'] = policestations['FACSUBGRP']
        policestations['name'] = "Police Station"
        policestations = policestations[['fclass', 'name', 'geometry']]
        print(f"Number of policestations in prepared dataset: {len(policestations)}")
        datasets['policestations'] = policestations

    # Postoffices
    print("\nPrepare postoffices...")
    postoffices = datasets['postoffices']
    if postoffices is not None:
        postoffices['fclass'] = "Post Office"
        postoffices = postoffices[['fclass', 'name', 'geometry']]
        print(f"Number of postoffices in prepared dataset: {len(postoffices)}")
        datasets['postoffices'] = postoffices

    # Merge Points Datasets
    print("\nMerge points datasets...")
    points_keys = ['pois', 'pofw', 'firehouses', 'libraries', 'schools', 'busdepots', 'policestations', 'postoffices']
    points_dfs = [datasets.get(k) for k in points_keys if datasets.get(k) is not None]

    required_columns = ['fclass', 'name', 'geometry']
    for df in points_dfs:
        if not all(col in df.columns for col in required_columns):
            raise ValueError("One of the point datasets does not contain the required columns.")

    if len(points_dfs) > 0:
        bldg_pts = pd.concat(points_dfs, ignore_index=True)[required_columns]
        bldg_pts['ObjectID'] = bldg_pts.index + 1
        print("Columns in merged dataset:", bldg_pts.columns.tolist())
        print("\nCount by fclass:")
        print(bldg_pts['fclass'].value_counts())
        print(f"Total building points: {len(bldg_pts)}")
    else:
        bldg_pts = gpd.GeoDataFrame(columns=required_columns + ['ObjectID'], crs=TARGET_CRS)

    bldg_pts = gpd.GeoDataFrame(bldg_pts, geometry='geometry', crs=TARGET_CRS)

    # Extract data from lots
    print("\nExtract data from lots...")
    lots = datasets['lots']
    if lots is not None and len(bldg_pts) > 0:
        lot_fields = ['Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 'ComArea',
                      'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea',
                      'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR']
        bldg_pts = gpd.sjoin(bldg_pts, lots[lot_fields + ['geometry']], how='left', predicate='within')
        if 'index_right' in bldg_pts.columns:
            bldg_pts.drop(columns=['index_right'], inplace=True)

        matched = bldg_pts[~bldg_pts['Address'].isna()].shape[0]
        unmatched = bldg_pts[bldg_pts['Address'].isna()].shape[0]
        print(f"Points successfully extracted data from lots: {matched}")
        print(f"Points did not extract data from lots: {unmatched}")

    # Combine points data with building footprints
    print("\nCombine points data with building footprints...")
    buildings = datasets['buildings']
    if buildings is not None and len(bldg_pts) > 0:
        buildings = buildings[['geometry', 'groundelev', 'heightroof', 'lststatype', 'cnstrct_yr']]
        joined = gpd.sjoin(buildings, bldg_pts, how='inner', predicate='contains')
        print("Columns in joined after sjoin:", joined.columns)

        agg_fields = [
            'fclass', 'name', 'Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 
            'ComArea', 'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea', 
            'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR'
        ]

        def combine_values(series):
            vals = series.dropna().unique()
            return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan

        agg_fields = [f for f in agg_fields if f in joined.columns]
        if agg_fields:
            grouped = joined.groupby(joined.index)
            agg_dict = {f: combine_values for f in agg_fields}
            agg_result = grouped.agg(agg_dict)
            buildings = buildings.loc[agg_result.index]
            for f in agg_fields:
                buildings[f] = agg_result[f]

        datasets['buildings'] = buildings
    else:
        print("No buildings or points to combine or no overlapping features found.")

    # Extract data from NSI
    print("\nExtract data from national structures inventory...")
    nsi = datasets['nsi']
    if nsi is not None and buildings is not None:
        nsi_fields = ['bldgtype', 'num_story', 'found_type', 'found_ht']
        joined_nsi = gpd.sjoin(buildings, nsi[nsi_fields + ['geometry']], how='left', predicate='contains')

        def combine_values(series):
            vals = series.dropna().unique()
            return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan

        agg_dict_nsi = {f: combine_values for f in nsi_fields}
        grouped_nsi = joined_nsi.groupby(joined_nsi.index).agg(agg_dict_nsi)
        for f in nsi_fields:
            buildings[f] = grouped_nsi[f]

        matched_nsi = buildings[~buildings['bldgtype'].isna()].shape[0]
        unmatched_nsi = buildings[buildings['bldgtype'].isna()].shape[0]
        print(f"Buildings successfully extracted data from nsi: {matched_nsi}")
        print(f"Buildings did not extract data from nsi: {unmatched_nsi}")

    # Isolate parking lots
    def clean_and_validate_geometry(gdf):
        """Clean and validate geometries in a GeoDataFrame."""
        # Make valid geometries and buffer by tiny amount to fix topology
        gdf.geometry = gdf.geometry.make_valid().buffer(0.01).buffer(-0.01)

        # Remove empty or invalid geometries
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]

        # Ensure all geometries are polygons or multipolygons
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

        return gdf

    def explode_multipart_features(gdf):
        """Explode multipart features into single part features."""
        # Explode multipart geometries
        exploded = gdf.explode(index_parts=True)

        # Reset index and drop the multiindex parts
        exploded = exploded.reset_index(drop=True)

        return exploded

    def safe_dissolve(gdf, dissolve_field):
        """Safely dissolve geometries while handling topology."""
        try:
            # Group by dissolve field
            groups = gdf.groupby(dissolve_field)
            dissolved_parts = []

            for name, group in groups:
                # Clean group geometries
                group = clean_and_validate_geometry(group)

                if len(group) > 0:
                    # Union geometries within group
                    unified = group.geometry.unary_union

                    # Create new GeoDataFrame with dissolved geometry
                    dissolved_part = gpd.GeoDataFrame(
                        {dissolve_field: [name]},
                        geometry=[unified],
                        crs=group.crs
                    )

                    # Explode any multipart features
                    dissolved_part = explode_multipart_features(dissolved_part)
                    dissolved_parts.append(dissolved_part)

            # Combine all dissolved parts
            if dissolved_parts:
                result = pd.concat(dissolved_parts, ignore_index=True)
                return clean_and_validate_geometry(result)
            return gdf

        except Exception as e:
            print(f"Error during dissolve operation: {e}")
            return gdf

    print("\nIsolate parking lots from PLUTO data...")
    if lots is not None:
        try:
            lots['LandUse'] = lots['LandUse'].astype(str)
            parking = lots[lots['LandUse'] == '10'].copy()

            # Initial geometry cleaning
            parking = clean_and_validate_geometry(parking)

            # Filter by area
            parking = parking[parking['LotArea'] >= MIN_PARKING_AREA].copy()

            parking['fclass'] = "Parking"
            parking['name'] = parking['Address']

            # Dissolve with improved handling
            print("Dissolving parking lots...")
            parking = safe_dissolve(parking, 'OwnerName')

            # Ensure no multipart features
            parking = explode_multipart_features(parking)

            # Final cleanup
            parking = clean_and_validate_geometry(parking)

            print(f"Number of parking lots and structures (>= {MIN_PARKING_AREA} sq ft): {len(parking)}")

            # Verify final geometries
            invalid_count = sum(~parking.geometry.is_valid)
            if invalid_count > 0:
                print(f"Warning: {invalid_count} invalid geometries found after processing")
                parking = parking[parking.geometry.is_valid]

        except Exception as e:
            print(f"Error processing parking lots: {e}")
            parking = gpd.GeoDataFrame(columns=['fclass', 'name', 'geometry'], crs=TARGET_CRS)
    else:
        parking = gpd.GeoDataFrame(columns=['fclass', 'name', 'geometry'], crs=TARGET_CRS)

        
    # Merge parking lots and buildings
    print("\nMerge parking lots and buildings...")
    # Ensure buildings have an fclass column so that it aligns with parking
    if buildings is not None:
        if 'fclass' not in buildings.columns:
            buildings['fclass'] = 'Building'
    else:
        # If no buildings, create an empty GeoDataFrame with required columns
        buildings = gpd.GeoDataFrame(columns=['fclass', 'geometry'], crs=TARGET_CRS)
        buildings['fclass'] = 'Building'

    # Now we unify parking with buildings based on columns in buildings
    bldg_fields = set(buildings.columns)
    parking_fields = set(parking.columns)
    extra_in_parking = parking_fields - bldg_fields

    # Drop columns from parking that are not in buildings
    parking = parking.drop(columns=list(extra_in_parking), errors='ignore')

    # Add missing fields to parking with default 'Unknown'
    missing_in_parking = bldg_fields - set(parking.columns)
    for f in missing_in_parking:
        if f == 'geometry':  # geometry must be present
            continue
        if f == 'fclass':
            # If fclass is missing in parking (it shouldn't be since we set it to 'Parking')
            # just ensure it is set
            parking[f] = 'Parking'
        else:
            parking[f] = 'Unknown'

    # At this point, parking and buildings have the same columns including fclass
    parking = parking[buildings.columns]

    # Merge them together
    sites = pd.concat([buildings, parking], ignore_index=True)
    print(f"Number of sites of interest: {len(sites)}")

    # Export final sites
    sites.to_file("./output/shapefiles/preprocessed_sites_solar.shp", driver="ESRI Shapefile")
    sites.to_file("./output/preprocessed_sites_solar.geojson", driver="GeoJSON")

    total_duration = time.time() - total_start_time
    print(f"\nTotal processing time: {timedelta(seconds=total_duration)}")
    
except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Close raster datasets
    for name, dataset in datasets.items():
        if isinstance(dataset, rasterio.DatasetReader):
            dataset.close()

    # Clean up temporary files
    for temp_file in temp_files:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {e}")