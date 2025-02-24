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

# Toggle whether to include parking lots in final output
INCLUDE_PARKING = False  # Set to False to exclude parking lots

# Dictionary of data files
data_files = {
    'nsi': {'path': INPUT_FOLDER / 'NYC_NSI.geojson', 'columns': None},
    'lots': {'path': INPUT_FOLDER / 'MapPLUTO.geojson', 'columns': None},
    'pofw': {'path': INPUT_FOLDER / 'NYC_POFW.geojson', 'columns': None},
    'buildings': {'path': INPUT_FOLDER / 'NYC_Buildings.geojson', 'columns': None},
    'facilities': {'path': INPUT_FOLDER / 'NYC_Facilities.geojson', 'columns': None},
}

# Initialize empty dictionaries and lists
datasets = {}
temp_files = []


# ---------- DATASET-LOADING HELPERS ---------- #
def print_dataset_info(name, data):
    """
    Print summary information about a loaded dataset.
    """
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

def load_geojson(file_path, columns=None):
    """
    Load a GeoJSON file into a GeoDataFrame and reproject if needed.
    """
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

def load_raster(file_path):
    """
    Load a raster file. If it's not in the target CRS, reproject on the fly
    and return a handle to the reprojected file.
    """
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


# ---------- GEOMETRY CLEANUP HELPERS ---------- #
def clean_and_validate_geometry(gdf):
    """
    Clean and validate geometries in a GeoDataFrame by making them valid
    and removing invalid or empty rows.
    """
    # Make valid geometries and buffer by tiny amount to fix topology
    gdf.geometry = gdf.geometry.make_valid().buffer(0.01).buffer(-0.01)

    # Remove empty or invalid geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]

    # Ensure all geometries are polygons or multipolygons
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    return gdf

def explode_multipart_features(gdf):
    """
    Explode multipart features into single part features.
    """
    exploded = gdf.explode(index_parts=True)
    exploded = exploded.reset_index(drop=True)
    return exploded

def safe_dissolve(gdf, dissolve_field):
    """
    Safely dissolve geometries while handling topology errors.
    """
    try:
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
                dissolved_part = explode_multipart_features(dissolved_part)
                dissolved_parts.append(dissolved_part)

        if dissolved_parts:
            result = pd.concat(dissolved_parts, ignore_index=True)
            return clean_and_validate_geometry(result)
        return gdf

    except Exception as e:
        print(f"Error during dissolve operation: {e}")
        return gdf


# ---------- DATA PROCESSING FUNCTIONS ---------- #
def process_facilities(gdf):
    """
    Prepare the NYC_Facilities dataset.
    Only includes specific FACTYPEs.
    """
    print("\nPrepare NYC Facilities...")

    # Define allowed facility types
    allowed_factypes = {
        'OTHER SCHOOL - NON-PUBLIC', 'COMMUNITY SERVICES',
        'ELEMENTARY SCHOOL - PUBLIC', 'CHARTER SCHOOL',
        'COMPASS ELEMENTARY', 'LICENSED PRIVATE SCHOOLS', 'PUBLIC LIBRARY', 'FIREHOUSE',
        'NURSING HOME', 'VISUAL ARTS', 'K-8 SCHOOL - PUBLIC', 'TRANSIT FACILITY',
        'MUSEUM', 'SCHOOL BUS DEPOT', 'TRANSPORTATION FACILITY',
        'NYCHA COMMUNITY CENTER - COMMUNITY CENTER - CORNERSTONE',
        'ELEMENTARY SCHOOL - CHARTER', 'K-8 SCHOOL - CHARTER',
        'PRE-SCHOOL FOR STUDENTS WITH DISABILITIES',
        'POLICE STATION', 'COMBINED MAINTENANCE/STORAGE FACILITY', 'HOSPITAL',
         'ADULT HOME', 'SENIORS', 'MANNED TRANSPORTATION FACILITY',
        'TRANSIT SUBSTATION', 'WASTEWATER PUMPING STATION', 'TRANSIT YARD',
        'NYCT SUBWAY YARD', 'MALL', 'MTA BUS DEPOT', 'TRANSFER STATION',
        'NYCT MAINTENANCE AND OTHER FACILITY', 
        'INDOOR STORAGE (WAREHOUSE)', 'WASTEWATER TREATMENT PLANT',
        'INDOOR STORAGE - EQUIPMENT', 'LIRR MAINTENANCE AND OTHER FACILITY',
        'SOLID WASTE LANDFILL'
    }

    # Filter for allowed facility types
    gdf = gdf[gdf['FACTYPE'].isin(allowed_factypes)]

    # Just ensure fclass and name columns exist
    if 'fclass' not in gdf.columns:
        gdf['fclass'] = gdf['FACTYPE'].fillna('Facilities')

    # Create a 'name' column if it doesn't exist
    if 'FACNAME' in gdf.columns:
        gdf['name'] = gdf['FACNAME'].fillna('Unknown')
    else:
        gdf['name'] = 'Unknown'

    # Ensure standard columns exist; fill if missing
    text_cols = ['FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN',
                 'OPNAME', 'OPABBREV', 'OPTYPE']
    for col in text_cols:
        if col not in gdf.columns:
            gdf[col] = 'Unknown'

    # For numeric columns that appear in the final output, fill with 0 if missing
    if 'CAPACITY' not in gdf.columns:
        gdf['CAPACITY'] = 0

    # Keep only these columns + geometry
    keep_cols = [
        'geometry', 'fclass', 'name',
        'FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN',
        'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY'
    ]
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[existing_cols]

    print("\nFacility counts by FACTYPE:")
    print(gdf['FACTYPE'].value_counts(dropna=False))
    print(f"\nTotal number of facilities: {len(gdf)}")
    return gdf


def process_pofw(gdf):
    """
    Prepare places of worship.
    Retain them all and add the standard columns so they can merge with Facilities.
    """
    print("\nPrepare OSM Places of Worship...")

    # If 'fclass' not present, set it to "OSM_POFW"
    if 'fclass' not in gdf.columns:
        gdf['fclass'] = 'OSM_POFW'
    else:
        gdf['fclass'] = 'OSM_POFW'  # standardize

    # Ensure a 'name' column
    if 'name' not in gdf.columns:
        gdf['name'] = 'Unknown'
    else:
        gdf['name'] = gdf['name'].fillna('Unknown')

    # Make sure the facility-related columns exist (fill with 'Unknown' or 0)
    text_columns = ['FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN',
                    'OPNAME', 'OPABBREV', 'OPTYPE']
    for col in text_columns:
        gdf[col] = 'Unknown'

    gdf['CAPACITY'] = 0  # numeric column in final
    # Keep the geometry and columns consistent with Facilities
    keep_cols = [
        'geometry', 'fclass', 'name',
        'FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN',
        'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY'
    ]
    gdf = gdf[keep_cols]

    print(f"Number of places of worship: {len(gdf)}")
    return gdf

def merge_point_datasets(datasets):
    """
    Merge the Facilities dataset with the POFW dataset into a single
    GeoDataFrame of 'points' that we can then join with PLUTO & building footprints.
    """
    print("\nMerge points datasets...")

    required_columns = [
        'fclass', 'name', 'FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN',
        'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'geometry'
    ]

    points_dfs = []
    for key in ['pofw', 'facilities']:
        if key in datasets and datasets[key] is not None:
            df = datasets[key]
            # Check if the df has all required columns; if not, skip
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: {key} is missing columns: {missing_cols}")
                continue
            points_dfs.append(df[required_columns])

    if not points_dfs:
        print("No valid datasets to merge!")
        return gpd.GeoDataFrame(columns=required_columns, crs=TARGET_CRS)

    # Concatenate them
    merged_pts = pd.concat(points_dfs, ignore_index=True)

    # Add an ObjectID
    merged_pts['ObjectID'] = merged_pts.index + 1

    # Print some summary info
    print("\nColumns in merged dataset:", merged_pts.columns.tolist())
    print("\nCount by fclass:")
    print(merged_pts['fclass'].value_counts(dropna=False))
    print(f"\nTotal points from facilities + places of worship: {len(merged_pts)}")

    return gpd.GeoDataFrame(merged_pts, geometry='geometry', crs=TARGET_CRS)


# ---------- MAIN PROCESSING BLOCK ---------- #
if not INPUT_FOLDER.exists():
    raise FileNotFoundError(f"Input folder not found at: {INPUT_FOLDER}")

n_cores = max(1, mp.cpu_count() - 1)

try:
    total_start_time = time.time()

    # 1) Load all datasets
    for key, file_info in data_files.items():
        file_path = file_info['path']
        columns = file_info['columns']

        start_time = time.time()
        print(f"\nLoading {file_path.name}...")
        if file_path.suffix == '.geojson':
            datasets[key] = load_geojson(file_path, columns)
        elif file_path.suffix == '.tif':
            datasets[key] = load_raster(file_path)
            # If a reprojected raster is created, keep track of the temp file
            if datasets[key] is not None and datasets[key].name != str(file_path):
                temp_files.append(Path(datasets[key].name))

        duration = time.time() - start_time
        print(f"Loading time for {key}: {timedelta(seconds=duration)}")

        # Print immediate summary
        if datasets[key] is not None:
            if isinstance(datasets[key], gpd.GeoDataFrame):
                print_dataset_info(key, datasets[key])
            else:
                print(f"{key} loaded but not a GeoDataFrame.")
        else:
            print(f"{key} dataset could not be loaded or is empty.")

    # 2) Process Facilities (keep them all)
    if 'facilities' in datasets and datasets['facilities'] is not None:
        datasets['facilities'] = process_facilities(datasets['facilities'])

    # 3) Process POFW (keep them all)
    if 'pofw' in datasets and datasets['pofw'] is not None:
        datasets['pofw'] = process_pofw(datasets['pofw'])

    # 4) Merge Facilities + POFW
    bldg_pts = merge_point_datasets(datasets)

    # 5) Extract data from PLUTO ("lots") into those points
    print("\nExtract data from lots (PLUTO)...")
    lots = datasets.get('lots')
    if lots is not None and not bldg_pts.empty:
        lot_fields = [
            'Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 'ComArea',
            'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea',
            'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR', 'LandUse'
        ]
        # Only join if these columns exist in PLUTO
        existing_lot_fields = [f for f in lot_fields if f in lots.columns]

        # Spatial join for points
        bldg_pts = gpd.sjoin(
            bldg_pts, 
            lots[existing_lot_fields + ['geometry']], 
            how='left', 
            predicate='within'
        )
        if 'index_right' in bldg_pts.columns:
            bldg_pts.drop(columns=['index_right'], inplace=True)

        matched = bldg_pts[~bldg_pts['Address'].isna()].shape[0]
        unmatched = bldg_pts[bldg_pts['Address'].isna()].shape[0]
        print(f"Points successfully extracted data from PLUTO: {matched}")
        print(f"Points did not extract data from PLUTO: {unmatched}")

    # 6) Combine points data with building footprints
    print("\nCombine points data with building footprints...")
    buildings = datasets.get('buildings')
    if buildings is not None and not bldg_pts.empty:
        # Keep only essential columns in buildings
        bldg_fields = ['geometry', 'groundelev', 'heightroof', 'lststatype', 'cnstrct_yr']
        existing_bldg_fields = [f for f in bldg_fields if f in buildings.columns]
        buildings = buildings[existing_bldg_fields]

        joined = gpd.sjoin(buildings, bldg_pts, how='inner', predicate='contains')
        print("Columns in joined after sjoin:", joined.columns)

        def combine_values(series):
            vals = series.dropna().unique()
            return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan

        # We'll aggregate everything in bldg_pts that we want to keep
        agg_fields = [
            'fclass', 'name', 'Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 
            'ComArea', 'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea', 
            'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR',
            'FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN', 
            'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'LandUse'
        ]
        # Only aggregate fields that exist in the joined data
        agg_fields = [f for f in agg_fields if f in joined.columns]

        if agg_fields:
            grouped = joined.groupby(joined.index)
            agg_dict = {f: combine_values for f in agg_fields}
            if 'CAPACITY' in joined.columns:
                agg_dict['CAPACITY'] = 'sum'  # numeric sum for capacity

            agg_result = grouped.agg(agg_dict)
            buildings = buildings.loc[agg_result.index]  # re-align to grouped index
            for f in agg_fields:
                buildings[f] = agg_result[f]

        datasets['buildings'] = buildings
    else:
        print("No buildings or point datasets to combine, or no overlaps found.")

    # 7) Extract data from NSI
    print("\nExtract data from national structures inventory (NSI)...")
    nsi = datasets.get('nsi')
    if nsi is not None and 'buildings' in datasets and datasets['buildings'] is not None:
        buildings = datasets['buildings']
        nsi_fields = ['bldgtype', 'num_story', 'found_type', 'found_ht']
        existing_nsi = [f for f in nsi_fields if f in nsi.columns]
        joined_nsi = gpd.sjoin(buildings, nsi[existing_nsi + ['geometry']], 
                               how='left', predicate='contains')

        def combine_values(series):
            vals = series.dropna().unique()
            return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan

        agg_dict_nsi = {f: combine_values for f in existing_nsi}
        grouped_nsi = joined_nsi.groupby(joined_nsi.index).agg(agg_dict_nsi)
        for f in existing_nsi:
            buildings[f] = grouped_nsi[f]

        matched_nsi = 0
        if 'bldgtype' in buildings.columns:
            matched_nsi = buildings[~buildings['bldgtype'].isna()].shape[0]
        unmatched_nsi = buildings.shape[0] - matched_nsi
        print(f"Buildings successfully extracted data from NSI: {matched_nsi}")
        print(f"Buildings did not extract data from NSI: {unmatched_nsi}")

    # ------------------ PARKING LOTS ------------------ #
    # Isolate parking lots from PLUTO data...
    print("\nIsolate parking lots from PLUTO data...")
    if lots is not None:
        try:
            # Ensure LandUse is a string
            lots['LandUse'] = lots['LandUse'].astype(str)

            # LandUse == '10' generally indicates parking
            parking = lots[lots['LandUse'] == '10'].copy()

            # Filter only large lots, based on a minimum area
            parking = parking[parking['LotArea'] >= MIN_PARKING_AREA].copy()

            # Clean geometries before dissolve
            parking = clean_and_validate_geometry(parking)

            # --------------------------------------------------
            # 1) Assign or copy columns BEFORE dissolve
            # --------------------------------------------------
            parking['fclass'] = "Parking"
            # For name, we might use the Address; fill with Unknown if missing
            parking['name'] = parking['Address'].fillna('Unknown')
            
            # Add fields to match building dataset
            parking['FACTYPE'] = 'Parking Lot'
            parking['FACSUBGRP'] = 'Transportation'
            parking['FACGROUP'] = 'Transportation Infrastructure'
            parking['FACDOMAIN'] = 'Transportation'
            parking['OPNAME'] = parking['OwnerName'].fillna('Unknown')
            parking['OPABBREV'] = 'Unknown'
            parking['OPTYPE'] = 'Private'
            parking['CAPACITY'] = 0

            print("Dissolving parking lots...")
            # --------------------------------------------------
            # 2) DISSOLVE: everything except 'OwnerName' gets lost
            # --------------------------------------------------
            parking = safe_dissolve(parking, 'OwnerName')

            # Now 'parking' only has ['OwnerName', 'geometry'] from the dissolve

            # --------------------------------------------------
            # 3) Reassign columns after dissolve
            # --------------------------------------------------
            parking['fclass'] = "Parking"
            # You can choose any naming convention you like:
            parking['name'] = "Parking Lot (Dissolved)"
            
            # If you want to keep a single address or a single lot area,
            # you'd need a custom group-by aggregation before dissolving.
            # For now, we just use a generic placeholder:
            parking['FACTYPE'] = 'Parking Lot'
            parking['FACSUBGRP'] = 'Transportation'
            parking['FACGROUP'] = 'Transportation Infrastructure'
            parking['FACDOMAIN'] = 'Transportation'
            parking['OPNAME'] = parking['OwnerName'].fillna('Unknown')
            parking['OPABBREV'] = 'Unknown'
            parking['OPTYPE'] = 'Private'
            parking['CAPACITY'] = 0

            # Continue with explode & clean
            parking = explode_multipart_features(parking)
            parking = clean_and_validate_geometry(parking)

            invalid_count = sum(~parking.geometry.is_valid)
            if invalid_count > 0:
                print(f"Warning: {invalid_count} invalid geometries found after processing")
                parking = parking[parking.geometry.is_valid]

            print(f"Number of parking lots (LandUse=10) >= {MIN_PARKING_AREA} sq ft: {len(parking)}")

        except Exception as e:
            print(f"Error processing parking lots: {e}")
            parking = gpd.GeoDataFrame(columns=['fclass', 'name', 'geometry'], crs=TARGET_CRS)
    else:
        parking = gpd.GeoDataFrame(columns=['fclass', 'name', 'geometry'], crs=TARGET_CRS)

    # 8) Merge parking lots and buildings into a single final dataset
    print("\nMerge parking lots and buildings into final sites dataset...")
    # Retrieve the buildings dataset (which has merged building footprints + points)
    sites = datasets.get('buildings')
    if sites is None:
        # If no buildings, create an empty GeoDataFrame with needed columns
        sites = gpd.GeoDataFrame(
            columns=[
                'fclass', 'name', 'geometry', 'FACTYPE', 'FACSUBGRP', 'FACGROUP',
                'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY'
            ],
            crs=TARGET_CRS
        )
        sites['fclass'] = 'Building'

    # Ensure 'fclass' in buildings
    if 'fclass' not in sites.columns:
        sites['fclass'] = 'Building'

    # For merging, ensure both have the same columns
    required_cols = [
        'fclass', 'name', 'geometry', 'FACTYPE', 'FACSUBGRP', 'FACGROUP',
        'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY'
    ]
    # Add any missing columns in buildings
    for col in required_cols:
        if col not in sites.columns and col != 'geometry':
            # CAPACITY is numeric, all others can be "Unknown"
            if col == 'CAPACITY':
                sites[col] = 0
            else:
                sites[col] = 'Unknown'

    if INCLUDE_PARKING:
        # Add any missing columns in parking
        for col in required_cols:
            if col not in parking.columns and col != 'geometry':
                if col == 'CAPACITY':
                    parking[col] = 0
                else:
                    parking[col] = 'Unknown'

        # Conform parking columns to match the buildings
        parking = parking[required_cols]

        # Combine them
        sites = pd.concat([sites, parking], ignore_index=True)
        print(f"Number of sites of interest (Buildings + Parking): {len(sites)}")
    else:
        print(f"Number of sites of interest (Buildings only): {len(sites)}")


    # ---------- OUTPUT ---------- #
    # 1) Shapefile in ./output/shapefiles
    # 2) GeoJSON in ./output
    shp_dir = Path("./output/shapefiles")
    shp_dir.mkdir(parents=True, exist_ok=True)

    # Write shapefile
    sites.to_file(shp_dir / "preprocessed_sites_solar.shp", driver="ESRI Shapefile")
    # Write GeoJSON
    sites.to_file("./output/preprocessed_sites_solar.geojson", driver="GeoJSON")

    print(f"\nFinal sites output: {len(sites)} features.")

    total_duration = time.time() - total_start_time
    print(f"\nTotal processing time: {timedelta(seconds=total_duration)}")

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Close raster datasets
    for name, dataset in datasets.items():
        if isinstance(dataset, rasterio.DatasetReader):
            dataset.close()

    # Clean up temporary reprojected raster files
    for temp_file in temp_files:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {e}")