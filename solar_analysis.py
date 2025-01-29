"""
solar_analysis.py
Updated to:
- Load preprocessed_sites.geojson
- Convert heightroof to a numeric value and fill missing
- Analyze solar potential directly.

This script:
- Loads preprocessed sites from ./output/preprocessed_sites.geojson
- Performs solar potential analysis on these sites
- Outputs the result as ./output/sites_solar.geojson and ./output/shapefiles/sites_solar.shp
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import numba
from numba import njit
import logging
import pvlib
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import warnings
from pathlib import Path
import time
import shapely.geometry

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@numba.njit
def calculate_shadow_impact_numba(height_diff_array, sun_altitude, distance_array):
    """
    Calculate the shadow impact for arrays of buildings.

    Args:
        height_diff_array (np.array): Array of height differences
        sun_altitude (float): Solar altitude angle in degrees
        distance_array (np.array): Distances between buildings

    Returns:
        np.array: Array of shadow impact values between 0 and 1
    """
    impacts = np.zeros_like(height_diff_array)
    tan_sun_altitude = np.tan(np.radians(sun_altitude))
    for i in range(len(height_diff_array)):
        height_diff = height_diff_array[i]
        distance = distance_array[i]

        if sun_altitude <= 0:
            impacts[i] = 1.0
            continue
        if height_diff <= 0:
            impacts[i] = 0.0
            continue

        shadow_length = height_diff / tan_sun_altitude
        if distance <= shadow_length:
            impacts[i] = 1.0 - (distance / shadow_length)
        else:
            impacts[i] = 0.0

    return impacts

def calculate_shadow_impact_vectorized(building_row, buildings_gdf, solar_pos, spatial_index):
    """
    Calculate shadow impact for a building considering nearby buildings.

    Args:
        building_row (GeoSeries): Row containing building data
        buildings_gdf (GeoDataFrame): GeoDataFrame containing all buildings
        solar_pos (Series): Solar position data
        spatial_index: Spatial index for buildings

    Returns:
        float: Shadow factor between 0 and 1
    """
    try:
        # Set maximum distance to check for shadows
        MAX_SHADOW_DISTANCE = 3 * building_row['heightroof']
        bounds = building_row.geometry.bounds

        # Expand bounds by MAX_SHADOW_DISTANCE
        bounds = (
            bounds[0] - MAX_SHADOW_DISTANCE,
            bounds[1] - MAX_SHADOW_DISTANCE,
            bounds[2] + MAX_SHADOW_DISTANCE,
            bounds[3] + MAX_SHADOW_DISTANCE
        )

        # Use spatial index to find nearby buildings
        possible_matches_idx = list(spatial_index.intersection(bounds))
        nearby_buildings = buildings_gdf.iloc[possible_matches_idx]
        nearby_buildings = nearby_buildings[nearby_buildings.index != building_row.name]

        if nearby_buildings.empty:
            return 1.0

        ref_centroid = building_row.geometry.centroid.coords[0]
        nearby_centroids = np.array([geom.centroid.coords[0] for geom in nearby_buildings.geometry])

        dx = nearby_centroids[:, 0] - ref_centroid[0]
        dy = nearby_centroids[:, 1] - ref_centroid[1]
        distances = np.sqrt(dx**2 + dy**2)

        height_diff = nearby_buildings['heightroof'].values - building_row['heightroof']
        sun_altitude = solar_pos['apparent_elevation']

        shadow_impacts = calculate_shadow_impact_numba(
            height_diff,
            sun_altitude,
            distances
        )

        total_shadow_impact = np.mean(shadow_impacts)
        shadow_factor = max(0.0, 1.0 - total_shadow_impact)
        return shadow_factor

    except Exception as e:
        logger.error(f"Error calculating shadow impact for building {building_row.name}: {str(e)}")
        return 1.0

# Global variables for worker processes
global_buildings_gdf = None
global_spatial_index = None
global_solar_position = None
global_annual_radiation = None
global_panel_density = None
global_panel_efficiency = None
global_performance_ratio = None

def init_worker(buildings_gdf, spatial_index, solar_position, annual_radiation,
                panel_density, panel_efficiency, performance_ratio):
    """
    Initialize global variables in worker processes.
    """
    global global_buildings_gdf
    global global_spatial_index
    global global_solar_position
    global global_annual_radiation
    global global_panel_density
    global global_panel_efficiency
    global global_performance_ratio

    global_buildings_gdf = buildings_gdf
    global_spatial_index = spatial_index
    global_solar_position = solar_position
    global_annual_radiation = annual_radiation
    global_panel_density = panel_density
    global_panel_efficiency = panel_efficiency
    global_performance_ratio = performance_ratio

def worker_process_building(args):
    """
    Worker function to process a single building using global variables.
    """
    idx, building = args
    try:
        # Get the geometry from the building
        geometry = building.geometry

        # Calculate area in square feet (assuming input is already in EPSG:6539)
        area_ft2 = geometry.area
        area_m2 = area_ft2 * 0.092903

        # Grab global variables
        buildings_gdf = global_buildings_gdf
        spatial_index = global_spatial_index
        solar_pos = global_solar_position
        annual_radiation = global_annual_radiation
        panel_density = global_panel_density
        panel_efficiency = global_panel_efficiency
        performance_ratio = global_performance_ratio

        # Calculate shadow factor
        shadow_factor = calculate_shadow_impact_vectorized(
            building, buildings_gdf, solar_pos, spatial_index
        )

        # Annual solar potential (kWh/year)
        solar_potential = (
            annual_radiation *
            area_m2 *
            panel_density *
            panel_efficiency *
            performance_ratio *
            shadow_factor
        )

        # Peak system size (kW) under STC
        # 1 m2 at 100% efficiency => 1 kW
        # so at 20% efficiency => 0.20 kW per m2
        peak_power_kW = float(area_m2 * panel_density * panel_efficiency)

        # Calculate average monthly output (kWh/month)
        avg_monthly_kwh = solar_potential / 12.0

        # Estimate the number of households powered (assuming 300 kWh/month)
        households_powered = avg_monthly_kwh / 300.0

        return {
            'solar_potential': float(solar_potential),      # kWh/year
            'effective_area': float(area_m2 * panel_density),
            'peak_power': peak_power_kW,                    # kW
            'shadow_factor': float(shadow_factor),
            'annual_radiation': float(annual_radiation),
            'performance_ratio': float(performance_ratio),
            'area_ft2': float(area_ft2),
            'area_m2': float(area_m2),
            'avg_monthly_kwh': float(avg_monthly_kwh),
            'households_powered': float(households_powered)
        }

    except Exception as e:
        logger.error(f"Error processing building {idx}: {str(e)}")
        return {
            'solar_potential': 0.0,
            'effective_area': 0.0,
            'peak_power': 0.0,
            'shadow_factor': 0.0,
            'annual_radiation': 0.0,
            'performance_ratio': 0.0,
            'area_ft2': 0.0,
            'area_m2': 0.0,
            'avg_monthly_kwh': 0.0,
            'households_powered': 0.0
        }


class SolarAnalyzer:
    """Class for analyzing solar potential of buildings."""

    def __init__(self):
        logger.info("Initializing SolarAnalyzer...")
        self._initialize_constants()
        self._initialize_cache()
        self.buildings_gdf = None
        self.spatial_index = None

    def _initialize_constants(self):
        self.NYC_LAT = 40.7128
        self.NYC_LON = -74.0060
        self.PANEL_EFFICIENCY = 0.20
        self.PERFORMANCE_RATIO = 0.75
        self.PANEL_DENSITY = 0.70
        self._solar_position = None

    def _initialize_cache(self):
        self._monthly_radiation = self._initialize_radiation_data()
        self._annual_radiation = self._calculate_annual_radiation()

    def _initialize_radiation_data(self):
        return {
            '01': 2.45, '02': 3.42, '03': 4.53, '04': 5.64,
            '05': 6.48, '06': 6.89, '07': 6.75, '08': 5.98,
            '09': 4.92, '10': 3.67, '11': 2.56, '12': 2.12
        }

    def _calculate_annual_radiation(self):
        monthly_sum = sum(self._monthly_radiation.values())
        return monthly_sum * 365 / 12

    def initialize_spatial_index(self, buildings_gdf):
        self.spatial_index = buildings_gdf.sindex
        return buildings_gdf

    def get_solar_position(self):
        if self._solar_position is None:
            times = pd.date_range('2020-06-21 12:00:00', periods=1, freq='H', tz='UTC')
            location = pvlib.location.Location(latitude=self.NYC_LAT, longitude=self.NYC_LON)
            self._solar_position = location.get_solarposition(times).iloc[0]
        return self._solar_position

def process_buildings_parallel(analyzer, buildings_chunk, buildings_gdf):
    """
    Process buildings using multiprocessing.
    """
    analyzer.buildings_gdf = buildings_gdf
    analyzer.initialize_spatial_index(buildings_gdf)

    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Using {num_processes} processes for parallel processing.")

    args_list = [(idx, row) for idx, row in buildings_chunk.iterrows()]

    solar_position = analyzer.get_solar_position()
    spatial_index = analyzer.spatial_index
    annual_radiation = analyzer._annual_radiation
    panel_density = analyzer.PANEL_DENSITY
    panel_efficiency = analyzer.PANEL_EFFICIENCY
    performance_ratio = analyzer.PERFORMANCE_RATIO

    with Pool(processes=num_processes, initializer=init_worker, initargs=(
        buildings_gdf, spatial_index, solar_position, annual_radiation,
        panel_density, panel_efficiency, performance_ratio
    )) as pool:
        results_list = pool.map(worker_process_building, args_list)

    return pd.DataFrame(results_list, index=buildings_chunk.index)

def analyze_solar_potential(candidate_buildings, full_buildings):
    """
    Analyze solar potential for candidate buildings.
    """
    try:
        analyzer = SolarAnalyzer()
        logger.info("Initialized SolarAnalyzer")

        # Ensure CRS is set and transform to EPSG:6539
        if candidate_buildings.crs is None:
            logger.warning("Input data missing CRS, assuming EPSG:4326")
            candidate_buildings = candidate_buildings.set_crs('EPSG:4326')
        if full_buildings.crs is None:
            full_buildings = full_buildings.set_crs('EPSG:4326')

        # Project to NAD83(2011) / New York Long Island (ftUS)
        candidate_projected = candidate_buildings.to_crs('EPSG:6539')
        buildings_projected = full_buildings.to_crs('EPSG:6539')

        analyzer.initialize_spatial_index(buildings_projected)

        results = process_buildings_parallel(
            analyzer,
            candidate_projected,
            buildings_projected
        )

        # Create a new GeoDataFrame with only the geometry from candidate_projected
        analyzed_buildings = gpd.GeoDataFrame(
            geometry=candidate_projected.geometry,
            crs=candidate_projected.crs
        )

        # Add original columns from candidate_projected, excluding area calculations
        for col in candidate_projected.columns:
            if col not in ['area_ft2', 'area_m2', 'geometry']:
                analyzed_buildings[col] = candidate_projected[col]

        # Add results columns
        for col in results.columns:
            analyzed_buildings[col] = results[col]

        return analyzed_buildings

    except Exception as e:
        logger.error(f"Error in solar potential analysis: {str(e)}")
        return None


def ensure_2d_geometry(gdf):
    """
    Ensure geometries are 2D by dropping Z coordinates if present.
    """
    try:
        if gdf is None:
            return None

        def drop_z(geom):
            if geom is None:
                return None
            if geom.has_z:
                if geom.geom_type == 'Polygon':
                    return shapely.geometry.Polygon(
                        shell=[(x, y) for x, y, *_ in geom.exterior.coords],
                        holes=[[(x, y) for x, y, *_ in inner.coords] for inner in geom.interiors]
                    )
                elif geom.geom_type == 'MultiPolygon':
                    return shapely.geometry.MultiPolygon([
                        shapely.geometry.Polygon(
                            shell=[(x, y) for x, y, *_ in poly.exterior.coords],
                            holes=[[(x, y) for x, y, *_ in inner.coords] for inner in poly.interiors]
                        )
                        for poly in geom.geoms
                    ])
            return geom

        gdf.geometry = gdf.geometry.apply(drop_z)
        return gdf
    except Exception as e:
        logger.error(f"Error ensuring 2D geometry: {str(e)}")
        return gdf

def create_centroids_within_buildings(buildings_gdf):
    """
    Create a single point within each building polygon, preferably at the centroid
    or at a point guaranteed to be within irregular polygons.

    Args:
        buildings_gdf (GeoDataFrame): The buildings GeoDataFrame
    Returns:
        GeoDataFrame: One point per building with all building attributes
    """
    # Input validation
    if buildings_gdf is None:
        logger.error("Input GeoDataFrame is None")
        return None

    if len(buildings_gdf) == 0:
        logger.error("Input GeoDataFrame is empty")
        return None

    try:
        logger.info(f"Creating centroids for {len(buildings_gdf)} buildings...")
        points_list = []
        attributes_list = []

        # Process each building
        for idx, building in buildings_gdf.iterrows():
            try:
                # Skip invalid geometries
                if building.geometry is None or building.geometry.is_empty or not building.geometry.is_valid:
                    continue

                # Try to use centroid first
                point = building.geometry.centroid

                # If centroid is not within polygon, use point on surface
                if not building.geometry.contains(point):
                    point = building.geometry.representative_point()

                # Create attributes dictionary for the point
                point_attributes = {
                    'building_id': idx,
                    'solar_potential': building['solar_potential'],
                    'effective_area': building['effective_area'],
                    'peak_power': building['peak_power'],
                    'shadow_factor': building['shadow_factor'],
                    'area_ft2': building['area_ft2']
                }

                # Add any additional columns that exist in the building data
                for col in buildings_gdf.columns:
                    if col not in point_attributes and col != 'geometry':
                        point_attributes[col] = building[col]

                points_list.append(point)
                attributes_list.append(point_attributes)

            except Exception as e:
                logger.error(f"Error processing building {idx}: {str(e)}")
                continue

        if not points_list:
            logger.error("No valid points were created")
            return None

        # Create GeoDataFrame from points
        points_gdf = gpd.GeoDataFrame(
            attributes_list,
            geometry=points_list,
            crs=buildings_gdf.crs
        )

        # Add point ID field
        points_gdf['point_id'] = range(1, len(points_gdf) + 1)

        logger.info(f"Successfully created {len(points_gdf)} centroids for {len(buildings_gdf)} buildings")
        return points_gdf

    except Exception as e:
        logger.error(f"Error creating centroids within buildings: {str(e)}")
        return None

if __name__ == "__main__":
    start_time = time.time()

    # Load the preprocessed sites
    input_file = Path('./output/preprocessed_sites_solar.geojson')
    if not input_file.exists():
        logger.error("preprocessed_sites_solar.geojson not found. Run preprocessing first.")
        exit(1)

    sites = gpd.read_file(input_file)
    logger.info(f"Loaded {len(sites)} sites from preprocessed_sites_solar.geojson")

    # Project to equal-area projection
    if sites.crs is None:
        logger.warning("Input CRS is None, setting to EPSG:4326 (WGS84)")
        sites = sites.set_crs('EPSG:4326')

    # Project to NAD83(2011) / New York Long Island (ftUS)
    sites = sites.to_crs('EPSG:6539')
    logger.info("Projected data to EPSG:6539 for accurate area calculations")

    # Ensure required columns
    if 'heightroof' not in sites.columns:
        logger.warning("heightroof not found, defaulting to 0 ft for demonstration.")
        sites['heightroof'] = 0.0
    else:
        # Convert heightroof to numeric and fill missing
        sites['heightroof'] = pd.to_numeric(sites['heightroof'], errors='coerce')
        missing_count = sites['heightroof'].isna().sum()
        if missing_count > 0:
            logger.warning(f"{missing_count} buildings had non-numeric heightroof values; assigning default of 0 ft.")
            sites['heightroof'].fillna(0.0, inplace=True)

    # Calculate area in the projected CRS
    sites['area_ft2'] = sites.geometry.area
    sites['area_m2'] = sites['area_ft2'] * 0.092903
    logger.info("Calculated areas in projected CRS")

    full_buildings = sites.copy()

    # Analyze solar potential
    logger.info("Starting solar potential analysis...")
    analyzed_buildings = analyze_solar_potential(sites, full_buildings)

    if analyzed_buildings is not None and not analyzed_buildings.empty:
        logger.info("Solar analysis completed successfully.")

        # Project back to WGS84 for output
        analyzed_buildings = analyzed_buildings.to_crs('EPSG:4326')

        # Ensure 2D geometries before saving
        analyzed_buildings = ensure_2d_geometry(analyzed_buildings)

        # Save buildings as GeoJSON
        geojson_output = Path("./output/sites_solar.geojson")
        analyzed_buildings.to_file(
            geojson_output,
            driver='GeoJSON',
            encoding='utf-8'
        )
        logger.info(f"Building results saved to {geojson_output}")

    if analyzed_buildings is not None and not analyzed_buildings.empty:
        logger.info("Solar analysis completed successfully.")

        # Project back to WGS84 for output
        analyzed_buildings = analyzed_buildings.to_crs('EPSG:4326')

        # Ensure 2D geometries before saving
        analyzed_buildings = ensure_2d_geometry(analyzed_buildings)

        # Save buildings as GeoJSON
        geojson_output = Path("./output/sites_solar.geojson")
        analyzed_buildings.to_file(
            geojson_output,
            driver='GeoJSON',
            encoding='utf-8'
        )
        logger.info(f"Building results saved to {geojson_output}")

        # Create centroids within buildings
        points_gdf = create_centroids_within_buildings(analyzed_buildings)

        if points_gdf is not None and not points_gdf.empty:
            # Save points as GeoJSON
            points_geojson_output = Path("./output/sites_solar_points.geojson")
            points_gdf.to_file(
                points_geojson_output,
                driver='GeoJSON',
                encoding='utf-8'
            )
            logger.info(f"Point results saved to {points_geojson_output}")

            # Prepare points for shapefile
            points_for_shapefile = points_gdf.copy()

            # Convert any non-standard data types to string
            for col in points_for_shapefile.select_dtypes(include=['object']).columns:
                points_for_shapefile[col] = points_for_shapefile[col].astype(str)

            # Limit field names to 10 characters
            points_for_shapefile.columns = [col[:10] for col in points_for_shapefile.columns]

            # Save points as Shapefile
            points_shapefile_output = Path("./output/shapefiles/sites_solar_points.shp")
            points_for_shapefile.to_file(
                points_shapefile_output,
                driver='ESRI Shapefile',
                encoding='utf-8'
            )
            logger.info(f"Point results saved to {points_shapefile_output}")
        else:
            logger.error("Failed to create centroids within buildings")



    end_time = time.time()
    logger.info(f"Total solar analysis execution time: {end_time - start_time:.2f} seconds")