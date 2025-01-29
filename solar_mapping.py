import folium
from folium import plugins
from branca import colormap as cm
from folium.plugins import MeasureControl, Fullscreen
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import shutil

def create_detailed_solar_map(buildings_gdf, points_gdf):
    """
    Create interactive map with visualization of solar analysis results.
    Includes buildings, points, and CSC Neighborhoods.
    """
    print("Creating solar potential map...")
    try:
        # Data validation and cleaning for mapping
        buildings_clean = buildings_gdf[
            (buildings_gdf['solar_potential'] > 0) &
            (buildings_gdf['effective_area'] > 0) &
            (buildings_gdf['peak_power'] > 0) &
            (buildings_gdf['solar_potential'].notna())
        ].copy()

        # Load CSC Neighborhoods
        neighborhoods_file = Path('input') / 'CSC_Neighborhoods.geojson'
        if neighborhoods_file.exists():
            neighborhoods_gdf = gpd.read_file(neighborhoods_file)
            neighborhoods_wgs84 = neighborhoods_gdf.to_crs('EPSG:4326')
        else:
            print("Warning: CSC_Neighborhoods.geojson not found")
            neighborhoods_wgs84 = None

        # Convert buildings to WGS84 for mapping
        if not buildings_clean.empty:
            buildings_wgs84 = buildings_clean.to_crs('EPSG:4326')
        else:
            buildings_wgs84 = gpd.GeoDataFrame(crs='EPSG:4326', geometry=[])

        # Convert points to WGS84
        points_wgs84 = points_gdf.to_crs('EPSG:4326')

        # Calculate map bounds for auto-zoom
        all_bounds = []
        if not buildings_wgs84.empty:
            all_bounds.append(buildings_wgs84.total_bounds)
        if not points_wgs84.empty:
            all_bounds.append(points_wgs84.total_bounds)
        if neighborhoods_wgs84 is not None:
            all_bounds.append(neighborhoods_wgs84.total_bounds)

        if all_bounds:
            combined_bounds = np.vstack(all_bounds)
            minx = np.min(combined_bounds[:, 0])
            miny = np.min(combined_bounds[:, 1])
            maxx = np.max(combined_bounds[:, 2])
            maxy = np.max(combined_bounds[:, 3])
            bounds = [minx, miny, maxx, maxy]
        else:
            # Default to NYC
            bounds = [-74.0060, 40.7128, -74.0060, 40.7128]

        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        # Fallback if bounds are invalid
        if np.isnan(center_lat) or np.isnan(center_lon):
            center_lat = 40.7128
            center_lon = -74.0060
            print("Warning: Using default NYC coordinates for map center")

        # ---------------------------------------------------------
        # CREATE THE FOLIUM MAP WITH NO DEFAULT BASE TILES
        # ---------------------------------------------------------
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            prefer_canvas=True,
            tiles=None  # <-- No default tiles so we can add exactly two
        )

        # ---------------------------------------------------------
        # ADD TWO BASE LAYERS:
        # 1) LIGHT MAP (OSM/CARTODB)
        # 2) SATELLITE (ESRI WORLD IMAGERY)
        # ---------------------------------------------------------
        folium.TileLayer(
            tiles='CartoDB Positron',
            name='Light Map',
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri WorldImagery',
            name='Satellite Imagery',
            control=True
        ).add_to(m)

        # Fit bounds if valid
        if not np.isnan(center_lat) and not np.isnan(center_lon):
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Gather valid solar potential values (excluding zero/negative)
        building_values = (
            buildings_wgs84['solar_potential'].dropna().values if not buildings_wgs84.empty else np.array([])
        )
        point_values = (
            points_wgs84['solar_potential'].dropna().values if not points_wgs84.empty else np.array([])
        )
        valid_values = np.concatenate([building_values, point_values])
        valid_values = valid_values[valid_values > 0]  # remove <= 0

        if len(valid_values) == 0:
            raise ValueError("No valid (positive) solar potential values found for color mapping")

        # Compute min and max from positive values only
        min_val = valid_values.min()
        max_val = valid_values.max()

        # Ensure min_val isn't zero for the log scale
        if min_val <= 0:
            min_val = 1e-6

        # Compute log scale boundaries
        log_min = np.log1p(min_val)
        log_max = np.log1p(max_val)

        # 7 breaks from log_min to log_max
        colors = ['#4d4d4d', '#fff7bc', '#fee391', '#fdb863', '#f87d43', '#e95224', '#cc3311']
        log_breaks = np.linspace(log_min, log_max, len(colors))
        value_breaks = np.expm1(log_breaks)

        # Create tick labels (real kWh values)
        tick_labels = []
        for value in value_breaks:
            if value >= 1_000_000:
                tick_labels.append(f"{int(value / 1_000_000):,}M")
            elif value >= 1_000:
                tick_labels.append(f"{int(value / 1_000):,}K")
            else:
                tick_labels.append(f"{int(value):,}")

        # Build the log-based colormap
        colormap = cm.LinearColormap(
            colors=colors,
            index=log_breaks,  # log-based breaks
            vmin=log_min,
            vmax=log_max,
            caption='Solar Potential (kWh/year)',
            tick_labels=tick_labels
        )
        colormap.add_to(m)

        # Create feature groups
        neighborhoods_fg = folium.FeatureGroup(name='CSC Neighborhoods', show=True)
        buildings_fg = folium.FeatureGroup(name='Buildings', show=True)
        points_fg = folium.FeatureGroup(name='Points', show=True)

        def map_color(value):
            """Map a real kWh/year value to the correct log-based color."""
            if pd.isna(value) or value <= 0:
                return '#4d4d4d'
            log_value = np.log1p(value)
            return colormap(log_value)

        # Keep your original log-based radius approach
        def calculate_radius(value):
            if pd.isna(value) or value <= 0:
                return 2
            log_value = np.log1p(value)

            min_point = np.log1p(points_wgs84['solar_potential'].dropna().replace(0, np.nan).min())
            max_point = np.log1p(points_wgs84['solar_potential'].dropna().max())

            if pd.isna(min_point) or pd.isna(max_point) or min_point == max_point:
                return 4

            normalized = (log_value - min_point) / (max_point - min_point)
            min_radius = 1
            max_radius = 6

            radius = min_radius + (np.exp(normalized * 2) - 1) * (max_radius - min_radius) / (np.e - 1)
            return radius

        # Add neighborhoods
        if neighborhoods_wgs84 is not None:
            folium.GeoJson(
                neighborhoods_wgs84,
                style_function=lambda x: {
                    'fillColor': '#ffffff',
                    'fillOpacity': 0,
                    'color': '#000000',
                    'weight': 2
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['Name'],
                    aliases=['Neighborhood:'],
                    style=('background-color: white; color: #333333;')
                )
            ).add_to(neighborhoods_fg)

        # Add buildings
        if not buildings_wgs84.empty:
            folium.GeoJson(
                buildings_wgs84,
                style_function=lambda feature: {
                    'fillColor': map_color(feature['properties']['solar_potential']),
                    'color': '#666666',
                    'weight': 1,
                    'fillOpacity': 0.7
                }
            ).add_to(buildings_fg)

        # Add points
        for idx, point in points_wgs84.iterrows():
            try:
                color = map_color(point['solar_potential'])
                radius = calculate_radius(point['solar_potential'])
                name = point['name'] if pd.notna(point['name']) else 'N/A'
                fclass = point['fclass'] if pd.notna(point['fclass']) else 'N/A'
                owner = point['OwnerName'] if pd.notna(point['OwnerName']) else 'N/A'
                area_val = point['area_ft2'] if pd.notna(point['area_ft2']) else 0

                popup_content = f"""
                <div style="width:300px">
                    <h4>{name}</h4>
                    <b>Type:</b> {fclass}<br>
                    <b>Owner:</b> {owner}<br>
                    <b>Area:</b> ~{round(area_val)} sq ft<br>
                    <b>Annual Generation:</b> ~{point['solar_potential']:,.0f} kWh/year<br>
                    <b>System Size:</b> {point['peak_power']:,.1f} kW<br>
                    <b>Households Powered:</b> ~{point['households_powered']:,.0f}<br>
                </div>
                """

                tooltip = (
                    f"{fclass}, {point['peak_power']:,.1f} kW\n"
                    f"({point['households_powered']:,.0f} households)"
                )

                folium.CircleMarker(
                    location=[point.geometry.y, point.geometry.x],
                    radius=radius,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.9,
                    weight=1,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=tooltip
                ).add_to(points_fg)

            except Exception as e:
                print(f"Error adding point {idx} to map: {str(e)}")
                continue

        # Add feature groups
        neighborhoods_fg.add_to(m)
        buildings_fg.add_to(m)
        points_fg.add_to(m)

        # Add measure control
        m.add_child(MeasureControl(
            position='topleft',
            primary_length_unit='miles',
            secondary_length_unit='feet',
            primary_area_unit='acres',
            secondary_area_unit='sqfeet'
        ))

        # Add layer control - user can switch between Light Map & Satellite
        folium.LayerControl(position='topright').add_to(m)

        return m

    except Exception as e:
        print(f"Error in map creation: {str(e)}")
        raise

if __name__ == "__main__":
    # Load datasets
    buildings_file = Path('output') / 'sites_solar.geojson'
    points_file = Path('output') / 'sites_solar_points.geojson'

    if not buildings_file.exists():
        print("sites_solar.geojson not found. Please run solar_analyzer.py first.")
        exit(1)
    if not points_file.exists():
        print("sites_solar_points.geojson not found. Please run solar_analyzer.py first.")
        exit(1)

    analyzed_buildings = gpd.read_file(buildings_file)
    analyzed_points = gpd.read_file(points_file)

    # Create the deployment directory
    solarwebmap_deploy_dir = Path('output') / 'solarwebmap-deploy'
    solarwebmap_deploy_dir.mkdir(parents=True, exist_ok=True)

    # Create and save the map
    solar_map = create_detailed_solar_map(analyzed_buildings, analyzed_points)
    map_file = solarwebmap_deploy_dir / 'solar_potential_map.html'
    solar_map.save(str(map_file))
    print(f"\nInteractive map saved to {map_file}")

    # Copy as index.html
    index_file = solarwebmap_deploy_dir / 'index.html'
    shutil.copy(str(map_file), str(index_file))
    print(f"Webmap deployed at {index_file}")