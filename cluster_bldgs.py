#!/usr/bin/env python3
import os
import json
import geopandas as gpd
import networkx as nx
from shapely.ops import unary_union
import folium
import branca.colormap as cm
from branca.element import Template, MacroElement
import webbrowser

def dissolve_buildings(input_path, output_path, preserve_original=False):
    """
    Reads the input buildings dataset and dissolves groups of polygons that share edges (or intersect)
    into single features. Each dissolved feature gets a new attribute 'num_bldgs_merged' with the count
    of buildings that were merged.
    
    Parameters:
    -----------
    input_path : str
        Path to input GeoJSON file
    output_path : str
        Path to output GeoJSON file
    preserve_original : bool, default False
        If True, keeps the original geometries and adds a cluster_id instead of dissolving
    """
    # Read the input GeoJSON
    gdf = gpd.read_file(input_path)
    
    # Fix any invalid geometries by buffering with 0
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    
    # Create a graph where each node represents a building (polygon)
    G = nx.Graph()
    for idx in gdf.index:
        G.add_node(idx)
    
    # Build a spatial index for efficiency
    sindex = gdf.sindex
    for idx, row in gdf.iterrows():
        # Get potential neighboring polygons (by bounding box)
        possible_matches = list(sindex.intersection(row.geometry.bounds))
        for j in possible_matches:
            if idx < j:
                # Check if the geometries touch or intersect
                if row.geometry.touches(gdf.loc[j, 'geometry']) or row.geometry.intersects(gdf.loc[j, 'geometry']):
                    G.add_edge(idx, j)
    
    # Group polygons by connected components
    clusters = list(nx.connected_components(G))
    
    if preserve_original:
        # Add cluster_id to original geometries instead of dissolving
        gdf['cluster_id'] = -1
        gdf['num_bldgs_in_cluster'] = 0
        
        for i, comp in enumerate(clusters):
            indices = list(comp)
            count = len(indices)
            gdf.loc[indices, 'cluster_id'] = i
            gdf.loc[indices, 'num_bldgs_in_cluster'] = count
        
        # Write to output
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"Clustered buildings (original geometries) saved to {output_path}")
        return gdf
    else:
        # Improved approach: dissolve geometries with care to avoid triangulation
        dissolved_records = []
        for comp in clusters:
            indices = list(comp)
            
            # Extract the geometries for this cluster
            geoms = gdf.loc[indices, 'geometry'].tolist()
            
            # Use cascaded_union instead of unary_union to maintain polygon structure
            # This should avoid the triangulation issue
            from shapely.ops import cascaded_union
            merged_geom = cascaded_union(geoms)
            
            # If for some reason we get a geometry collection, try to convert to a proper polygon
            if merged_geom.geom_type == 'GeometryCollection':
                # Try to create a proper polygon from the collection
                polygons = [g for g in merged_geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
                if polygons:
                    merged_geom = cascaded_union(polygons)
            
            count = len(indices)
            dissolved_records.append({'num_bldgs_merged': count, 'geometry': merged_geom})
        
        # Create a GeoDataFrame with the dissolved features and write to output
        dissolved_gdf = gpd.GeoDataFrame(dissolved_records, crs=gdf.crs)
        dissolved_gdf.to_file(output_path, driver='GeoJSON')
        print(f"Dissolved buildings saved to {output_path}")
        return dissolved_gdf

def create_webmap(dissolved_geojson_path, output_html, count_field='num_bldgs_merged'):
    """
    Creates a folium webmap using the dissolved GeoJSON. The map uses a CartoDB Positron basemap,
    colors the features using a linear color ramp based on the count field, and includes a tooltip,
    legend, title, and logo with the requested styles.
    
    Parameters:
    -----------
    dissolved_geojson_path : str
        Path to the GeoJSON file
    output_html : str
        Path to output HTML file
    count_field : str, default 'num_bldgs_merged'
        Field name that contains the count of merged buildings
    """
    # Load the dissolved data with geopandas
    gdf = gpd.read_file(dissolved_geojson_path)
    
    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326) for web mapping
    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Fix any invalid geometries and compute the centroid for map initialization
    valid_geoms = [geom if geom.is_valid else geom.buffer(0) for geom in gdf.geometry]
    centroid = gpd.GeoSeries(valid_geoms, crs=gdf.crs).union_all().centroid

    # Initialize the folium map
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12, tiles='CartoDB positron')
    
    # Convert GeoDataFrame to GeoJSON for Folium
    geo_data = gdf.to_json()
    data = json.loads(geo_data)
    
    # Extract the values for the count field to set up the color scale
    values = [feature['properties'][count_field] for feature in data['features']]
    min_val, max_val = min(values), max(values)
    
    # Create a linear colormap (from light yellow to dark red)
    colormap = cm.LinearColormap(colors=['#ffffcc', '#800026'], vmin=min_val, vmax=max_val)
    
    # Define a style function for the GeoJSON layer using the colormap
    def style_function(feature):
        value = feature['properties'][count_field]
        return {
            'fillColor': colormap(value),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
        }
    
    # Add a GeoJson layer with a tooltip that displays the number of merged buildings
    tooltip = folium.GeoJsonTooltip(
        fields=[count_field],
        aliases=['# of buildings in cluster:'],
        localize=True
    )
    
    folium.GeoJson(
        data,
        style_function=style_function,
        tooltip=tooltip
    ).add_to(m)
    
    # Define styles for the legend, title, and logo as provided
    LEGEND_STYLES = {
        "container": "position: fixed; bottom: 50px; right: 50px; z-index:9999; background: white; padding: 15px; border-radius: 15px; font-family: Arial, sans-serif; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);",
        "header": "margin-top:0; margin-bottom: 12px; font-size: 16px; font-weight: bold;",
        "sectionHeader": "margin-top:10px; margin-bottom: 5px; font-size: 14px; font-weight: bold;",
        "itemContainer": "display: flex; align-items: center; margin-bottom: 5px;",
        "colorBox": "width: 20px; height: 20px; margin-right: 5px;",
        "label": "font-size: 13px;"
    }

    TITLE_STYLE = {
        "container": "position: fixed; top: 10px; left: 10px; z-index:9999; background: white; padding: 10px 20px; border-radius: 10px; font-family: Arial, sans-serif; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);",
        "title": "margin: 0; font-size: 20px; font-weight: bold; color: #333;"
    }

    LOGO_STYLE = {
        "container": "position: fixed; top: 100px; left: 10px; z-index:9999; background: white; padding: 5px 10px; border-radius: 5px; font-family: 'Futura', Arial, sans-serif; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);"
    }
    
    # Create Legend HTML with a horizontal color ramp
    legend_html = f'''
    <div style="{LEGEND_STYLES['container']}">
        <div style="{LEGEND_STYLES['header']}"># of merged buildings</div>
        <div style="height: 20px; width: 200px; background: linear-gradient(to right, {colormap.colors[0]}, {colormap.colors[-1]});"></div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 5px;">
            <span>lower</span>
            <span>higher</span>
        </div>
    </div>
    '''
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)
    
    # Add a title box with 'Building Clusters'
    title_html = f'''
    <div style="{TITLE_STYLE['container']}">
        <h1 style="{TITLE_STYLE['title']}">Building Clusters</h1>
    </div>
    '''
    title = MacroElement()
    title._template = Template(title_html)
    m.get_root().add_child(title)
    
    # Add a logo box with the text "One Analysis" (with "One" in Futura bold and "Analysis" in Futura light)
    logo_html = f'''
    <div style="{LOGO_STYLE['container']}">
        <span style="font-family: 'Futura', sans-serif; font-weight: bold;">One</span>
        <span style="font-family: 'Futura', sans-serif; font-weight: lighter;"> Analysis</span>
    </div>
    '''
    logo = MacroElement()
    logo._template = Template(logo_html)
    m.get_root().add_child(logo)
    
    # Save the webmap to an HTML file
    m.save(output_html)
    print(f"Webmap saved to {output_html}")

def main():
    import webbrowser
    
    input_path = './input/buildings.geojson'
    output_geojson = './output/bldgs_dissolved.geojson'
    output_html = 'webmap.html'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
    
    # Process the buildings: Set preserve_original to False to dissolve polygons in each cluster
    preserve_original = False  # Changed from True to False to properly dissolve polygons
    result_gdf = dissolve_buildings(input_path, output_geojson, preserve_original)
    
    # Create and save the folium webmap
    count_field = 'num_bldgs_merged'  # Using the correct field for dissolved polygons
    create_webmap(output_geojson, output_html, count_field)
    
    # Automatically open the HTML file in the default browser
    print(f"Opening {output_html} in default browser...")
    webbrowser.open('file://' + os.path.realpath(output_html))

if __name__ == '__main__':
    main()