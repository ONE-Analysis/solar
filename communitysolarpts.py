import re
import json
import os

# Regex pattern to capture each record.
# It assumes each record is in the following format:
#
# [new google.maps.LatLng(41.101154, -73.807220),
# "1",
# "Ridge Hill Community Solar",
# "ridge-hill-community-solar",
# "https://account.powermarket.io/img/project/349-marketplace-profile-pic.png",
# {lat: 41.101154, lng: -73.807220},
# false
# ],
#
# Group breakdown:
#   1. Lat from LatLng
#   2. Lng from LatLng
#   3. ID (as a string)
#   4. Name
#   5. Slug
#   6. Image URL
#   7. Lat from object literal
#   8. Lng from object literal
#   9. Boolean value ("true" or "false")
pattern = re.compile(
    r"""\[\s*new\s+google\.maps\.LatLng\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*\{lat:\s*([-\d.]+)\s*,\s*lng:\s*([-\d.]+)\s*\}\s*,\s*(true|false)\s*\]""",
    re.DOTALL,
)

def parse_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    matches = pattern.findall(text)
    features = []

    for match in matches:
        try:
            # Use the object literal values (group 7 and 8) for the point geometry.
            lat = float(match[6])
            lng = float(match[7])
        except ValueError:
            continue  # Skip record if coordinates cannot be parsed

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lng, lat]  # GeoJSON expects [longitude, latitude]
            },
            "properties": {
                "id": match[2],
                "name": match[3],
                "slug": match[4],
                "image": match[5],
                "active": True if match[8].lower() == "true" else False
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

def main():
    # Set the input and output file paths.
    input_file = os.path.join(".", "Input", "communitysolarsites_powermarket.txt")
    output_file = os.path.join(".", "output", "communitysolar_powermarket.geojson")
    
    geojson = parse_file(input_file)
    
    # Ensure output directory exists.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    print(f"GeoJSON successfully written to {output_file}")

if __name__ == "__main__":
    main()