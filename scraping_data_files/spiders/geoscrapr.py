import json
import math

# 📁 Input/output paths
INPUT_JSON = "/users/zachfelber/desktop/travelhunters/data_acquisition/json_final/unique_cities.json"
OUTPUT_JSON = "/users/zachfelber/desktop/geo_cities/geo_cities.json"

DISTANCE_M = 100  # Distance from center point (≈100 meters)
LAT_OFFSET = DISTANCE_M / 111000  # Latitude conversion (~0.0009 degrees)

def generate_all_points(city, lat, lon):
    # Longitude correction using latitude
    LON_OFFSET = DISTANCE_M / (111000 * math.cos(math.radians(lat)))
    
    return [
        {"city": city, "latitude": round(lat, 6), "longitude": round(lon, 6), "direction": "center"},
        {"city": city, "latitude": round(lat + LAT_OFFSET, 6), "longitude": round(lon, 6), "direction": "north"},
        {"city": city, "latitude": round(lat - LAT_OFFSET, 6), "longitude": round(lon, 6), "direction": "south"},
        {"city": city, "latitude": round(lat, 6), "longitude": round(lon + LON_OFFSET, 6), "direction": "east"},
        {"city": city, "latitude": round(lat, 6), "longitude": round(lon - LON_OFFSET, 6), "direction": "west"},
        {"city": city, "latitude": round(lat + LAT_OFFSET, 6), "longitude": round(lon + LON_OFFSET, 6), "direction": "northeast"},
        {"city": city, "latitude": round(lat + LAT_OFFSET, 6), "longitude": round(lon - LON_OFFSET, 6), "direction": "northwest"},
        {"city": city, "latitude": round(lat - LAT_OFFSET, 6), "longitude": round(lon + LON_OFFSET, 6), "direction": "southeast"},
        {"city": city, "latitude": round(lat - LAT_OFFSET, 6), "longitude": round(lon - LON_OFFSET, 6), "direction": "southwest"},
    ]

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        cities = json.load(f)

    all_coords = []

    for entry in cities:
        city = entry.get("city")
        lat = entry.get("latitude")
        lon = entry.get("longitude")

        if not all([city, lat, lon]):
            continue

        expanded = generate_all_points(city, lat, lon)
        all_coords.extend(expanded)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_coords, f, indent=2)

    print(f"✅ Generated {len(all_coords)} coordinate entries across {len(cities)} cities.")
    print(f"📁 Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
