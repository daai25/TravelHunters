import json
from geopy.geocoders import Nominatim
from time import sleep

# Load data from the original JSON file
input_file = r"c:\Users\evanb\TravelHunters\Destinations_and_Coords\booking_worldwide.json"
output_file = r"c:\Users\evanb\TravelHunters\Destinations_and_Coords\locations_with_coordinates.json"


with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract unique locations
locations = sorted({entry["location"] for entry in data if "location" in entry})

# Set up the geocoder
geolocator = Nominatim(user_agent="location_geocoder_script")

# Dictionary to store results
location_coords = []

# Loop through each unique location and geocode
for loc in locations:
    try:
        print(f"Geocoding: {loc}")
        geo = geolocator.geocode(loc)
        if geo:
            location_coords.append({
                "location": loc,
                "latitude": geo.latitude,
                "longitude": geo.longitude
            })
        else:
            location_coords.append({
                "location": loc,
                "latitude": None,
                "longitude": None
            })
    except Exception as e:
        print(f"Error geocoding {loc}: {e}")
        location_coords.append({
            "location": loc,
            "latitude": None,
            "longitude": None
        })
    sleep(1)  # Respect Nominatim's rate limit

# Save to new JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(location_coords, f, ensure_ascii=False, indent=2)

print(f"\n✅ Geocoding complete. Results saved to '{output_file}'")
