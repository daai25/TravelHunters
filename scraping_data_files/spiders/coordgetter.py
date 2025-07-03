import json
import time
from geopy.geocoders import Nominatim

# Initialize geocoder
geolocator = Nominatim(user_agent="destination_geocoder")

# Read the destinations with UTF-16 encoding
with open('destinations.json', 'r', encoding='utf-16') as f:
    destinations = [line.strip() for line in f if line.strip()]

print(f"Found {len(destinations)} destinations")

# Geocode each destination
geocoded_destinations = []
for i, destination in enumerate(destinations):
    print(f"Processing {i+1}/{len(destinations)}: {destination}")
    try:
        location = geolocator.geocode(destination)
        if location:
            geocoded_destinations.append({
                "name": destination,
                "latitude": location.latitude,
                "longitude": location.longitude
            })
        else:
            geocoded_destinations.append({
                "name": destination,
                "latitude": None,
                "longitude": None,
                "error": "Location not found"
            })
    except Exception as e:
        geocoded_destinations.append({
            "name": destination,
            "latitude": None,
            "longitude": None,
            "error": str(e)
        })
    
    # Be nice to the API - add a small delay
    #time.sleep(1)

# Save to new file
with open('destinations_with_coordinates2.json', 'w', encoding='utf-8') as f:
    json.dump(geocoded_destinations, f, indent=2, ensure_ascii=False)

print(f"Processed {len(geocoded_destinations)} destinations")