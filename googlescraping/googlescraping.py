import requests
import json
import time
import os

# Get current script directory to handle relative paths correctly
script_dir = os.path.dirname(os.path.abspath(__file__))

#Get api key from api_key.txt
api_key_path = os.path.join(script_dir, "api_key.txt")
with open(api_key_path, "r") as file:
    API_KEY = file.read().strip()

# Load destinations with coordinates
destinations_file = os.path.join(script_dir, "../Destinations_and_Coords/destinations_with_coordinates.json")
with open(destinations_file, "r") as file:
    destinations = json.load(file)

# Filter destinations that have valid coordinates
valid_destinations = [dest for dest in destinations if dest.get("latitude") and dest.get("longitude")]

print(f"Found {len(valid_destinations)} destinations with valid coordinates")

radius = 5000  # in Metern (increased for better coverage)
type_place = "lodging,point_of_interest"  # Für Hotels

all_hotels = []

for i, destination in enumerate(valid_destinations[:10]):  # Limit to first 10 for testing
    print(f"\n🌍 Processing {i+1}/{min(10, len(valid_destinations))}: {destination['name']}")
    
    location = f"{destination['latitude']}, {destination['longitude']}"
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius={radius}&type={type_place}&key={API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if data.get("status") == "OK":
        hotels_in_city = data["results"]
        print(f"  ✅ Found {len(hotels_in_city)} places in {destination['name']}")
        
        # Add destination info to each hotel
        for hotel in hotels_in_city:
            hotel["destination_name"] = destination["name"]
            hotel["destination_lat"] = destination["latitude"]
            hotel["destination_lng"] = destination["longitude"]
        
        all_hotels.extend(hotels_in_city)
    else:
        print(f"  ❌ Error for {destination['name']}: {data.get('status')}")
    
    # Add delay to avoid rate limiting
    time.sleep(1)

# Save all hotels data to file
hotels_output_path = os.path.join(script_dir, "hotels.json")
print(f"\n💾 Saving {len(all_hotels)} hotels to {hotels_output_path}")
with open(hotels_output_path, "w") as file:
    json.dump(all_hotels, file, indent=2)

# Print summary statistics
print(f"\n📊 Summary:")
print(f"Total hotels found: {len(all_hotels)}")

# Group by destination
destinations_count = {}
for hotel in all_hotels:
    dest_name = hotel.get("destination_name", "Unknown")
    destinations_count[dest_name] = destinations_count.get(dest_name, 0) + 1

print(f"Hotels per destination:")
for dest, count in sorted(destinations_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {dest}: {count} hotels")

# Print some example hotel data
print(f"\n🏨 Example hotels:")
for i, hotel in enumerate(all_hotels[:5]):  # Show first 5 hotels
    name = hotel.get("name", "Unknown")
    address = hotel.get("vicinity", "Unknown address")
    rating = hotel.get("rating", "No rating")
    destination = hotel.get("destination_name", "Unknown destination")
    user_ratings_total = hotel.get("user_ratings_total", 0)
    
    geometry = hotel.get("geometry", {}).get("location", {})
    lat = geometry.get("lat", "Unknown")
    lng = geometry.get("lng", "Unknown")
    
    print(f"  {i+1}. {name} in {destination}")
    print(f"     📍 {address}")
    print(f"     ⭐ {rating} ({user_ratings_total} reviews)")
    print(f"     🌍 Lat: {lat}, Lng: {lng}")
    print()
