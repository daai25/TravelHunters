import requests

#Get api key from api_key.txt
with open("api_key.txt", "r") as file:
    API_KEY = file.read().strip()

# API_KEY = "DEIN_GOOGLE_API_KEY"
location = "47.384, 8.503"  # Berlin (Lat, Lng)
radius = 2000  # in Metern
type_place = "lodging,point_of_interest"  # Für Hotels

url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius={radius}&type={type_place}&key={API_KEY}"

response = requests.get(url)
data = response.json()

# Output from some Hoteldata
print(data["results"])
#save the data to a file
with open("hotels.json", "w") as file:
    file.write(response.text)

for hotel in data["results"]:
    print("Hotel:")
    name = hotel.get("name")
    address = hotel.get("vicinity")
    rating = hotel.get("rating")
    print(f"{name} — {address} — ⭐ {rating}")
    print("Critrics:")
    user_ratings_total = hotel.get("user_ratings_total")
    print("Location:")
    geometry = hotel.get("geometry", {}).get("location", {})
    lat = geometry.get("lat")
    lng = geometry.get("lng")
    print(f"Latitude: {lat}, Longitude: {lng}")
