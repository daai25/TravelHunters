import requests

#Get api key from api_key.txt
with open("api_key.txt", "r") as file:
    API_KEY = file.read().strip()

# API_KEY = "DEIN_GOOGLE_API_KEY"
location = "47.384, 8.503"  # Berlin (Lat, Lng)
radius = 2000  # in Metern
type_place = "lodging"  # Für Hotels

url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius={radius}&type={type_place}&key={API_KEY}"

response = requests.get(url)
data = response.json()

# Ausgabe einiger Hoteldaten
for hotel in data["results"]:
    name = hotel.get("name")
    address = hotel.get("vicinity")
    rating = hotel.get("rating")
    print(f"{name} — {address} — ⭐ {rating}")
