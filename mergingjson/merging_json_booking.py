import json
import requests
import time
import os

# Aktuellen Pfad erkennen (wie vorher)
script_dir = os.path.dirname(os.path.abspath(__file__))

# API-Key laden
api_key_path = os.path.join(script_dir, "api_key.txt")
with open(api_key_path, "r") as file:
    API_KEY = file.read().strip()

def get_coordinates_by_name_and_location(hotel_name, location):
    query = f"{hotel_name}, {location}"  # z. B. "The Ritz-Carlton, Berlin"
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "geometry",
        "key": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("candidates"):
        coords = data["candidates"][0]["geometry"]["location"]
        print(f"✅ {hotel_name} → Lat: {coords['lat']}, Lng: {coords['lng']}")
        return coords["lat"], coords["lng"]
    else:
        print(f"❌ Keine Koordinaten gefunden für: {query}")
        return None, None

def enrich_hotels_with_coordinates(input_path, output_path):
    # JSON einlesen
    with open(input_path, "r", encoding="utf-8") as f:
        hotels = json.load(f)

    # Für jeden Eintrag: Koordinaten holen
    for hotel in hotels:
        name = hotel.get("name", "")
        location = hotel.get("location") or hotel.get("city") or hotel.get("destination", "")
        lat, lng = get_coordinates_by_name_and_location(name, location)
        hotel["latitude"] = lat
        hotel["longitude"] = lng
        time.sleep(0.2)  # API nicht überlasten

    # Speichern
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hotels, f, ensure_ascii=False, indent=2)

    print(f"✅ Fertig! Datei gespeichert unter: {output_path}")

# ▶ Start
enrich_hotels_with_coordinates(
    "../data_acquisition/json_final/booking_worldwide.json",
    "../data_acquisition/json_final/booking_worldwide_enriched.json"
)
