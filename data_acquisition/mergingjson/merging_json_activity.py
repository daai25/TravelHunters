import json
import requests
import time
import os

# Get current script directory to handle relative paths correctly
script_dir = os.path.dirname(os.path.abspath(__file__))

#Get api key from api_key.txt
api_key_path = os.path.join(script_dir, "api_key.txt")
with open(api_key_path, "r") as file:
    API_KEY = file.read().strip()

def get_coordinates(location):
    query = f"{location}"
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
        print(f"✅ Koordinaten gefunden für: {query} -> Lat: {coords['lat']}, Lng: {coords['lng']}")
        return coords["lat"], coords["lng"]
    else:
        print(f"❌ Keine Koordinaten gefunden für: {query}")
        return None, None

def enrich_json_file(input_path, output_path):
    # Schritt 1: Lade die JSON-Daten
    with open(input_path, "r", encoding="utf-8") as f:
        activities = json.load(f)

    # Schritt 2: Ergänze jede Aktivität mit Koordinaten
    for activity in activities:
        # name = activity.get("name")
        location = activity.get("location", activity.get("destination", ""))
        lat, lng = get_coordinates(location)
        activity["latitude"] = lat
        activity["longitude"] = lng
        time.sleep(0.2)  # API nicht überlasten

    # Schritt 3: Speichere die neuen Daten
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(activities, f, ensure_ascii=False, indent=2)

    print(f"✅ Fertig! Datei gespeichert unter: {output_path}")

# ▶ Ausführen
enrich_json_file("../json_final/activities_worldwide.json", "../data_acquisition/json_final/activities_worldwide_enriched.json")
