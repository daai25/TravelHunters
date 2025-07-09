import json
import requests
import os
from pathlib import Path
from duckduckgo_search import DDGS

def get_images_for_city(city_name, count=100):
    try:
        with DDGS() as ddgs:
            results = ddgs.images(keywords=city_name)
            return [img["image"] for _, img in zip(range(count), results)]
    except Exception as e:
        print(f"❌ DuckDuckGo error for {city_name}: {e}")
        return []


# Load city data
json_path = "C:/Users/evanb/Downloads/unique_cities.json"
with open(json_path, "r", encoding="utf-8") as file:
    cities = json.load(file)

# Set destination folder
dest_folder = r"C:\Users\evanb\TravelHunters\cityImages"
os.makedirs(dest_folder, exist_ok=True)

def download_image(url, save_path):
    try:
        img_data = requests.get(url, timeout=10)
        img_data.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(img_data.content)
        print(f"✅ Saved: {save_path}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

# Main loop
for city in cities:
    raw_name = city["city"]
    clean_name = raw_name.replace(" ", "_")
    print(f"🔍 Searching images for: {raw_name}")
    image_urls = get_images_for_city(raw_name, count=100)
    for i, url in enumerate(image_urls):
        filename = f"{clean_name}_{str(i+1).zfill(3)}.jpg"
        save_path = Path(dest_folder) / filename
        download_image(url, save_path)
