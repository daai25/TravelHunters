import requests
import os
from pathlib import Path
from duckduckgo_search import DDGS

def get_images_for_city(city_search_term, count=50):
    try:
        with DDGS() as ddgs:
            results = ddgs.images(keywords=city_search_term)
            return [img["image"] for _, img in zip(range(count), results)]
    except Exception as e:
        print(f"❌ DuckDuckGo error for {city_search_term}: {e}")
        return []

# 🌍 Manually set your city here
city_name = "Noonu"  # ← Change this to any city you like
search_term = f"vacation to {city_name}"
clean_name = city_name.replace(" ", "_")

# 📁 Destination folder
dest_folder = r"C:\Users\evanb\TravelHunters\cityImages_additional"
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

# 🔍 Search and download loop
print(f"🔍 Searching images for: {search_term}")
image_urls = get_images_for_city(search_term, count=50)
for i, url in enumerate(image_urls, start=101):
    filename = f"{clean_name}_{str(i)}.jpg"
    save_path = Path(dest_folder) / filename
    download_image(url, save_path)
