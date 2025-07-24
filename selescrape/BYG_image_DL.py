import os
import re
import requests
from PIL import Image
from io import BytesIO

INPUT_FILE = "/Users/zachfelber/Desktop/images.txt"  # Your text file (not real JSON)
OUTPUT_DIR = os.path.expanduser("~/Desktop/BYG_images")

def parse_custom_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    data = {}
    city = None

    for line in lines:
        line = line.strip()
        if line.startswith('['):
            match = re.match(r'\[\d+\]\s+(.+?):\s+https?://', line)
            if match:
                city = match.group(1).strip()
        elif line.startswith('↳ Image:') and city:
            img_url = line.split('↳ Image:')[-1].strip()
            data[city] = img_url
            city = None  # Reset to avoid pairing one image with multiple cities

    return data

def download_image(url, city_name):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        filename = os.path.join(OUTPUT_DIR, f"{sanitize_filename(city_name)}.jpg")
        img.save(filename, "JPEG", quality=85)
        print(f"✅ Saved: {filename}")
    except Exception as e:
        print(f"[!] Failed to download {city_name}: {e}")

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    location_to_url = parse_custom_file(INPUT_FILE)

    for city, img_url in location_to_url.items():
        print(f"Downloading image for {city}...")
        download_image(img_url, city)

if __name__ == "__main__":
    main()
