import os
import time
import requests
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

OUTPUT_DIR = os.path.expanduser("~/Desktop/travelhunters/selescrape/BYG_images")

def init_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--log-level=3")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0")
    return webdriver.Chrome(service=Service(), options=options)

def get_insider_guide_links(driver, explorer_url, limit=5):
    driver.get(explorer_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'a.article'))
    )
    seen = set()
    links = []
    for a in driver.find_elements(By.CSS_SELECTOR, 'a.article'):
        href = a.get_attribute('href')
        if href and '/explorer/' in href and href not in seen:
            seen.add(href)
            links.append(href)
            if len(links) >= limit:
                break
    return links

import re

def extract_city_country(url):
    slug = url.split('/explorer/')[-1].strip('/')
    parts = slug.split('-')
    
    # Remove invalid characters and sanitize
    def clean(text):
        text = re.sub(r'[\\/:"*?<>|]+', '', text)  # remove invalid filename chars
        text = text.replace(',', '')               # remove commas
        text = re.sub(r'\s+', ' ', text).strip()   # normalize spaces
        return text.title()

    if '-' in slug:
        if '/' in slug:
            # E.g., travel-inspiration/best-places-for-cheese
            parts = slug.split('/')
            city = clean(parts[-1].replace('-', ' '))
            country = clean(parts[0].replace('-', ' '))
        else:
            city = clean(' '.join(parts[:-1]))
            country = clean(parts[-1])
    else:
        city = clean(slug.replace('-', ' '))
        country = ''

    return f"{city}, {country}".strip(', ')


def get_first_image(driver, url):
    driver.get(url)
    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'img'))
        )
        for img in driver.find_elements(By.CSS_SELECTOR, 'img'):
            src = img.get_attribute('src')
            if src and src.startswith('http') and 'svg' not in src:
                return src
    except:
        return None
    return None

def download_and_convert_image(url, filename):
    try:
        r = requests.get(url, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(filename, "JPEG", quality=85)
        return True
    except Exception as e:
        print(f"[!] Failed to save {filename}: {e}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    driver = init_driver()

    try:
        guide_links = get_insider_guide_links(driver, "https://www.getyourguide.com/explorer/", limit=10)
        for i, url in enumerate(guide_links):
            print(f"[{i+1}] {url}")
            city_name = extract_city_country(url)
            img_url = get_first_image(driver, url)
            if img_url:
                out_path = os.path.join(OUTPUT_DIR, f"{city_name}.jpg")
                if download_and_convert_image(img_url, out_path):
                    print(f"✅ Saved: {out_path}")
                else:
                    print(f"[!] Error downloading {city_name}")
            else:
                print(f"[!] No image found for {city_name}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
