from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

def init_driver():
    options = Options()
    options.add_argument("--headless=new")  # Use headless mode for speed
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    service = Service()
    return webdriver.Chrome(service=service, options=options)

def scroll_and_collect_guide_links(driver, explorer_url):
    driver.get(explorer_url)
    wait = WebDriverWait(driver, 8)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a.article')))
    
    seen = set()
    guide_links = []
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        articles = driver.find_elements(By.CSS_SELECTOR, 'a.article')
        new_found = 0
        for a in articles:
            href = a.get_attribute('href')
            if href and '/explorer/' in href and href not in seen:
                seen.add(href)
                guide_links.append(href)
                new_found += 1

        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(0.6)  # Lowered delay

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height and new_found == 0:
            break
        last_height = new_height

    return guide_links

def extract_city_country_from_url(url):
    slug = url.split("/explorer/")[-1].strip("/")
    if not slug:
        return "Unknown"

    parts = slug.replace('-', ' ').split('/')
    city_part = parts[-1].title()
    country_part = parts[0].title() if len(parts) > 1 else ""
    city_country = f"{city_part}, {country_part}".strip(", ")
    return re.sub(r'[\\/:*?"<>|]', '', city_country)

def scrape_single_image_url(driver, url):
    try:
        driver.get(url)
        WebDriverWait(driver, 6).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'img'))
        )
        for img in driver.find_elements(By.CSS_SELECTOR, 'img'):
            src = img.get_attribute('src')
            if src and src.startswith("http") and 'svg' not in src:
                return src
    except:
        pass
    return None

def main():
    explorer_url = "https://www.getyourguide.com/explorer/"
    driver = init_driver()

    try:
        guide_links = scroll_and_collect_guide_links(driver, explorer_url)
        print(f"\n✅ Found {len(guide_links)} guide pages.\n")

        for i, guide_url in enumerate(guide_links, 1):
            city_country = extract_city_country_from_url(guide_url)
            print(f"[{i}] {city_country}: {guide_url}")
            img_url = scrape_single_image_url(driver, guide_url)
            if img_url:
                print(f"    ↳ Image: {img_url}")
            else:
                print("    ↳ No image found.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
