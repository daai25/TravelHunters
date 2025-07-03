from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def init_driver():
    options = Options()
    # options.add_argument("--headless")  # Uncomment to run without UI
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    service = Service()  # Assumes chromedriver is in PATH
    return webdriver.Chrome(service=service, options=options)

def scroll_to_bottom(driver, pause_time=1.0, max_attempts=20):
    last_height = driver.execute_script("return document.body.scrollHeight")
    attempts = 0

    while attempts < max_attempts:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        attempts += 1

def get_insider_guide_links(driver, explorer_url):
    driver.get(explorer_url)
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'a.article'))
    )
    
    scroll_to_bottom(driver)  # Ensure all guides are loaded
    
    links = set()
    articles = driver.find_elements(By.CSS_SELECTOR, 'a.article')
    for a in articles:
        href = a.get_attribute('href')
        if href and '/explorer/' in href:
            links.add(href)
    return list(links)

def get_explore_city_link(driver, guide_url):
    driver.get(guide_url)
    try:
        explore_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, 'Explore'))
        )
        return explore_link.get_attribute('href')
    except:
        return None

def get_activity_titles(driver, explore_city_url):
    driver.get(explore_city_url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'span'))
        )
        span_elements = driver.find_elements(By.CSS_SELECTOR, 'span')
        titles = [span.text.strip() for span in span_elements if span.text.strip()]
        return titles
    except:
        return []

def main():
    explorer_url = "https://www.getyourguide.com/explorer/"
    driver = init_driver()

    try:
        print("Collecting all Insider Guide links...")
        guide_links = get_insider_guide_links(driver, explorer_url)
        print(f"Found {len(guide_links)} insider guides.\n")

        all_titles = []
        for i, guide_url in enumerate(guide_links):
            print(f"[{i+1}/{len(guide_links)}] Visiting guide: {guide_url}")
            explore_link = get_explore_city_link(driver, guide_url)
            if explore_link:
                print(f"  → Found explore link: {explore_link}")
                titles = get_activity_titles(driver, explore_link)
                print(f"  → Found {len(titles)} activity titles.")
                all_titles.extend(titles)
            else:
                print("  → No explore link found.")
            time.sleep(0.5)

        unique_titles = list(set(all_titles))
        print("\n--- All Extracted Activity Titles ---")
        for idx, title in enumerate(unique_titles, 1):
            print(f"{idx}. {title}")

        print(f"\nTotal unique activity titles extracted: {len(unique_titles)}")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
