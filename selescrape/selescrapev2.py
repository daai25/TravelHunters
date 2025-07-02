from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def init_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run without UI
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    service = Service()  # Assumes chromedriver is in PATH
    return webdriver.Chrome(service=service, options=chrome_options)

def scrape_explorer_links(driver, url, max_items=3):
    driver.get(url)
    print(f"Loading main explorer page: {url}")
    
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '.article__info h2.header'))
    )

    # Find all article containers with links
    article_cards = driver.find_elements(By.CSS_SELECTOR, 'a.article')

    results = []
    for card in article_cards:
        if len(results) >= max_items:
            break
        try:
            title_element = card.find_element(By.CSS_SELECTOR, 'h2.header')
            title = title_element.text.strip()
            link = card.get_attribute('href')
            if title and link:
                results.append({'title': title, 'url': link})
        except Exception as e:
            print(f"Error extracting a card: {e}")
    return results

def scrape_detail_page(driver, url):
    print(f"Visiting detail page: {url}")
    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'h1'))
        )
    except:
        print("Page took too long to load or content missing.")
        return {}

    data = {'url': url}

    # Title
    try:
        data['title'] = driver.find_element(By.CSS_SELECTOR, 'h1').text.strip()
    except:
        data['title'] = None

    # Subtitle
    try:
        subtitle = driver.find_element(By.CSS_SELECTOR, 'h2')
        data['subtitle'] = subtitle.text.strip()
    except:
        data['subtitle'] = None

    # Price
    try:
        price = driver.find_element(By.CSS_SELECTOR, '[data-testid="price-block"]')
        data['price'] = price.text.strip()
    except:
        data['price'] = None

    # Scroll to bottom to load all sections (important!)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    # Get all text content from all sections on the page
    try:
        main_content = driver.find_element(By.CSS_SELECTOR, '[data-testid="product-detail-page"]')
        data['full_description'] = main_content.text.strip()
    except:
        data['full_description'] = None

    return data



def main():
    start_url = "https://www.getyourguide.com/explorer/"
    driver = init_driver()

    try:
        # Step 1: Get only 3 linked titles and URLs from explorer page
        links = scrape_explorer_links(driver, start_url, max_items=3)
        print(f"Collected {len(links)} items.")

        # Step 2: Visit each link and scrape details
        detailed_results = []
        for i, link_info in enumerate(links):
            print(f"\nScraping {i+1}/{len(links)}: {link_info['title']}")
            detail_data = scrape_detail_page(driver, link_info['url'])
            detail_data['listing_title'] = link_info['title']
            detailed_results.append(detail_data)
            time.sleep(0.5)

        # Step 3: Display collected data
        print("\n--- Scraped Data ---")
        for i, item in enumerate(detailed_results):
            print(f"\n{i+1}. {item['listing_title']}")
            print(f"URL: {item['url']}")
            print(f"Title: {item.get('title')}")
            print(f"Subtitle: {item.get('subtitle')}")
            print(f"Price: {item.get('price')}")
            full_description = item.get('full_description')
            desc = item.get('full_description')
            if desc:
                print(f"Full Description (truncated):\n{desc[:500]}...\n")
            else:
                print("Full Description: N/A\n")


    finally:
        print("Closing browser...")
        driver.quit()

if __name__ == "__main__":
    main()
