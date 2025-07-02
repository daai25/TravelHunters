from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def scrape_getyourguide_titles(url):
    """
    Scrapes article titles from the given GetYourGuide Explorer URL using Selenium.

    Args:
        url (str): The URL of the GetYourGuide Explorer page.

    Returns:
        list: A list of extracted article titles.
    """
    # Configure Chrome options for headless mode
    chrome_options = Options()
    #chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no UI)
    chrome_options.add_argument("--no-sandbox") # Required for running headless Chrome in some environments (e.g., Docker)
    chrome_options.add_argument("--disable-gpu") # Recommended for headless mode
    chrome_options.add_argument("--window-size=1920,1080") # Set a consistent window size
    # Add a common User-Agent to mimic a regular browser
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

    # Set up the Chrome service
    # Ensure chromedriver is in your system's PATH, or provide the full path to it.
    # Example: service = Service('/usr/local/bin/chromedriver')
    service = Service() # Assumes chromedriver is in PATH

    driver = None # Initialize driver to None
    titles = [] # Initialize titles list

    try:
        # Initialize the WebDriver
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print(f"Navigating to {url}...")
        driver.get(url)

        # Wait for the page content to load dynamically.
        # We wait for an element with class 'article' and 'container' to be present.
        # This is a robust way to ensure the main content has rendered.
        print("Waiting for page content to load...")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.article.container'))
        )
        print("Page content loaded. Extracting titles...")

        # Find all h2 elements that are descendants of a div with class "article__info"
        # and also have the class "header".
        # We use find_elements (plural) to get a list of all matching elements.
        h2_elements = driver.find_elements(By.CSS_SELECTOR, '.article__info h2.header')

        if not h2_elements:
            print("No h2 elements found with the specified selector. The page structure might have changed or content is not yet visible.")
            # Optionally print the full page source for debugging if no elements are found
            # print("\n--- Full Page Source for Debugging ---")
            # print(driver.page_source)
            # print("--- End Full Page Source ---")
        else:
            for element in h2_elements:
                # Get the text content of each h2 element
                title = element.text.strip()
                if title:
                    titles.append(title)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if driver:
            print("Closing browser...")
            driver.quit() # Close the browser when done

    return titles

if __name__ == "__main__":
    target_url = "https://www.getyourguide.com/explorer/"
    extracted_titles = scrape_getyourguide_titles(target_url)

    if extracted_titles:
        print("\n--- Extracted Titles ---")
        for i, title in enumerate(extracted_titles):
            print(f"{i+1}. {title}")
        print(f"\nTotal titles extracted: {len(extracted_titles)}")
    else:
        print("No titles were extracted.")

