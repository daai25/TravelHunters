def main():
    start_url = "https://www.getyourguide.com/explorer/"
    driver = init_driver()

    try:
        # Step 1: Get all linked titles and URLs from explorer page
        links = scrape_explorer_links(driver, start_url)
        print(f"Found {len(links)} items total.")
        
        # Limit to first 10
        links = links[:10]
        print("Limiting scrape to first 10 items.")

        # Step 2: Visit each link and scrape details
        detailed_results = []
        for i, link_info in enumerate(links):
            print(f"\nScraping {i+1}/{len(links)}: {link_info['title']}")
            detail_data = scrape_detail_page(driver, link_info['url'])
            detail_data['listing_title'] = link_info['title']
            detailed_results.append(detail_data)
            time.sleep(2)  # Be polite to the server

        # Step 3: Display collected data
        print("\n--- Scraped Data ---")
        for i, item in enumerate(detailed_results):
            print(f"\n{i+1}. {item['listing_title']}")
            print(f"URL: {item.get('url')}")
            print(f"Title: {item.get('title')}")
            print(f"Subtitle: {item.get('subtitle')}")
            print(f"Price: {item.get('price')}")
            desc = item.get('description')
            if desc:
                print(f"Description: {desc[:200]}...")
            else:
                print("Description: N/A")

    finally:
        print("Closing browser...")
        driver.quit()
