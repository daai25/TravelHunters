import scrapy
from scrapy.spiders import Spider
import re
import json


class AttractionsSpider(Spider):
    name = 'attractions'
    start_urls = [
        'https://www.tripadvisor.com/Attractions-g188113-Activities-Zurich.html',
        'https://www.tripadvisor.com/Attractions-g188099-Activities-Bern.html',
        'https://www.tripadvisor.com/Attractions-g188064-Activities-Geneva.html',
    ]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 3,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'HTTPERROR_ALLOWED_CODES': [404, 403, 500, 502, 503],
        'CONCURRENT_REQUESTS': 1,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 2,
        'AUTOTHROTTLE_MAX_DELAY': 6,
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    }

    def parse(self, response):
        print(f"🌐 Parsing Attractions response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Extract city from URL
        city = self.extract_city_from_url(response.url)
        
        # Debug: Save HTML
        with open('debug_attractions_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_attractions_response.html")
        
        # TripAdvisor selectors for attractions
        attraction_selectors = [
            '[data-automation="cardWrapper"]',
            '.listing_title',
            '.attraction_element',
            '.result-card',
            '[data-test-target="shops-attraction-card"]'
        ]
        
        attractions_found = []
        for selector in attraction_selectors:
            attractions = response.css(selector)
            if attractions:
                print(f"✅ Found {len(attractions)} attractions with selector: {selector}")
                attractions_found = attractions
                break
        
        if not attractions_found:
            print("❌ No attractions found with specific selectors. Searching for general content...")
            
            # Fallback: Search for links with attraction indicators
            attraction_links = response.css('a[href*="Attraction"], a[href*="Activities"]::attr(href)').getall()
            print(f"Found {len(attraction_links)} potential attraction links")
            
            if not attraction_links:
                # Last resort: Basic info
                yield {
                    'source': 'TripAdvisor-Attractions',
                    'name': 'Page accessed successfully',
                    'link': response.url,
                    'rating': None,
                    'price': None,
                    'location': city,
                    'description': f'Page loaded with status {response.status}, check debug_attractions_response.html for content',
                    'category': 'attraction'
                }
                return
        
        # Extract attraction information
        for i, attraction in enumerate(attractions_found[:20]):
            try:
                # Attraction Name
                name_selectors = [
                    '[data-automation="name"]::text',
                    '.listing_title a::text',
                    'h3 a::text',
                    'h2 a::text',
                    '.result-title::text'
                ]
                name = self.extract_with_selectors(attraction, name_selectors)
                
                # Attraction Link
                link_selectors = [
                    'a::attr(href)',
                    '[data-automation="name"]::attr(href)',
                    '.listing_title a::attr(href)'
                ]
                link = self.extract_with_selectors(attraction, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Rating
                rating_selectors = [
                    '[data-automation="rating"]::attr(aria-label)',
                    '.ui_bubble_rating::attr(alt)',
                    '.rating::attr(title)',
                    '.stars::attr(title)'
                ]
                rating = self.extract_with_selectors(attraction, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Price (if available)
                price_selectors = [
                    '.price::text',
                    '[data-automation="price"]::text',
                    '.attraction_pricing::text'
                ]
                price = self.extract_with_selectors(attraction, price_selectors)
                price = self.clean_price(price)
                
                # Description/Category
                desc_selectors = [
                    '.attraction_clarification::text',
                    '.result-description::text',
                    '[data-automation="description"]::text'
                ]
                description = self.extract_with_selectors(attraction, desc_selectors)
                
                # Fallback for name
                if not name:
                    all_texts = attraction.css('a::text, h3::text, h2::text').getall()
                    for text in all_texts:
                        text = text.strip()
                        if len(text) > 5 and len(text) < 100:
                            name = text
                            break
                    if not name:
                        name = f"Attraction {i+1}"
                
                # Fallback for link
                if not link:
                    link = response.url
                
                attraction_data = {
                    'source': 'TripAdvisor-Attractions',
                    'name': name,
                    'link': link,
                    'rating': rating,
                    'price': price,
                    'location': city,
                    'description': description,
                    'category': 'attraction'
                }
                
                print(f"🎯 Attraction {i+1}: {name} - Rating: {rating} - City: {city}")
                yield attraction_data
                
            except Exception as e:
                print(f"❌ Error processing attraction {i+1}: {str(e)}")
                continue

    def extract_city_from_url(self, url):
        """Extract city name from TripAdvisor URL"""
        city_mappings = {
            'Zurich': 'Zürich, Switzerland',
            'Bern': 'Bern, Switzerland',
            'Geneva': 'Geneva, Switzerland'
        }
        
        for city_key, city_full in city_mappings.items():
            if city_key in url:
                return city_full
        
        return 'Switzerland'

    def extract_with_selectors(self, element, selectors):
        """Try different selectors until one works"""
        for selector in selectors:
            try:
                result = element.css(selector).get()
                if result and result.strip():
                    return result.strip()
            except:
                continue
        return None

    def clean_rating(self, rating):
        """Clean rating data"""
        if not rating:
            return None
        
        # Extract numbers from rating text
        numbers = re.findall(r'(\d+\.?\d*)', rating)
        if numbers:
            return str(float(numbers[0]))
        return None

    def clean_price(self, price):
        """Clean price data"""
        if not price:
            return None
        
        # Remove currency symbols and extract numbers
        numbers = re.findall(r'(\d+)', price.replace(',', ''))
        if numbers:
            return numbers[0]
        return None
