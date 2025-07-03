import scrapy
from scrapy.spiders import Spider
import re
import json


class ExpediaSpider(Spider):
    name = 'expedia'
    start_urls = [
        'https://www.expedia.com/Hotel-Search?destination=Zurich,%20Switzerland&startDate=2025-08-01&endDate=2025-08-03&rooms=1&adults=2',
        'https://www.expedia.com/Hotel-Search?destination=Bern,%20Switzerland&startDate=2025-08-01&endDate=2025-08-03&rooms=1&adults=2',
    ]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 3,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'HTTPERROR_ALLOWED_CODES': [404, 403, 500, 502, 503],
        'CONCURRENT_REQUESTS': 1,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 2,
        'AUTOTHROTTLE_MAX_DELAY': 8,
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    }

    def parse(self, response):
        print(f"🌐 Parsing Expedia response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open('debug_expedia_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_expedia_response.html")
        
        # Expedia Selektoren
        hotel_selectors = [
            '[data-stid="section-results"] [data-stid="lodging-card"]',
            '.uitk-card',
            '[data-testid="property-listing"]',
            '.property-listing',
            '[data-stid="property-listing"]',
            '.hotel-wrap',
            '[data-stid="lodging-card"]'
        ]
        
        hotels_found = []
        for selector in hotel_selectors:
            hotels = response.css(selector)
            if hotels:
                print(f"✅ Found {len(hotels)} hotels with selector: {selector}")
                hotels_found = hotels
                break
        
        if not hotels_found:
            print("❌ No hotels found with any selector. Trying to extract any hotel-related content...")
            
            # Fallback: Suche nach Hotel-Links
            hotel_links = response.css('a[href*="hotel"], a[href*="Hotel"]::attr(href)').getall()
            print(f"Found {len(hotel_links)} potential hotel links")
            
            if not hotel_links:
                # Als letzter Ausweg: Basis-Info zurückgeben
                yield {
                    'source': 'Expedia',
                    'name': 'Page accessed successfully',
                    'link': response.url,
                    'rating': None,
                    'price': None,
                    'location': None,
                    'description': f'Page loaded with status {response.status}, check debug_expedia_response.html for content'
                }
                return
        
        # Extrahiere Hotel-Informationen
        for i, hotel in enumerate(hotels_found[:25]):  # Maximal 25 Hotels pro Seite
            try:
                # Hotel Name
                name_selectors = [
                    '[data-stid="content-hotel-title"]::text',
                    '.uitk-heading::text',
                    'h3::text',
                    '.property-name::text',
                    '[data-testid="title"]::text',
                    'h2::text'
                ]
                name = self.extract_with_selectors(hotel, name_selectors)
                
                # Hotel Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '[data-stid="open-hotel-information"]::attr(href)'
                ]
                link = self.extract_with_selectors(hotel, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Bewertung
                rating_selectors = [
                    '[data-stid="content-review-text"]::text',
                    '.uitk-text[data-stid="content-review-text"]::text',
                    '.review-score::text',
                    '[aria-label*="rating"]::text',
                    '.star-rating::attr(aria-label)'
                ]
                rating = self.extract_with_selectors(hotel, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Preis
                price_selectors = [
                    '[data-stid="price-display-field"]::text',
                    '.uitk-text[data-stid="price-display-field"]::text',
                    '.price::text',
                    '.rate-info::text'
                ]
                price = self.extract_with_selectors(hotel, price_selectors)
                price = self.clean_price(price)
                
                # Standort
                location_selectors = [
                    '[data-stid="content-hotel-neighborhood"]::text',
                    '.uitk-text[data-stid="content-hotel-location"]::text',
                    '.location::text',
                    '.neighborhood::text',
                    '.locality::text'
                ]
                location = self.extract_with_selectors(hotel, location_selectors)
                
                # Beschreibung/Features
                desc_selectors = [
                    '[data-stid="content-hotel-amenities"]::text',
                    '.uitk-text::text',
                    '.amenities::text',
                    '.description::text'
                ]
                description_parts = hotel.css(' '.join(desc_selectors)).getall()
                description = ' '.join(description_parts[:3]) if description_parts else None
                
                # Fallback für Name wenn leer
                if not name:
                    name = f"Hotel {i+1}"
                
                # Fallback für Link wenn leer
                if not link:
                    link = response.url
                
                hotel_data = {
                    'source': 'Expedia',
                    'name': name,
                    'link': link,
                    'rating': rating,
                    'price': price,
                    'location': location,
                    'description': description
                }
                
                print(f"🏨 Hotel {i+1}: {name} - Rating: {rating} - Price: {price}")
                yield hotel_data
                
            except Exception as e:
                print(f"❌ Error processing hotel {i+1}: {str(e)}")
                continue

    def extract_with_selectors(self, element, selectors):
        """Versuche verschiedene Selektoren bis einer funktioniert"""
        for selector in selectors:
            try:
                result = element.css(selector).get()
                if result and result.strip():
                    return result.strip()
            except:
                continue
        return None

    def clean_rating(self, rating):
        """Bereinige Bewertung"""
        if not rating:
            return None
        
        # Extrahiere Zahlen aus dem Rating-Text
        numbers = re.findall(r'(\d+\.?\d*)', rating)
        if numbers:
            rating_num = float(numbers[0])
            # Normalisiere auf 10er Skala falls nötig
            if rating_num <= 5:
                rating_num = rating_num * 2
            return str(rating_num)
        return None

    def clean_price(self, price):
        """Bereinige Preis"""
        if not price:
            return None
        
        # Entferne Währungssymbole und extrahiere Zahlen
        numbers = re.findall(r'(\d+)', price.replace(',', ''))
        if numbers:
            return numbers[0]
        return None
