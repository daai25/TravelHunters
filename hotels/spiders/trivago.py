import scrapy
from scrapy.spiders import Spider
import re
import json


class TrivagoSpider(Spider):
    name = 'trivago'
    start_urls = [
        'https://www.trivago.com/en/srl?search=200-30765;dr-20250801-20250803',  # Zürich
        'https://www.trivago.com/en/srl?search=200-6894;dr-20250801-20250803',   # Bern
    ]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'HTTPERROR_ALLOWED_CODES': [404, 403, 500, 502, 503],
        'CONCURRENT_REQUESTS': 1,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 4,
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    }

    def parse(self, response):
        print(f"🌐 Parsing Trivago response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open('debug_trivago_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_trivago_response.html")
        
        # Trivago Selektoren
        hotel_selectors = [
            '[data-testid="item-wrapper"]',
            '.item-wrapper',
            '.accommodation-list-item',
            '.hotel-item',
            '[data-qa="hotel-item"]',
            '.item'
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
                    'source': 'Trivago',
                    'name': 'Page accessed successfully',
                    'link': response.url,
                    'rating': None,
                    'price': None,
                    'location': None,
                    'description': f'Page loaded with status {response.status}, check debug_trivago_response.html for content'
                }
                return
        
        # Extrahiere Hotel-Informationen
        for i, hotel in enumerate(hotels_found[:25]):  # Maximal 25 Hotels pro Seite
            try:
                # Hotel Name
                name_selectors = [
                    '[data-testid="item-name"]::text',
                    '.item-name::text',
                    '.hotel-name::text',
                    'h3::text',
                    'h2::text',
                    '.name::text'
                ]
                name = self.extract_with_selectors(hotel, name_selectors)
                
                # Hotel Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '[data-testid="item-link"]::attr(href)'
                ]
                link = self.extract_with_selectors(hotel, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Bewertung
                rating_selectors = [
                    '[data-testid="rating-value"]::text',
                    '.rating-value::text',
                    '.score::text',
                    '[data-qa="rating"]::text',
                    '.accommodation-rating::text'
                ]
                rating = self.extract_with_selectors(hotel, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Preis
                price_selectors = [
                    '[data-testid="recommended-price"]::text',
                    '.recommended-price::text',
                    '.price::text',
                    '.rate::text',
                    '[data-qa="price"]::text'
                ]
                price = self.extract_with_selectors(hotel, price_selectors)
                price = self.clean_price(price)
                
                # Standort
                location_selectors = [
                    '[data-testid="item-location"]::text',
                    '.item-location::text',
                    '.location::text',
                    '.neighborhood::text',
                    '.address::text'
                ]
                location = self.extract_with_selectors(hotel, location_selectors)
                
                # Beschreibung/Features
                desc_selectors = [
                    '.amenities::text',
                    '.description::text',
                    '.features::text'
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
                    'source': 'Trivago',
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
