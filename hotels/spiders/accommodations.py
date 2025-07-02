import scrapy
from scrapy.spiders import Spider
import re
import json


class AccommodationsSpider(Spider):
    name = 'accommodations'
    start_urls = [
        'https://www.hostelworld.com/search?search_keywords=Zurich,%20Switzerland',
        'https://www.hostelworld.com/search?search_keywords=Bern,%20Switzerland',
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
        print(f"🌐 Parsing Accommodations response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open('debug_accommodations_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_accommodations_response.html")
        
        # Hostelworld/Accommodation Selektoren
        accommodation_selectors = [
            '.property-card',
            '.accommodation-item',
            '.hostel-card',
            '.property-item',
            '[data-cy="property-card"]',
            '.listing-item'
        ]
        
        accommodations_found = []
        for selector in accommodation_selectors:
            accommodations = response.css(selector)
            if accommodations:
                print(f"✅ Found {len(accommodations)} accommodations with selector: {selector}")
                accommodations_found = accommodations
                break
        
        if not accommodations_found:
            print("❌ No accommodations found with specific selectors. Searching for general content...")
            
            # Fallback: Suche nach Hotel/Accommodation-ähnlichen Links
            fallback_selectors = [
                'div[class*="property"]',
                'div[class*="hotel"]',
                'div[class*="accommodation"]',
                'div[class*="listing"]',
                'article',
                'li[class*="item"]'
            ]
            
            for selector in fallback_selectors:
                accommodations = response.css(selector)
                if len(accommodations) > 3:  # Mindestens ein paar Ergebnisse
                    print(f"✅ Found {len(accommodations)} potential accommodations with fallback selector: {selector}")
                    accommodations_found = accommodations[:15]  # Begrenzen auf 15
                    break
            
            if not accommodations_found:
                # Als letzter Ausweg: Basis-Info zurückgeben
                yield {
                    'source': 'Accommodations',
                    'name': 'Page accessed successfully',
                    'link': response.url,
                    'rating': None,
                    'price': None,
                    'location': self.get_location_from_url(response.url),
                    'description': f'Page loaded with status {response.status}, check debug_accommodations_response.html for content'
                }
                return
        
        # Extrahiere Unterkunfts-Informationen
        for i, accommodation in enumerate(accommodations_found[:20]):  # Maximal 20 pro Seite
            try:
                # Name der Unterkunft
                name_selectors = [
                    '.property-name::text',
                    '.hostel-name::text',
                    'h3::text',
                    'h2::text',
                    '.title::text',
                    '.name::text',
                    '[class*="name"]::text',
                    '[class*="title"]::text'
                ]
                name = self.extract_with_selectors(accommodation, name_selectors)
                
                # Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '.property-link::attr(href)',
                    'h3 a::attr(href)',
                    'h2 a::attr(href)'
                ]
                link = self.extract_with_selectors(accommodation, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Bewertung
                rating_selectors = [
                    '.rating::text',
                    '.score::text',
                    '[class*="rating"]::text',
                    '[class*="score"]::text',
                    '.review-score::text'
                ]
                rating = self.extract_with_selectors(accommodation, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Preis
                price_selectors = [
                    '.price::text',
                    '.cost::text',
                    '[class*="price"]::text',
                    '[class*="rate"]::text',
                    '.nightly-rate::text'
                ]
                price = self.extract_with_selectors(accommodation, price_selectors)
                price = self.clean_price(price)
                
                # Standort
                location = self.get_location_from_url(response.url)
                location_selectors = [
                    '.location::text',
                    '.address::text',
                    '.neighborhood::text',
                    '[class*="location"]::text'
                ]
                specific_location = self.extract_with_selectors(accommodation, location_selectors)
                if specific_location:
                    location = f"{specific_location}, {location}"
                
                # Beschreibung
                desc_selectors = [
                    '.description::text',
                    '.summary::text',
                    'p::text',
                    '.amenities::text'
                ]
                description_parts = []
                for desc_sel in desc_selectors:
                    texts = accommodation.css(desc_sel).getall()
                    description_parts.extend(texts[:2])
                
                description = ' '.join(description_parts[:2]) if description_parts else None
                if description:
                    description = description.strip()[:200]  # Begrenzen auf 200 Zeichen
                
                # Fallback für Name wenn leer
                if not name:
                    # Versuche, irgendeinen sinnvollen Text als Name zu finden
                    all_texts = accommodation.css('::text').getall()
                    for text in all_texts:
                        text = text.strip()
                        if len(text) > 5 and len(text) < 100 and not re.match(r'^\d+', text):
                            name = text
                            break
                    if not name:
                        name = f"Accommodation {i+1}"
                
                # Fallback für Link wenn leer
                if not link:
                    link = response.url
                
                accommodation_data = {
                    'source': 'Accommodations',
                    'name': name,
                    'link': link,
                    'rating': rating,
                    'price': price,
                    'location': location,
                    'description': description
                }
                
                print(f"🏨 Accommodation {i+1}: {name} - Price: {price} - Location: {location}")
                yield accommodation_data
                
            except Exception as e:
                print(f"❌ Error processing accommodation {i+1}: {str(e)}")
                continue

    def get_location_from_url(self, url):
        """Extrahiere Standort aus der URL"""
        if 'zurich' in url.lower():
            return 'Zürich, Switzerland'
        elif 'bern' in url.lower():
            return 'Bern, Switzerland'
        elif 'switzerland' in url.lower():
            return 'Switzerland'
        else:
            return 'Unknown Location'

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
