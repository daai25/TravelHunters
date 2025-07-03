import scrapy
from scrapy.spiders import Spider
import re
import json


class KayakSpider(Spider):
    name = 'kayak'
    start_urls = [
        'https://www.kayak.com/hotels/Zurich,Switzerland-c9033/2025-08-01/2025-08-03/2guests',
        'https://www.kayak.com/hotels/Bern,Switzerland-c8490/2025-08-01/2025-08-03/2guests',
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
        print(f"🌐 Parsing Kayak response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open('debug_kayak_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_kayak_response.html")
        
        # Kayak Selektoren - einfache und robuste
        hotel_selectors = [
            '.resultInner',
            '.hotelResultInner',
            '.result',
            '.Common-Booking-MultiBookProvider',
            '.resultContainer',
            '[data-resultid]'
        ]
        
        hotels_found = []
        for selector in hotel_selectors:
            hotels = response.css(selector)
            if hotels:
                print(f"✅ Found {len(hotels)} hotels with selector: {selector}")
                hotels_found = hotels
                break
        
        if not hotels_found:
            print("❌ No hotels found with specific selectors. Searching for any hotels...")
            
            # Erweiterte Suche nach beliebigen Hotel-Containern
            fallback_selectors = [
                'div[class*="result"]',
                'div[class*="hotel"]',
                'div[class*="card"]',
                'article',
                'li[class*="result"]'
            ]
            
            for selector in fallback_selectors:
                hotels = response.css(selector)
                if len(hotels) > 5:  # Mindestens ein paar Ergebnisse
                    print(f"✅ Found {len(hotels)} potential hotels with fallback selector: {selector}")
                    hotels_found = hotels[:20]  # Begrenzen auf 20
                    break
            
            if not hotels_found:
                # Als allerletzter Ausweg: Basis-Info zurückgeben
                yield {
                    'source': 'Kayak',
                    'name': 'Page accessed successfully',
                    'link': response.url,
                    'rating': None,
                    'price': None,
                    'location': None,
                    'description': f'Page loaded with status {response.status}, check debug_kayak_response.html for content'
                }
                return
        
        # Extrahiere Hotel-Informationen
        for i, hotel in enumerate(hotels_found[:25]):  # Maximal 25 Hotels pro Seite
            try:
                # Hotel Name - sehr flexible Suche
                name_selectors = [
                    '.hotelName::text',
                    '.Common-Booking-MultiBookProvider-name::text',
                    '.keel-grid-row h3::text',
                    'h3::text',
                    'h2::text',
                    'h4::text',
                    '[class*="name"]::text',
                    '[class*="title"]::text'
                ]
                name = self.extract_with_selectors(hotel, name_selectors)
                
                # Hotel Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '.hotelName a::attr(href)',
                    'h3 a::attr(href)',
                    'h2 a::attr(href)'
                ]
                link = self.extract_with_selectors(hotel, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Bewertung
                rating_selectors = [
                    '.rating::text',
                    '.keel-rating::text',
                    '[class*="rating"]::text',
                    '[class*="score"]::text',
                    '.reviews::text'
                ]
                rating = self.extract_with_selectors(hotel, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Preis
                price_selectors = [
                    '.price::text',
                    '.Common-Booking-MultiBookProvider-rate::text',
                    '[class*="price"]::text',
                    '[class*="rate"]::text',
                    '[class*="cost"]::text'
                ]
                price = self.extract_with_selectors(hotel, price_selectors)
                price = self.clean_price(price)
                
                # Standort - auch aus dem Text extrahieren
                location_selectors = [
                    '.location::text',
                    '.neighborhood::text',
                    '[class*="location"]::text',
                    '[class*="address"]::text'
                ]
                location = self.extract_with_selectors(hotel, location_selectors)
                
                # Beschreibung - sammle verschiedene Texte
                description_parts = []
                desc_selectors = [
                    '.amenities::text',
                    '.description::text',
                    '.features::text',
                    'p::text'
                ]
                for desc_sel in desc_selectors:
                    texts = hotel.css(desc_sel).getall()
                    description_parts.extend(texts[:2])  # Max 2 pro Selektor
                
                description = ' '.join(description_parts[:3]) if description_parts else None
                if description:
                    description = description.strip()
                
                # Fallback für Name wenn leer
                if not name:
                    # Versuche, irgendeinen Text als Name zu finden
                    all_texts = hotel.css('::text').getall()
                    for text in all_texts:
                        text = text.strip()
                        if len(text) > 5 and len(text) < 100 and not re.match(r'^\d+', text):
                            name = text
                            break
                    if not name:
                        name = f"Hotel {i+1}"
                
                # Fallback für Link wenn leer
                if not link:
                    link = response.url
                
                hotel_data = {
                    'source': 'Kayak',
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
