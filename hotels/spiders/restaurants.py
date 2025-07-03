import scrapy
from scrapy.spiders import Spider
import re
import json


class RestaurantSpider(Spider):
    name = 'restaurants'
    start_urls = [
        'https://www.yelp.com/search?find_desc=Hotels&find_loc=Zurich%2C%20Switzerland',
        'https://www.yelp.com/search?find_desc=Hotels&find_loc=Bern%2C%20Switzerland',
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
        print(f"🌐 Parsing Yelp response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open('debug_yelp_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_yelp_response.html")
        
        # Yelp Selektoren für Hotels/Restaurants
        business_selectors = [
            '[data-testid="serp-ia-card"]',
            '.container__373c0__1lKS2',
            '.searchResult__373c0__17w8F',
            '[data-analytics-label="biz-name"]',
            '.result',
            '.search-result'
        ]
        
        businesses_found = []
        for selector in business_selectors:
            businesses = response.css(selector)
            if businesses:
                print(f"✅ Found {len(businesses)} businesses with selector: {selector}")
                businesses_found = businesses
                break
        
        if not businesses_found:
            print("❌ No businesses found with any selector. Trying to extract any business-related content...")
            
            # Fallback: Suche nach Links
            business_links = response.css('a[href*="biz"]::attr(href)').getall()
            print(f"Found {len(business_links)} potential business links")
            
            if not business_links:
                # Als letzter Ausweg: Basis-Info zurückgeben
                yield {
                    'source': 'Yelp',
                    'name': 'Page accessed successfully',
                    'link': response.url,
                    'rating': None,
                    'price': None,
                    'location': None,
                    'description': f'Page loaded with status {response.status}, check debug_yelp_response.html for content'
                }
                return
        
        # Extrahiere Business-Informationen
        for i, business in enumerate(businesses_found[:20]):  # Maximal 20 Businesses pro Seite
            try:
                # Business Name
                name_selectors = [
                    '[data-analytics-label="biz-name"]::text',
                    'h3 a::text',
                    'h4 a::text',
                    '.css-19v1rkv::text',
                    'a[class*="business-name"]::text',
                    '.business-name::text'
                ]
                name = self.extract_with_selectors(business, name_selectors)
                
                # Business Link
                link_selectors = [
                    '[data-analytics-label="biz-name"]::attr(href)',
                    'h3 a::attr(href)',
                    'h4 a::attr(href)',
                    'a[href*="biz"]::attr(href)'
                ]
                link = self.extract_with_selectors(business, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Bewertung
                rating_selectors = [
                    '[aria-label*="star rating"]::attr(aria-label)',
                    '.css-gutk1c::attr(aria-label)',
                    '.i-stars::attr(title)',
                    '[class*="rating"]::text'
                ]
                rating = self.extract_with_selectors(business, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Preis
                price_selectors = [
                    '[aria-label*="Price range"]::text',
                    '.css-1xe5s96::text',
                    '.price-range::text',
                    '[class*="price"]::text'
                ]
                price = self.extract_with_selectors(business, price_selectors)
                price = self.clean_price(price)
                
                # Standort
                location_selectors = [
                    '.css-chan6m::text',
                    '.secondary-attributes::text',
                    '.address::text',
                    '[class*="address"]::text'
                ]
                location = self.extract_with_selectors(business, location_selectors)
                
                # Kategorie/Beschreibung
                desc_selectors = [
                    '[class*="category"]::text',
                    '.css-1jq13cf::text',
                    '.category-str-list::text'
                ]
                description_parts = business.css(' '.join(desc_selectors)).getall()
                description = ', '.join(description_parts[:3]) if description_parts else None
                
                # Fallback für Name wenn leer
                if not name:
                    # Versuche, irgendeinen relevanten Text zu finden
                    all_texts = business.css('a::text, h3::text, h4::text').getall()
                    for text in all_texts:
                        text = text.strip()
                        if len(text) > 3 and len(text) < 100:
                            name = text
                            break
                    if not name:
                        name = f"Business {i+1}"
                
                # Fallback für Link wenn leer
                if not link:
                    link = response.url
                
                business_data = {
                    'source': 'Yelp',
                    'name': name,
                    'link': link,
                    'rating': rating,
                    'price': price,
                    'location': location,
                    'description': description
                }
                
                print(f"🏨 Business {i+1}: {name} - Rating: {rating} - Price: {price}")
                yield business_data
                
            except Exception as e:
                print(f"❌ Error processing business {i+1}: {str(e)}")
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
            return str(rating_num)
        return None

    def clean_price(self, price):
        """Bereinige Preis - für Yelp meist $ Anzahl"""
        if not price:
            return None
        
        # Zähle $ Zeichen für Preisklasse
        dollar_count = price.count('$')
        if dollar_count > 0:
            return str(dollar_count)
        
        # Fallback: Extrahiere Zahlen
        numbers = re.findall(r'(\d+)', price.replace(',', ''))
        if numbers:
            return numbers[0]
        return None
