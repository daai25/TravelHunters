import scrapy
from scrapy.spiders import Spider
import re
import json


class DestinationsSpider(Spider):
    name = 'destinations'
    
    # Weltweite beliebte Destinationen
    destinations = [
        # Europa
        'Paris', 'London', 'Rome', 'Barcelona', 'Amsterdam', 'Prague', 'Vienna', 'Berlin',
        'Zurich', 'Munich', 'Florence', 'Venice', 'Dublin', 'Edinburgh', 'Stockholm', 'Copenhagen',
        'Athens', 'Istanbul', 'Lisbon', 'Madrid', 'Brussels', 'Budapest', 'Krakow', 'Oslo',
        
        # Asien
        'Tokyo', 'Kyoto', 'Seoul', 'Bangkok', 'Singapore', 'Hong Kong', 'Shanghai', 'Beijing',
        'Mumbai', 'Delhi', 'Goa', 'Dubai', 'Abu Dhabi', 'Doha', 'Manila', 'Jakarta',
        'Kuala Lumpur', 'Ho Chi Minh City', 'Hanoi', 'Phnom Penh', 'Yangon', 'Kathmandu',
        
        # Afrika
        'Cape Town', 'Johannesburg', 'Cairo', 'Marrakech', 'Casablanca', 'Nairobi', 'Addis Ababa',
        'Lagos', 'Accra', 'Dakar', 'Tunis', 'Algiers', 'Khartoum', 'Kampala',
        
        # Nordamerika
        'New York', 'Los Angeles', 'Las Vegas', 'San Francisco', 'Chicago', 'Miami', 'Boston',
        'Washington DC', 'Seattle', 'Toronto', 'Vancouver', 'Montreal', 'Mexico City', 'Cancun',
        
        # Südamerika
        'Rio de Janeiro', 'São Paulo', 'Buenos Aires', 'Lima', 'Santiago', 'Bogota', 'Quito',
        'La Paz', 'Montevideo', 'Asuncion', 'Caracas', 'Georgetown', 'Paramaribo',
        
        # Ozeanien
        'Sydney', 'Melbourne', 'Auckland', 'Wellington', 'Brisbane', 'Perth', 'Adelaide',
        'Suva', 'Port Moresby', 'Nuku\'alofa', 'Apia', 'Port Vila',
        
        # Beliebte Inseln
        'Bali', 'Phuket', 'Maldives', 'Santorini', 'Mykonos', 'Ibiza', 'Mallorca', 'Cyprus',
        'Malta', 'Crete', 'Rhodes', 'Zakynthos', 'Corfu', 'Capri', 'Sicily', 'Sardinia'
    ]
    
    def start_requests(self):
        """Generiere Requests für alle Destinationen"""
        for destination in self.destinations[:20]:  # Erste 20 für Test
            # Lonely Planet URLs
            lonely_planet_url = f"https://www.lonelyplanet.com/search?q={destination.replace(' ', '+')}"
            yield scrapy.Request(
                url=lonely_planet_url,
                callback=self.parse,
                meta={'destination': destination, 'page': 1, 'source': 'lonely_planet'}
            )
            
            # Alternative: TripAdvisor URLs
            tripadvisor_url = f"https://www.tripadvisor.com/Search?q={destination.replace(' ', '+')}"
            yield scrapy.Request(
                url=tripadvisor_url,
                callback=self.parse_tripadvisor,
                meta={'destination': destination, 'page': 1, 'source': 'tripadvisor'}
            )
    
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'HTTPERROR_ALLOWED_CODES': [404, 403, 500, 502, 503],
        'CONCURRENT_REQUESTS': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 5,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
        'DOWNLOAD_TIMEOUT': 30,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/avif,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    }

    def parse(self, response):
        destination = response.meta.get('destination', 'Unknown')
        current_page = response.meta.get('page', 1)
        source = response.meta.get('source', 'lonely_planet')
        
        print(f"🌐 Parsing {source} destinations for {destination} (Page {current_page})")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open(f'debug_destinations_{source}_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"💾 HTML saved to debug_destinations_{source}_response.html")
        
        # Lonely Planet Selektoren für Attraktionen und Destinationen
        attraction_selectors = [
            '.card--poi',
            '.poi-card',
            '.attraction-card',
            '.place-card',
            '[data-name="poi-card"]',
            '.listing-card',
            '.search-result',
            '.destination-card',
            '.place-item',
            '.result-item'
        ]
        
        attractions_found = []
        for selector in attraction_selectors:
            attractions = response.css(selector)
            if attractions:
                print(f"✅ Found {len(attractions)} destinations with selector: {selector}")
                attractions_found = attractions
                break
        
        if not attractions_found:
            print("❌ No destinations found with specific selectors. Searching for general content...")
            
            # Fallback: Suche nach Links und Content
            links = response.css('a[href*="attraction"], a[href*="place"], a[href*="destination"], a[href*="things"], a[href*="guide"]')
            if links:
                print(f"✅ Found {len(links)} potential destination links")
                attractions_found = links[:15]  # Begrenzen auf 15
            else:
                # Als letzter Ausweg: Basis-Info zurückgeben
                yield {
                    'source': f'Destinations-{source}',
                    'name': f'{destination} - Page accessed',
                    'link': response.url,
                    'rating': None,
                    'price': None,
                    'location': destination,
                    'description': f'Page loaded with status {response.status}',
                    'category': 'destination',
                    'image': None,
                    'image_url': None,
                    'destination': destination
                }
                return
        
        # Extrahiere Destinations/Attraktions-Informationen
        for i, attraction in enumerate(attractions_found[:20]):  # Maximal 20 pro Seite
            try:
                # Name der Attraktion/Destination
                name_selectors = [
                    '.card__title::text',
                    '.poi-name::text',
                    'h3::text',
                    'h2::text',
                    '.title::text',
                    '.name::text',
                    '.result-title::text',
                    '.place-name::text',
                    '::text'
                ]
                name = self.extract_with_selectors(attraction, name_selectors)
                
                # Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '.card__link::attr(href)',
                    '.result-link::attr(href)'
                ]
                link = self.extract_with_selectors(attraction, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Bewertung (falls vorhanden)
                rating_selectors = [
                    '.rating::text',
                    '.score::text',
                    '[class*="rating"]::text',
                    '.stars::text',
                    '.review-score::text'
                ]
                rating = self.extract_with_selectors(attraction, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Preis (falls vorhanden)
                price_selectors = [
                    '.price::text',
                    '.cost::text',
                    '[class*="price"]::text',
                    '.fee::text',
                    '.from-price::text'
                ]
                price = self.extract_with_selectors(attraction, price_selectors)
                price = self.clean_price(price)
                
                # Standort
                location = destination
                location_selectors = [
                    '.location::text',
                    '.address::text',
                    '.place::text',
                    '.destination::text'
                ]
                specific_location = self.extract_with_selectors(attraction, location_selectors)
                if specific_location:
                    location = f"{specific_location}, {destination}"
                
                # Beschreibung
                desc_selectors = [
                    '.description::text',
                    '.summary::text',
                    'p::text',
                    '.excerpt::text',
                    '.snippet::text'
                ]
                description_parts = []
                for desc_sel in desc_selectors:
                    texts = attraction.css(desc_sel).getall()
                    description_parts.extend(texts[:2])
                
                description = ' '.join(description_parts[:2]) if description_parts else None
                if description:
                    description = description.strip()[:300]  # Begrenzen auf 300 Zeichen
                
                # Bild-URL extrahieren
                image_selectors = [
                    'img::attr(src)',
                    'img::attr(data-src)',
                    'img::attr(data-lazy-src)',
                    '.image img::attr(src)',
                    '.photo img::attr(src)',
                    '.thumbnail img::attr(src)',
                    '[class*="image"] img::attr(src)',
                    'picture img::attr(src)',
                    'figure img::attr(src)'
                ]
                
                image = self.extract_with_selectors(attraction, image_selectors)
                if image:
                    # Vollständige URL erstellen falls relative URL
                    if image.startswith('//'):
                        image = f"https:{image}"
                    elif image.startswith('/'):
                        image = response.urljoin(image)
                    elif not image.startswith('http'):
                        image = response.urljoin(image)
                    
                    # Filter für gültige Bild-URLs
                    if any(ext in image.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.avif']):
                        # Entferne URL-Parameter die die Bildqualität reduzieren könnten
                        if '?' in image:
                            base_image = image.split('?')[0]
                            # Behalte nur wichtige Parameter
                            if any(param in image for param in ['w=', 'width=', 'h=', 'height=', 'q=', 'quality=']):
                                # Hochauflösende Version anfordern
                                image = image.replace('w=150', 'w=800').replace('w=300', 'w=800')
                                image = image.replace('h=150', 'h=600').replace('h=300', 'h=600')
                            else:
                                image = base_image
                
                # Fallback für Name wenn leer
                if not name:
                    # Versuche, irgendeinen sinnvollen Text als Name zu finden
                    all_texts = attraction.css('::text').getall()
                    for text in all_texts:
                        text = text.strip()
                        if len(text) > 3 and len(text) < 100 and not re.match(r'^\d+', text):
                            name = text
                            break
                    if not name:
                        name = f"{destination} Attraction {i+1}"
                
                # Fallback für Link wenn leer
                if not link:
                    link = response.url
                
                destination_data = {
                    'source': f'Destinations-{source}',
                    'name': name,
                    'link': link,
                    'rating': rating,
                    'price': price,
                    'location': location,
                    'description': description,
                    'category': 'destination',
                    'image': image,
                    'image_url': image,  # Add explicit image_url field for consistency
                    'destination': destination
                }
                
                print(f"🏖️ Destination {i+1}/{len(attractions_found)}")
                print(f"Destination: {name}, Rating: {rating}, Location: {location}, Images: {1 if image else 0}")
                yield destination_data
                
            except Exception as e:
                print(f"❌ Error processing destination {i+1}: {str(e)}")
                continue
        
        # Pagination: Look for next page (limit to 3 pages per destination)
        if current_page < 3:
            next_page_selectors = [
                'a[aria-label="Next page"]::attr(href)',
                '.pagination-next::attr(href)',
                '[data-testid="next-page"]::attr(href)',
                'a[rel="next"]::attr(href)',
                '.next-page::attr(href)',
                '.pager-next::attr(href)'
            ]
            
            # Try to find next page URL
            next_page_url = None
            for selector in next_page_selectors:
                next_url = response.css(selector).get()
                if next_url:
                    next_page_url = next_url
                    break
            
            # Alternative: construct next page URL manually
            if not next_page_url and 'lonelyplanet.com' in response.url:
                if 'page=' in response.url:
                    base_url = response.url.split('&page=')[0].split('?page=')[0]
                    next_page = current_page + 1
                    if '?' in response.url:
                        next_page_url = f"{base_url}&page={next_page}"
                    else:
                        next_page_url = f"{base_url}?page={next_page}"
                else:
                    if '?' in response.url:
                        next_page_url = f"{response.url}&page={current_page + 1}"
                    else:
                        next_page_url = f"{response.url}?page={current_page + 1}"
            
            if next_page_url:
                if not next_page_url.startswith('http'):
                    next_page_url = response.urljoin(next_page_url)
                
                print(f"🔄 Following next page: {next_page_url}")
                yield scrapy.Request(
                    url=next_page_url,
                    callback=self.parse,
                    meta={'destination': destination, 'page': current_page + 1, 'source': source}
                )
            else:
                print(f"🏁 No more pages found for {destination}")
        else:
            print(f"🏁 Reached page limit for {destination}")

    def parse_tripadvisor(self, response):
        """Parse TripAdvisor results"""
        destination = response.meta.get('destination', 'Unknown')
        current_page = response.meta.get('page', 1)
        
        print(f"🌐 Parsing TripAdvisor destinations for {destination} (Page {current_page})")
        print(f"Response status: {response.status}")
        
        # TripAdvisor specific selectors
        attraction_selectors = [
            '.result-card',
            '.attraction-card',
            '.listing-card',
            '.result-item',
            '[data-automation="hotel-card"]',
            '.search-result'
        ]
        
        attractions_found = []
        for selector in attraction_selectors:
            attractions = response.css(selector)
            if attractions:
                print(f"✅ Found {len(attractions)} TripAdvisor destinations with selector: {selector}")
                attractions_found = attractions
                break
        
        # Process found attractions similar to parse method
        for i, attraction in enumerate(attractions_found[:15]):  # Limit to 15 per page
            try:
                name_selectors = [
                    '.result-title::text',
                    'h3::text',
                    'h2::text',
                    '.title::text',
                    '[data-automation="hotel-name"]::text'
                ]
                name = self.extract_with_selectors(attraction, name_selectors)
                
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '.result-link::attr(href)'
                ]
                link = self.extract_with_selectors(attraction, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Rating for TripAdvisor
                rating_selectors = [
                    '[class*="rating"]::text',
                    '.review-score::text',
                    '.rating::text'
                ]
                rating = self.extract_with_selectors(attraction, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Image extraction
                image_selectors = [
                    'img::attr(src)',
                    'img::attr(data-src)',
                    '.photo img::attr(src)'
                ]
                image = self.extract_with_selectors(attraction, image_selectors)
                if image:
                    if image.startswith('//'):
                        image = f"https:{image}"
                    elif image.startswith('/'):
                        image = response.urljoin(image)
                
                if name:
                    yield {
                        'source': 'Destinations-tripadvisor',
                        'name': name,
                        'link': link or response.url,
                        'rating': rating,
                        'price': None,
                        'location': destination,
                        'description': None,
                        'category': 'destination',
                        'image': image,
                        'image_url': image,
                        'destination': destination
                    }
                    
                    print(f"🏖️ TripAdvisor Destination {i+1}: {name}")
                    
            except Exception as e:
                print(f"❌ Error processing TripAdvisor destination {i+1}: {str(e)}")
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
