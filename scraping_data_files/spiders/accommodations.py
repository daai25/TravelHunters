import scrapy
from scrapy.spiders import Spider
import re
import json


class AccommodationsSpider(Spider):
    name = 'accommodations'
    
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
        """Generiere Requests für alle Destinationen mit besseren Accommodation-Websites"""
        for destination in self.destinations:
            # Alternative Accommodation-Websites verwenden, die weniger Anti-Scraping-Maßnahmen haben
            
            # Hostelbookers URLs
            hostelbookers_url = f"https://www.hostelbookers.com/hostels/{destination.replace(' ', '-').lower()}/"
            yield scrapy.Request(
                url=hostelbookers_url,
                callback=self.parse,
                meta={'destination': destination, 'page': 1, 'source': 'hostelbookers'}
            )
            
            # Booking.com Hostels/Budget URLs
            booking_url = f"https://www.booking.com/searchresults.html?ss={destination.replace(' ', '+')}&accommodation_type=21,22,23&order=popularity"
            yield scrapy.Request(
                url=booking_url,
                callback=self.parse_booking,
                meta={'destination': destination, 'page': 1, 'source': 'booking'}
            )
            
            # Hotels.com Budget URLs
            hotels_url = f"https://www.hotels.com/search.do?resolved-location=CITY:{destination.replace(' ', '%20')}&f-price=0-100"
            yield scrapy.Request(
                url=hotels_url,
                callback=self.parse_hotels,
                meta={'destination': destination, 'page': 1, 'source': 'hotels'}
            )
    
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
        destination = response.meta.get('destination', 'Unknown')
        page = response.meta.get('page', 1)
        source = response.meta.get('source', 'hostelworld')
        
        print(f"🌐 Parsing Accommodations for {destination} (Page {page}) from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Handle error responses
        if response.status >= 500:
            print(f"⚠️ Server error {response.status} for {destination}, trying next destination")
            return
        
        # Debug: Speichere HTML für Analyse (nur für erste Seite)
        if page == 1:
            with open(f'debug_accommodations_{source}_response.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"💾 HTML saved to debug_accommodations_{source}_response.html")
        
        # Hostelbookers/General Accommodation Selektoren
        accommodation_selectors = [
            '.property-card',
            '.accommodation-item',
            '.hostel-card',
            '.property-item',
            '.listing-item',
            '.hotel-item',
            '.property-listing',
            '[data-cy="property-card"]',
            '.search-result-item',
            '.hostel-item'
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
                
                # Bild-URL Extraktion mit verschiedenen Selektoren
                image_selectors = [
                    'img::attr(src)',
                    'img::attr(data-src)',
                    'img::attr(data-lazy)',
                    '.property-image img::attr(src)',
                    '.hostel-image img::attr(src)',
                    '.photo img::attr(src)',
                    '.image img::attr(src)',
                    '[class*="image"] img::attr(src)',
                    '[class*="photo"] img::attr(src)',
                    'picture img::attr(src)',
                    '.thumbnail img::attr(src)',
                    '.gallery img::attr(src)',
                    'img::attr(data-original)',
                    'img::attr(data-lazy-src)',
                    '.property-card img::attr(src)',
                    '.accommodation-item img::attr(src)'
                ]
                
                image_url = self.extract_with_selectors(accommodation, image_selectors)
                
                # Bild-URL Optimierung
                if image_url:
                    # Relative URLs zu absoluten machen
                    if image_url.startswith('//'):
                        image_url = 'https:' + image_url
                    elif image_url.startswith('/'):
                        image_url = response.urljoin(image_url)
                    elif not image_url.startswith('http'):
                        image_url = response.urljoin(image_url)
                    
                    # URL-Parameter für bessere Bildqualität optimieren
                    if '?w=' in image_url or '&w=' in image_url:
                        # Ersetze kleine Bildbreiten mit größeren
                        image_url = re.sub(r'[?&]w=\d+', '?w=800', image_url)
                    if '?h=' in image_url or '&h=' in image_url:
                        image_url = re.sub(r'[?&]h=\d+', '&h=600', image_url)
                
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
                
                # Standort - verbesserte Extraktion
                location = self.get_location_from_destination(destination)
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
                        name = f"Accommodation {i+1} in {destination}"
                
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
                    'description': description,
                    'image': image_url,  # Für Kompatibilität
                    'image_url': image_url,
                    'destination': destination,
                    'page': page,
                    'accommodation_source': source
                }
                
                print(f"🏨 Accommodation {i+1}: {name} - Price: {price} - Location: {location}")
                if image_url:
                    print(f"  📸 Image: {image_url[:80]}...")
                
                yield accommodation_data
                
            except Exception as e:
                print(f"❌ Error processing accommodation {i+1}: {str(e)}")
                continue
        
        # Pagination: Gehe zu nächster Seite (maximal 3 Seiten pro Destination)
        if page < 3:
            next_page_selectors = [
                '.pagination .next::attr(href)',
                '.pager .next::attr(href)',
                'a[rel="next"]::attr(href)',
                '.pagination a[aria-label="Next"]::attr(href)',
                '.next-page::attr(href)',
                'a[class*="next"]::attr(href)',
                '.pagination li:last-child a::attr(href)'
            ]
            
            next_page_url = None
            for selector in next_page_selectors:
                next_page_url = response.css(selector).get()
                if next_page_url:
                    break
            
            # Alternative: Konstruiere nächste Seite basierend auf aktueller URL
            if not next_page_url and source == 'hostelworld':
                if '&page=' in response.url:
                    next_page_url = re.sub(r'&page=\d+', f'&page={page + 1}', response.url)
                elif '?page=' in response.url:
                    next_page_url = re.sub(r'\?page=\d+', f'?page={page + 1}', response.url)
                else:
                    next_page_url = f"{response.url}&page={page + 1}"
            
            if next_page_url:
                if not next_page_url.startswith('http'):
                    next_page_url = response.urljoin(next_page_url)
                
                print(f"🔄 Going to page {page + 1} for {destination}: {next_page_url}")
                yield scrapy.Request(
                    url=next_page_url,
                    callback=self.parse,
                    meta={
                        'destination': destination,
                        'page': page + 1,
                        'source': source
                    }
                )
            else:
                print(f"📄 No more pages found for {destination} (stopped at page {page})")

    def parse_booking(self, response):
        """Parse Booking.com budget accommodation listings"""
        destination = response.meta.get('destination', 'Unknown')
        page = response.meta.get('page', 1)
        
        print(f"🏨 Parsing Booking.com for {destination} (Page {page}) from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Handle error responses
        if response.status >= 500:
            print(f"⚠️ Server error {response.status} for {destination}, trying next destination")
            return
        
        # Booking.com selectors (ähnlich der existing booking spider)
        booking_selectors = [
            '[data-testid="property-card"]',
            '.sr_property_block',
            '.sr_item',
            '.property_block',
            '[data-testid="property-card-container"]'
        ]
        
        accommodations_found = []
        for selector in booking_selectors:
            accommodations = response.css(selector)
            if accommodations:
                print(f"✅ Found {len(accommodations)} Booking.com accommodations with selector: {selector}")
                accommodations_found = accommodations
                break
        
        if not accommodations_found:
            print("❌ No Booking.com accommodations found with specific selectors.")
            return
        
        # Extrahiere Informationen
        for i, accommodation in enumerate(accommodations_found[:15]):
            try:
                # Name
                name_selectors = [
                    '[data-testid="title"]::text',
                    '.sr-hotel__name::text',
                    'h3::text',
                    'h2::text',
                    '.property_title::text'
                ]
                name = self.extract_with_selectors(accommodation, name_selectors)
                
                # Bild-URL
                image_selectors = [
                    'img::attr(src)',
                    'img::attr(data-src)',
                    '[data-testid="image"] img::attr(src)',
                    '.hotel_image img::attr(src)',
                    '.sr_item_photo img::attr(src)'
                ]
                
                image_url = self.extract_with_selectors(accommodation, image_selectors)
                
                # Bild-URL Optimierung
                if image_url:
                    if image_url.startswith('//'):
                        image_url = 'https:' + image_url
                    elif image_url.startswith('/'):
                        image_url = response.urljoin(image_url)
                    elif not image_url.startswith('http'):
                        image_url = response.urljoin(image_url)
                
                # Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '[data-testid="title-link"]::attr(href)'
                ]
                link = self.extract_with_selectors(accommodation, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Preis
                price_selectors = [
                    '[data-testid="price-and-discounted-price"]::text',
                    '.bui-price-display__value::text',
                    '.price::text',
                    '.sr_price::text'
                ]
                price = self.extract_with_selectors(accommodation, price_selectors)
                price = self.clean_price(price)
                
                # Bewertung
                rating_selectors = [
                    '[data-testid="review-score"]::text',
                    '.bui-review-score__badge::text',
                    '.review-score-badge::text'
                ]
                rating = self.extract_with_selectors(accommodation, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Fallbacks
                if not name:
                    name = f"Booking.com Accommodation {i+1} in {destination}"
                if not link:
                    link = response.url
                
                accommodation_data = {
                    'source': 'Accommodations',
                    'name': name,
                    'link': link,
                    'rating': rating,
                    'price': price,
                    'location': self.get_location_from_destination(destination),
                    'description': f'Budget accommodation in {destination} from Booking.com',
                    'image': image_url,
                    'image_url': image_url,
                    'destination': destination,
                    'page': page,
                    'accommodation_source': 'booking'
                }
                
                print(f"🏨 Booking.com {i+1}: {name} - Price: {price}")
                if image_url:
                    print(f"  📸 Image: {image_url[:80]}...")
                
                yield accommodation_data
                
            except Exception as e:
                print(f"❌ Error processing Booking.com accommodation {i+1}: {str(e)}")
                continue

    def parse_hotels(self, response):
        """Parse Hotels.com budget accommodation listings"""
        destination = response.meta.get('destination', 'Unknown')
        page = response.meta.get('page', 1)
        
        print(f"🏩 Parsing Hotels.com for {destination} (Page {page}) from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Handle error responses
        if response.status >= 500:
            print(f"⚠️ Server error {response.status} for {destination}, trying next destination")
            return
        
        # Hotels.com selectors
        hotels_selectors = [
            '[data-stid="section-results"] [data-stid]',
            '.hotel-wrap',
            '.results-item',
            '.listing',
            '[data-testid="property-listing"]'
        ]
        
        accommodations_found = []
        for selector in hotels_selectors:
            accommodations = response.css(selector)
            if accommodations:
                print(f"✅ Found {len(accommodations)} Hotels.com accommodations with selector: {selector}")
                accommodations_found = accommodations
                break
        
        if not accommodations_found:
            print("❌ No Hotels.com accommodations found with specific selectors.")
            return
        
        # Extrahiere Informationen
        for i, accommodation in enumerate(accommodations_found[:15]):
            try:
                # Name
                name_selectors = [
                    '[data-stid="content-hotel-title"]::text',
                    'h3::text',
                    'h4::text',
                    '.hotel-name::text',
                    '.property-name::text'
                ]
                name = self.extract_with_selectors(accommodation, name_selectors)
                
                # Bild-URL
                image_selectors = [
                    'img::attr(src)',
                    'img::attr(data-src)',
                    '[data-stid="lodging-photo-featured"] img::attr(src)',
                    '.property-image img::attr(src)'
                ]
                
                image_url = self.extract_with_selectors(accommodation, image_selectors)
                
                # Bild-URL Optimierung
                if image_url:
                    if image_url.startswith('//'):
                        image_url = 'https:' + image_url
                    elif image_url.startswith('/'):
                        image_url = response.urljoin(image_url)
                    elif not image_url.startswith('http'):
                        image_url = response.urljoin(image_url)
                
                # Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '[data-stid="open-hotel-information"]::attr(href)'
                ]
                link = self.extract_with_selectors(accommodation, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Preis
                price_selectors = [
                    '[data-stid="price-display-value"]::text',
                    '.price-current::text',
                    '.rate::text',
                    '.price::text'
                ]
                price = self.extract_with_selectors(accommodation, price_selectors)
                price = self.clean_price(price)
                
                # Bewertung
                rating_selectors = [
                    '[data-stid="content-hotel-rating"]::text',
                    '.star-rating::text',
                    '.rating::text'
                ]
                rating = self.extract_with_selectors(accommodation, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Fallbacks
                if not name:
                    name = f"Hotels.com Accommodation {i+1} in {destination}"
                if not link:
                    link = response.url
                
                accommodation_data = {
                    'source': 'Accommodations',
                    'name': name,
                    'link': link,
                    'rating': rating,
                    'price': price,
                    'location': self.get_location_from_destination(destination),
                    'description': f'Budget accommodation in {destination} from Hotels.com',
                    'image': image_url,
                    'image_url': image_url,
                    'destination': destination,
                    'page': page,
                    'accommodation_source': 'hotels'
                }
                
                print(f"🏩 Hotels.com {i+1}: {name} - Price: {price}")
                if image_url:
                    print(f"  📸 Image: {image_url[:80]}...")
                
                yield accommodation_data
                
            except Exception as e:
                print(f"❌ Error processing Hotels.com accommodation {i+1}: {str(e)}")
                continue
        """Parse Airbnb-style accommodation listings"""
        destination = response.meta.get('destination', 'Unknown')
        page = response.meta.get('page', 1)
        
        print(f"🏠 Parsing Airbnb-style for {destination} (Page {page}) from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Handle error responses
        if response.status >= 500:
            print(f"⚠️ Server error {response.status} for {destination}, trying next destination")
            return
        
        # Airbnb-specific selectors
        airbnb_selectors = [
            '[data-testid="listing-card"]',
            '.listing-card',
            '.place-item',
            '[itemprop="itemListElement"]',
            '.search-result',
            '.property-card'
        ]
        
        accommodations_found = []
        for selector in airbnb_selectors:
            accommodations = response.css(selector)
            if accommodations:
                print(f"✅ Found {len(accommodations)} Airbnb accommodations with selector: {selector}")
                accommodations_found = accommodations
                break
        
        if not accommodations_found:
            print("❌ No Airbnb accommodations found. Trying fallback selectors...")
            # Fallback für Airbnb-ähnliche Seiten
            fallback_selectors = [
                'div[class*="listing"]',
                'div[class*="property"]',
                'div[class*="place"]',
                '[data-id]',
                '.rental-item'
            ]
            
            for selector in fallback_selectors:
                accommodations = response.css(selector)
                if len(accommodations) > 2:
                    print(f"✅ Found {len(accommodations)} accommodations with fallback selector: {selector}")
                    accommodations_found = accommodations[:15]
                    break
        
        # Extrahiere Informationen
        for i, accommodation in enumerate(accommodations_found[:15]):
            try:
                # Name
                name_selectors = [
                    '[data-testid="listing-card-title"]::text',
                    '.listing-title::text',
                    'h3::text',
                    'h2::text',
                    '.place-name::text',
                    '[class*="title"]::text'
                ]
                name = self.extract_with_selectors(accommodation, name_selectors)
                
                # Bild-URL für Airbnb
                image_selectors = [
                    'img::attr(src)',
                    'img::attr(data-src)',
                    '[data-testid="listing-card-image"] img::attr(src)',
                    '.listing-image img::attr(src)',
                    '.place-photo img::attr(src)',
                    'picture img::attr(src)',
                    'img::attr(data-original)',
                    'img::attr(data-lazy-src)'
                ]
                
                image_url = self.extract_with_selectors(accommodation, image_selectors)
                
                # Bild-URL Optimierung
                if image_url:
                    if image_url.startswith('//'):
                        image_url = 'https:' + image_url
                    elif image_url.startswith('/'):
                        image_url = response.urljoin(image_url)
                    elif not image_url.startswith('http'):
                        image_url = response.urljoin(image_url)
                
                # Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '[data-testid="listing-card-link"]::attr(href)'
                ]
                link = self.extract_with_selectors(accommodation, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Preis
                price_selectors = [
                    '[data-testid="price"]::text',
                    '.price::text',
                    '.rate::text',
                    '[class*="price"]::text',
                    '.cost::text'
                ]
                price = self.extract_with_selectors(accommodation, price_selectors)
                price = self.clean_price(price)
                
                # Bewertung
                rating_selectors = [
                    '[data-testid="listing-card-rating"]::text',
                    '.rating::text',
                    '[class*="rating"]::text',
                    '.review-score::text'
                ]
                rating = self.extract_with_selectors(accommodation, rating_selectors)
                rating = self.clean_rating(rating)
                
                # Fallbacks
                if not name:
                    name = f"Airbnb Accommodation {i+1} in {destination}"
                if not link:
                    link = response.url
                
                accommodation_data = {
                    'source': 'Accommodations',
                    'name': name,
                    'link': link,
                    'rating': rating,
                    'price': price,
                    'location': self.get_location_from_destination(destination),
                    'description': f'Airbnb-style accommodation in {destination}',
                    'image': image_url,
                    'image_url': image_url,
                    'destination': destination,
                    'page': page,
                    'accommodation_source': 'airbnb'
                }
                
                print(f"🏠 Airbnb {i+1}: {name} - Price: {price}")
                if image_url:
                    print(f"  📸 Image: {image_url[:80]}...")
                
                yield accommodation_data
                
            except Exception as e:
                print(f"❌ Error processing Airbnb accommodation {i+1}: {str(e)}")
                continue

    def get_location_from_destination(self, destination):
        """Extrahiere Standort aus Destination"""
        location_map = {
            # Europa
            'Paris': 'Paris, France',
            'London': 'London, United Kingdom',
            'Rome': 'Rome, Italy',
            'Barcelona': 'Barcelona, Spain',
            'Amsterdam': 'Amsterdam, Netherlands',
            'Prague': 'Prague, Czech Republic',
            'Vienna': 'Vienna, Austria',
            'Berlin': 'Berlin, Germany',
            'Zurich': 'Zürich, Switzerland',
            'Munich': 'Munich, Germany',
            'Florence': 'Florence, Italy',
            'Venice': 'Venice, Italy',
            'Dublin': 'Dublin, Ireland',
            'Edinburgh': 'Edinburgh, Scotland',
            'Stockholm': 'Stockholm, Sweden',
            'Copenhagen': 'Copenhagen, Denmark',
            'Athens': 'Athens, Greece',
            'Istanbul': 'Istanbul, Turkey',
            'Lisbon': 'Lisbon, Portugal',
            'Madrid': 'Madrid, Spain',
            
            # Asien
            'Tokyo': 'Tokyo, Japan',
            'Kyoto': 'Kyoto, Japan',
            'Seoul': 'Seoul, South Korea',
            'Bangkok': 'Bangkok, Thailand',
            'Singapore': 'Singapore',
            'Hong Kong': 'Hong Kong',
            'Shanghai': 'Shanghai, China',
            'Beijing': 'Beijing, China',
            'Mumbai': 'Mumbai, India',
            'Delhi': 'Delhi, India',
            'Dubai': 'Dubai, UAE',
            
            # Amerika
            'New York': 'New York, USA',
            'Los Angeles': 'Los Angeles, USA',
            'San Francisco': 'San Francisco, USA',
            'Toronto': 'Toronto, Canada',
            'Mexico City': 'Mexico City, Mexico',
            'Rio de Janeiro': 'Rio de Janeiro, Brazil',
            'Buenos Aires': 'Buenos Aires, Argentina',
            
            # Weitere Destinationen
            'Sydney': 'Sydney, Australia',
            'Melbourne': 'Melbourne, Australia',
            'Cape Town': 'Cape Town, South Africa',
            'Cairo': 'Cairo, Egypt',
            'Bali': 'Bali, Indonesia',
            'Phuket': 'Phuket, Thailand'
        }
        
        return location_map.get(destination, destination)

    def get_location_from_url(self, url):
        """Extrahiere Standort aus der URL (Legacy-Funktion)"""
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
