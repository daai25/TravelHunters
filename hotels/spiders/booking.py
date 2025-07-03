import scrapy
from scrapy.spiders import Spider
import re


class BookingSpider(Spider):
    name = 'booking'
    start_urls = [
        # Europa
        'https://www.booking.com/searchresults.html?ss=Paris&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=London&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Rome&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Barcelona&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Amsterdam&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Berlin&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Vienna&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Prague&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Budapest&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Lisbon&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Madrid&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Stockholm&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Copenhagen&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Oslo&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Helsinki&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Zurich&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Geneva&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        
        # Nordamerika
        'https://www.booking.com/searchresults.html?ss=New+York&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Los+Angeles&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Chicago&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=San+Francisco&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Miami&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Las+Vegas&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Toronto&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Vancouver&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Montreal&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        
        # Asien
        'https://www.booking.com/searchresults.html?ss=Tokyo&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Osaka&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Seoul&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Beijing&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Shanghai&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Hong+Kong&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Singapore&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Bangkok&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Kuala+Lumpur&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Mumbai&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Delhi&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Dubai&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        
        # Australien & Ozeanien
        'https://www.booking.com/searchresults.html?ss=Sydney&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Melbourne&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Brisbane&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Perth&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Auckland&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        
        # Südamerika
        'https://www.booking.com/searchresults.html?ss=Rio+de+Janeiro&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Sao+Paulo&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Buenos+Aires&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Lima&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Santiago&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Bogota&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        
        # Afrika
        'https://www.booking.com/searchresults.html?ss=Cape+Town&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Johannesburg&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Cairo&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Marrakech&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Casablanca&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        
        # Beliebte Urlaubsziele
        'https://www.booking.com/searchresults.html?ss=Bali&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Phuket&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Maldives&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Santorini&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Mykonos&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Ibiza&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Cancun&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Tulum&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Playa+del+Carmen&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
    ]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2,  # Schneller für mehr URLs
        'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'HTTPERROR_ALLOWED_CODES': [404, 403, 500, 502, 503],
        'CONCURRENT_REQUESTS': 2,  # Mehr parallele Requests
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 5,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
        'COOKIES_ENABLED': True,
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
    }

    def parse(self, response):
        print(f"🌐 Parsing response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open('debug_booking_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_booking_response.html")
        
        # Folge den Hotels auf der aktuellen Seite
        yield from self.extract_hotels(response)
        
        # Suche nach "Nächste Seite" Links für Pagination
        next_page_selectors = [
            'a[aria-label*="Next"]::attr(href)',
            'a[aria-label*="Weiter"]::attr(href)', 
            '.bui-pagination__next-arrow a::attr(href)',
            '.sr-pagination__next a::attr(href)',
            '.paging-next a::attr(href)',
            'a[data-testid="pagination-next"]::attr(href)',
            'a.bui-pagination__link--next::attr(href)'
        ]
        
        next_page_url = None
        for selector in next_page_selectors:
            next_url = response.css(selector).get()
            if next_url:
                next_page_url = response.urljoin(next_url)
                break
        
        # Auch manuell die nächste Seite generieren (offset-basiert)
        if not next_page_url:
            current_url = response.url
            if 'offset=' in current_url:
                # Aktuelle Offset extrahieren
                import re
                offset_match = re.search(r'offset=(\d+)', current_url)
                if offset_match:
                    current_offset = int(offset_match.group(1))
                    next_offset = current_offset + 25  # Booking.com zeigt normalerweise 25 pro Seite
                    if next_offset <= 100:  # Limitiere auf 4 Seiten pro Stadt
                        next_page_url = re.sub(r'offset=\d+', f'offset={next_offset}', current_url)
            else:
                # Erste Pagination - füge offset=25 hinzu
                separator = '&' if '?' in current_url else '?'
                next_page_url = f"{current_url}{separator}offset=25"
        
        # Folge der nächsten Seite
        if next_page_url:
            print(f"🔄 Following next page: {next_page_url}")
            yield response.follow(next_page_url, callback=self.parse)
        else:
            print("🏁 No more pages found for this search")
    
    def extract_hotels(self, response):
        # Verschiedene Selektoren für Booking.com Hotels
        hotel_selectors = [
            '[data-testid="property-card"]',
            '.sr_property_block',
            '[data-testid="property-card-container"]',
            '.sr-card',
            '.listingsContainer > div',
            'div[data-hotelid]'
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
            
            # Fallback: Suche nach Links, die auf Hotels hindeuten
            hotel_links = response.css('a[href*="hotel"]::attr(href), a[href*="property"]::attr(href)').getall()
            print(f"Found {len(hotel_links)} potential hotel links")
            
            # Extrahiere auch alle Texte, die wie Hotelnamen aussehen
            potential_names = response.css('h1::text, h2::text, h3::text, h4::text').getall()
            
            # Extrahiere verfügbare Bilder von der Seite
            all_images = response.css('img::attr(src), img::attr(data-src)').getall()
            quality_images = []
            for img_url in all_images:
                if img_url and any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    if not any(skip in img_url.lower() for skip in ['icon', 'logo', 'sprite', 'pixel', 'svg']):
                        quality_images.append(response.urljoin(img_url))
            
            for i, link in enumerate(hotel_links[:15]):  # Limitiere auf erste 15
                name = potential_names[i] if i < len(potential_names) else f"Hotel {i+1}"
                # Verteile Bilder auf Hotels
                hotel_images = quality_images[i*2:(i+1)*2] if i*2 < len(quality_images) else []
                yield {
                    "source": "Booking.com",
                    "name": name.strip() if name else None,
                    "link": response.urljoin(link),
                    "rating": None,
                    "price": None,
                    "location": None,
                    "description": "Extracted from general page content",
                    "images": hotel_images,
                    "image_count": len(hotel_images)
                }
            
            # Wenn immer noch nichts gefunden, gib wenigstens die URL-Info zurück
            if not hotel_links:
                # Sammle trotzdem verfügbare Bilder
                page_images = response.css('img::attr(src), img::attr(data-src)').getall()
                sample_images = []
                for img_url in page_images[:5]:
                    if img_url and any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        if not any(skip in img_url.lower() for skip in ['icon', 'logo', 'sprite']):
                            sample_images.append(response.urljoin(img_url))
                
                yield {
                    "source": "Booking.com",
                    "name": "Page accessed successfully",
                    "link": response.url,
                    "rating": None,
                    "price": None,
                    "location": None,
                    "description": f"Page loaded with status {response.status}, check debug_booking_response.html for content",
                    "images": sample_images,
                    "image_count": len(sample_images)
                }
            return
        
        # Verarbeite gefundene Hotels
        print(f"🏨 Processing {len(hotels_found)} hotels...")
        for i, hotel in enumerate(hotels_found):
            print(f"Processing hotel {i+1}/{len(hotels_found)}")
            
            # Extrahiere Hotelnamen
            name_selectors = [
                '[data-testid="title"]::text',
                'h3 a::text', 'h3::text', 'h2::text', 'h1::text',
                '.sr-hotel__name::text', '.sr-hotel__title::text',
                '[data-testid="property-card-title"]::text',
                'a[data-testid="property-card-title-link"]::text',
                '.fcab3ed991::text',
                '.fde444d7ef::text'
            ]
            name = None
            for name_sel in name_selectors:
                name_text = hotel.css(name_sel).get()
                if name_text and name_text.strip():
                    name = name_text.strip()
                    break
            
            # Extrahiere Link
            link_selectors = [
                'a::attr(href)',
                '[data-testid="property-card-title-link"]::attr(href)',
                'h3 a::attr(href)'
            ]
            link = None
            for link_sel in link_selectors:
                link_url = hotel.css(link_sel).get()
                if link_url:
                    link = response.urljoin(link_url)
                    break
            
            # Extrahiere Bewertung
            rating_selectors = [
                '[data-testid="review-score"] div::text',
                '.bui-review-score__badge::text',
                '.review-score-badge::text',
                '[aria-label*="Scored"]::text',
                '.a3b8729ab1::text',
                '[data-testid="rating-stars"] span::text'
            ]
            rating = None
            for rating_sel in rating_selectors:
                rating_text = hotel.css(rating_sel).get()
                if rating_text:
                    # Extrahiere Nummer aus Text
                    rating_match = re.search(r'(\d+[.,]?\d*)', rating_text.replace(',', '.'))
                    if rating_match:
                        rating = rating_match.group(1)
                        break
            
            # Extrahiere Preis
            price_selectors = [
                '[data-testid="price-and-discounted-price"] span::text',
                '.priceDisplay span::text', '.bui-price-display__value::text',
                '.sr-card__price .bui-price-display__value::text',
                '.fccdc0e4f3::text', '.dd5a8c3738::text',
                'span[aria-hidden="true"]::text'
            ]
            price = None
            for price_sel in price_selectors:
                price_texts = hotel.css(price_sel).getall()
                for price_text in price_texts:
                    if price_text and ('€' in price_text or 'CHF' in price_text or '$' in price_text):
                        # Extrahiere Nummer aus Preis
                        price_match = re.search(r'(\d+[.,]?\d*)', price_text.replace(',', '').replace('.', ''))
                        if price_match:
                            price = price_match.group(1)
                            break
                if price:
                    break
            
            # Extrahiere Standort
            location_selectors = [
                '[data-testid="address"]::text',
                '.sr-card__address::text',
                '.f4bd0794db::text',
                '.f419a93f12::text'
            ]
            location = None
            for loc_sel in location_selectors:
                location_text = hotel.css(loc_sel).get()
                if location_text and location_text.strip():
                    location = location_text.strip()
                    break
            
            # Extrahiere Bild-URLs
            image_selectors = [
                'img::attr(src)',
                'img::attr(data-src)',
                '[data-testid="property-card-image"] img::attr(src)',
                '.sr-card__image img::attr(src)',
                '.bui-card__image img::attr(src)',
                '.hotel_image img::attr(src)',
                'picture img::attr(src)',
                'img[alt*="hotel" i]::attr(src)',
                'img[alt*="property" i]::attr(src)'
            ]
            images = []
            for img_sel in image_selectors:
                img_urls = hotel.css(img_sel).getall()
                for img_url in img_urls:
                    if img_url and any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        # Vollständige URL erstellen
                        full_img_url = response.urljoin(img_url)
                        # Nur hochwertige Bilder (nicht Icons/Logos)
                        if not any(skip in img_url.lower() for skip in ['icon', 'logo', 'sprite', 'pixel']):
                            images.append(full_img_url)
                if images:
                    break
            
            # Limitiere auf die ersten 3 Bilder
            images = images[:3] if images else []
            
            # Beschreibung aus allem Text extrahieren
            description = ' '.join(hotel.css('::text').getall()).strip()[:200]
            
            print(f"Hotel: {name}, Price: {price}, Rating: {rating}, Location: {location}, Images: {len(images)}")
            
            if name or link or description:  # Ausgeben wenn irgendetwas gefunden wurde
                yield {
                    "source": "Booking.com",
                    "name": name,
                    "link": link,
                    "rating": rating,
                    "price": price,
                    "location": location,
                    "description": description if description else None,
                    "images": images,
                    "image_count": len(images)
                }