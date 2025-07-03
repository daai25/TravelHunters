import scrapy
from scrapy.spiders import Spider
import re


class BookingSpider(Spider):
    name = 'booking'
    start_urls = [
        'https://www.booking.com/searchresults.html?ss=Zurich&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
        'https://www.booking.com/searchresults.html?ss=Bern&checkin=2025-08-01&checkout=2025-08-03&group_adults=2&no_rooms=1',
    ]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 3,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'HTTPERROR_ALLOWED_CODES': [404, 403, 500, 502, 503],
        'CONCURRENT_REQUESTS': 1,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 2,
        'AUTOTHROTTLE_MAX_DELAY': 10,
    }

    def parse(self, response):
        print(f"🌐 Parsing response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open('debug_booking_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_booking_response.html")
        
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