import scrapy
from scrapy.spiders import Spider
from bs4 import BeautifulSoup
import re


class HolidayCheckSpider(Spider):
    name = 'holidaycheck'
    start_urls = [
        'https://www.holidaycheck.de/pi/hotels-in-mallorca/4a90b043-68be-35de-9f79-ddc56529b8ce',
        'https://www.holidaycheck.de/'
    ]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'HTTPERROR_ALLOWED_CODES': [404, 403, 500, 502, 503]
    }

    def parse(self, response):
        print(f"🌐 Parsing response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        with open('debug_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("💾 HTML saved to debug_response.html")
        
        # Versuche verschiedene mögliche Selektoren für Hotels
        hotel_selectors = [
            'div[data-testid="hotel-card"]',
            '.hotelTeaser',
            '.hotel-teaser',
            '.hotel-card',
            '.search-result-item',
            '.hotel-item',
            '.accommodation-item',
            'article',
            '.result-item'
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
            hotel_links = response.css('a[href*="hotel"]::attr(href), a[href*="accommodation"]::attr(href)').getall()
            print(f"Found {len(hotel_links)} potential hotel links")
            
            # Extrahiere auch alle Texte, die wie Hotelnamen aussehen
            potential_names = response.css('h1::text, h2::text, h3::text, h4::text, css-1xu06z6').getall()
            
            for i, link in enumerate(hotel_links[:10]):  # Limitiere auf erste 10
                name = potential_names[i] if i < len(potential_names) else f"Hotel {i+1}"
                yield {
                    "source": "HolidayCheck",
                    "name": name.strip() if name else None,
                    "link": response.urljoin(link),
                    "rating": None,
                    "price": None,
                    "location": None,
                    "description": "Extracted from general page content"
                }
            
            # Wenn immer noch nichts gefunden, gib wenigstens die URL-Info zurück
            if not hotel_links:
                yield {
                    "source": "HolidayCheck",
                    "name": "Page accessed successfully",
                    "link": response.url,
                    "rating": None,
                    "price": None,
                    "location": None,
                    "description": f"Page loaded with status {response.status}, check debug_response.html for content"
                }
            return
        
        # Verarbeite gefundene Hotels
        for hotel in hotels_found:
            # Extrahiere Hotelnamen
            name_selectors = [
                '.css-1xu06z6',
            ]
            name = None
            for name_sel in name_selectors:
                name_text = hotel.css(name_sel).get()
                if name_text and name_text.strip():
                    name = name_text.strip()
                    break
            
            # Extrahiere Link
            link = hotel.css('a::attr(href)').get()
            if link:
                link = response.urljoin(link)
            
            # Extrahiere Bewertung
            rating_selectors = [
                '.css-wx75r6'
            ]
            rating = None
            for rating_sel in rating_selectors:
                rating_text = hotel.css(rating_sel).get()
                if rating_text:
                    rating_match = re.search(r'(\d+[.,]?\d*)', rating_text)
                    if rating_match:
                        rating = rating_match.group(1).replace(',', '.')
                        break
            
            # Extrahiere Preis
            price_selectors = [
                '.css-14h7dlz'
            ]
            price = None
            for price_sel in price_selectors:
                price_text = hotel.css(price_sel).get()
                if price_text:
                    price_match = re.search(r'(\d+[.,]?\d*)', price_text.replace('€', '').replace('CHF', ''))
                    if price_match:
                        price = price_match.group(1).replace(',', '.')
                        break
            
            # Extrahiere Standort
            location_selectors = [
                '.css-oh7js0'
            ]
            location = None
            for loc_sel in location_selectors:
                location_text = hotel.css(loc_sel).get()
                if location_text and location_text.strip():
                    location = location_text.strip()
                    break
            
            # Beschreibung aus allem Text extrahieren
            description = ' '.join(hotel.css('::text').getall()).strip()[:200]
            
            if name or link or description:  # Ausgeben wenn irgendetwas gefunden wurde
                yield {
                    "source": "HolidayCheck",
                    "name": name,
                    "link": link,
                    "rating": rating,
                    "price": price,
                    "location": location,
                    "description": description if description else None
                }
