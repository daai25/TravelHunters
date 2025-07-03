import scrapy
from scrapy.spiders import Spider
import re


class DemoSpider(Spider):
    name = 'demo'
    # Verwende eine einfache, öffentlich zugängliche Test-Seite
    start_urls = [
        'https://quotes.toscrape.com/',
        'https://books.toscrape.com/',
    ]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'CONCURRENT_REQUESTS': 1,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 3,
    }

    def parse(self, response):
        print(f"🌐 Parsing Demo response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Debug: Speichere HTML für Analyse
        filename = f'debug_demo_response_{response.url.split("/")[-2] or "root"}.html'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"💾 HTML saved to {filename}")
        
        if 'quotes.toscrape.com' in response.url:
            # Scrape Zitate als "Hotel"-Daten für Demo-Zwecke
            quotes = response.css('div.quote')
            print(f"✅ Found {len(quotes)} quotes on quotes page")
            
            for i, quote in enumerate(quotes[:10]):  # Nur erste 10
                text = quote.css('span.text::text').get()
                author = quote.css('small.author::text').get()
                tags = quote.css('div.tags a.tag::text').getall()
                
                # Konvertiere zu "Hotel"-Format für Konsistenz
                yield {
                    'source': 'Demo-Quotes',
                    'name': f"Quote by {author}" if author else f"Quote {i+1}",
                    'link': response.url,
                    'rating': str(len(tags)),  # Anzahl Tags als "Bewertung"
                    'price': str(len(text)) if text else None,  # Textlänge als "Preis"
                    'location': 'Quotes Website',
                    'description': text[:100] + '...' if text and len(text) > 100 else text
                }
                
        elif 'books.toscrape.com' in response.url:
            # Scrape Bücher als "Hotel"-Daten für Demo-Zwecke
            books = response.css('article.product_pod')
            print(f"✅ Found {len(books)} books on books page")
            
            for i, book in enumerate(books[:10]):  # Nur erste 10
                title = book.css('h3 a::attr(title)').get()
                price = book.css('div.product_price p.price_color::text').get()
                rating_class = book.css('p.star-rating::attr(class)').get()
                link = book.css('h3 a::attr(href)').get()
                
                # Extrahiere Rating aus CSS-Klasse
                rating = None
                if rating_class:
                    rating_words = {'One': '1', 'Two': '2', 'Three': '3', 'Four': '4', 'Five': '5'}
                    for word, num in rating_words.items():
                        if word in rating_class:
                            rating = num
                            break
                
                # Clean price
                clean_price = None
                if price:
                    numbers = re.findall(r'(\d+\.?\d*)', price)
                    if numbers:
                        clean_price = numbers[0]
                
                # Konvertiere zu "Hotel"-Format
                yield {
                    'source': 'Demo-Books',
                    'name': title if title else f"Book {i+1}",
                    'link': response.urljoin(link) if link else response.url,
                    'rating': rating,
                    'price': clean_price,
                    'location': 'Books Website',
                    'description': f"A book titled: {title}" if title else "A book from the collection"
                }
        
        else:
            # Fallback für andere Seiten
            yield {
                'source': 'Demo-Fallback',
                'name': 'Demo page accessed',
                'link': response.url,
                'rating': None,
                'price': None,
                'location': 'Demo Website',
                'description': f'Successfully accessed {response.url} with status {response.status}'
            }
