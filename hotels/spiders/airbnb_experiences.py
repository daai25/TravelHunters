import scrapy
import json
import re


class AirbnbExperiencesSpider(scrapy.Spider):
    name = 'airbnb_experiences'
    allowed_domains = ['airbnb.com']
    start_urls = [
        'https://www.airbnb.com/s/paris/experiences',
        'https://www.airbnb.com/s/london/experiences',
        'https://www.airbnb.com/s/new-york/experiences',
        'https://www.airbnb.com/s/tokyo/experiences',
        'https://www.airbnb.com/s/rome/experiences',
    ]

    custom_settings = {
        'FEEDS': {
            'airbnb_experiences_output.json': {
                'format': 'json',
                'overwrite': True,
            },
        },
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    def parse(self, response):
        # Extract city from URL
        city_match = re.search(r'/s/([^/]+)/experiences', response.url)
        city = city_match.group(1).replace('-', ' ') if city_match else 'unknown'
        
        # Extract experience links
        experience_links = response.css('a[href*="/experiences/"]::attr(href)').getall()
        
        for link in experience_links[:10]:  # Limit for testing
            if '/experiences/' in link and link.count('/') > 3:  # Avoid category pages
                full_url = response.urljoin(link)
                yield response.follow(full_url, self.parse_experience, meta={'city': city})
        
        # Fallback: extract basic info from cards
        yield from self.parse_experience_cards(response, city)

    def parse_experience_cards(self, response, city):
        """Extract basic experience info from listing cards"""
        # Try different selectors for experience cards
        cards = response.css('[data-testid="card-container"], .listing-card, .experience-card')
        
        for card in cards:
            name = card.css('h3::text, .listing-link::text').get()
            price = card.css('[data-testid="price"]::text, .price::text').get()
            rating = card.css('[data-testid="rating"]::text, .rating::text').get()
            image = card.css('img::attr(src)').get()
            link = card.css('a::attr(href)').get()
            
            if name:
                # Clean price
                if price:
                    price_match = re.search(r'[\d,]+', price.replace(',', ''))
                    price_clean = price_match.group() if price_match else price
                else:
                    price_clean = None
                
                # Clean rating
                if rating:
                    rating_match = re.search(r'(\d+\.?\d*)', rating)
                    rating_clean = float(rating_match.group(1)) if rating_match else None
                else:
                    rating_clean = None
                
                yield {
                    'name': name.strip(),
                    'price': price_clean,
                    'rating': rating_clean,
                    'city': city,
                    'image': image,
                    'url': response.urljoin(link) if link else response.url,
                    'source': 'Airbnb Experiences',
                    'category': 'experience',
                    'type': 'local_experience'
                }

    def parse_experience(self, response):
        city = response.meta.get('city', 'unknown')
        
        # Extract experience details
        name = response.css('h1::text').get()
        if not name:
            name = response.css('[data-testid="experience-title"]::text').get()
        
        # Extract description
        description = response.css('[data-section-id="DESCRIPTION_DEFAULT"] span::text').getall()
        description = ' '.join(description) if description else ''
        
        # Extract what you'll do sections
        what_youll_do = response.css('[data-section-id="WHAT_YOULL_DO"] span::text').getall()
        what_youll_do = ' '.join(what_youll_do) if what_youll_do else ''
        
        # Extract price
        price = response.css('[data-testid="book-it-price"]::text').get()
        if not price:
            price = response.css('.price::text').get()
        
        if price:
            price_match = re.search(r'[\d,]+', price.replace(',', ''))
            price_clean = price_match.group() if price_match else price
        else:
            price_clean = None
        
        # Extract rating and review count
        rating = response.css('[data-testid="pdp-reviews-highlight-banner-host-rating"]::text').get()
        if rating:
            rating_match = re.search(r'(\d+\.?\d*)', rating)
            rating_clean = float(rating_match.group(1)) if rating_match else None
        else:
            rating_clean = None
        
        review_count = response.css('[data-testid="pdp-reviews-highlight-banner-review-count"]::text').get()
        if review_count:
            count_match = re.search(r'(\d+)', review_count)
            review_count_clean = int(count_match.group(1)) if count_match else None
        else:
            review_count_clean = None
        
        # Extract duration
        duration = response.css('[data-testid="experience-duration"]::text').get()
        if not duration:
            # Look for duration in text
            duration_text = response.css('*:contains("hour"):text()').getall()
            for text in duration_text:
                if 'hour' in text.lower():
                    duration = text.strip()
                    break
        
        # Extract host info
        host_name = response.css('[data-testid="experience-host-name"]::text').get()
        
        # Extract what's included
        included = response.css('[data-section-id="WHAT_I_PROVIDE"] li::text').getall()
        
        # Extract group size
        group_size = response.css('[data-testid="experience-group-size"]::text').get()
        
        # Extract languages
        languages = response.css('[data-testid="experience-language"] span::text').getall()
        
        # Extract location/meeting point
        location = response.css('[data-testid="experience-location"]::text').get()
        
        # Extract photos
        photos = response.css('img[data-testid="experience-photo"]::attr(src)').getall()
        if not photos:
            photos = response.css('.experience-photos img::attr(src)').getall()
        
        # Extract availability info
        availability = response.css('[data-testid="experience-availability"]::text').get()
        
        # Extract categories/tags
        categories = response.css('[data-testid="experience-category"] span::text').getall()
        
        # Extract age requirements
        age_requirement = response.css('*:contains("age"):text()').get()
        if age_requirement and 'age' in age_requirement.lower():
            age_match = re.search(r'(\d+).+age', age_requirement.lower())
            min_age = age_match.group(1) if age_match else None
        else:
            min_age = None
        
        yield {
            'name': name.strip() if name else 'Unknown Experience',
            'description': description.strip() if description else '',
            'what_youll_do': what_youll_do.strip() if what_youll_do else '',
            'price': price_clean,
            'rating': rating_clean,
            'review_count': review_count_clean,
            'duration': duration.strip() if duration else '',
            'host_name': host_name.strip() if host_name else '',
            'city': city,
            'included': [i.strip() for i in included] if included else [],
            'group_size': group_size.strip() if group_size else '',
            'languages': [l.strip() for l in languages] if languages else [],
            'location': location.strip() if location else '',
            'photos': photos[:3] if photos else [],
            'availability': availability.strip() if availability else '',
            'categories': [c.strip() for c in categories] if categories else [],
            'min_age': min_age,
            'url': response.url,
            'source': 'Airbnb Experiences',
            'category': 'local_experience',
            'type': 'local_experience'
        }
