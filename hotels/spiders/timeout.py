import scrapy
import json
import re


class TimeOutSpider(scrapy.Spider):
    name = 'timeout'
    allowed_domains = ['timeout.com']
    start_urls = [
        'https://www.timeout.com/paris/things-to-do',
        'https://www.timeout.com/london/things-to-do',
        'https://www.timeout.com/newyork/things-to-do',
        'https://www.timeout.com/berlin/things-to-do',
        'https://www.timeout.com/barcelona/things-to-do',
    ]

    custom_settings = {
        'FEEDS': {
            'timeout_output.json': {
                'format': 'json',
                'overwrite': True,
            },
        },
        'DOWNLOAD_DELAY': 1,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    def parse(self, response):
        # Extract city from URL
        city_match = re.search(r'timeout\.com/([^/]+)/', response.url)
        city = city_match.group(1) if city_match else 'unknown'
        
        # Extract article/venue links
        article_links = response.css('a[href*="/things-to-do/"]::attr(href)').getall()
        venue_links = response.css('a[href*="/venues/"]::attr(href)').getall()
        
        # Combine and deduplicate links
        all_links = list(set(article_links + venue_links))
        
        for link in all_links[:20]:  # Limit for testing
            full_url = response.urljoin(link)
            yield response.follow(full_url, self.parse_venue, meta={'city': city})
        
        # Also try to extract direct venue info from listing
        venues = response.css('.card, .venue-card, [data-testid="venue-card"]')
        for venue in venues:
            yield from self.parse_venue_card(venue, response, city)

    def parse_venue_card(self, venue_selector, response, city):
        """Extract basic venue info from card on listing page"""
        name = venue_selector.css('h3::text, .card-title::text, .venue-name::text').get()
        description = venue_selector.css('.card-description::text, .venue-description::text').get()
        category = venue_selector.css('.card-category::text, .venue-category::text').get()
        image = venue_selector.css('img::attr(src)').get()
        link = venue_selector.css('a::attr(href)').get()
        
        if name:
            yield {
                'name': name.strip(),
                'description': description.strip() if description else '',
                'category': category.strip() if category else '',
                'city': city,
                'image': response.urljoin(image) if image else '',
                'url': response.urljoin(link) if link else response.url,
                'source': 'Time Out',
                'type': 'activity'
            }

    def parse_venue(self, response):
        city = response.meta.get('city', 'unknown')
        
        # Extract venue/activity details
        name = response.css('h1::text').get()
        if not name:
            name = response.css('.article-title::text, .venue-title::text').get()
        
        # Extract description/content
        description_paragraphs = response.css('.article-content p::text, .venue-content p::text').getall()
        description = ' '.join(description_paragraphs) if description_paragraphs else ''
        
        # Extract category/type
        category = response.css('.article-category::text, .venue-category::text').get()
        if not category:
            # Try to extract from breadcrumbs
            breadcrumbs = response.css('.breadcrumb a::text').getall()
            category = breadcrumbs[-1] if breadcrumbs else ''
        
        # Extract address/location
        address = response.css('.venue-address::text, .address::text').get()
        
        # Extract rating if available
        rating = response.css('.rating::text, .score::text').get()
        if rating:
            rating_match = re.search(r'(\d+\.?\d*)', rating)
            rating = float(rating_match.group(1)) if rating_match else None
        
        # Extract price information
        price = response.css('.price::text, .venue-price::text').get()
        
        # Extract opening hours
        hours = response.css('.opening-hours::text, .venue-hours::text').getall()
        hours = ' '.join(hours) if hours else ''
        
        # Extract images
        images = response.css('.article-images img::attr(src), .venue-gallery img::attr(src)').getall()
        
        # Extract tags/features
        tags = response.css('.tags a::text, .venue-tags a::text').getall()
        
        # Extract contact info
        phone = response.css('.phone::text, .venue-phone::text').get()
        website = response.css('.website::attr(href), .venue-website::attr(href)').get()
        
        # Determine activity type
        activity_type = 'general'
        if any(word in (category.lower() if category else '') for word in ['restaurant', 'bar', 'cafe']):
            activity_type = 'dining'
        elif any(word in (category.lower() if category else '') for word in ['museum', 'gallery', 'art']):
            activity_type = 'culture'
        elif any(word in (category.lower() if category else '') for word in ['park', 'outdoor', 'nature']):
            activity_type = 'outdoor'
        elif any(word in (category.lower() if category else '') for word in ['shop', 'market', 'store']):
            activity_type = 'shopping'
        
        yield {
            'name': name.strip() if name else 'Unknown Venue',
            'description': description.strip() if description else '',
            'category': category.strip() if category else '',
            'city': city,
            'address': address.strip() if address else '',
            'rating': rating,
            'price': price.strip() if price else '',
            'opening_hours': hours.strip() if hours else '',
            'images': [response.urljoin(img) for img in images[:3]] if images else [],
            'tags': [tag.strip() for tag in tags] if tags else [],
            'phone': phone.strip() if phone else '',
            'website': website,
            'url': response.url,
            'source': 'Time Out',
            'activity_type': activity_type,
            'type': 'activity'
        }
