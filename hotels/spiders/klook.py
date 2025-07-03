import scrapy
import json
import re


class KlookSpider(scrapy.Spider):
    name = 'klook'
    allowed_domains = ['klook.com']
    start_urls = [
        'https://www.klook.com/en-US/city/2-tokyo-things-to-do/',
        'https://www.klook.com/en-US/city/5-singapore-things-to-do/',
        'https://www.klook.com/en-US/city/7-hong-kong-things-to-do/', 
        'https://www.klook.com/en-US/city/63-seoul-things-to-do/',
        'https://www.klook.com/en-US/city/47-bangkok-things-to-do/',
        'https://www.klook.com/en-US/city/17-paris-things-to-do/',
        'https://www.klook.com/en-US/city/8-london-things-to-do/',
    ]

    custom_settings = {
        'FEEDS': {
            'klook_output.json': {
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
        city_match = re.search(r'/city/\d+-([^-]+)', response.url)
        city = city_match.group(1) if city_match else 'unknown'
        
        # Extract activity links
        activity_links = response.css('a[href*="/activity/"]::attr(href)').getall()
        
        # Alternative selectors for activity cards
        if not activity_links:
            activity_links = response.css('.card a::attr(href), .activity-card a::attr(href)').getall()
        
        for link in activity_links[:15]:  # Limit for testing
            if '/activity/' in link:
                full_url = response.urljoin(link)
                yield response.follow(full_url, self.parse_activity, meta={'city': city})
        
        # Fallback: extract basic info from listing page
        if not activity_links:
            yield from self.parse_activity_list(response, city)

    def parse_activity_list(self, response, city):
        """Fallback method to extract basic activity info from listing page"""
        activities = response.css('.card, .activity-card, [data-testid="activity-card"]')
        
        for activity in activities:
            name = activity.css('h3::text, .card-title::text, .activity-name::text').get()
            price = activity.css('.price::text, .activity-price::text').get()
            rating = activity.css('.rating::text').get()
            image = activity.css('img::attr(src)').get()
            link = activity.css('a::attr(href)').get()
            
            if name:
                # Clean price
                if price:
                    price_match = re.search(r'[\d,]+\.?\d*', price.replace(',', ''))
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
                    'image': response.urljoin(image) if image else '',
                    'url': response.urljoin(link) if link else response.url,
                    'source': 'Klook',
                    'category': 'activity',
                    'type': 'tour_activity'
                }

    def parse_activity(self, response):
        city = response.meta.get('city', 'unknown')
        
        # Extract activity details
        name = response.css('h1::text').get()
        if not name:
            name = response.css('.activity-title::text, .product-title::text').get()
        
        # Extract description
        description_parts = response.css('.description p::text, .activity-description p::text').getall()
        description = ' '.join(description_parts) if description_parts else ''
        
        # Extract highlights/features
        highlights = response.css('.highlights li::text, .features li::text').getall()
        
        # Extract price
        price = response.css('.price::text, .current-price::text').get()
        if price:
            price_match = re.search(r'[\d,]+\.?\d*', price.replace(',', ''))
            price_clean = price_match.group() if price_match else price
        else:
            price_clean = None
        
        # Extract rating and reviews
        rating = response.css('.rating-score::text, .average-rating::text').get()
        if rating:
            rating_match = re.search(r'(\d+\.?\d*)', rating)
            rating_clean = float(rating_match.group(1)) if rating_match else None
        else:
            rating_clean = None
        
        review_count = response.css('.review-count::text').get()
        if review_count:
            count_match = re.search(r'(\d+)', review_count)
            review_count_clean = int(count_match.group(1)) if count_match else None
        else:
            review_count_clean = None
        
        # Extract duration
        duration = response.css('.duration::text, .activity-duration::text').get()
        
        # Extract category/type
        category = response.css('.category::text, .activity-category::text').get()
        if not category:
            # Try to extract from breadcrumbs
            breadcrumbs = response.css('.breadcrumb a::text').getall()
            category = breadcrumbs[-2] if len(breadcrumbs) > 1 else ''
        
        # Extract images
        images = response.css('.gallery img::attr(src), .activity-images img::attr(src)').getall()
        
        # Extract what's included
        included = response.css('.included li::text, .whats-included li::text').getall()
        
        # Extract meeting point/location
        meeting_point = response.css('.meeting-point::text, .location::text').get()
        
        # Extract cancellation policy
        cancellation = response.css('.cancellation::text, .cancellation-policy::text').get()
        
        # Extract languages available
        languages = response.css('.languages li::text, .available-languages li::text').getall()
        
        # Determine activity type based on category or name
        activity_type = 'general'
        name_lower = name.lower() if name else ''
        category_lower = category.lower() if category else ''
        
        if any(word in name_lower or word in category_lower for word in ['tour', 'walk', 'guide']):
            activity_type = 'tour'
        elif any(word in name_lower or word in category_lower for word in ['ticket', 'admission', 'entry']):
            activity_type = 'attraction'
        elif any(word in name_lower or word in category_lower for word in ['food', 'dining', 'cooking']):
            activity_type = 'food_experience'
        elif any(word in name_lower or word in category_lower for word in ['transport', 'transfer', 'car', 'bus']):
            activity_type = 'transport'
        elif any(word in name_lower or word in category_lower for word in ['spa', 'massage', 'wellness']):
            activity_type = 'wellness'
        
        yield {
            'name': name.strip() if name else 'Unknown Activity',
            'description': description.strip() if description else '',
            'highlights': [h.strip() for h in highlights] if highlights else [],
            'price': price_clean,
            'rating': rating_clean,
            'review_count': review_count_clean,
            'duration': duration.strip() if duration else '',
            'category': category.strip() if category else '',
            'city': city,
            'images': [response.urljoin(img) for img in images[:3]] if images else [],
            'included': [i.strip() for i in included] if included else [],
            'meeting_point': meeting_point.strip() if meeting_point else '',
            'cancellation': cancellation.strip() if cancellation else '',
            'languages': [l.strip() for l in languages] if languages else [],
            'url': response.url,
            'source': 'Klook',
            'activity_type': activity_type,
            'type': 'tour_activity'
        }
