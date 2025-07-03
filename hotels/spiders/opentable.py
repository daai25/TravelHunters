import scrapy
import json
import re


class OpenTableSpider(scrapy.Spider):
    name = 'opentable'
    allowed_domains = ['opentable.com']
    start_urls = [
        'https://www.opentable.com/lists/best-restaurants-paris',
        'https://www.opentable.com/lists/best-restaurants-london', 
        'https://www.opentable.com/lists/best-restaurants-new-york',
        'https://www.opentable.com/lists/best-restaurants-rome',
        'https://www.opentable.com/lists/best-restaurants-tokyo',
    ]

    custom_settings = {
        'FEEDS': {
            'opentable_output.json': {
                'format': 'json',
                'overwrite': True,
            },
        },
        'DOWNLOAD_DELAY': 1.5,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    def parse(self, response):
        # Extract restaurant cards/links
        restaurant_links = response.css('a[data-test="restaurant-list-item"]::attr(href)').getall()
        
        # Alternative selectors if the above doesn't work
        if not restaurant_links:
            restaurant_links = response.css('.rest-row-info a::attr(href)').getall()
        
        if not restaurant_links:
            restaurant_links = response.css('a[href*="/restaurants/"]::attr(href)').getall()
        
        for link in restaurant_links[:15]:  # Limit for testing
            if '/restaurants/' in link:
                full_url = response.urljoin(link)
                yield response.follow(full_url, self.parse_restaurant)
        
        # Fallback: extract basic info from listing page
        if not restaurant_links:
            yield from self.parse_restaurant_list(response)

    def parse_restaurant_list(self, response):
        """Fallback method to extract basic restaurant info from listing page"""
        restaurants = response.css('[data-test="restaurant-list-item"]')
        
        for restaurant in restaurants:
            name = restaurant.css('h3::text, .rest-row-name::text').get()
            cuisine = restaurant.css('.rest-row-meta-cuisine::text').get()
            location = restaurant.css('.rest-row-meta-location::text').get()
            rating = restaurant.css('.star-rating::attr(aria-label)').get()
            price = restaurant.css('.rest-row-pricing::text').get()
            
            if name:
                yield {
                    'name': name.strip(),
                    'cuisine': cuisine.strip() if cuisine else '',
                    'location': location.strip() if location else '',
                    'rating': rating,
                    'price_range': price.strip() if price else '',
                    'url': response.url,
                    'source': 'OpenTable',
                    'category': 'restaurant',
                    'type': 'dining'
                }

    def parse_restaurant(self, response):
        # Extract restaurant details
        name = response.css('h1[data-test="restaurant-name"]::text').get()
        if not name:
            name = response.css('.restaurant-title h1::text').get()
        
        cuisine = response.css('[data-test="restaurant-cuisine"]::text').get()
        if not cuisine:
            cuisine = response.css('.restaurant-cuisine::text').get()
        
        description = response.css('[data-test="restaurant-description"] p::text').get()
        if not description:
            description = response.css('.restaurant-description::text').get()
        
        # Extract location/address
        address = response.css('[data-test="restaurant-address"]::text').get()
        if not address:
            address = response.css('.restaurant-address::text').get()
        
        # Extract rating
        rating = response.css('.star-rating::attr(aria-label)').get()
        if rating:
            rating_match = re.search(r'(\d+\.?\d*)', rating)
            rating = float(rating_match.group(1)) if rating_match else None
        
        # Extract price range
        price_range = response.css('.restaurant-pricing::text').get()
        
        # Extract photos
        photos = response.css('.restaurant-photos img::attr(src)').getall()
        
        # Extract menu highlights
        menu_items = response.css('.menu-item-name::text').getall()
        
        # Extract amenities/features
        amenities = response.css('.restaurant-amenities li::text').getall()
        
        # Extract phone and website
        phone = response.css('[data-test="restaurant-phone"]::attr(href)').get()
        if phone and phone.startswith('tel:'):
            phone = phone[4:]
        
        website = response.css('[data-test="restaurant-website"]::attr(href)').get()
        
        yield {
            'name': name.strip() if name else 'Unknown Restaurant',
            'cuisine': cuisine.strip() if cuisine else '',
            'description': description.strip() if description else '',
            'address': address.strip() if address else '',
            'rating': rating,
            'price_range': price_range.strip() if price_range else '',
            'photos': photos[:3] if photos else [],
            'menu_highlights': menu_items[:5] if menu_items else [],
            'amenities': [a.strip() for a in amenities] if amenities else [],
            'phone': phone,
            'website': website,
            'url': response.url,
            'source': 'OpenTable',
            'category': 'restaurant',
            'type': 'dining'
        }
