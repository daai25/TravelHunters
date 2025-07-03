import scrapy
import json
import re


class AtlasObscuraSpider(scrapy.Spider):
    name = 'atlas_obscura'
    allowed_domains = ['atlasobscura.com']
    start_urls = [
        'https://www.atlasobscura.com/places?sort=random',
        'https://www.atlasobscura.com/places/filter/type/food-and-drink',
        'https://www.atlasobscura.com/places/filter/type/museums-and-attractions',
        'https://www.atlasobscura.com/places/filter/type/nature',
    ]

    custom_settings = {
        'FEEDS': {
            'atlas_obscura_output.json': {
                'format': 'json',
                'overwrite': True,
            },
        },
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
    }

    def parse(self, response):
        # Extract place links
        place_links = response.css('a.content-card::attr(href)').getall()
        
        for link in place_links[:20]:  # Limit for testing
            if link.startswith('/places/'):
                full_url = response.urljoin(link)
                yield response.follow(full_url, self.parse_place)
        
        # Follow pagination (if exists)
        next_page = response.css('a.pagination-next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_place(self, response):
        # Extract place details
        name = response.css('h1.DDPage__header-title::text').get()
        if not name:
            name = response.css('.place-title::text').get()
        
        description = response.css('.DDPage__body p::text').getall()
        description = ' '.join(description) if description else None
        
        location = response.css('.DDPage__header-subtitle::text').get()
        if not location:
            location = response.css('.place-location::text').get()
        
        # Extract coordinates if available
        coordinates = None
        map_data = response.css('script:contains("latitude")::text').get()
        if map_data:
            lat_match = re.search(r'"latitude":\s*([+-]?\d+\.?\d*)', map_data)
            lng_match = re.search(r'"longitude":\s*([+-]?\d+\.?\d*)', map_data)
            if lat_match and lng_match:
                coordinates = {
                    'latitude': float(lat_match.group(1)),
                    'longitude': float(lng_match.group(1))
                }
        
        # Extract tags/categories
        tags = response.css('.place-tags .tag::text').getall()
        
        # Extract images
        images = response.css('.place-gallery img::attr(src)').getall()
        
        # Rating/popularity indicators
        visits = response.css('.visit-count::text').get()
        
        yield {
            'name': name.strip() if name else 'Unknown Place',
            'description': description.strip() if description else '',
            'location': location.strip() if location else '',
            'coordinates': coordinates,
            'tags': [tag.strip() for tag in tags] if tags else [],
            'images': images[:3] if images else [],  # Limit to first 3 images
            'visits': visits.strip() if visits else None,
            'url': response.url,
            'source': 'Atlas Obscura',
            'category': 'unique_places',
            'type': 'attraction'
        }
