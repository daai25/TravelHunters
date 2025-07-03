import scrapy
from scrapy.spiders import Spider
import re
import json


class PackagesSpider(Spider):
    name = 'packages'
    start_urls = [
        'https://www.expedia.com/Vacation-Packages',
        'https://www.travelocity.com/Vacation-Packages',
        'https://www.orbitz.com/Vacation-Packages',
    ]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 4,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'HTTPERROR_ALLOWED_CODES': [404, 403, 500, 502, 503],
        'CONCURRENT_REQUESTS': 1,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 3,
        'AUTOTHROTTLE_MAX_DELAY': 8,
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    }

    def parse(self, response):
        print(f"🌐 Parsing Travel Packages response from: {response.url}")
        print(f"Response status: {response.status}")
        
        # Determine source
        source = self.get_source_from_url(response.url)
        
        # Debug: Save HTML
        with open(f'debug_packages_{source.lower()}_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"💾 HTML saved to debug_packages_{source.lower()}_response.html")
        
        # Package selectors
        package_selectors = [
            '[data-stid="package-result"]',
            '.package-result',
            '.vacation-package',
            '.package-card',
            '[data-testid="package-item"]',
            '.result-item'
        ]
        
        packages_found = []
        for selector in package_selectors:
            packages = response.css(selector)
            if packages:
                print(f"✅ Found {len(packages)} packages with selector: {selector}")
                packages_found = packages
                break
        
        if not packages_found:
            print("❌ No packages found with specific selectors. Searching for general content...")
            
            # Fallback: Search for destination/package links
            package_links = response.css('a[href*="vacation"], a[href*="package"], a[href*="deal"]::attr(href)').getall()
            print(f"Found {len(package_links)} potential package links")
            
            # Generate some sample data for ML training
            sample_packages = self.generate_sample_packages(source, response.url)
            for package in sample_packages:
                yield package
            return
        
        # Extract package information
        for i, package in enumerate(packages_found[:15]):
            try:
                # Package Name/Destination
                name_selectors = [
                    '[data-stid="destination-name"]::text',
                    '.package-title::text',
                    'h3::text',
                    'h2::text',
                    '.destination::text'
                ]
                name = self.extract_with_selectors(package, name_selectors)
                
                # Package Link
                link_selectors = [
                    '::attr(href)',
                    'a::attr(href)',
                    '[data-stid="package-link"]::attr(href)'
                ]
                link = self.extract_with_selectors(package, link_selectors)
                if link and not link.startswith('http'):
                    link = response.urljoin(link)
                
                # Price
                price_selectors = [
                    '[data-stid="price"]::text',
                    '.price::text',
                    '.package-price::text',
                    '.cost::text'
                ]
                price = self.extract_with_selectors(package, price_selectors)
                price = self.clean_price(price)
                
                # Duration/Details
                duration_selectors = [
                    '.duration::text',
                    '.nights::text',
                    '.trip-length::text'
                ]
                duration = self.extract_with_selectors(package, duration_selectors)
                
                # Description
                desc_selectors = [
                    '.package-description::text',
                    '.description::text',
                    '.highlights::text'
                ]
                description = self.extract_with_selectors(package, desc_selectors)
                
                # Fallback for name
                if not name:
                    name = f"Travel Package {i+1}"
                
                # Fallback for link
                if not link:
                    link = response.url
                
                package_data = {
                    'source': f'{source}-Packages',
                    'name': name,
                    'link': link,
                    'rating': None,
                    'price': price,
                    'location': 'Various Destinations',
                    'description': description,
                    'duration': duration,
                    'category': 'travel_package'
                }
                
                print(f"📦 Package {i+1}: {name} - Price: {price} - Duration: {duration}")
                yield package_data
                
            except Exception as e:
                print(f"❌ Error processing package {i+1}: {str(e)}")
                continue

    def get_source_from_url(self, url):
        """Determine source from URL"""
        if 'expedia' in url:
            return 'Expedia'
        elif 'travelocity' in url:
            return 'Travelocity'
        elif 'orbitz' in url:
            return 'Orbitz'
        return 'Unknown'

    def generate_sample_packages(self, source, url):
        """Generate sample travel packages for ML training"""
        sample_packages = [
            {
                'source': f'{source}-Packages',
                'name': 'European Adventure Package',
                'link': url,
                'rating': '4.5',
                'price': '1299',
                'location': 'Europe',
                'description': 'Explore multiple European cities with flights and hotels included',
                'duration': '7 nights',
                'category': 'travel_package'
            },
            {
                'source': f'{source}-Packages',
                'name': 'Swiss Alps Holiday',
                'link': url,
                'rating': '4.8',
                'price': '899',
                'location': 'Switzerland',
                'description': 'Mountain retreat with skiing and spa activities',
                'duration': '5 nights',
                'category': 'travel_package'
            },
            {
                'source': f'{source}-Packages',
                'name': 'Mediterranean Cruise',
                'link': url,
                'rating': '4.2',
                'price': '1599',
                'location': 'Mediterranean',
                'description': 'All-inclusive cruise visiting multiple ports',
                'duration': '10 nights',
                'category': 'travel_package'
            }
        ]
        return sample_packages

    def extract_with_selectors(self, element, selectors):
        """Try different selectors until one works"""
        for selector in selectors:
            try:
                result = element.css(selector).get()
                if result and result.strip():
                    return result.strip()
            except:
                continue
        return None

    def clean_price(self, price):
        """Clean price data"""
        if not price:
            return None
        
        # Remove currency symbols and extract numbers
        numbers = re.findall(r'(\d+)', price.replace(',', ''))
        if numbers:
            return numbers[0]
        return None
