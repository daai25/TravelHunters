# spiders/travel_spider.py
import scrapy
from scrapy_selenium import SeleniumRequest
from urllib.parse import urlparse


class TravelSpider(scrapy.Spider):
    """
    Ein Spider für alle Reise-Datenquellen.
    • Statische Quelle (Kaggle-Dataset) wird per normalem Request abgeholt.
    • Dynamische JavaScript-Seiten werden über SeleniumRequest gerendert.
    • Für jede Domain gibt es eine eigene Parse-Routine.
    """
    name = "travelspider"
    allowed_domains = [
        "kaggle.com",
        "tripadvisor.de",
        "booking.com",
        "getyourguide.de",
        "holidaycheck.de"
    ]

    # Erste Start-URLs – ergänze hier gerne weitere Listing-Seiten oder Such-Resultate
    start_urls = [
        "https://www.kaggle.com/datasets/rkiattisak/traveler-trip-data",
        # Beispiel-Listing „Hotels in Zürich“ – Landing-Page für TripAdvisor
        "https://www.tripadvisor.de/Hotels-g188113-Zurich-Hotels.html",
        # Booking: beliebige Such-URL, hier Zürich
        "https://www.booking.com/searchresults.de.html?ss=Z%C3%BCrich",
        # GetYourGuide: Top-Erlebnisse Zürich
        "https://www.getyourguide.de/zuerich-l55/",
        # HolidayCheck: Hotel-Liste Schweiz
        "https://www.holidaycheck.de/dh/hotels-schweiz/7c1716d2-4c24-3d44-9dc5-66aac6c5c0fb"
    ]

    custom_settings = {
        # Schonendes Crawlen
        "DOWNLOAD_DELAY": 2,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        # Selenium-Middleware aktivieren (siehe settings.py)
        "DOWNLOADER_MIDDLEWARES": {
            "scrapy_selenium.SeleniumMiddleware": 800
        }
    }

    # ─────────────────────────────────────────────────────────────
    #  ROUTING – jede Domain bekommt ihren eigenen Callback
    # ─────────────────────────────────────────────────────────────
    def start_requests(self):
        for url in self.start_urls:
            domain = urlparse(url).netloc
            if "kaggle.com" in domain:
                yield scrapy.Request(url, callback=self.parse_kaggle, dont_filter=True)
            elif "tripadvisor" in domain:
                yield SeleniumRequest(url, callback=self.parse_tripadvisor, wait_time=8, dont_filter=True)
            elif "booking.com" in domain:
                yield SeleniumRequest(url, callback=self.parse_booking, wait_time=8, dont_filter=True)
            elif "getyourguide" in domain:
                yield SeleniumRequest(url, callback=self.parse_getyourguide, wait_time=8, dont_filter=True)
            elif "holidaycheck" in domain:
                yield SeleniumRequest(url, callback=self.parse_holidaycheck, wait_time=8, dont_filter=True)

    # ─────────────────────────────────────────────────────────────
    #  1) Kaggle – statischer Datensatz
    # ─────────────────────────────────────────────────────────────
    def parse_kaggle(self, response):
        """Liefert Direkt-Links zu den CSV-Dateien des Datensatzes."""
        for href in response.css("a::attr(href)").getall():
            if href.endswith(".csv"):
                yield {"site": "Kaggle", "file_url": response.urljoin(href)}

    # ─────────────────────────────────────────────────────────────
    #  2) TripAdvisor
    # ─────────────────────────────────────────────────────────────
    def parse_tripadvisor(self, response):
        """Fängt Links auf einzelne Hotel-Detailseiten ab."""
        hotel_links = response.css("div.listing_title a::attr(href)").getall()
        for href in hotel_links:
            yield SeleniumRequest(
                url=response.urljoin(href),
                callback=self.parse_tripadvisor_hotel,
                wait_time=6
            )

    def parse_tripadvisor_hotel(self, response):
        yield {
            "site": "TripAdvisor",
            "name": response.css("h1::text").get(default="").strip(),
            "rating": response.css("span[class*=rating] span::text").get(),
            "address": response.css("span.public-business-listing-ContactInfo__nonWebLinkText--nGymU::text").get(),
        }

    # ─────────────────────────────────────────────────────────────
    #  3) Booking.com
    # ─────────────────────────────────────────────────────────────
    def parse_booking(self, response):
        property_links = response.css('a[data-testid="title-link"]::attr(href)').getall()
        for href in property_links:
            yield SeleniumRequest(
                url=response.urljoin(href),
                callback=self.parse_booking_property,
                wait_time=6
            )

    def parse_booking_property(self, response):
        yield {
            "site": "Booking",
            "name": response.css("h2#hp_hotel_name::text").get(default="").strip(),
            "rating": response.css("div.b5cd09854e.d10a6220b4::text").get(),
            "address": response.css("span.hp_address_subtitle::text").get(default="").strip()
        }

    # ─────────────────────────────────────────────────────────────
    #  4) GetYourGuide
    # ─────────────────────────────────────────────────────────────
    def parse_getyourguide(self, response):
        tour_links = response.css("a.tour-card::attr(href)").getall()
        for href in tour_links:
            yield SeleniumRequest(
                url=response.urljoin(href),
                callback=self.parse_getyourguide_tour,
                wait_time=6
            )

    def parse_getyourguide_tour(self, response):
        yield {
            "site": "GetYourGuide",
            "title": response.css("h1::text").get(),
            "rating": response.css("div.rating-overall__rating-average::text").get(),
            "price": response.css("div.price-display__price::text").get()
        }

    # ─────────────────────────────────────────────────────────────
    #  5) HolidayCheck
    # ─────────────────────────────────────────────────────────────
    def parse_holidaycheck(self, response):
        hotel_links = response.css("a.hotel-link::attr(href)").getall()
        for href in hotel_links:
            yield SeleniumRequest(
                url=response.urljoin(href),
                callback=self.parse_holidaycheck_hotel,
                wait_time=6
            )

    def parse_holidaycheck_hotel(self, response):
        yield {
            "site": "HolidayCheck",
            "name": response.css("h1.hotel-name::text").get(),
            "rating": response.css("span.score::text").get(),
            "locality": response.css("span.region::text").get()
        }