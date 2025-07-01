import scrapy
from scrapy_selenium import SeleniumRequest

class HolidayCheckSeleniumSpider(scrapy.Spider):
    name = "holidaycheck_selenium"

    def start_requests(self):
        yield SeleniumRequest(
            url="https://www.holidaycheck.de/dh/hotels-schweiz/7c1716d2-4c24-3d44-9dc5-66aac6c5c0fb",
            callback=self.parse,
            wait_time=5,
        )

    def parse(self, response):
        for hotel in response.css("div.hotelTeaser"):
            yield {
                "source": "HolidayCheck",
                "name": hotel.css("span span::text").get(),
                "link": response.urljoin(hotel.css("a::attr(href)").get()),
                "rating": hotel.css("hc-rating::attr(rating)").get()
            }
