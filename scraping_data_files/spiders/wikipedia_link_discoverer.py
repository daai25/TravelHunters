#!/usr/bin/env python3
"""
Wikipedia Link Discoverer - Massive Edition
Automatisches Entdecken von mehr als 1000 Wikipedia-Destinationen
"""

import requests
import re
import time
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MassiveWikipediaLinkDiscoverer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TravelHunters-MassiveScraper 1.0 (Educational Project)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        self.discovered_urls = set()
        
    def get_country_capitals(self):
        """Sammelt alle Hauptstädte der Welt"""
        capitals = []
        
        try:
            url = "https://en.wikipedia.org/wiki/List_of_national_capitals"
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Finde alle Links zu Hauptstädten
            for table in soup.find_all('table', class_='wikitable'):
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    for cell in cells:
                        links = cell.find_all('a', href=True)
                        for link in links:
                            href = link.get('href')
                            if href and href.startswith('/wiki/') and ':' not in href:
                                full_url = urljoin("https://en.wikipedia.org", href)
                                if self.is_valid_destination(link.get_text()):
                                    capitals.append(full_url)
            
            logger.info(f"Gefundene Hauptstädte: {len(capitals)}")
            return capitals
            
        except Exception as e:
            logger.error(f"Fehler beim Sammeln der Hauptstädte: {e}")
            return []
    
    def get_largest_cities(self):
        """Sammelt die größten Städte der Welt"""
        cities = []
        
        city_lists = [
            "https://en.wikipedia.org/wiki/List_of_largest_cities",
            "https://en.wikipedia.org/wiki/List_of_metropolitan_areas_by_population",
            "https://en.wikipedia.org/wiki/List_of_urban_areas_by_population",
        ]
        
        for list_url in city_lists:
            try:
                response = self.session.get(list_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for table in soup.find_all('table', class_='wikitable'):
                    for row in table.find_all('tr'):
                        cells = row.find_all(['td', 'th'])
                        for cell in cells:
                            links = cell.find_all('a', href=True)
                            for link in links:
                                href = link.get('href')
                                if href and href.startswith('/wiki/') and ':' not in href:
                                    full_url = urljoin("https://en.wikipedia.org", href)
                                    if self.is_valid_destination(link.get_text()):
                                        cities.append(full_url)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Fehler beim Sammeln von {list_url}: {e}")
        
        logger.info(f"Gefundene große Städte: {len(cities)}")
        return cities
    
    def get_tourist_destinations(self):
        """Sammelt beliebte Touristenziele"""
        destinations = []
        
        tourism_lists = [
            "https://en.wikipedia.org/wiki/World_Tourism_rankings",
            "https://en.wikipedia.org/wiki/List_of_World_Heritage_Sites",
            "https://en.wikipedia.org/wiki/New7Wonders_of_the_World",
            "https://en.wikipedia.org/wiki/Wonders_of_the_World",
        ]
        
        for list_url in tourism_lists:
            try:
                response = self.session.get(list_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Sammle alle relevanten Links
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and href.startswith('/wiki/') and ':' not in href:
                        full_url = urljoin("https://en.wikipedia.org", href)
                        if self.is_valid_destination(link.get_text()):
                            destinations.append(full_url)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Fehler beim Sammeln von {list_url}: {e}")
        
        logger.info(f"Gefundene Touristenziele: {len(destinations)}")
        return destinations
    
    def get_cities_by_region(self):
        """Sammelt Städte nach Regionen"""
        cities = []
        
        region_lists = [
            "https://en.wikipedia.org/wiki/List_of_cities_in_Europe",
            "https://en.wikipedia.org/wiki/List_of_cities_in_Asia", 
            "https://en.wikipedia.org/wiki/List_of_cities_in_Africa",
            "https://en.wikipedia.org/wiki/List_of_cities_in_North_America",
            "https://en.wikipedia.org/wiki/List_of_cities_in_South_America",
            "https://en.wikipedia.org/wiki/List_of_cities_in_Oceania",
        ]
        
        for region_url in region_lists:
            try:
                response = self.session.get(region_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Sammle Links aus Listen und Tabellen
                for element in soup.find_all(['ul', 'ol', 'table']):
                    for link in element.find_all('a', href=True):
                        href = link.get('href')
                        if href and href.startswith('/wiki/') and ':' not in href:
                            full_url = urljoin("https://en.wikipedia.org", href)
                            if self.is_valid_destination(link.get_text()):
                                cities.append(full_url)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Fehler beim Sammeln von {region_url}: {e}")
        
        logger.info(f"Gefundene regionale Städte: {len(cities)}")
        return cities
    
    def get_airports_and_ports(self):
        """Sammelt Städte mit großen Flughäfen und Häfen"""
        destinations = []
        
        transport_lists = [
            "https://en.wikipedia.org/wiki/List_of_busiest_airports_by_passenger_traffic",
            "https://en.wikipedia.org/wiki/List_of_largest_ports",
            "https://en.wikipedia.org/wiki/List_of_busiest_cruise_ports_by_passengers",
        ]
        
        for list_url in transport_lists:
            try:
                response = self.session.get(list_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for table in soup.find_all('table', class_='wikitable'):
                    for row in table.find_all('tr'):
                        cells = row.find_all(['td', 'th'])
                        for cell in cells:
                            links = cell.find_all('a', href=True)
                            for link in links:
                                href = link.get('href')
                                if href and href.startswith('/wiki/') and ':' not in href:
                                    full_url = urljoin("https://en.wikipedia.org", href)
                                    text = link.get_text()
                                    if self.is_valid_destination(text) and not self.is_airport_code(text):
                                        destinations.append(full_url)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Fehler beim Sammeln von {list_url}: {e}")
        
        logger.info(f"Gefundene Transport-Städte: {len(destinations)}")
        return destinations
    
    def is_valid_destination(self, text):
        """Prüft, ob ein Text ein gültiges Reiseziel darstellt"""
        if not text or len(text) < 2:
            return False
        
        # Ausschließen
        invalid_patterns = [
            r'^(the|and|or|of|in|at|to|for|with|by)$',
            r'^\d+$',  # Nur Zahlen
            r'^[A-Z]{2,4}$',  # Flughafencodes
            r'(edit|source|citation|reference|note)',
            r'(airport|international|domestic|code)',
            r'^(list|category|template|file|image)',
        ]
        
        text_lower = text.lower().strip()
        for pattern in invalid_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Mindestens 2 Buchstaben
        if not re.search(r'[a-zA-Z]{2,}', text):
            return False
            
        return True
    
    def is_airport_code(self, text):
        """Prüft, ob Text ein Flughafencode ist"""
        return bool(re.match(r'^[A-Z]{3,4}$', text.strip()))
    
    def discover_massive_links(self, target_count=1500):
        """Entdeckt eine massive Anzahl von Destination-Links"""
        logger.info(f"🚀 Starte massive Link-Entdeckung (Ziel: {target_count} Links)")
        
        all_urls = []
        
        # 1. Hauptstädte sammeln
        logger.info("📍 Sammle Hauptstädte...")
        all_urls.extend(self.get_country_capitals())
        
        # 2. Große Städte sammeln
        logger.info("🏙️ Sammle große Städte...")
        all_urls.extend(self.get_largest_cities())
        
        # 3. Touristenziele sammeln  
        logger.info("🗺️ Sammle Touristenziele...")
        all_urls.extend(self.get_tourist_destinations())
        
        # 4. Regionale Städte sammeln
        logger.info("🌍 Sammle regionale Städte...")
        all_urls.extend(self.get_cities_by_region())
        
        # 5. Transport-Hubs sammeln
        logger.info("✈️ Sammle Transport-Hubs...")
        all_urls.extend(self.get_airports_and_ports())
        
        # Duplikate entfernen und sortieren
        unique_urls = list(set(all_urls))
        unique_urls.sort()
        
        logger.info(f"✅ {len(unique_urls)} einzigartige Destinationen gefunden!")
        
        # Falls noch nicht genug, Top-URLs nehmen
        if len(unique_urls) > target_count:
            unique_urls = unique_urls[:target_count]
            logger.info(f"🎯 Auf {target_count} Destinationen begrenzt")
        
        return unique_urls


def main():
    """Hauptfunktion für massive Link-Entdeckung"""
    discoverer = MassiveWikipediaLinkDiscoverer()
    
    # Entdecke 1500+ Destinationen
    massive_urls = discoverer.discover_massive_links(target_count=1500)
    
    print(f"\n{'='*60}")
    print(f"🌍 MASSIVE WIKIPEDIA DESTINATION DISCOVERY")
    print(f"{'='*60}")
    print(f"✅ {len(massive_urls)} Destinationen entdeckt!")
    print(f"🎯 Bereit für massive Scraping-Operation!")
    
    # Zeige erste 10 Beispiele
    print(f"\n📋 Erste 10 Destinationen:")
    for i, url in enumerate(massive_urls[:10], 1):
        print(f"   {i:2d}. {url}")
    
    print(f"\n💾 Speichere massive Link-Liste...")
    
    return massive_urls


if __name__ == "__main__":
    main()
