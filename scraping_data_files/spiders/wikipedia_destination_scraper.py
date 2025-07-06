#!/usr/bin/env python3
"""
Wikipedia Destinations Scraper
Eigenständiges Script zum Scrapen von Destinationen von Wikipedia
"""

import requests
import json
import os
import hashlib
import time
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging

# Import der Destination URLs aus separatem Modul
from wikipedia_destination_links import wikipedia_destination_urls_extended as wikipedia_destination_urls


class WikipediaDestinationsScraper:
    def __init__(self):
        # Verzeichnisse erstellen
        self.images_dir = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/destination_images"
        self.output_dir = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/json_backup"
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Session für Requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TravelHunters-Bot 1.0 (Educational Project)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Daten
        self.destinations = []
        self.processed_destinations = set()
        
        # Logging Setup
        log_file = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/wikipedia_scraping.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def clean_text(self, text):
        """Bereinigt Text von Wikipedia-spezifischen Elementen"""
        if not text:
            return ""
        
        # Entferne Referenzen in eckigen Klammern
        text = re.sub(r'\[[^\]]*\]', '', text)
        # Entferne mehrfache Leerzeichen
        text = re.sub(r'\s+', ' ', text)
        # Entferne führende/nachfolgende Leerzeichen
        text = text.strip()
        
        return text
    
    def extract_main_image(self, soup, base_url):
        """Extrahiert das Hauptbild der Wikipedia-Seite"""
        try:
            # Suche nach dem infobox-Bild
            infobox = soup.find('table', class_='infobox')
            if infobox:
                img = infobox.find('img')
                if img and img.get('src'):
                    img_url = urljoin(base_url, img.get('src'))
                    # Konvertiere zu hochauflösender Version
                    if '/thumb/' in img_url:
                        img_url = img_url.replace('/thumb/', '/').split('/')
                        # Entferne die Größenangabe am Ende
                        img_url = '/'.join(img_url[:-1])
                    return img_url
            
            # Fallback: Erstes größeres Bild finden
            images = soup.find_all('img')
            for img in images:
                if img.get('src') and img.get('width'):
                    try:
                        width = int(img.get('width', 0))
                        if width > 200:  # Nur größere Bilder
                            img_url = urljoin(base_url, img.get('src'))
                            if '/thumb/' in img_url:
                                img_url = img_url.replace('/thumb/', '/').split('/')
                                img_url = '/'.join(img_url[:-1])
                            return img_url
                    except ValueError:
                        continue
                        
        except Exception as e:
            self.logger.warning(f"Fehler beim Extrahieren des Hauptbildes: {e}")
        
        return None
    
    def download_image(self, img_url, destination_name):
        """Lädt ein Bild herunter und speichert es lokal"""
        try:
            if not img_url:
                return None
                
            response = self.session.get(img_url, timeout=30)
            response.raise_for_status()
            
            # Erstelle Dateinamen
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
            safe_name = re.sub(r'[^\w\s-]', '', destination_name.lower())
            safe_name = re.sub(r'[-\s]+', '_', safe_name)
            
            # Bestimme Dateierweiterung
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                ext = '.gif'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                # Fallback aus URL
                parsed = urlparse(img_url)
                ext = os.path.splitext(parsed.path)[1] or '.jpg'
            
            filename = f"{safe_name}_main_{url_hash}{ext}"
            filepath = os.path.join(self.images_dir, filename)
            
            # Prüfe Dateigröße (max 50MB für Git LFS)
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 50 * 1024 * 1024:
                self.logger.warning(f"Bild zu groß ({content_length} bytes): {img_url}")
                return None
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Bild heruntergeladen: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Fehler beim Herunterladen des Bildes {img_url}: {e}")
            return None
    
    def extract_coordinates(self, soup):
        """Extrahiert Koordinaten aus der Wikipedia-Seite"""
        try:
            # Suche nach Koordinaten in verschiedenen Formaten
            coord_span = soup.find('span', class_='geo')
            if coord_span:
                coord_text = coord_span.get_text().strip()
                # Parse Koordinaten im Format "52.520008; 13.404954"
                if ';' in coord_text:
                    lat, lon = coord_text.split(';')
                    return {
                        'latitude': float(lat.strip()),
                        'longitude': float(lon.strip())
                    }
            
            # Alternative: Suche nach geo-microformat
            geo_div = soup.find('div', class_='geo')
            if geo_div:
                coord_text = geo_div.get_text().strip()
                if ';' in coord_text:
                    lat, lon = coord_text.split(';')
                    return {
                        'latitude': float(lat.strip()),
                        'longitude': float(lon.strip())
                    }
                    
        except Exception as e:
            self.logger.warning(f"Fehler beim Extrahieren der Koordinaten: {e}")
        
        return None
    
    def extract_basic_info(self, soup):
        """Extrahiert grundlegende Informationen aus der Infobox"""
        info = {}
        
        try:
            infobox = soup.find('table', class_='infobox')
            if infobox:
                rows = infobox.find_all('tr')
                for row in rows:
                    th = row.find('th')
                    td = row.find('td')
                    if th and td:
                        key = self.clean_text(th.get_text())
                        value = self.clean_text(td.get_text())
                        if key and value:
                            info[key.lower()] = value
        except Exception as e:
            self.logger.warning(f"Fehler beim Extrahieren der Basis-Infos: {e}")
        
        return info
    
    def scrape_destination(self, url):
        """Scrapt eine einzelne Destination von Wikipedia"""
        try:
            self.logger.info(f"Scraping: {url}")
            
            # Check if already processed
            if url in self.processed_destinations:
                self.logger.info(f"Bereits verarbeitet: {url}")
                return None
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Titel extrahieren
            title_element = soup.find('h1', class_='firstHeading')
            title = title_element.get_text().strip() if title_element else "Unknown"
            
            # Erste Absätze für Beschreibung
            description_paragraphs = []
            content_div = soup.find('div', class_='mw-parser-output')
            if content_div:
                paragraphs = content_div.find_all('p', limit=3)
                for p in paragraphs:
                    text = self.clean_text(p.get_text())
                    if text and len(text) > 50:  # Nur substantielle Absätze
                        description_paragraphs.append(text)
            
            description = ' '.join(description_paragraphs[:2])  # Erste 2 Absätze
            
            # Hauptbild extrahieren und herunterladen
            main_image_url = self.extract_main_image(soup, url)
            image_filename = None
            if main_image_url:
                image_filename = self.download_image(main_image_url, title)
            
            # Koordinaten extrahieren
            coordinates = self.extract_coordinates(soup)
            
            # Basis-Informationen
            basic_info = self.extract_basic_info(soup)
            
            # Destination-Objekt erstellen
            destination = {
                'name': title,
                'url': url,
                'description': description,
                'image_url': main_image_url,
                'image_file': image_filename,
                'coordinates': coordinates,
                'basic_info': basic_info,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'wikipedia'
            }
            
            self.destinations.append(destination)
            self.processed_destinations.add(url)
            
            self.logger.info(f"✓ Destination verarbeitet: {title}")
            
            # Kurze Pause zwischen Requests
            time.sleep(1)
            
            return destination
            
        except Exception as e:
            self.logger.error(f"Fehler beim Scraping von {url}: {e}")
            return None
    
    def save_data(self):
        """Speichert die gesammelten Daten als JSON (Booking-Format: eine Zeile pro Objekt)"""
        output_file = os.path.join(self.output_dir, 'wikipedia_destinations.json')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for destination in self.destinations:
                    # Jedes Objekt in einer separaten Zeile, kompakt formatiert (wie booking.json)
                    json.dump(destination, f, ensure_ascii=False, separators=(',', ':'))
                    f.write('\n')
            
            self.logger.info(f"Daten gespeichert: {output_file}")
            self.logger.info(f"Anzahl Destinationen: {len(self.destinations)}")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern: {e}")
    
    def run(self, max_destinations=None):
        """Führt den Scraping-Prozess aus"""
        self.logger.info("Wikipedia Destinations Scraper gestartet")
        self.logger.info(f"Zu verarbeitende URLs: {len(wikipedia_destination_urls)}")
        
        if max_destinations:
            urls_to_process = wikipedia_destination_urls[:max_destinations]
            self.logger.info(f"Limitiert auf {max_destinations} Destinationen")
        else:
            urls_to_process = wikipedia_destination_urls
        
        successful = 0
        failed = 0
        
        for i, url in enumerate(urls_to_process, 1):
            self.logger.info(f"Fortschritt: {i}/{len(urls_to_process)}")
            
            result = self.scrape_destination(url)
            if result:
                successful += 1
            else:
                failed += 1
            
            # Speichere zwischendurch (alle 10 Destinationen)
            if i % 10 == 0:
                self.save_data()
                self.logger.info(f"Zwischenspeicherung nach {i} Destinationen")
        
        # Finale Speicherung
        self.save_data()
        
        self.logger.info("Scraping abgeschlossen!")
        self.logger.info(f"Erfolgreich: {successful}")
        self.logger.info(f"Fehlgeschlagen: {failed}")
        
        return self.destinations


def main():
    """Hauptfunktion zum Ausführen des Scrapers"""
    scraper = WikipediaDestinationsScraper()
    
    # Für Tests: nur erste 5 Destinationen
    # destinations = scraper.run(max_destinations=5)
    
    # Für vollständigen Scraping-Lauf: alle Destinationen
    destinations = scraper.run()
    
    print(f"\n{'='*50}")
    print(f"SCRAPING ABGESCHLOSSEN")
    print(f"{'='*50}")
    print(f"Destinationen verarbeitet: {len(destinations)}")
    print(f"Bilder heruntergeladen: {len([d for d in destinations if d.get('image_file')])}")
    print(f"Mit Koordinaten: {len([d for d in destinations if d.get('coordinates')])}")


if __name__ == "__main__":
    main()
