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
from wikipedia_destination_links import get_massive_destination_urls, get_extended_destination_urls

# Lade die URL-Liste - starte mit der erweiterten Liste und erweitere bei Bedarf
print("🚀 Wikipedia Scraper startet...")
print("📋 Lade Destination-URLs...")

# Nutze erst die bewährte erweiterte Liste (106 Destinationen)
wikipedia_destination_urls = get_extended_destination_urls()
print(f"✅ Basis-Liste geladen: {len(wikipedia_destination_urls)} Destinationen")

# Falls mehr als 200 Destinationen gewünscht sind, erweitere mit automatischer Generierung
TARGET_DESTINATIONS = 200  # ✅ ECHTES SCRAPING: Erweiterte Liste für Produktion

if TARGET_DESTINATIONS > len(wikipedia_destination_urls):
    print(f"🔧 Erweitere auf {TARGET_DESTINATIONS} Destinationen...")
    additional_needed = TARGET_DESTINATIONS - len(wikipedia_destination_urls)
    massive_urls = get_massive_destination_urls(max_destinations=additional_needed)
    
    # Kombiniere beide Listen und entferne Duplikate
    combined_urls = list(set(wikipedia_destination_urls + massive_urls))
    wikipedia_destination_urls = combined_urls[:TARGET_DESTINATIONS]
    print(f"✅ Finale Liste: {len(wikipedia_destination_urls)} Destinationen")
else:
    print(f"✅ Verwende Basis-Liste: {len(wikipedia_destination_urls)} Destinationen")


class WikipediaDestinationsScraper:
    def __init__(self):
        # Verzeichnisse erstellen
        self.images_dir = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/destination_images"
        self.output_dir = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/json_backup"
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # JSON Output-Datei
        self.output_file = os.path.join(self.output_dir, 'wikipedia_destinations.json')
        
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
        self.processed_destinations = set()
        self.destinations_count = 0
        
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
        """Bereinigt Text von Wikipedia-spezifischen Elementen - VERBESSERTE VERSION"""
        if not text:
            return ""
        
        # Entferne Referenzen in eckigen Klammern [1], [citation needed], etc.
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Entferne Wikipedia-spezifische Elemente
        text = re.sub(r'\(listen\)', '', text)  # (listen) Audio-Links
        text = re.sub(r'\(help·info\)', '', text)  # (help·info) Links
        text = re.sub(r'pronunciation:', '', text, flags=re.IGNORECASE)
        
        # Entferne mehrfache Leerzeichen und Zeilenumbrüche
        text = re.sub(r'\s+', ' ', text)
        
        # Entferne führende/nachfolgende Leerzeichen
        text = text.strip()
        
        # Entferne leere Klammern
        text = re.sub(r'\(\s*\)', '', text)
        
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
        
    def append_destination_to_file(self, destination):
        """Fügt eine Destination sofort zur JSON-Datei hinzu (streaming/live output)"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                # Jedes Objekt in einer separaten Zeile, kompakt formatiert (wie booking.json)
                json.dump(destination, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
            
            self.destinations_count += 1
            self.logger.info(f"✓ Destination zur Datei hinzugefügt: {destination['name']} (#{self.destinations_count})")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Destination: {e}")

    def initialize_output_file(self):
        """Initialisiert die Output-Datei (leert sie falls sie existiert)"""
        try:
            # Erstelle/leere die Datei
            with open(self.output_file, 'w', encoding='utf-8') as f:
                pass  # Leere Datei erstellen
            self.logger.info(f"Output-Datei initialisiert: {self.output_file}")
        except Exception as e:
            self.logger.error(f"Fehler beim Initialisieren der Output-Datei: {e}")
    
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
            
            # OPTIMIERTE Beschreibungsextraktion für Wikipedia
            description = None
            
            # Hauptstrategie: Finde das mw-parser-output div und verwende den ersten substantiellen Absatz
            content_text_div = soup.find('div', id='mw-content-text')
            if content_text_div:
                parser_output = content_text_div.find('div', class_='mw-parser-output')
                if parser_output:
                    # Hole alle Absätze - der erste ist oft leer, der zweite enthält die Hauptbeschreibung
                    paragraphs = parser_output.find_all('p', recursive=False)
                    
                    substantial_paragraphs = []
                    for p in paragraphs:
                        text = self.clean_text(p.get_text())
                        
                        # Überspringe leere oder sehr kurze Absätze
                        if len(text) < 50:
                            continue
                            
                        # Überspringe reine Koordinaten- oder Pronunciations-Zeilen
                        if ('coordinates:' in text.lower() or 
                            (len(text) < 150 and ('pronunciation' in text.lower() or 'ⓘ' in text))):
                            continue
                        
                        # Überspringe Absätze die nur aus Markierungen bestehen
                        words = text.split()
                        if len(words) < 15:  # Zu wenige Wörter
                            continue
                            
                        # Das ist ein guter substantieller Absatz
                        substantial_paragraphs.append(text)
                        
                        # Stoppe nach 2 guten Absätzen
                        if len(substantial_paragraphs) >= 2:
                            break
                    
                    # Erstelle Beschreibung aus den ersten 1-2 substantiellen Absätzen
                    if substantial_paragraphs:
                        if len(substantial_paragraphs) == 1:
                            description = substantial_paragraphs[0]
                        else:
                            # Kombiniere zwei Absätze, aber begrenze die Gesamtlänge
                            combined = substantial_paragraphs[0] + ' ' + substantial_paragraphs[1]
                            if len(combined) > 800:
                                description = substantial_paragraphs[0]  # Nur der erste wenn zu lang
                            else:
                                description = combined
                        
                        # Begrenze auf maximale Länge und ende bei einem Satzende
                        if len(description) > 600:
                            sentences = description.split('. ')
                            truncated = ''
                            for sentence in sentences:
                                if len(truncated + sentence + '. ') > 600:
                                    break
                                truncated += sentence + '. '
                            description = truncated.rstrip() if truncated else description[:600] + '...'
            
            # Fallback falls die Hauptstrategie nicht funktioniert
            if not description:
                # Suche in allen Absätzen nach dem ersten substantiellen
                all_paragraphs = soup.find_all('p')
                for p in all_paragraphs:
                    text = self.clean_text(p.get_text())
                    if (len(text) > 100 and 
                        len(text.split()) > 20 and 
                        'coordinates:' not in text.lower()):
                        description = text[:500] + ('...' if len(text) > 500 else '')
                        break
            
            # Letzter Fallback
            if not description:
                description = f"Wikipedia-Artikel über {title}"
            
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
            
            # Destination SOFORT zur Datei hinzufügen (streaming/live output)
            self.append_destination_to_file(destination)
            self.processed_destinations.add(url)
            
            self.logger.info(f"✓ Destination verarbeitet: {title}")
            
            # Kurze Pause zwischen Requests
            time.sleep(1)
            
            return destination
            
        except Exception as e:
            self.logger.error(f"Fehler beim Scraping von {url}: {e}")
            return None
    
    def save_data(self):
        """Legacy-Methode für Kompatibilität - wird nicht mehr benötigt da live geschrieben wird"""
        self.logger.info(f"Live-Streaming aktiv - Daten bereits gespeichert in: {self.output_file}")
        self.logger.info(f"Anzahl verarbeiteter Destinationen: {self.destinations_count}")

    def run(self, max_destinations=None):
        """Führt den Scraping-Prozess aus"""
        self.logger.info("Wikipedia Destinations Scraper gestartet")
        self.logger.info(f"Zu verarbeitende URLs: {len(wikipedia_destination_urls)}")
        
        # Initialisiere die Output-Datei (leere sie)
        self.initialize_output_file()
        
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
        
        self.logger.info("Scraping abgeschlossen!")
        self.logger.info(f"Erfolgreich: {successful}")
        self.logger.info(f"Fehlgeschlagen: {failed}")
        self.logger.info(f"Output-Datei: {self.output_file}")
        
        return successful  # Rückgabe der Anzahl erfolgreicher Destinationen


def main():
    """Hauptfunktion zum Ausführen des Scrapers"""
    scraper = WikipediaDestinationsScraper()
    
    # Vollständiges Scraping aller Destinationen
    print("🚀 VOLLSTÄNDIGES WIKIPEDIA SCRAPING")
    print(f"📋 Verarbeite {len(wikipedia_destination_urls)} Destinationen...")
    
    # Führe vollständiges Scraping aus
    successful = scraper.run()
    
    print(f"\n🎯 SCRAPING ABGESCHLOSSEN!")
    print(f"✅ Erfolgreich verarbeitet: {successful} Destinationen")
    print(f"� Output-Datei: {scraper.output_file}")
    print(f"�️  Bilder-Verzeichnis: {scraper.images_dir}")
    
    return successful
