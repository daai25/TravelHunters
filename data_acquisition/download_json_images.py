#!/usr/bin/env python3
"""
Bilder-Download Skript für JSON-Dateien
========================================
Lädt alle Bilder aus booking_worldwide.json und activities_worldwide.json 
in den lokalen OneDrive-Ordner herunter.

Erstellt von: GitHub Copilot
Datum: 7. Juli 2025
"""

import json
import os
import requests
import time
from urllib.parse import urlparse
from pathlib import Path
import hashlib
import logging

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('json_image_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class JSONImageDownloader:
    def __init__(self):
        self.onedrive_path = Path.home() / "OneDrive - ZHAW"
        self.base_dir = self.onedrive_path / "Travel Hunter" / "Image"
        
        # Erstelle Zielordner
        self.booking_dir = self.base_dir / "Booking_Hotels"
        self.activities_dir = self.base_dir / "Activities"
        
        self.booking_dir.mkdir(parents=True, exist_ok=True)
        self.activities_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON-Dateien Pfade
        self.project_root = Path(__file__).parent.parent
        self.booking_json = self.project_root / "data_acquisition" / "json_final" / "booking_worldwide.json"
        self.activities_json = self.project_root / "data_acquisition" / "json_final" / "activities_worldwide.json"
        
        # Download-Statistiken
        self.stats = {
            'booking': {'total': 0, 'downloaded': 0, 'skipped': 0, 'errors': 0, 'no_images': 0, 'invalid_urls': 0},
            'activities': {'total': 0, 'downloaded': 0, 'skipped': 0, 'errors': 0, 'no_images': 0, 'invalid_urls': 0}
        }
        
        # Request Session für bessere Performance
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def get_file_extension(self, url):
        """Ermittelt die Dateiendung aus der URL"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Bekannte Bildformate
        if '.jpg' in path or '.jpeg' in path:
            return '.jpg'
        elif '.png' in path:
            return '.png'
        elif '.gif' in path:
            return '.gif'
        elif '.webp' in path:
            return '.webp'
        elif '.svg' in path:
            return '.svg'
        else:
            return '.jpg'  # Standard Fallback

    def create_filename(self, item_name, location, url, category='hotel'):
        """Erstellt einen sauberen Dateinamen"""
        # Bereinige den Namen
        clean_name = "".join(c for c in item_name if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_location = "".join(c for c in location if c.isalnum() or c in (' ', '-', '_')).strip()
        
        # Kürze zu lange Namen
        if len(clean_name) > 50:
            clean_name = clean_name[:50]
        
        # URL-Hash für Eindeutigkeit
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Dateiendung
        ext = self.get_file_extension(url)
        
        # Zusammensetzen
        filename = f"{clean_location}_{clean_name}_{category}_{url_hash}{ext}"
        
        # Entferne doppelte Leerzeichen und ersetze durch Unterstriche
        filename = "_".join(filename.split())
        filename = filename.replace(" ", "_")
        
        return filename

    def download_image(self, url, filepath):
        """Lädt ein einzelnes Bild herunter"""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Content-Type Check
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                logger.warning(f"⚠️  URL ist kein Bild: {url} (Content-Type: {content_type})")
                return False
            
            # Bild speichern
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = filepath.stat().st_size
            if file_size < 1024:  # Kleiner als 1KB
                logger.warning(f"⚠️  Bild zu klein: {filepath} ({file_size} bytes)")
                filepath.unlink()  # Lösche kleine Dateien
                return False
            
            logger.info(f"✅ Heruntergeladen: {filepath.name} ({file_size:,} bytes)")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Download-Fehler für {url}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler bei {url}: {str(e)}")
            return False

    def process_booking_images(self):
        """Verarbeitet Booking.com Bilder"""
        logger.info("🏨 Starte Download von Booking.com Bildern...")
        
        if not self.booking_json.exists():
            logger.error(f"❌ Booking JSON-Datei nicht gefunden: {self.booking_json}")
            return
        
        with open(self.booking_json, 'r', encoding='utf-8') as f:
            booking_data = json.load(f)
        
        logger.info(f"📊 Gefunden: {len(booking_data)} Hotels")
        
        for item in booking_data:
            # Hotel-Name und Standort
            hotel_name = item.get('name', 'Unknown Hotel')
            location = item.get('location', 'Unknown Location')
            
            # Alle Bild-URLs finden
            image_urls = []
            
            # Einzelnes Bild
            if 'image' in item and item['image']:
                if isinstance(item['image'], str):
                    image_urls.append(item['image'])
                elif isinstance(item['image'], list):
                    image_urls.extend(item['image'])
            
            # Bilder-Array
            if 'images' in item and item['images']:
                if isinstance(item['images'], list):
                    image_urls.extend(item['images'])
                elif isinstance(item['images'], str):
                    image_urls.append(item['images'])
            
            # URLs verarbeiten
            for i, url in enumerate(image_urls):
                if not url or not isinstance(url, str):
                    continue
                
                self.stats['booking']['total'] += 1
                
                # Dateiname erstellen
                filename = self.create_filename(hotel_name, location, url, 'hotel')
                filepath = self.booking_dir / filename
                
                # Prüfe ob Datei bereits existiert
                if filepath.exists():
                    logger.info(f"⏭️  Bereits vorhanden: {filename}")
                    self.stats['booking']['skipped'] += 1
                    continue
                
                # Bild herunterladen
                if self.download_image(url, filepath):
                    self.stats['booking']['downloaded'] += 1
                else:
                    self.stats['booking']['errors'] += 1
                
                # Kurze Pause zwischen Downloads
                time.sleep(0.5)

    def process_activities_images(self):
        """Verarbeitet Activities Bilder"""
        logger.info("🎯 Starte Download von Activities Bildern...")
        
        if not self.activities_json.exists():
            logger.error(f"❌ Activities JSON-Datei nicht gefunden: {self.activities_json}")
            return
        
        with open(self.activities_json, 'r', encoding='utf-8') as f:
            activities_data = json.load(f)
        
        logger.info(f"📊 Gefunden: {len(activities_data)} Activities")
        
        # Zähle einzigartige URLs
        unique_urls = set()
        items_without_images = 0
        
        for item in activities_data:
            # Activity-Name und Standort
            activity_name = item.get('name', 'Unknown Activity')
            location = item.get('location', item.get('destination', 'Unknown Location'))
            
            # Alle Bild-URLs finden
            image_urls = []
            
            # Einzelnes Bild
            if 'image' in item and item['image']:
                image_urls.append(item['image'])
            
            # image_url Feld
            if 'image_url' in item and item['image_url']:
                if item['image_url'] not in image_urls:
                    image_urls.append(item['image_url'])
            
            # Statistik für Items ohne Bilder
            if not image_urls:
                items_without_images += 1
                self.stats['activities']['no_images'] += 1
                continue
            
            # URLs verarbeiten - JEDE Activity bekommt ihr eigenes Bild
            for i, url in enumerate(image_urls):
                if not url or not isinstance(url, str):
                    self.stats['activities']['invalid_urls'] += 1
                    continue
                
                self.stats['activities']['total'] += 1
                unique_urls.add(url)
                
                # Dateiname erstellen - IMMER mit Activity-Namen für Eindeutigkeit
                filename = self.create_filename(activity_name, location, url, 'activity')
                filepath = self.activities_dir / filename
                
                # Prüfe ob Datei bereits existiert
                if filepath.exists():
                    logger.info(f"⏭️  Bereits vorhanden: {filename}")
                    self.stats['activities']['skipped'] += 1
                    continue
                
                # Bild herunterladen
                if self.download_image(url, filepath):
                    self.stats['activities']['downloaded'] += 1
                else:
                    self.stats['activities']['errors'] += 1
                
                # Kurze Pause zwischen Downloads
                time.sleep(0.5)
        
        # Zusätzliche Statistiken loggen
        logger.info(f"📊 Activities-Analyse:")
        logger.info(f"   🔗 Einzigartige URLs: {len(unique_urls)}")
        logger.info(f"   🚫 Ohne Bilder: {items_without_images}")
        logger.info(f"   ♻️  URL-Wiederverwendung: {self.stats['activities']['total'] - len(unique_urls)} mal")

    def print_summary(self):
        """Zeigt eine Zusammenfassung der Download-Statistiken"""
        logger.info("\n" + "="*60)
        logger.info("📊 DOWNLOAD-ZUSAMMENFASSUNG")
        logger.info("="*60)
        
        # Booking Statistiken
        booking = self.stats['booking']
        logger.info(f"🏨 BOOKING.COM HOTELS:")
        logger.info(f"   Total URLs:        {booking['total']:>6}")
        logger.info(f"   Heruntergeladen:   {booking['downloaded']:>6}")
        logger.info(f"   Übersprungen:      {booking['skipped']:>6}")
        logger.info(f"   Fehler:            {booking['errors']:>6}")
        
        # Activities Statistiken
        activities = self.stats['activities']
        logger.info(f"🎯 ACTIVITIES:")
        logger.info(f"   Total URLs:        {activities['total']:>6}")
        logger.info(f"   Heruntergeladen:   {activities['downloaded']:>6}")
        logger.info(f"   Übersprungen:      {activities['skipped']:>6}")
        logger.info(f"   Fehler:            {activities['errors']:>6}")
        logger.info(f"   Ohne Bilder:       {activities['no_images']:>6}")
        logger.info(f"   Ungültige URLs:    {activities['invalid_urls']:>6}")
        
        # Gesamt
        total_urls = booking['total'] + activities['total']
        total_downloaded = booking['downloaded'] + activities['downloaded']
        total_skipped = booking['skipped'] + activities['skipped']
        total_errors = booking['errors'] + activities['errors']
        
        logger.info(f"📈 GESAMT:")
        logger.info(f"   Total URLs:        {total_urls:>6}")
        logger.info(f"   Heruntergeladen:   {total_downloaded:>6}")
        logger.info(f"   Übersprungen:      {total_skipped:>6}")
        logger.info(f"   Fehler:            {total_errors:>6}")
        
        logger.info("="*60)
        logger.info(f"📁 Bilder gespeichert in:")
        logger.info(f"   Hotels:     {self.booking_dir}")
        logger.info(f"   Activities: {self.activities_dir}")
        logger.info("="*60)

    def run(self):
        """Hauptfunktion - führt den gesamten Download-Prozess aus"""
        logger.info("🚀 Starte JSON-Bilder Download...")
        logger.info(f"📁 OneDrive-Pfad: {self.onedrive_path}")
        
        start_time = time.time()
        
        try:
            # Booking.com Bilder herunterladen
            self.process_booking_images()
            
            # Activities Bilder herunterladen
            self.process_activities_images()
            
        except KeyboardInterrupt:
            logger.info("\n❌ Download durch Benutzer abgebrochen")
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler: {str(e)}")
        finally:
            # Session schließen
            self.session.close()
            
            # Zusammenfassung anzeigen
            end_time = time.time()
            duration = end_time - start_time
            
            self.print_summary()
            logger.info(f"⏱️  Gesamtdauer: {duration:.1f} Sekunden")
            logger.info("✅ Download abgeschlossen!")

def main():
    """Hauptfunktion"""
    downloader = JSONImageDownloader()
    downloader.run()

if __name__ == "__main__":
    main()
