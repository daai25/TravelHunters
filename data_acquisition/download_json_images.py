#!/usr/bin/env python3
"""
Hotel Image Download Script for TravelHunters
==============================================
Downloads hotel images from booking_worldwide.json to local OneDrive folder.
Note: Activities functionality removed as per project refactoring.

Created by: TravelHunters Team
Date: July 2025
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

class HotelImageDownloader:
    def __init__(self):
        self.onedrive_path = Path.home() / "OneDrive - ZHAW"
        self.base_dir = self.onedrive_path / "Travel Hunter" / "Image"
        
        # Create destination folder for hotels only
        self.booking_dir = self.base_dir / "Booking_Hotels"
        self.booking_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON file path (hotels only)
        self.project_root = Path(__file__).parent.parent
        self.booking_json = self.project_root / "data_acquisition" / "json_final" / "booking_worldwide.json"
        
        # Download statistics
        self.stats = {
            'booking': {'total': 0, 'downloaded': 0, 'skipped': 0, 'errors': 0, 'no_images': 0, 'invalid_urls': 0}
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

    def print_summary(self):
        """Shows summary of download statistics"""
        logger.info("\n" + "="*60)
        logger.info("📊 DOWNLOAD SUMMARY")
        logger.info("="*60)
        
        # Booking statistics
        booking = self.stats['booking']
        logger.info(f"🏨 BOOKING.COM HOTELS:")
        logger.info(f"   Total URLs:        {booking['total']:>6}")
        logger.info(f"   Downloaded:        {booking['downloaded']:>6}")
        logger.info(f"   Skipped:           {booking['skipped']:>6}")
        logger.info(f"   Errors:            {booking['errors']:>6}")
        logger.info(f"   No images:         {booking['no_images']:>6}")
        logger.info(f"   Invalid URLs:      {booking['invalid_urls']:>6}")
        
        logger.info("="*60)
        logger.info(f"📁 Images saved to: {self.booking_dir}")
        logger.info("="*60)

    def run(self):
        """Main function - executes the entire download process"""
        logger.info("🚀 Starting hotel image download...")
        logger.info(f"📁 OneDrive path: {self.onedrive_path}")
        
        start_time = time.time()
        
        try:
            # Download Booking.com hotel images
            self.process_booking_images()
            
        except KeyboardInterrupt:
            logger.info("\n❌ Download cancelled by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
        finally:
            # Close session
            self.session.close()
            
            # Show summary
            end_time = time.time()
            duration = end_time - start_time
            
            self.print_summary()
            logger.info(f"⏱️  Total duration: {duration:.1f} seconds")
            logger.info("✅ Download completed!")

def main():
    """Main function"""
    downloader = HotelImageDownloader()
    downloader.run()

if __name__ == "__main__":
    main()
