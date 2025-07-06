#!/usr/bin/env python3
"""
Wikipedia Destination Scraper
Vollständiger Scraper für weltweite Wikipedia-Destinationen mit Koordinaten, Bildern und detaillierten Beschreibungen
"""

import requests
import json
import os
import hashlib
import time
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from wikipedia_destination_links import get_destination_urls

def clean_text(text):
    """Bereinigt Text von Wikipedia-spezifischen Elementen"""
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

def extract_coordinates(soup):
    """Extrahiert Koordinaten aus Wikipedia-Seite"""
    try:
        # Methode 1: Koordinaten aus dem span mit id="coordinates"
        coord_span = soup.find('span', id='coordinates')
        if coord_span:
            # Suche nach Latitude/Longitude in verschiedenen Formaten
            coord_text = coord_span.get_text()
            
            # Dezimalgrad-Format suchen (z.B. 48.8566°N 2.3522°E)
            decimal_pattern = r'(\d+\.?\d*)°([NS])\s+(\d+\.?\d*)°([EW])'
            match = re.search(decimal_pattern, coord_text)
            if match:
                lat_val, lat_dir, lon_val, lon_dir = match.groups()
                lat = float(lat_val) * (1 if lat_dir == 'N' else -1)
                lon = float(lon_val) * (1 if lon_dir == 'E' else -1)
                return lat, lon
        
        # Methode 2: Koordinaten aus data-lat und data-lon Attributen
        coord_element = soup.find(['span', 'div'], attrs={'data-lat': True, 'data-lon': True})
        if coord_element:
            lat = float(coord_element.get('data-lat'))
            lon = float(coord_element.get('data-lon'))
            return lat, lon
        
        # Methode 3: Koordinaten aus der Infobox
        infobox = soup.find('table', class_='infobox')
        if infobox:
            # Suche nach "Coordinates" Zeile
            rows = infobox.find_all('tr')
            for row in rows:
                header = row.find('th')
                if header and 'coordinates' in header.get_text().lower():
                    coord_cell = row.find('td')
                    if coord_cell:
                        coord_text = coord_cell.get_text()
                        
                        # Dezimalgrad-Format suchen
                        decimal_pattern = r'(\d+\.?\d*)°([NS])\s+(\d+\.?\d*)°([EW])'
                        match = re.search(decimal_pattern, coord_text)
                        if match:
                            lat_val, lat_dir, lon_val, lon_dir = match.groups()
                            lat = float(lat_val) * (1 if lat_dir == 'N' else -1)
                            lon = float(lon_val) * (1 if lon_dir == 'E' else -1)
                            return lat, lon
        
        # Methode 4: Koordinaten aus #coordinates Link
        coord_links = soup.find_all('a', href=re.compile(r'geo:'))
        for link in coord_links:
            href = link.get('href')
            # geo:48.8566,2.3522 Format
            geo_match = re.search(r'geo:(-?\d+\.?\d*),(-?\d+\.?\d*)', href)
            if geo_match:
                lat, lon = float(geo_match.group(1)), float(geo_match.group(2))
                return lat, lon
        
        # Methode 5: Koordinaten aus Microformat
        geo_spans = soup.find_all('span', class_='geo')
        for span in geo_spans:
            geo_text = span.get_text()
            # Format: "48.8566; 2.3522"
            parts = geo_text.split(';')
            if len(parts) == 2:
                try:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    return lat, lon
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"⚠️  Fehler beim Extrahieren der Koordinaten: {e}")
    
    return None, None

def extract_main_image(soup, base_url):
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
        print(f"⚠️  Fehler beim Extrahieren des Hauptbildes: {e}")
    
    return None

def download_image(img_url, destination_name, images_dir):
    """Lädt ein Bild herunter und speichert es lokal"""
    try:
        if not img_url:
            return None
            
        print(f"🖼️  Lade Bild herunter: {img_url}")
        
        # Korrekte Headers für Wikimedia
        headers = {
            'User-Agent': 'TravelHunters-Bot/1.0 (https://github.com/example/travelhunters; contact@example.com) Educational Project',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        response = requests.get(img_url, headers=headers, timeout=30)
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
        elif 'svg' in content_type:
            ext = '.svg'
        else:
            # Fallback aus URL
            parsed = urlparse(img_url)
            ext = os.path.splitext(parsed.path)[1] or '.jpg'
        
        filename = f"{safe_name}_main_{url_hash}{ext}"
        filepath = os.path.join(images_dir, filename)
        
        # Prüfe Dateigröße (max 50MB für Git LFS)
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 50 * 1024 * 1024:
            print(f"⚠️  Bild zu groß ({content_length} bytes): {img_url}")
            return None
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(filepath)
        print(f"✅ Bild heruntergeladen: {filename} ({file_size:,} bytes)")
        return filename
        
    except Exception as e:
        print(f"❌ Fehler beim Herunterladen des Bildes {img_url}: {e}")
        return None

def extract_long_description(soup):
    """Extrahiert eine lange, detaillierte Beschreibung von der Wikipedia-Seite"""
    try:
        content_div = soup.find('div', id='mw-content-text')
        if not content_div:
            return "Keine Beschreibung gefunden"
        
        parser_output = content_div.find('div', class_='mw-parser-output')
        if not parser_output:
            return "Keine Beschreibung gefunden"
        
        # Sammle alle substantiellen Absätze
        description_parts = []
        
        # 1. Hauptabsätze am Anfang
        paragraphs = parser_output.find_all('p', recursive=False)
        substantial_paragraphs = []
        
        for p in paragraphs:
            text = clean_text(p.get_text())
            
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
            
            # Stoppe nach 3 guten Absätzen für die Einleitung
            if len(substantial_paragraphs) >= 3:
                break
        
        description_parts.extend(substantial_paragraphs)
        
        # 2. Zusätzliche Informationen aus wichtigen Abschnitten
        section_keywords = ['history', 'geography', 'culture', 'tourism', 'economy', 'demographics', 'climate']
        
        for keyword in section_keywords:
            if len(' '.join(description_parts)) > 2000:  # Begrenze Gesamtlänge
                break
                
            # Suche nach Überschrift mit diesem Keyword
            headings = parser_output.find_all(['h2', 'h3'], string=re.compile(keyword, re.IGNORECASE))
            
            for heading in headings:
                if len(' '.join(description_parts)) > 2000:
                    break
                    
                # Finde den nächsten Absatz nach der Überschrift
                next_element = heading.find_next_sibling()
                while next_element:
                    if next_element.name == 'p':
                        text = clean_text(next_element.get_text())
                        if len(text) > 100:  # Nur substantielle Absätze
                            description_parts.append(text)
                            break
                    elif next_element.name in ['h2', 'h3']:  # Nächste Überschrift erreicht
                        break
                    next_element = next_element.find_next_sibling()
        
        # Erstelle finale Beschreibung
        if not description_parts:
            return "Keine Beschreibung gefunden"
        
        # Kombiniere alle Teile
        full_description = ' '.join(description_parts)
        
        # Begrenze auf eine sinnvolle Länge (ca. 1500-2500 Zeichen)
        if len(full_description) > 2500:
            # Schneide am letzten Satzende ab
            sentences = full_description.split('. ')
            truncated = ''
            for sentence in sentences:
                if len(truncated + sentence + '. ') > 2500:
                    break
                truncated += sentence + '. '
            full_description = truncated.rstrip() if truncated else full_description[:2500] + '...'
        
        return full_description if full_description else "Keine Beschreibung gefunden"
        
    except Exception as e:
        print(f"⚠️  Fehler beim Extrahieren der Beschreibung: {e}")
        return "Keine Beschreibung gefunden"

def scrape_destination(url, images_dir):
    """Scrapt eine komplette Destination mit allen Informationen"""
    try:
        print(f"🔍 Scraping: {url}")
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Titel
        title_element = soup.find('h1', class_='firstHeading')
        title = title_element.get_text().strip() if title_element else "Unknown"
        
        # Beschreibung (lang und detailliert)
        description = extract_long_description(soup)
        
        # Koordinaten
        latitude, longitude = extract_coordinates(soup)
        
        # Hauptbild extrahieren und herunterladen
        main_image_url = extract_main_image(soup, url)
        image_filename = None
        
        if main_image_url:
            image_filename = download_image(main_image_url, title, images_dir)
        
        result = {
            'name': title,
            'url': url,
            'description': description,
            'latitude': latitude,
            'longitude': longitude,
            'image_url': main_image_url,
            'image_file': image_filename,
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'wikipedia'
        }
        
        print(f"✅ Erfolgreich: {title}")
        if latitude and longitude:
            print(f"📍 Koordinaten: {latitude:.4f}, {longitude:.4f}")
        else:
            print(f"⚠️  Keine Koordinaten gefunden")
        if image_filename:
            print(f"🖼️  Bild gespeichert: {image_filename}")
        else:
            print(f"⚠️  Kein Bild gefunden")
        print(f"📝 Beschreibung: {len(description)} Zeichen")
        
        return result
        
    except Exception as e:
        print(f"❌ Fehler bei {url}: {e}")
        return None

def main():
    """Hauptfunktion für vollständiges Wikipedia-Scraping"""
    print("🚀 WIKIPEDIA DESTINATION SCRAPER")
    print("🌍 Vollständiges Scraping mit Koordinaten, Bildern und detaillierten Beschreibungen")
    print("=" * 80)
    
    # Verzeichnisse erstellen
    images_dir = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/destination_images"
    output_dir = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/json_backup"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Lade Destination-Links
    print("📥 Lade Destination-Links...")
    try:
        destination_links = get_destination_urls()
        print(f"✅ {len(destination_links)} Destination-Links geladen")
    except Exception as e:
        print(f"❌ Fehler beim Laden der Links: {e}")
        return
    
    # Ausgabedatei
    output_file = os.path.join(output_dir, "wikipedia_destinations.json")
    
    # Teste mit wenigen Destinationen oder verwende alle
    destination_links = destination_links[:25]  # Für Tests: ersten 25 Destinationen
    # Für Vollbetrieb: alle Links verwenden
    
    print(f"🎯 Starte Scraping von {len(destination_links)} Destinationen...")
    print(f"💾 Streaming-Ausgabe nach: {output_file}")
    
    successful_count = 0
    error_count = 0
    
    # Öffne Ausgabedatei für Streaming (append mode)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, url in enumerate(destination_links, 1):
            print(f"\n--- Destination {i}/{len(destination_links)} ---")
            
            result = scrape_destination(url, images_dir)
            
            if result:
                # Schreibe sofort in Datei (streaming)
                json.dump(result, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
                f.flush()  # Stelle sicher, dass es sofort geschrieben wird
                successful_count += 1
            else:
                error_count += 1
            
            # Fortschrittsbericht alle 50 Destinationen
            if i % 50 == 0:
                print(f"\n📊 ZWISCHENBERICHT:")
                print(f"✅ Erfolgreich: {successful_count}")
                print(f"❌ Fehler: {error_count}")
                print(f"📈 Fortschritt: {i}/{len(destination_links)} ({(i/len(destination_links)*100):.1f}%)")
            
            # Pause zwischen Requests (höflich gegenüber Wikipedia)
            time.sleep(1)  # 1 Sekunde Pause
    
    print(f"\n🎯 ENDERGEBNIS:")
    print(f"✅ Erfolgreich gescrapt: {successful_count}/{len(destination_links)} Destinationen")
    print(f"❌ Fehler: {error_count}")
    print(f"📁 JSON gespeichert: {output_file}")
    print(f"🖼️  Bilder-Verzeichnis: {images_dir}")
    
    # Zeige heruntergeladene Bilder
    try:
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.gif', '.webp', '.svg'))]
        print(f"📸 Heruntergeladene Bilder: {len(image_files)}")
    except:
        print("⚠️  Keine Bilder-Statistik verfügbar")

if __name__ == "__main__":
    main()
