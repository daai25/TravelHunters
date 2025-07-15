import os
import sqlite3
from PIL import Image
import io
from pathlib import Path

# Bestimme das Arbeitsverzeichnis relativ zum Skript
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Konfiguration
travelhunters_db = 'travelhunters.db'
image_dbs = ['byg_images.db', 'city_images.db', 'geo_city_images.db', 'wiki_images.db']
output_dir = 'extracted_images'

# Überprüfe ob alle benötigten Dateien existieren
missing_files = []
if not os.path.exists(travelhunters_db):
    missing_files.append(travelhunters_db)
for db in image_dbs:
    if not os.path.exists(db):
        missing_files.append(db)

if missing_files:
    print(f"❌ Fehlende Datenbankdateien: {missing_files}")
    print(f"📂 Aktuelles Arbeitsverzeichnis: {os.getcwd()}")
    print("📋 Verfügbare .db Dateien:")
    for file in Path(".").glob("*.db"):
        print(f"   {file}")
    exit(1)

# Stelle sicher, dass das Ausgabeverzeichnis existiert
os.makedirs(output_dir, exist_ok=True)

# Lade alle Städte in ein Dictionary: city_id -> city_name
def load_cities():
    """Lädt alle Städte aus der travelhunters Datenbank"""
    print(f"🔍 Verbinde mit {travelhunters_db}...")
    try:
        cities = {}
        conn = sqlite3.connect(travelhunters_db)
        cursor = conn.cursor()
        
        print("📊 Lade Städte aus der Datenbank...")
        cursor.execute("SELECT id, name FROM city")
        for city_id, name in cursor.fetchall():
            cities[city_id] = name
        conn.close()
        
        print(f"✅ {len(cities)} Städte erfolgreich geladen")
        return cities
    except Exception as e:
        print(f"❌ Fehler beim Laden der Städte: {e}")
        return {}

# Extrahiere Bilder aus einer Datenbank
def extract_images_from_db(db_path, cities_dict):
    """Extrahiert Bilder aus einer Bilddatenbank"""
    print(f"🖼️ Extrahiere Bilder aus {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Prüfe ob die Tabelle city_images existiert
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='city_images'")
        if not cursor.fetchone():
            print(f"⚠️ Tabelle 'city_images' nicht in {db_path} gefunden - überspringe")
            conn.close()
            return 0
        
        cursor.execute("SELECT id, filename, extension, file, city_id FROM city_images")
        results = cursor.fetchall()
        
        extracted_count = 0
        for img_id, filename, extension, blob_data, city_id in results:
            city_name = cities_dict.get(city_id, 'UnknownCity')
            city_folder = os.path.join(output_dir, city_name)
            os.makedirs(city_folder, exist_ok=True)

            # Bestimme Dateinamen
            file_extension = extension if extension.startswith('.') else f'.{extension}'
            safe_filename = f'{img_id}_{filename}{file_extension}'
            filepath = os.path.join(city_folder, safe_filename)

            # Bild speichern
            try:
                image = Image.open(io.BytesIO(blob_data))
                image.save(filepath)
                extracted_count += 1
            except Exception as e:
                print(f"❌ Fehler beim Speichern von {safe_filename}: {e}")

        conn.close()
        print(f"✅ {extracted_count} Bilder aus {db_path} extrahiert")
        return extracted_count
    except Exception as e:
        print(f"❌ Fehler beim Zugriff auf {db_path}: {e}")
        return 0

def main():
    """Haupt-Funktion zum Extrahieren aller Bilder"""
    print("🚀 Starte Bildextraktion...")
    cities = load_cities()
    
    if not cities:
        print("❌ Keine Städte geladen - Abbruch")
        return
    
    total_extracted = 0
    for db_path in image_dbs:
        if os.path.exists(db_path):
            total_extracted += extract_images_from_db(db_path, cities)
        else:
            print(f"⚠️ Datenbank {db_path} nicht gefunden - überspringe")
    
    print(f"🎉 Fertig! Insgesamt {total_extracted} Bilder extrahiert")

if __name__ == '__main__':
    main()
