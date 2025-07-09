import os
import sqlite3
from PIL import Image
import io

# === CONFIG ===
IMAGES_DIR = ''  # z.B. '/Users/deinname/OneDrive/CityImages'
CITY_DB_PATH = 'travelhunters.db'
IMAGES_DB_PATH = 'byg_images.db'
TARGET_SIZE = (224, 224)
MIN_FILE_SIZE_BYTES = 7000  # Optional: Bilder kleiner als 7 KB überspringen

# === Hilfsfunktion zur Normalisierung ===
def normalize_city_name(name):
    return name.replace("_", " ").replace("-", " ").strip().lower()

# === Verbindung zur travelhunters.db ===
city_conn = sqlite3.connect(CITY_DB_PATH)
city_cursor = city_conn.cursor()

# === Verbindung zur city_images.db ===
img_conn = sqlite3.connect(IMAGES_DB_PATH)
img_cursor = img_conn.cursor()

# === Tabelle erstellen, falls nicht vorhanden ===
img_cursor.execute("""
    CREATE TABLE IF NOT EXISTS city_images (
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        extension TEXT,
        file BLOB,
        bytes INTEGER,
        width INTEGER,
        height INTEGER,
        city_id INTEGER,
        FOREIGN KEY(city_id) REFERENCES city(id)
    );
""")

# === Alle Städte aus der travelhunters.db laden ===
city_cursor.execute("SELECT id, name FROM city")
cities = city_cursor.fetchall()
city_lookup = {normalize_city_name(name): city_id for city_id, name in cities}

# === Bildverarbeitung starten ===
for root, _, files in os.walk(IMAGES_DIR):
    for file in files:
        if not file.lower().endswith('.jpg'):
            continue

        file_path = os.path.join(root, file)

        # Optional: Kleine Dateien überspringen
        if os.path.getsize(file_path) < MIN_FILE_SIZE_BYTES:
            print(f"⚠️ Übersprungen (zu klein): {file}")
            continue

        base_name = os.path.splitext(file)[0]
        name_parts = base_name.split(' ')

        # Stadtnamen iterativ zusammensetzen, von lang nach kurz
        found_city_id = None
        for i in range(len(name_parts), 0, -1):
            possible_city = normalize_city_name(" ".join(name_parts[:i]))
            city_id = city_lookup.get(possible_city)
            if city_id:
                found_city_id = city_id
                break

        if not found_city_id:
            print(f"❌ Stadt nicht gefunden: '{base_name}' (Datei: {file})")
            continue

        # Bild laden und auf 224x224 skalieren
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                img = img.resize(TARGET_SIZE)

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()

                width, height = img.size
                size_in_bytes = len(img_bytes)

                # Bild einfügen
                img_cursor.execute("""
                    INSERT INTO city_images (filename, extension, file, bytes, width, height, city_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (file, 'jpg', img_bytes, size_in_bytes, width, height, found_city_id))

                print(f"✔️ Eingefügt: {file} (City ID: {found_city_id})")

        except Exception as e:
            print(f"❗ Fehler bei Datei {file}: {e}")

# === Änderungen speichern und Verbindungen schließen ===
img_conn.commit()
img_conn.close()
city_conn.close()

print("✅ Alle Bilder wurden verarbeitet.")
