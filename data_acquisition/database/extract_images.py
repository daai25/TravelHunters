import os
import sqlite3
from PIL import Image
import io

# Konfiguration
travelhunters_db = 'travelhunters.db'
image_dbs = ['byg_images.db', 'city_images.db', 'geo_city_images.db', 'wiki_images.db']  # Passe das ggf. an
output_dir = 'extracted_images'

# Stelle sicher, dass das Ausgabeverzeichnis existiert
os.makedirs(output_dir, exist_ok=True)

# Lade alle Städte in ein Dictionary: city_id -> city_name
def load_cities():
    cities = {}
    conn = sqlite3.connect(travelhunters_db)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM city")
    for city_id, name in cursor.fetchall():
        cities[city_id] = name
    conn.close()
    return cities

# Extrahiere Bilder aus einer Datenbank
def extract_images_from_db(db_path, cities_dict):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, extension, file, city_id FROM city_images")

    for img_id, filename, extension, blob_data, city_id in cursor.fetchall():
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
            print(f"Gespeichert: {filepath}")
        except Exception as e:
            print(f"Fehler beim Verarbeiten von Bild {img_id} aus {db_path}: {e}")

    conn.close()

def main():
    cities = load_cities()
    for db_path in image_dbs:
        print(f"Verarbeite {db_path} ...")
        extract_images_from_db(db_path, cities)
    print("Fertig.")

if __name__ == '__main__':
    main()
