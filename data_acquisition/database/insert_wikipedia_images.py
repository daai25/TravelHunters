
import os
import sqlite3
from PIL import Image, ImageOps
from io import BytesIO

# === CONFIG ===
image_folder = "YOUR_IMAGE_FOLDER"  # Replace with your actual image folder path
source_db = "./travelhunters.db"         # DB with "city" table
target_db = "./wiki_images.db" # DB with "wikipedia_images" table
target_size = (250, 250)
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# === CONNECT TO DATABASES ===
city_conn = sqlite3.connect(source_db)
city_cursor = city_conn.cursor()

image_conn = sqlite3.connect(target_db)
image_cursor = image_conn.cursor()

# === READ CITY TABLE ===
city_cursor.execute("SELECT id, name FROM city")
city_rows = city_cursor.fetchall()

# Normalized name → id lookup
city_lookup = {
    name.strip().lower().replace(" ", "_"): id
    for id, name in city_rows
}

# === HELPER FUNCTIONS ===
def resize_and_pad(image, size=(250, 250)):
    return ImageOps.pad(image, size, color=(255, 255, 255))

def file_already_exists(filename):
    image_cursor.execute("SELECT 1 FROM wikipedia_images WHERE filename = ?", (filename,))
    return image_cursor.fetchone() is not None

# === PROCESS FILES ===
counter = 0

for root, _, files in os.walk(image_folder):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        full_path = os.path.join(root, file)
        file_lower = file.lower()
        filename_only = os.path.basename(file)

        # Find matching city by prefix
        matching_city_name = next(
            (name for name in city_lookup if file_lower.startswith(name)),
            None
        )
        if not matching_city_name:
            continue

        city_id = city_lookup[matching_city_name]

        # Skip duplicates
        if file_already_exists(filename_only):
            continue

        try:
            with open(full_path, 'rb') as f:
                raw_bytes = f.read()

            # Load and resize image
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
            resized_image = resize_and_pad(image, size=target_size)

            # Convert to bytes
            output_buffer = BytesIO()
            resized_image.save(output_buffer, format='JPEG', quality=90)
            resized_bytes = output_buffer.getvalue()
            byte_size = len(resized_bytes)
            width, height = resized_image.size

            # Insert into DB
            image_cursor.execute("""
                INSERT INTO wikipedia_images (filename, extension, file, bytes, width, height, city_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                filename_only,
                'jpg',
                resized_bytes,
                byte_size,
                width,
                height,
                city_id
            ))

            counter += 1
            if counter % 100 == 0:
                image_conn.commit()
                print(f"{counter} images processed...")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

# === CLEANUP ===
image_conn.commit()
city_conn.close()
image_conn.close()
print(f"✅ Done. Stored {counter} resized city-linked images.")
