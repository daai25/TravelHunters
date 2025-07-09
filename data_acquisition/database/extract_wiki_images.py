import os
import sqlite3

# === CONFIG ===
database_file = "wiki_images.db"
output_folder = "./extracted_images"  # Create this folder if it doesn't exist

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Connect to database
conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# Fetch all images
cursor.execute("SELECT filename, file FROM wikipedia_images")
images = cursor.fetchall()

print(f"Exporting {len(images)} images...")

counter = 0
for filename, blob in images:
    try:
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'wb') as f:
            f.write(blob)
        counter += 1
        if counter % 100 == 0:
            print(f"{counter} images exported...")

    except Exception as e:
        print(f"❌ Error writing {filename}: {e}")

conn.close()
print(f"✅ Done. Exported {counter} images to '{output_folder}'.")
