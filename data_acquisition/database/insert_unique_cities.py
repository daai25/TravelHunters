import json
import sqlite3

# --- Config ---
json_file = '../json_final/unique_cities.json'
sqlite_db = 'travelhunters.db'

# --- Load JSON data ---
with open(json_file, 'r', encoding='utf-8') as f:
    cities = json.load(f)

# --- Connect to SQLite ---
conn = sqlite3.connect(sqlite_db)
cursor = conn.cursor()

# --- Insert data into city table ---
for entry in cities:
    name = entry.get('city')
    latitude = entry.get('latitude')
    longitude = entry.get('longitude')

    cursor.execute('''
        INSERT INTO city (name, latitude, longitude)
        VALUES (?, ?, ?)
    ''', (name, latitude, longitude))

# --- Commit and close ---
conn.commit()
conn.close()

print("City data successfully inserted into database.")