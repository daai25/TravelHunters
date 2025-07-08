import sqlite3
import json


# Function to read JSON file
def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Function to insert data into SQLite database
def insert_data(data, db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    for item in data:
        # Extract fields from JSON
        name = item['name']
        link = item['link']
        rating = item['rating']
        price = item['price']
        location = item['location']
        description = item['description']
        image_url = item['images'][0] if item['images'] else None
        latitude = item['latitude']
        longitude = item['longitude']

        # Insert into database
        c.execute(
            '''INSERT INTO booking_worldwide (name, link, rating, price, location, description, image_url, latitude,
                                              longitude)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (name, link, rating, price, location, description, image_url, latitude, longitude))

    conn.commit()
    conn.close()

# Main function to run the script
def main():
    json_file = '../json_final/booking_worldwide_enriched.json'
    db_file = 'travelhunters.db'

    data = read_json(json_file)
    insert_data(data, db_file)

    print("Data inserted successfully!")

if __name__ == '__main__':
    main()
