#!/usr/bin/env python3
"""
Generiert pro Hotel-ID mehrere realistische User-Queries (nur hotel_id, user_query)
Die Queries klingen wie echte Nutzereingaben und basieren auf der Beschreibung
"""
import sqlite3
import random
from pathlib import Path
import csv

# Datenbankpfad
project_root = Path(__file__).parent.parent.parent.parent
DB_PATH = project_root / "data_acquisition" / "database" / "travelhunters.db"

# Ziel-Datei für Testdatensatz
OUT_CSV = Path(__file__).parent / "user_queries.csv"

def fetch_hotel_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, description, location, price, rating FROM booking_worldwide
        WHERE description IS NOT NULL AND description != ''
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def generate_user_queries(hotel_id, description, location, price, rating, n_variants=3):
    # Natürlichere, alltagsnahe Query-Templates (preis- und ortsbezogen)
    templates = [
        "I want a hotel at the beach with my family, maximum ${price:.0f}.",
        "Looking for a family-friendly hotel in {location} with a pool, up to ${price:.0f} per night.",
        "Can you recommend a romantic hotel in {location} for less than ${price:.0f}?",
        "I need a hotel in {location} with breakfast included, not more than ${price:.0f}.",
        "Where can I find a hotel in {location} with a spa for under ${price:.0f}?",
        "Is there a hotel in {location} with sea view and good reviews, max ${price:.0f}?",
        "I want to stay in {location} with my kids, close to the city center, budget ${price:.0f}.",
        "Looking for a quiet hotel in {location} for a relaxing holiday, max ${price:.0f}.",
        "Any hotel in {location} with free parking and WiFi, up to ${price:.0f}?",
        "I want a hotel in {location} with {desc_kw}, not more than ${price:.0f} per night.",
        "Can you suggest a hotel in {location} for a business trip, max ${price:.0f}?",
        "Where can I book a hotel in {location} with a gym and breakfast, under ${price:.0f}?",
        "Looking for a pet-friendly hotel in {location}, budget ${price:.0f}.",
        "I want a hotel in {location} with a balcony and nice view, max ${price:.0f}.",
        "Any recommendations for a hotel in {location} with {desc_kw}, up to ${price:.0f}?"
    ]
    # Zwei zusätzliche, weniger preis-/ortsbezogene Templates
    generic_templates = [
        "I'm searching for a place that feels like home and has {desc_kw}.",
        "I want a hotel with excellent service and {desc_kw} for my next vacation.",
        "A hotel with a cozy atmosphere and {desc_kw} would be perfect.",
        "Looking for a unique experience, maybe a hotel with {desc_kw}.",
        "Can you recommend something special with {desc_kw}?"
    ]
    # Extrahiere Keywords aus der Beschreibung
    desc_kw = None
    if description:
        words = [w.strip('.,;:!?') for w in description.split() if len(w) > 4]
        if words:
            desc_kw = random.choice(words)
        else:
            desc_kw = description.split()[0]
    else:
        desc_kw = "nice amenities"
    # Generiere die ersten 3 Varianten wie bisher (preis-/ortsbezogen)
    queries = set()
    for _ in range(n_variants * 3):
        template = random.choice(templates)
        query = template.format(
            location=location or "a nice place",
            price=price or 150,
            rating=rating or 7.0,
            desc_kw=desc_kw or "good service"
        )
        queries.add(query)
        if len(queries) >= n_variants:
            break
    # Füge 2 generische Varianten hinzu (weniger preis-/ortsbezogen)
    for _ in range(10):
        template = random.choice(generic_templates)
        query = template.format(
            location=location or "a nice place",
            price=price or 150,
            rating=rating or 7.0,
            desc_kw=desc_kw or "good service"
        )
        queries.add(query)
        if len(queries) >= n_variants + 2:
            break
    # Reihenfolge: erst die ersten 3, dann die generischen
    queries_list = list(queries)
    return queries_list[:n_variants] + queries_list[n_variants:n_variants+2]

def main():
    print("Generiere minimalen User-Query-Testdatensatz...")
    hotels = fetch_hotel_data()
    rows = []
    for hotel_id, description, location, price, rating in hotels:
        queries = generate_user_queries(hotel_id, description, location, price, rating, n_variants=3)
        for q in queries:
            rows.append({"hotel_id": hotel_id, "user_query": q})
    # Schreibe als CSV
    with open(OUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hotel_id", "user_query"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Fertig! {len(rows)} User-Queries gespeichert in {OUT_CSV}")

if __name__ == "__main__":
    main()
