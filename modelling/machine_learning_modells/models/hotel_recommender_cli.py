# hotel_recommender_cli.py
"""
Konsolen-Tool: Nutzer gibt eine Anfrage ein, das Skript schlägt das Hotel mit dem höchsten Similarity Score vor.
"""
def recommend_hotel(user_query):
    import sqlite3
    from sentence_transformers import SentenceTransformer
    import numpy as np

    DB_PATH = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/database/travelhunters.db"
    MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

    import re
    # Lade alle Hotel-Descriptions und Preise
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Preis aus Query extrahieren (z.B. "maximal 250", "unter 200", "bis 300")
    price_limit = None
    price_patterns = [
        r"maximal[\s:]*([0-9]+)",
        r"unter[\s:]*([0-9]+)",
        r"bis[\s:]*([0-9]+)",
        r"<=?[\s:]*([0-9]+)",
        r"([0-9]+)[\s]*([eE]uro|€|franc|franken|chf|usd|dollar)?[\s]*(pro nacht|/nacht|per night|night)?"
    ]
    for pat in price_patterns:
        m = re.search(pat, user_query, re.IGNORECASE)
        if m:
            try:
                price_limit = int(m.group(1))
                break
            except Exception:
                continue

    if price_limit is not None:
        cur.execute("SELECT id, description, price FROM booking_worldwide WHERE description IS NOT NULL AND description != '' AND price IS NOT NULL AND price <= ?", (price_limit,))
        hotels = cur.fetchall()
        if not hotels:
            print(f"Keine Hotels mit Preis <= {price_limit} gefunden. Zeige beste Empfehlungen ohne Preisfilter.")
            cur.execute("SELECT id, description, price FROM booking_worldwide WHERE description IS NOT NULL AND description != ''")
            hotels = cur.fetchall()
    else:
        cur.execute("SELECT id, description, price FROM booking_worldwide WHERE description IS NOT NULL AND description != ''")
        hotels = cur.fetchall()

    # Embedding-Modell laden
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    user_emb = model.encode([user_query], normalize_embeddings=True)[0]
    descs = [desc for _, desc, _ in hotels]
    hotel_embs = model.encode(descs, normalize_embeddings=True)

    # Similarity berechnen
    scores = np.dot(hotel_embs, user_emb)
    top_k = 3
    best_indices = np.argsort(scores)[-top_k:][::-1]
    print(f"\n🏨 Top {top_k} Empfehlungen:")
    print("=" * 60)
    for rank, idx in enumerate(best_indices, 1):
        hotel_id = hotels[idx][0]
        score = scores[idx]
        cur.execute("SELECT * FROM booking_worldwide WHERE id=?", (hotel_id,))
        hotel_info = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
        info = dict(zip(columns, hotel_info))
        # Felder holen
        name = info.get("name") or info.get("hotel_name") or f"Hotel {hotel_id}"
        location = info.get("location", "Unbekannt")
        price = info.get("price", 0)
        rating = info.get("rating", 0)
        desc = info.get("description", "")
        desc_short = (desc[:100] + "...") if len(desc) > 100 else desc
        print(f"\n{rank}. {name}")
        print(f"   📍 {location}")
        print(f"   💰 {price} pro Nacht")
        print(f"   ⭐ {rating}/10.0")
        print(f"   🎯 Score: {score:.3f}")
        if desc_short:
            print(f"   📝 {desc_short}")
        print("-" * 60)
    print("=" * 60)
    conn.close()

if __name__ == "__main__":
    user_query = input("Bitte gib deine Hotelsuche ein: ")
    recommend_hotel(user_query)
