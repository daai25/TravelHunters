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

    # Lade alle Hotel-Descriptions
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, description FROM booking_worldwide WHERE description IS NOT NULL AND description != ''")
    hotels = cur.fetchall()
    conn.close()

    # Embedding-Modell laden
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    user_emb = model.encode([user_query], normalize_embeddings=True)[0]
    descs = [desc for _, desc in hotels]
    hotel_embs = model.encode(descs, normalize_embeddings=True)

    # Similarity berechnen
    scores = np.dot(hotel_embs, user_emb)
    best_idx = int(np.argmax(scores))
    best_hotel_id, best_desc = hotels[best_idx]
    best_score = scores[best_idx]
    print(f"Empfohlenes Hotel: ID {best_hotel_id}\nBeschreibung: {best_desc}\nScore: {best_score:.3f}")

if __name__ == "__main__":
    user_query = input("Bitte gib deine Hotelsuche ein: ")
    recommend_hotel(user_query)
