# embedding_score_generator.py
"""
Dieses Skript berechnet Similarity Scores zwischen Hotel-Descriptions aus der travelhunters.db und den zugehörigen user_queries aus user_queries.csv.
Die Ergebnisse werden in einer neuen Tabelle 'embeddings' in der travelhunters.db gespeichert.
"""
import sqlite3
import csv
from sentence_transformers import SentenceTransformer
import numpy as np

DB_PATH = "../../travelhunters.db"
CSV_PATH = "../machine_learning_modells/models/user_queries.csv"
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# 1. Lade alle Descriptions aus der Datenbank (hotel_id -> description)
def load_descriptions():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, description FROM booking_worldwide")
    desc_dict = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return desc_dict

# 2. Lade alle user_queries aus der CSV (hotel_id -> [user_query,...])
def load_user_queries():
    queries = {}
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hid = int(row['hotel_id'])
            queries.setdefault(hid, []).append(row['user_query'])
    return queries

# 3. Berechne Similarity Scores und speichere in DB
def main():
    desc_dict = load_descriptions()
    queries = load_user_queries()
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    # DB vorbereiten
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            hotel_id INTEGER,
            user_query TEXT,
            description TEXT,
            score REAL
        )
    """)
    conn.commit()

    for hotel_id, user_query_list in queries.items():
        if hotel_id not in desc_dict:
            continue  # skip, falls keine Description vorhanden
        desc = desc_dict[hotel_id]
        desc_emb = model.encode([desc], normalize_embeddings=True)[0]
        user_embs = model.encode(user_query_list, normalize_embeddings=True)
        # Similarity Score für jede Query
        for uq, uq_emb in zip(user_query_list, user_embs):
            score = float(np.dot(desc_emb, uq_emb))
            cur.execute(
                "INSERT INTO embeddings (hotel_id, user_query, description, score) VALUES (?, ?, ?, ?)",
                (hotel_id, uq, desc, score)
            )
    conn.commit()
    conn.close()
    print("Fertig! Alle Scores gespeichert.")

if __name__ == "__main__":
    main()
