"""
Flask Web-Server für Hotelempfehlungen basierend auf semantischer Suche.
Konvertiert das ursprüngliche Konsolen-Tool zu einer Web-API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os

# Flask App erstellen
app = Flask(__name__)
CORS(app)  # Ermöglicht Requests vom Frontend

# Konfiguration
DB_PATH = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/database/travelhunters.db"
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# Globale Variablen für Performance
model = None
print("Lade ML-Modell...")

def load_model():
    """Lädt das Sentence-Transformer Modell einmalig beim Start"""
    global model
    if model is None:
        print(f"Lade Modell: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        print("Modell erfolgreich geladen!")
    return model

def extract_price_limit(user_query):
    """Extrahiert Preislimit aus der User-Query"""
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
    
    return price_limit

def get_hotels_from_db(price_limit=None):
    """Lädt Hotels aus der Datenbank mit optionalem Preisfilter"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Datenbank nicht gefunden: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    try:
        if price_limit is not None:
            cur.execute("""
                SELECT id, description, price 
                FROM booking_worldwide 
                WHERE description IS NOT NULL AND description != '' 
                AND price IS NOT NULL AND price <= ?
            """, (price_limit,))
            hotels = cur.fetchall()
            
            if not hotels:
                print(f"Keine Hotels mit Preis <= {price_limit} gefunden. Verwende alle Hotels.")
                cur.execute("""
                    SELECT id, description, price 
                    FROM booking_worldwide 
                    WHERE description IS NOT NULL AND description != ''
                """)
                hotels = cur.fetchall()
        else:
            cur.execute("""
                SELECT id, description, price 
                FROM booking_worldwide 
                WHERE description IS NOT NULL AND description != ''
            """)
            hotels = cur.fetchall()
        
        return hotels, cur, conn
    except Exception as e:
        conn.close()
        raise e

def get_hotel_details(hotel_id, cur):
    """Lädt vollständige Hotel-Details aus der Datenbank"""
    cur.execute("SELECT * FROM booking_worldwide WHERE id=?", (hotel_id,))
    hotel_info = cur.fetchone()
    
    if not hotel_info:
        return None
    
    columns = [desc[0] for desc in cur.description]
    info = dict(zip(columns, hotel_info))
    
    return {
        "id": hotel_id,
        "name": info.get("name") or info.get("hotel_name") or f"Hotel {hotel_id}",
        "location": info.get("location", "Unbekannt"),
        "price": f"CHF {info.get('price', 0)}",
        "rating": float(info.get("rating", 0)),
        "description": info.get("description", ""),
        "image": info.get("image_url", None),  # Falls vorhanden
        "amenities": []  # Kann erweitert werden
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Einfacher Health-Check Endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Hotel Recommender API ist bereit!",
        "model": MODEL_NAME
    })

@app.route('/recommend', methods=['POST'])
def recommend_hotels():
    """Hauptendpoint für Hotelempfehlungen"""
    try:
        # Request-Daten holen
        text_input = request.form.get('text_input', '').strip()
        language = request.form.get('language', 'de')
        
        # Validierung
        if not text_input:
            return jsonify({
                "error": "text_input ist erforderlich",
                "recommendations": []
            }), 400
        
        print(f"Query: '{text_input}' (Sprache: {language})")
        
        # Modell laden
        ml_model = load_model()
        
        # Preislimit extrahieren
        price_limit = extract_price_limit(text_input)
        if price_limit:
            print(f"💰 Preislimit erkannt: {price_limit}")
        
        # Hotels aus DB laden
        hotels, cur, conn = get_hotels_from_db(price_limit)
        print(f"{len(hotels)} Hotels gefunden")
        
        if not hotels:
            return jsonify({
                "error": "Keine Hotels in der Datenbank gefunden",
                "recommendations": []
            })
        
        # Embeddings berechnen
        user_emb = ml_model.encode([text_input], normalize_embeddings=True)[0]
        descs = [desc for _, desc, _ in hotels]
        hotel_embs = ml_model.encode(descs, normalize_embeddings=True)
        
        # Similarity berechnen
        scores = np.dot(hotel_embs, user_emb)
        
        # Top 3 Hotels finden
        top_k = 3
        best_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Empfehlungen zusammenstellen
        recommendations = []
        for rank, idx in enumerate(best_indices, 1):
            hotel_id = hotels[idx][0]
            score = scores[idx]
            
            hotel_details = get_hotel_details(hotel_id, cur)
            if hotel_details:
                # Score hinzufügen
                hotel_details["similarity_score"] = float(score)
                hotel_details["rank"] = rank
                
                # Beschreibung kürzen falls zu lang
                desc = hotel_details["description"]
                if len(desc) > 150:
                    hotel_details["description"] = desc[:150] + "..."
                
                recommendations.append(hotel_details)
        
        conn.close()
        
        print(f"{len(recommendations)} Empfehlungen generiert")
        
        return jsonify({
            "query": text_input,
            "language": language,
            "price_limit": price_limit,
            "recommendations": recommendations
        })
        
    except FileNotFoundError as e:
        return jsonify({
            "error": f"Datenbank nicht gefunden: {str(e)}",
            "recommendations": []
        }), 500
        
    except Exception as e:
        print(f"Fehler: {str(e)}")
        return jsonify({
            "error": f"Serverfehler: {str(e)}",
            "recommendations": []
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test-Endpoint mit Beispiel-Empfehlung"""
    return jsonify({
        "message": "Test erfolgreich!",
        "recommendations": [
            {
                "id": 1,
                "name": "Test Hotel Zürich",
                "location": "Zürich, Schweiz",
                "price": "CHF 250",
                "rating": 4.5,
                "description": "Ein wunderschönes Test-Hotel im Herzen von Zürich",
                "amenities": ["WiFi", "Spa", "Restaurant"],
                "similarity_score": 0.85,
                "rank": 1
            }
        ]
    })

if __name__ == "__main__":
    print(" Starte Hotel Recommender Server...")
    print(f" Datenbank: {DB_PATH}")
    print(f" Modell: {MODEL_NAME}")
    
    # Modell beim Start laden
    load_model()
    
    print("\n" + "="*50)
    print("Server läuft auf:")
    print("   http://localhost:8000")
    print("   http://127.0.0.1:8000")
    print("\n Endpoints:")
    print("   GET  /health  - Health Check")
    print("   GET  /test    - Test mit Beispieldaten") 
    print("   POST /recommend - Hotelempfehlungen")
    print("="*50)
    
    # Server starten
    app.run(
        host='0.0.0.0',  # Ermöglicht Zugriff von anderen IPs
        port=8000,       # Port 8000 statt 5000
        debug=True,      # Für Development
        threaded=True    # Für bessere Performance
    )