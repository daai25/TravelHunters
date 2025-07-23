"""
Unified Flask API that combines:
1. City prediction from images (PyTorch CNN)
2. Hotel recommendations (Semantic search)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import io
import base64
import tempfile
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Flask App
app = Flask(__name__)
CORS(app)

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))

# City Prediction Configuration
possible_model_paths = [
    os.path.join(current_dir, "city_model.pth"),
    os.path.join(current_dir, "..", "..", "cnn", "city_model.pth"),
    os.path.join(current_dir, "city_classifier_model.pth"),
    os.path.join(current_dir, "..", "..", "models", "city_model.pth"),
]

MODEL_PATH = None
for path in possible_model_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        print(f"Found city model at: {MODEL_PATH}")
        break

# Hotel Recommendation Configuration
DB_PATH = "/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/data_acquisition/database/travelhunters.db"
HOTEL_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# Global variables
city_model = None
hotel_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# City names (127 cities)
city_names = [
    "Agios Ioannis Mykonos", "Agios Sostis Mykonos", "Agios Stefanos", "Agrari", "Akrotiri",
    "Amed", "Amsterdam", "Ano Mera", "Auckland", "Baa Atoll",
    "Bangkok", "Barcelona", "Beijing", "Berlin", "Bogota",
    "Brisbane", "Budapest", "Buenos Aires", "Cairo", "Cala Llenya",
    "Cancún", "Canggu", "Cape Town", "Casablanca", "Chicago",
    "Copenhagen", "Dal", "Dhangethi", "Dhidhdhoo", "Dhiffushi",
    "Dhigurah", "Dubai", "Eidsvoll", "Elia", "Es Cana",
    "Fenfushi", "Fira", "Fulidhoo", "Gaafu Alifu Atoll", "Gardermoen",
    "Geneva", "Gjerdrum", "Gystad", "Hangnaameedhoo", "Helsinki",
    "Hong Kong", "Ibiza Town", "Imerovigli", "Jessheim", "Johannesburg",
    "Kintamani", "Klofta", "Klouvas", "Kuala Lumpur", "Kuta",
    "Las Vegas", "Lima", "Lisbon", "London", "Los Angeles",
    "Madrid", "Makunudhoo", "Male City", "Mandhoo", "Marrakech",
    "Meedhoo", "Meemu Atoll", "Megalokhori", "Melbourne", "Miami Beach",
    "Montréal", "Mumbai", "Mushimasgali", "Mýkonos City", "Nannestad",
    "New Delhi", "New York", "Nika Island", "Noonu", "North Male Atoll",
    "Nusa Dua", "Oia", "Osaka", "Oslo", "Paris",
    "Payangan", "Perivolos", "Perth", "Phuket Town", "Platis Yialos Mykonos",
    "Playa d'en Bossa", "Playa del Carmen", "Plintri", "Portinatx", "Prague",
    "Puerto de San Miguel", "Raa Atoll", "Rio de Janeiro", "Rome", "San Antonio",
    "San Antonio Bay", "San Francisco", "Sant Joan de Labritja", "Santa Agnès de Corona", "Santa Eularia des Riu",
    "Santiago de Compostela", "Sao Paulo", "Selemadeg", "Seminyak", "Seoul",
    "Shanghai", "Singapore", "South Male Atoll", "Stockholm", "Super Paradise Beach",
    "Sydney", "Tabanan", "Talamanca", "Thundufushi", "Tokyo", "Toronto", "Tourlos", 
    "Tulum", "Ubud", "Uluwatu", "Vancouver", "Vienna", "Zürich"
]

# === CITY PREDICTION FUNCTIONS ===

def load_city_model():
    """Load PyTorch city prediction model"""
    global city_model
    if city_model is None and MODEL_PATH:
        try:
            print("Loading city prediction model...")
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 127)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            city_model = model.to(device)
            print(f"✅ City model loaded successfully on {device}")
        except Exception as e:
            print(f"❌ Error loading city model: {e}")
    return city_model

def predict_city_from_image(image):
    """Predict city from PIL Image"""
    model = load_city_model()
    if not model:
        return None, 0.0
    
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor).squeeze()
            probs = F.softmax(output, dim=0)
            top_prob, top_idx = torch.topk(probs, 1)
        
        idx = top_idx.item()
        city = city_names[idx] if idx < len(city_names) else f"Class {idx}"
        confidence = round(top_prob.item(), 4)
        
        return city, confidence
    except Exception as e:
        print(f"Error during city prediction: {e}")
        return None, 0.0

# === HOTEL RECOMMENDATION FUNCTIONS ===

def load_hotel_model():
    """Load sentence transformer model for hotel recommendations"""
    global hotel_model
    if hotel_model is None:
        try:
            print("Loading hotel recommendation model...")
            hotel_model = SentenceTransformer(HOTEL_MODEL_NAME, trust_remote_code=True)
            print("✅ Hotel model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading hotel model: {e}")
    return hotel_model

def extract_price_limit(user_query):
    """Extract price limit from user query"""
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
    """Load hotels from database with optional price filter"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    
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
                print(f"No hotels with price <= {price_limit} found. Using all hotels.")
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
    """Load complete hotel details from database"""
    cur.execute("SELECT * FROM booking_worldwide WHERE id=?", (hotel_id,))
    hotel_info = cur.fetchone()
    
    if not hotel_info:
        return None
    
    columns = [desc[0] for desc in cur.description]
    info = dict(zip(columns, hotel_info))
    
    return {
        "id": hotel_id,
        "name": info.get("name") or info.get("hotel_name") or f"Hotel {hotel_id}",
        "location": info.get("location", "Unknown"),
        "price": f"CHF {info.get('price', 0)}",
        "rating": float(info.get("rating", 0)),
        "description": info.get("description", ""),
        "image": info.get("image_url", None),
    }

def recommend_hotels(user_query):
    """Get hotel recommendations based on user query"""
    model = load_hotel_model()
    if not model:
        return []
    
    try:
        price_limit = extract_price_limit(user_query)
        hotels, cur, conn = get_hotels_from_db(price_limit)
        
        if not hotels:
            return []
        
        # Calculate embeddings
        user_emb = model.encode([user_query], normalize_embeddings=True)[0]
        descs = [desc for _, desc, _ in hotels]
        hotel_embs = model.encode(descs, normalize_embeddings=True)
        
        # Calculate similarity
        scores = np.dot(hotel_embs, user_emb)
        
        # Get top 3 hotels
        top_k = 3
        best_indices = np.argsort(scores)[-top_k:][::-1]
        
        recommendations = []
        for rank, idx in enumerate(best_indices, 1):
            hotel_id = hotels[idx][0]
            score = scores[idx]
            
            hotel_details = get_hotel_details(hotel_id, cur)
            if hotel_details:
                hotel_details["similarity_score"] = float(score)
                hotel_details["rank"] = rank
                
                # Truncate description if too long
                desc = hotel_details["description"]
                if len(desc) > 150:
                    hotel_details["description"] = desc[:150] + "..."
                
                recommendations.append(hotel_details)
        
        conn.close()
        return recommendations
        
    except Exception as e:
        print(f"Error in hotel recommendations: {e}")
        return []

# === FLASK ROUTES ===

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'city_prediction': MODEL_PATH is not None,
            'hotel_recommendations': os.path.exists(DB_PATH) if DB_PATH else False
        },
        'device': str(device)
    })

@app.route('/predict_city', methods=['POST'])
def predict_city():
    """Predict city from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        image = Image.open(file.stream)
        city, confidence = predict_city_from_image(image)
        
        if city:
            return jsonify({
                'success': True,
                'prediction': {
                    'city': city,
                    'confidence': confidence
                }
            })
        else:
            return jsonify({'error': 'City prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/recommend_hotels', methods=['POST'])
def recommend_hotels_endpoint():
    """Get hotel recommendations"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        user_query = data['query'].strip()
        if not user_query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        recommendations = recommend_hotels(user_query)
        
        return jsonify({
            'success': True,
            'query': user_query,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/travel_recommendations', methods=['POST'])
def travel_recommendations():
    """Complete travel pipeline: city prediction + hotel recommendations"""
    try:
        # Check for image
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get user query
        user_query = request.form.get('query', '').strip()
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Step 1: Predict city from image
        image = Image.open(file.stream)
        city, confidence = predict_city_from_image(image)
        
        if not city:
            return jsonify({'error': 'City prediction failed'}), 500
        
        # Step 2: Modify query if confidence is high enough
        modified_query = user_query
        if confidence > 0.85:
            modified_query = f"{user_query} I would like to visit {city} with a {confidence:.2f} level of confidence."
        
        # Step 3: Get hotel recommendations
        recommendations = recommend_hotels(modified_query)
        
        return jsonify({
            'success': True,
            'city_prediction': {
                'city': city,
                'confidence': confidence
            },
            'query': {
                'original': user_query,
                'modified': modified_query,
                'confidence_threshold_met': confidence > 0.85
            },
            'hotel_recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/travel_recommendations_base64', methods=['POST'])
def travel_recommendations_base64():
    """Complete travel pipeline using base64 image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'query' not in data:
            return jsonify({'error': 'Missing image or query field'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Predict city
        city, confidence = predict_city_from_image(image)
        
        if not city:
            return jsonify({'error': 'City prediction failed'}), 500
        
        # Modify query
        user_query = data['query'].strip()
        modified_query = user_query
        if confidence > 0.85:
            modified_query = f"{user_query} I would like to visit {city} with a {confidence:.2f} level of confidence."
        
        # Get hotel recommendations
        recommendations = recommend_hotels(modified_query)
        
        return jsonify({
            'success': True,
            'city_prediction': {
                'city': city,
                'confidence': confidence
            },
            'query': {
                'original': user_query,
                'modified': modified_query,
                'confidence_threshold_met': confidence > 0.85
            },
            'hotel_recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Unified Travel API...")
    print(f"📍 City Model: {'✅' if MODEL_PATH else '❌'} {MODEL_PATH}")
    print(f"🏨 Hotel DB: {'✅' if os.path.exists(DB_PATH) else '❌'} {DB_PATH}")
    print(f"🔧 Device: {device}")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5002,  # Use a different port to avoid conflicts
        debug=True
    )