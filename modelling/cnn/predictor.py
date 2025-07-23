from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import io
import base64
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# --- Configuration ---
# Get the directory where this script is located and build path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try multiple possible locations for the model
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
        print(f"Found model at: {MODEL_PATH}")
        break

if MODEL_PATH is None:
    print("❌ Model file not found in any expected location:")
    for path in possible_model_paths:
        print(f"  Checked: {path}")
    exit(1)

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 127
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of cities
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

# --- Rebuild the model architecture ---
def build_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)

# Load model once at startup
try:
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction Function ---
def predict_image(image):
    """
    Predict city from PIL Image object
    Returns: (city_name, confidence_score)
    """
    try:
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor).squeeze()
            probs = F.softmax(output, dim=0)
            top_prob, top_idx = torch.topk(probs, 1)
        
        idx = top_idx.item()
        city = city_names[idx] if idx < len(city_names) else f"Class {idx}"
        confidence = round(top_prob.item(), 4)
        
        return city, confidence
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None, 0.0

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(device),
        'num_classes': NUM_CLASSES
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict city from uploaded image
    Accepts: multipart/form-data with 'image' file
    Returns: JSON with prediction results
    """
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process image
        image = Image.open(file.stream)
        city, confidence = predict_image(image)
        
        if city:
            return jsonify({
                'success': True,
                'prediction': {
                    'city': city,
                    'confidence': confidence
                }
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """
    Predict city from base64 encoded image
    Accepts: JSON with 'image' field containing base64 string
    Returns: JSON with prediction results
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No base64 image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix if present
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        city, confidence = predict_image(image)
        
        if city:
            return jsonify({
                'success': True,
                'prediction': {
                    'city': city,
                    'confidence': confidence
                }
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict_url', methods=['POST'])
def predict_url():
    """
    Predict city from image URL or file path
    Accepts: JSON with 'image_path' field
    Returns: JSON with prediction results
    """
    try:
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'error': 'No image path provided'}), 400
        
        image_path = data['image_path']
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        # Load and predict
        image = Image.open(image_path)
        city, confidence = predict_image(image)
        
        if city:
            return jsonify({
                'success': True,
                'prediction': {
                    'city': city,
                    'confidence': confidence
                }
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/cities', methods=['GET'])
def get_cities():
    """Get list of all possible cities"""
    return jsonify({
        'cities': city_names,
        'total_cities': len(city_names)
    })

if __name__ == '__main__':
    print("🚀 Starting Flask City Predictor API...")
    print(f"📍 Model: {MODEL_PATH}")
    print(f"🔧 Device: {device}")
    print(f"🏙️  Cities: {len(city_names)}")
    print("=" * 50)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,       # Default Flask port
        debug=True       # Enable debug mode for development
    )