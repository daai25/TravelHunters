'''
This script uses the classifier to predict a designation based on an input image.
'''


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Configuration ---
MODEL_PATH = r"city_classifier_model.h5"
IMAGE_SIZE = (224, 224)  # Based on the error, this is what your model expects

# Path to model - make configurable to support different locations
import os
if not os.path.exists(MODEL_PATH):
    # Try to find the model relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alternate_path = os.path.join(script_dir, MODEL_PATH)
    if os.path.exists(alternate_path):
        MODEL_PATH = alternate_path
    else:
        print(f"⚠️ Warnung: Modell nicht gefunden unter {MODEL_PATH}")
        print(f"   Versuche auch: {alternate_path}")
        print("   Bitte geben Sie den vollständigen Pfad zum Modell an.")

# List of cities for the predictor
city_names = [
    "Agios Ioannis Mykonos",
    "Agios Sostis Mykonos",
    "Agios Stefanos",
    "Agrari",
    "Akrotiri",
    "Amed",
    "Amsterdam",
    "Ano Mera",
    "Auckland",
    "Baa Atoll",
    "Bangkok",
    "Barcelona",
    "Beijing",
    "Berlin",
    "Bogota",
    "Brisbane",
    "Budapest",
    "Buenos Aires",
    "Cairo",
    "Cala Llenya",
    "Cancún",
    "Canggu",
    "Cape Town",
    "Casablanca",
    "Chicago",
    "Copenhagen",
    "Dal",
    "Dhangethi",
    "Dhidhdhoo",
    "Dhiffushi",
    "Dhigurah",
    "Dubai",
    "Eidsvoll",
    "Elia",
    "Es Cana",
    "Fenfushi",
    "Fira",
    "Fulidhoo",
    "Gaafu Alifu Atoll",
    "Gardermoen",
    "Geneva",
    "Gjerdrum",
    "Gystad",
    "Hangnaameedhoo",
    "Helsinki",
    "Hong Kong",
    "Ibiza Town",
    "Imerovigli",
    "Jessheim",
    "Johannesburg",
    "Kintamani",
    "Klofta",
    "Klouvas",
    "Kuala Lumpur",
    "Kuta",
    "Las Vegas",
    "Lima",
    "Lisbon",
    "London",
    "Los Angeles",
    "Madrid",
    "Makunudhoo",
    "Male City",
    "Mandhoo",
    "Marrakech",
    "Meedhoo",
    "Meemu Atoll",
    "Megalokhori",
    "Melbourne",
    "Miami Beach",
    "Montréal",
    "Mumbai",
    "Mushimasgali",
    "Mýkonos City",
    "Nannestad",
    "New Delhi",
    "New York",
    "Nika Island",
    "Noonu",
    "North Male Atoll",
    "Nusa Dua",
    "Oia",
    "Osaka",
    "Oslo",
    "Paris",
    "Payangan",
    "Perivolos",
    "Perth",
    "Phuket Town",
    "Platis Yialos Mykonos",
    "Playa d'en Bossa",
    "Playa del Carmen",
    "Plintri",
    "Portinatx",
    "Prague",
    "Puerto de San Miguel",
    "Raa Atoll",
    "Rio de Janeiro",
    "Rome",
    "San Antonio",
    "San Antonio Bay",
    "San Francisco",
    "Sant Joan de Labritja",
    "Santa Agnès de Corona",
    "Santa Eularia des Riu",
    "Santiago de Compostela",
    "Sao Paulo",
    "Selemadeg",
    "Seminyak",
    "Seoul",
    "Shanghai",
    "Singapore",
    "South Male Atoll",
    "Stockholm",
    "Super Paradise Beach",
    "Sydney",
    "Tabanan",
    "Talamanca",
    "Thundufushi",
    "Tokyo",
    "Toronto",
    "Tourlos",
    "Tulum",
    "Ubud",
    "Uluwatu",
    "Vancouver",
    "Vienna",
    "Zürich"]

# --- Load Model ---
try:
    print(f"Versuche Modell zu laden von: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Modell erfolgreich geladen!")
except Exception as e:
    print(f"❌ Fehler beim Laden des Modells: {e}")
    print("Stellen Sie sicher, dass die Modell-Datei existiert und TensorFlow korrekt installiert ist.")
    model = None

# --- Predict Function ---
def predict_image(image_path):
    # Check if model was successfully loaded
    if model is None:
        print("❌ Kann keine Vorhersage machen: Modell wurde nicht geladen.")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Bild nicht gefunden: {image_path}")
        return
        
    try:
        # Load and preprocess image (no flattening)
        print(f"Lade Bild: {image_path}")
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

        print("Processed input shape:", img_array.shape)

        # Predict
        print("Vorhersage wird durchgeführt...")
        prediction = model.predict(img_array)
        print("Raw Model Output:", prediction)
        #print(max(prediction[0]))
    except Exception as e:
        print(f"❌ Fehler bei der Bildverarbeitung oder Vorhersage: {e}")
        return

    # Interpret prediction - with error handling
    try:
        if prediction.shape[-1] > 1:
            # Ensure we don't try to access out-of-bounds indices
            num_classes = min(len(city_names), prediction.shape[-1])
            if num_classes < 3:
                print(f"⚠️ Weniger als 3 Klassen verfügbar ({num_classes})")
            
            top_indices = np.argsort(prediction[0])[::-1][:3]  # Sort descending and take top 3
            
            # Make sure indices are in range
            valid_indices = [idx for idx in top_indices if idx < len(city_names)]
            
            print("\n🌍 Top Städte zu besuchen:")
            for i, idx in enumerate(valid_indices):
                confidence = prediction[0][idx] * 100
                print(f"{i+1}. {city_names[idx]} (Konfidenz: {confidence:.1f}%)")
                
            if not valid_indices:
                print("❌ Keine gültigen Vorhersagen gefunden.")
        else:
            # Binary classification fallback
            predicted_class = (prediction[0][0] > 0.5).astype("int")
            confidence = abs(prediction[0][0] - 0.5) * 2 * 100  # Scale to 0-100%
            print(f"Binäre Klassifikation: {predicted_class} (Konfidenz: {confidence:.1f}%)")
    except Exception as e:
        print(f"❌ Fehler bei der Interpretation der Vorhersage: {e}")


# --- Run Manually ---
if __name__ == "__main__":
    print("\n🔍 TravelHunters Städteklassifizierer 🌆")
    print("=======================================")
    print("Dieser Classifier kann Städte anhand von Bildern erkennen.")
    print(f"Modellpfad: {MODEL_PATH}")
    print(f"Unterstützte Städte: {len(city_names)}")
    print("---------------------------------------")
    
    try:
        image_path = input("Geben Sie den vollständigen Pfad zum Bild ein: ").strip()
        if not image_path:
            print("❌ Kein Pfad eingegeben. Programm wird beendet.")
        else:
            predict_image(image_path)
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer abgebrochen.")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
    
    print("\n✅ Programm beendet.")
    

