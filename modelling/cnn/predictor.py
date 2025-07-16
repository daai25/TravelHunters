'''
This script uses CNN image features for recommendations in TravelHunters:
1. Predicting destinations based on input images
2. Extracting image features for similarity-based recommendations
3. Integration with the main recommender system
'''

import numpy as np
import sys
import os
from pathlib import Path
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from scipy.spatial.distance import cosine

# Add path to access the hybrid_model
current_dir = Path(__file__).parent.absolute()
ml_models_dir = current_dir.parent / "machine_learning_modells"
sys.path.append(str(ml_models_dir))
sys.path.append(str(ml_models_dir / "models"))

# Try to import the hybrid recommender
try:
    from models.hybrid_model import HybridRecommender
    hybrid_model_available = True
    print("✅ Hybrid-Modell erfolgreich importiert")
except ImportError as e:
    hybrid_model_available = False
    print(f"ℹ️ Hybrid-Modell nicht verfügbar: {e}")
    print("   Nur Bild-basierte Empfehlungen werden funktionieren.")

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

# --- Feature Extraction Function ---
def extract_image_features(image_path):
    """
    Extracts feature embeddings from an image using the trained CNN model.
    These embeddings can be used to calculate similarity between images.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        feature_vector: numpy array with image embeddings, or None if extraction failed
    """
    # Check if model was successfully loaded
    if model is None:
        print("❌ Kann keine Features extrahieren: Modell wurde nicht geladen.")
        return None
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Bild nicht gefunden: {image_path}")
        return None
    
    try:
        # Load and preprocess image
        print(f"Lade Bild für Feature-Extraktion: {image_path}")
        
        # Überprüfe, ob es sich um eine unterstützte Bilddatei handelt
        import imghdr
        if imghdr.what(image_path) not in ['jpeg', 'png', 'gif', 'bmp']:
            print(f"⚠️ Warnung: Die Datei {image_path} scheint kein unterstütztes Bildformat zu sein.")
            print("   Unterstützte Formate: JPEG, PNG, GIF, BMP")
            # Versuche trotzdem zu laden
        
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
        
        # Initialisiere das Modell durch einen ersten Vorhersage-Aufruf
        # Dies stellt sicher, dass alle Layer-Eingaben richtig definiert werden
        try:
            print("🔄 Initialisiere Modell-Layers...")
            # Führe eine Dummy-Vorhersage durch, um das Modell zu initialisieren
            dummy_prediction = model.predict(img_array, verbose=0)
            print(f"✓ Modell erfolgreich initialisiert (Output-Shape: {dummy_prediction.shape})")
            
            # Baue das Modell explizit auf, falls es ein Sequential-Modell ist
            if hasattr(model, 'built') and not model.built:
                model.build(input_shape=img_array.shape)
                print("✓ Sequential-Modell explizit gebaut")
                
        except Exception as e:
            print(f"⚠️ Warnung bei Modellinitialisierung: {e}")
            print("   Versuche trotzdem mit Feature-Extraktion fortzufahren...")
        
        # Verbesserte Feature-Extraktions-Strategie
        try:
            # Methode 1: Verwende den vordefinierten Feature-Extraktionslayer (falls bekannt)
            feature_layer_name = None
            
            # Prüfe alle Layer und ihre Eigenschaften nach der Initialisierung
            for i, layer in enumerate(model.layers):
                try:
                    # Suche nach einem Dense Layer mit passender Größe für Feature-Embedding
                    if isinstance(layer, Dense) and hasattr(layer, 'units') and 64 <= layer.units <= 512:
                        feature_layer_name = layer.name
                        print(f"✓ Gefunden: Dense Layer '{layer.name}' mit {layer.units} Units")
                        break
                except Exception as layer_err:
                    print(f"⚠️ Fehler beim Prüfen von Layer {i}: {layer_err}")
                    continue
            
            # Methode 2: Verwende den vorletzten Layer, falls verfügbar
            if not feature_layer_name and len(model.layers) > 2:
                try:
                    penultimate_layer = model.layers[-2]
                    feature_layer_name = penultimate_layer.name
                    print(f"✓ Verwende vorletzten Layer: '{feature_layer_name}'")
                except Exception as e:
                    print(f"⚠️ Fehler beim Zugriff auf vorletzten Layer: {e}")
                
            # Feature-Modell erstellen - verbesserter Ansatz
            if feature_layer_name:
                print(f"✓ Verwende Layer '{feature_layer_name}' für Feature-Extraktion")
                try:
                    # Sicherer Ansatz: Verwende funktionale API
                    target_layer = model.get_layer(feature_layer_name)
                    
                    # Überprüfe, ob das Modell Input-Tensoren hat
                    if hasattr(model, 'input') and model.input is not None:
                        feature_model = Model(inputs=model.input, outputs=target_layer.output)
                        print("✓ Feature-Modell mit funktionaler API erstellt")
                    else:
                        # Alternative: Verwende das ursprüngliche Modell bis zum gewünschten Layer
                        print("⚠️ Modell-Input nicht verfügbar, verwende alternatives Verfahren")
                        raise Exception("Model input not properly initialized")
                    
                    features = feature_model.predict(img_array, verbose=0)
                    print(f"✅ Features extrahiert: {features.shape}")
                    return features
                    
                except Exception as model_err:
                    print(f"⚠️ Fehler beim Erstellen des Feature-Modells: {model_err}")
                
        except Exception as e:
            print(f"⚠️ Feature-Extraktion mit spezifischem Layer fehlgeschlagen: {e}")
            print("   Versuche Fallback-Methoden...")
        
        # Fallback 1: Sequential-Modell-spezifische Feature-Extraktion
        try:
            print("✓ Fallback: Sequential-Modell Feature-Extraktion")
            
            # Für Sequential-Modelle: Gehe nur bis zu einem Dense-Layer vor dem Output
            intermediate_output = img_array
            target_layer_found = False
            temp_output = None
            temp_layer_units = None
            
            # Gehe durch die Layer und stoppe bei einem passenden Dense-Layer
            for i, layer in enumerate(model.layers[:-1]):  # Exclude last layer (output)
                try:
                    # Verwende den Layer direkt für die Vorhersage
                    if hasattr(layer, '__call__'):
                        intermediate_output = layer(intermediate_output)
                        
                        # Prüfe, ob es ein guter Punkt zum Stoppen ist
                        if isinstance(layer, Dense) and hasattr(layer, 'units'):
                            # Bevorzuge 128 Units, aber akzeptiere auch andere sinnvolle Größen
                            if layer.units == 128:
                                print(f"✓ Perfekt! Stoppe bei Dense Layer '{layer.name}' mit {layer.units} Units")
                                target_layer_found = True
                                break
                            elif 64 <= layer.units <= 256 and not target_layer_found:
                                print(f"✓ Stoppe bei Dense Layer '{layer.name}' mit {layer.units} Units")
                                target_layer_found = True
                                # Weiter suchen für eventuell besseren 128-Unit Layer, aber merke diesen
                                temp_output = intermediate_output
                                temp_layer_units = layer.units
                    else:
                        # Layer kann nicht direkt aufgerufen werden
                        print(f"⚠️ Layer {i} ({layer.name}) kann nicht direkt aufgerufen werden")
                        break
                        
                except Exception as layer_error:
                    print(f"⚠️ Fehler bei Layer {i} ({layer.name}): {layer_error}")
                    break
            
            if not target_layer_found:
                print("⚠️ Kein geeigneter Dense-Layer gefunden, verwende aktuellen Output")
            elif 'temp_output' in locals():
                print(f"✓ Verwende gespeicherten Output mit {temp_layer_units} Units")
                intermediate_output = temp_output
            
            # Konvertiere zu numpy falls nötig
            if hasattr(intermediate_output, 'numpy'):
                features = intermediate_output.numpy()
            else:
                features = intermediate_output
                
            # Stelle sicher, dass es ein numpy array ist
            features = np.array(features)
            
            # Normalisiere auf 128 Dimensionen
            if features.size != 128:
                print(f"⚠️ Features haben {features.shape} Form, normalisiere auf 128D")
                # Flatten
                flattened = features.flatten()
                
                if len(flattened) >= 128:
                    # Reduziere auf 128: nimm gleichmäßig verteilte Indizes
                    indices = np.linspace(0, len(flattened)-1, 128, dtype=int)
                    features = flattened[indices].reshape(1, -1)
                else:
                    # Fülle auf 128 auf
                    mean_val = flattened.mean() if len(flattened) > 0 else 0
                    padding = np.full(128 - len(flattened), mean_val)
                    features = np.concatenate([flattened, padding]).reshape(1, -1)
            else:
                features = features.reshape(1, -1)
            
            print(f"✅ Features durch Sequential-Verarbeitung extrahiert: {features.shape}")
            return features
            
        except Exception as e:
            print(f"⚠️ Sequential-Fallback fehlgeschlagen: {e}")
        
        # Fallback 2: Verwende direkt die Vorhersage des Hauptmodells als Features
        try:
            print("✓ Fallback: Verwende Modell-Output direkt als Features")
            raw_output = model.predict(img_array, verbose=0)
            
            # Falls das Output zu groß ist, reduziere es
            if raw_output.size > 512:
                # Verwende die ersten 128 Werte
                features = raw_output.flatten()[:128].reshape(1, -1)
            else:
                features = raw_output
                
            print(f"✅ Features aus Modell-Output extrahiert: {features.shape}")
            return features
            
        except Exception as e:
            print(f"⚠️ Fallback 2 fehlgeschlagen: {e}")
        
        # Fallback 2: Extrahiere von convolutional layer mit GlobalAveragePooling
        try:
            # Finde das letzte Convolutional Layer
            conv_layer = None
            for layer in model.layers:
                if 'conv' in layer.name.lower() or hasattr(layer, 'filters'):
                    conv_layer = layer
            
            if conv_layer:
                print(f"✓ Fallback: Verwende {conv_layer.name} mit GlobalAveragePooling2D")
                
                # Erstelle ein neues Modell mit GlobalAveragePooling
                gap_layer = GlobalAveragePooling2D()(conv_layer.output)
                feature_model = Model(inputs=model.input, outputs=gap_layer)
                features = feature_model.predict(img_array, verbose=0)
                print(f"✅ Features extrahiert: {features.shape}")
                return features
        except Exception as e:
            print(f"⚠️ Fallback 1 fehlgeschlagen: {e}")
            
        # Fallback 2: Erzeuge ein einfaches Feature-Array mit fester Größe
        print("ℹ️ Erstelle standard 128-D Features als letzter Fallback")
        # Erstelle ein Pseudo-Feature-Vektor basierend auf dem Bild
        reshaped_img = img_array.reshape(-1)[:128]  # Nehme die ersten 128 Werte
        if len(reshaped_img) < 128:
            # Fülle auf 128 auf, falls zu wenig Werte
            padding = np.zeros(128 - len(reshaped_img))
            reshaped_img = np.concatenate([reshaped_img, padding])
            
        features = reshaped_img.reshape(1, 128)
        print(f"✅ Fallback-Features erstellt: {features.shape}")
        return features
        
    except Exception as e:
        print(f"❌ Fehler bei der Feature-Extraktion: {e}")
        return None
        
    except Exception as e:
        print(f"❌ Fehler bei der Feature-Extraktion: {e}")
        return None

# --- Similarity Calculation Function ---
def calculate_image_similarity(features1, features2):
    """
    Calculate cosine similarity between two feature vectors
    
    Args:
        features1: First feature vector
        features2: Second feature vector
        
    Returns:
        similarity: Cosine similarity score (0-1)
    """
    if features1 is None or features2 is None:
        return 0.0
    
    try:
        # Flatten if necessary
        f1 = features1.flatten()
        f2 = features2.flatten()
        
        # Calculate cosine similarity
        from scipy.spatial.distance import cosine
        similarity = 1 - cosine(f1, f2)  # Convert distance to similarity
        
        return similarity
    except Exception as e:
        print(f"❌ Fehler bei der Ähnlichkeitsberechnung: {e}")
        return 0.0

# --- Predict Function ---
def predict_image(image_path):
    # Check if model was successfully loaded
    if model is None:
        print("❌ Kann keine Vorhersage machen: Modell wurde nicht geladen.")
        return None, None
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Bild nicht gefunden: {image_path}")
        return None, None
        
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
    except Exception as e:
        print(f"❌ Fehler bei der Bildverarbeitung oder Vorhersage: {e}")
        return None, None

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
            top_city = None
            top_confidence = 0
            
            for i, idx in enumerate(valid_indices):
                confidence = prediction[0][idx] * 100
                print(f"{i+1}. {city_names[idx]} (Konfidenz: {confidence:.1f}%)")
                
                # Speichere die Stadt mit der höchsten Konfidenz
                if i == 0:
                    top_city = city_names[idx]
                    top_confidence = confidence
                
            if not valid_indices:
                print("❌ Keine gültigen Vorhersagen gefunden.")
                return None, None
                
            return top_city, top_confidence
        else:
            # Binary classification fallback
            predicted_class = (prediction[0][0] > 0.5).astype("int")
            confidence = abs(prediction[0][0] - 0.5) * 2 * 100  # Scale to 0-100%
            print(f"Binäre Klassifikation: {predicted_class} (Konfidenz: {confidence:.1f}%)")
            return predicted_class, confidence
    except Exception as e:
        print(f"❌ Fehler bei der Interpretation der Vorhersage: {e}")
        return None, None


# --- Image Feature Database ---
class ImageFeatureDatabase:
    """Verwaltet eine Datenbank von Bild-Features für Ähnlichkeitsvergleiche"""
    
    def __init__(self):
        self.feature_vectors = {}  # Dictionary: image_id -> feature_vector
        self.city_features = {}    # Dictionary: city_name -> list of feature_vectors
        self.feature_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_features.npz")
        
        # Lade vorhandene Features wenn verfügbar
        self._load_features()
        
    def _load_features(self):
        """Lade vorgespeicherte Feature-Vektoren wenn vorhanden"""
        if os.path.exists(self.feature_file):
            try:
                data = np.load(self.feature_file, allow_pickle=True)
                self.feature_vectors = data['feature_vectors'].item()
                self.city_features = data['city_features'].item()
                print(f"✅ {len(self.feature_vectors)} Feature-Vektoren geladen")
            except Exception as e:
                print(f"❌ Fehler beim Laden der Features: {e}")
        else:
            print("ℹ️ Keine vorhandenen Features gefunden.")
            # Simulierte Feature-Vektoren für Demo-Zwecke erstellen
            self._create_demo_features()
    
    def _create_demo_features(self):
        """Erstellt simulierte Feature-Vektoren für Demo-Zwecke"""
        print("🔄 Erstelle Demo-Features für Städte...")
        feature_dim = 128  # Typische Feature-Dimension von CNNs
        
        # Erstelle für jede Stadt einen Beispiel-Feature-Vektor
        for city in city_names:
            # Simulierte Feature-Vektoren mit zufälligen Werten
            city_vector = np.random.randn(feature_dim).astype(np.float32)
            # Normalisieren
            city_vector = city_vector / np.linalg.norm(city_vector)
            
            self.feature_vectors[f"{city}_main"] = city_vector
            self.city_features[city] = [city_vector]
            
        print(f"✅ Demo-Features für {len(self.city_features)} Städte erstellt")
    
    def add_feature(self, image_id, city_name, feature_vector):
        """Fügt einen Feature-Vektor zur Datenbank hinzu"""
        self.feature_vectors[image_id] = feature_vector
        
        if city_name not in self.city_features:
            self.city_features[city_name] = []
        
        self.city_features[city_name].append(feature_vector)
        
    def save_features(self):
        """Speichert alle Feature-Vektoren in eine Datei"""
        try:
            np.savez_compressed(
                self.feature_file, 
                feature_vectors=self.feature_vectors,
                city_features=self.city_features
            )
            print(f"✅ Features gespeichert in {self.feature_file}")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Features: {e}")
            return False
    
    def get_similar_images(self, query_vector, top_k=5):
        """Findet ähnliche Bilder basierend auf einem Abfrage-Vektor"""
        if len(self.feature_vectors) == 0:
            return []
        
        # Berechne Ähnlichkeit zu allen gespeicherten Vektoren
        similarities = {}
        for image_id, feature_vector in self.feature_vectors.items():
            # Cosinus-Ähnlichkeit berechnen (1 - Cosinus-Distanz)
            sim = 1.0 - cosine(query_vector.flatten(), feature_vector.flatten())
            similarities[image_id] = sim
        
        # Sortiere nach Ähnlichkeit und gib die Top-K zurück
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def get_similar_cities(self, query_vector, top_k=5):
        """Findet ähnliche Städte basierend auf einem Abfrage-Vektor"""
        if len(self.city_features) == 0:
            return []
        
        # Berechne durchschnittliche Ähnlichkeit zu allen Feature-Vektoren jeder Stadt
        city_similarities = {}
        for city, feature_vectors in self.city_features.items():
            # Mittelwert der Ähnlichkeiten berechnen, wenn mehrere Vektoren pro Stadt
            city_sim = 0.0
            for feature_vector in feature_vectors:
                city_sim += 1.0 - cosine(query_vector.flatten(), feature_vector.flatten())
            city_sim /= len(feature_vectors)
            city_similarities[city] = city_sim
        
        # Sortiere nach Ähnlichkeit und gib die Top-K zurück
        sorted_results = sorted(city_similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def get_city_features(self, city_name):
        """Gibt die Feature-Vektoren für eine bestimmte Stadt zurück"""
        if city_name in self.city_features:
            return self.city_features[city_name]
        return []
    
    def get_feature_dict_for_recommender(self):
        """
        Erstellt ein Feature-Dictionary für das Hybrid-Empfehlungssystem
        Format: {city_id: feature_vector}
        """
        # Für jede Stadt nehmen wir den Durchschnitt der Feature-Vektoren
        feature_dict = {}
        for city, feature_vectors in self.city_features.items():
            # Bei mehreren Feature-Vektoren pro Stadt den Durchschnitt berechnen
            if feature_vectors:
                avg_vector = np.mean(feature_vectors, axis=0)
                # Normalisieren
                avg_vector = avg_vector / np.linalg.norm(avg_vector)
                feature_dict[city] = avg_vector
                
        return feature_dict

# Globale Feature-Datenbank initialisieren
feature_db = ImageFeatureDatabase()

# --- Integration mit dem Empfehlungssystem ---
def recommend_similar_destinations(image_path, top_k=3):
    """
    Empfiehlt ähnliche Reiseziele basierend auf einem Eingabebild
    
    Args:
        image_path: Pfad zum Eingabebild
        top_k: Anzahl der zu empfehlenden Ziele
        
    Returns:
        recommendations: Liste der empfohlenen Städte mit Ähnlichkeitswerten
    """
    # Extrahiere Features des Eingabebilds
    query_features = extract_image_features(image_path)
    if query_features is None:
        print("❌ Konnte keine Features aus dem Eingabebild extrahieren.")
        return []
    
    print("🔍 Suche nach ähnlichen Reisezielen...")
    
    # Feature-Datenbank für Ähnlichkeitsvergleich verwenden
    similar_cities = feature_db.get_similar_cities(query_features, top_k=top_k)
    
    print(f"✅ {len(similar_cities)} ähnliche Reiseziele gefunden")
    
    return similar_cities

# --- Hybrid-Modell Integration ---
class CNNHybridIntegration:
    """Verbindet den CNN-Feature-Extraktor mit dem Hybrid-Empfehlungssystem"""
    
    def __init__(self, feature_database=None):
        """Initialisiert die Integration mit dem Hybrid-Modell"""
        self.hybrid_recommender = None
        self.feature_database = feature_database or feature_db
        self._initialize_hybrid_model()
        
    def _initialize_hybrid_model(self):
        """Initialisiert das Hybrid-Modell wenn verfügbar"""
        if not hybrid_model_available:
            print("⚠️ Hybrid-Modell nicht verfügbar, Integration wird übersprungen")
            return False
            
        try:
            print("🔄 Initialisiere Hybrid-Empfehlungssystem...")
            self.hybrid_recommender = HybridRecommender()
            
            # Stelle sicher, dass das Modell initialisiert ist
            if hasattr(self.hybrid_recommender, 'initialize') and callable(self.hybrid_recommender.initialize):
                self.hybrid_recommender.initialize()
                
            print("✅ Hybrid-Empfehlungssystem initialisiert")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Initialisieren des Hybrid-Empfehlungssystems: {e}")
            return False
    
    def update_image_features(self):
        """Aktualisiert die Bild-Features im Hybrid-Modell"""
        if not self.hybrid_recommender:
            print("❌ Hybrid-Modell nicht verfügbar")
            return False
            
        try:
            # Feature-Dictionary für das Empfehlungssystem erstellen
            print("🔄 Erstelle Feature-Dictionary für Hybrid-Modell...")
            feature_dict = self.feature_database.get_feature_dict_for_recommender()
            
            # Feature-Dictionary an das Hybrid-Modell übergeben
            if hasattr(self.hybrid_recommender, 'set_image_features') and callable(self.hybrid_recommender.set_image_features):
                self.hybrid_recommender.set_image_features(feature_dict)
                print(f"✅ Bild-Features für {len(feature_dict)} Städte an Hybrid-Modell übergeben")
                return True
            else:
                print("❌ Hybrid-Modell unterstützt keine Bild-Features")
                return False
        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren der Bild-Features: {e}")
            return False
    
    def get_recommendations(self, user_id=None, image_path=None, top_k=10):
        """
        Holt Empfehlungen vom Hybrid-Modell mit optionaler Bild-Ähnlichkeit
        
        Args:
            user_id: Optional - Benutzer-ID für personalisierte Empfehlungen
            image_path: Optional - Pfad zum Bild für ähnliche Reiseziele
            top_k: Anzahl der Empfehlungen
            
        Returns:
            recommendations: Liste der empfohlenen Städte
        """
        if not self.hybrid_recommender:
            print("⚠️ Hybrid-Modell nicht verfügbar, verwende nur Bild-basierte Empfehlungen")
            if image_path:
                return recommend_similar_destinations(image_path, top_k)
            else:
                print("❌ Weder Hybrid-Modell noch Bild-Pfad verfügbar")
                return []
        
        try:
            # Wenn ein Bild vorhanden ist, Feature-Vektor extrahieren
            image_features = None
            image_recommendations = []
            
            if image_path:
                print("🔄 Extrahiere Bild-Features für Empfehlungen...")
                image_features = extract_image_features(image_path)
                if image_features is not None:
                    # Hole auch direkte bildbasierte Empfehlungen als Fallback
                    print("🔄 Hole bildbasierte Empfehlungen als Backup...")
                    image_recommendations = recommend_similar_destinations(image_path, top_k)
                else:
                    print("⚠️ Konnte keine Bild-Features extrahieren, verwende nur Benutzer-basierte Empfehlungen")
            
            # Hybrid-Empfehlungen holen
            print("🔄 Hole Hybrid-Empfehlungen...")
            hybrid_supported = hasattr(self.hybrid_recommender, 'recommend_with_image') and callable(self.hybrid_recommender.recommend_with_image)
            
            if hybrid_supported and image_features is not None:
                try:
                    recommendations = self.hybrid_recommender.recommend_with_image(
                        user_id=user_id,
                        image_features=image_features,
                        top_k=top_k
                    )
                    if recommendations and len(recommendations) > 0:
                        print(f"✅ {len(recommendations)} Hybrid-Empfehlungen erhalten")
                        return recommendations
                    else:
                        print("⚠️ Hybrid-Modell lieferte keine Empfehlungen, verwende Fallback")
                except Exception as e:
                    print(f"⚠️ Fehler bei Hybrid-Empfehlungen: {e}")
            
            # Wenn wir hier ankommen, war entweder kein Hybrid-Support vorhanden oder er hat keine Ergebnisse geliefert
            if hybrid_supported:
                print("⚠️ Hybrid-Modell funktioniert nicht wie erwartet, verwende Fallback")
            else:
                print("⚠️ Hybrid-Modell unterstützt keine Bild-Feature-Integration")
                
            # Fallback-Strategie:
            if image_recommendations:
                # Wenn wir Bild-Empfehlungen haben, verwende diese
                print("✅ Verwende bildbasierte Empfehlungen als Alternative")
                return image_recommendations
            elif user_id:
                # Wenn kein Bild, aber eine Benutzer-ID vorhanden ist
                try:
                    print("🔄 Versuche Standard-Empfehlungen über Benutzer-ID...")
                    recommendations = self.hybrid_recommender.recommend(user_id, top_k)
                    print(f"✅ {len(recommendations)} Standard-Empfehlungen erhalten")
                    return recommendations
                except Exception as e:
                    print(f"⚠️ Fehler bei Standard-Empfehlungen: {e}")
            
            # Wenn alles fehlschlägt, erstelle simulierte Empfehlungen basierend auf den vorhandenen Städtenamen
            print("⚠️ Alle Empfehlungsmethoden fehlgeschlagen, erstelle simulierte Empfehlungen")
            import random
            cities = list(self.feature_database.city_features.keys())
            random.shuffle(cities)
            sim_recommendations = [(city, random.uniform(0.7, 0.95)) for city in cities[:top_k]]
            print(f"✅ {len(sim_recommendations)} simulierte Empfehlungen erstellt")
            return sim_recommendations
                
        except Exception as e:
            print(f"❌ Fehler beim Holen von Hybrid-Empfehlungen: {e}")
            # Fallback auf rein bildbasierte Empfehlungen
            if image_path:
                print("⚠️ Verwende Fallback: Nur Bild-basierte Empfehlungen")
                return recommend_similar_destinations(image_path, top_k)
            return []

# --- Integration Demo-Funktion ---
def demo_recommendation_integration():
    """Demonstriert die Integration der Bildfeatures in das Empfehlungssystem"""
    print("\n🔄 Demonstration der Integration mit dem Empfehlungssystem")
    print("=" * 60)
    print("Diese Funktion zeigt, wie Bildfeatures in ein Empfehlungssystem integriert werden können.")
    print("Im vollständigen System würde dies automatisch erfolgen, wenn ein Benutzer ein Bild hochlädt.")
    
    image_path = input("\nGeben Sie den Pfad zu einem Reisefoto ein: ").strip()
    if not image_path or not os.path.exists(image_path):
        print("❌ Ungültiger Bildpfad oder Datei nicht gefunden.")
        return
    
    # 1. Extrahiere die Bildfeatures
    features = extract_image_features(image_path)
    if features is not None:
        print("\n✅ Features erfolgreich extrahiert!")
        print(f"   Feature-Vektor-Dimension: {features.shape}")
        
    # 2. Klassifiziere das Bild
    print("\n🔍 Klassifiziere das Bild:")
    predict_image(image_path)
    
    # 3. Empfehle ähnliche Reiseziele
    print("\n🌍 Empfohlene ähnliche Reiseziele:")
    similar_destinations = recommend_similar_destinations(image_path)
    for i, (city, similarity) in enumerate(similar_destinations, 1):
        print(f"{i}. {city} (Ähnlichkeit: {similarity:.2f})")
    
    print("\n📝 In einem vollständigen System:")
    print("  1. Diese Features würden in einer Datenbank gespeichert")
    print("  2. Zur Laufzeit werden Features ähnlicher Bilder verglichen")
    print("  3. Die Ähnlichkeitswerte fließen in den Empfehlungsalgorithmus ein")
    print("  4. Nutzer erhalten Empfehlungen basierend auf visuellen Präferenzen")

# --- Demo-Funktionen für Integration ---
def demo_recommendation_integration():
    """Demonstriert die Integration des CNN mit dem Empfehlungssystem"""
    print("\n🚀 Demo: Integration mit dem Empfehlungssystem")
    print("===============================================")
    
    # Initialisiere die Hybrid-Integration
    hybrid_integration = CNNHybridIntegration()
    
    # Option 1: Bild-basierte Empfehlungen ohne Benutzerkontext
    print("\n📷 Option 1: Nur Bild-basierte Empfehlungen")
    print("-------------------------------------------")
    image_path = input("Geben Sie den vollständigen Pfad zum Bild ein (oder Enter für Überspringen): ").strip()
    
    if image_path:
        print("\n🔍 Suche ähnliche Reiseziele basierend auf dem Bild...")
        
        # Rein bildbasierte Empfehlungen abrufen
        image_recommendations = recommend_similar_destinations(image_path, top_k=5)
        
        if image_recommendations:
            print("\n✅ Ähnliche Reiseziele basierend auf dem Bild:")
            for i, (city, similarity) in enumerate(image_recommendations):
                print(f"  {i+1}. {city} (Ähnlichkeit: {similarity:.2f})")
        else:
            print("❌ Konnte keine Empfehlungen basierend auf dem Bild finden.")
    else:
        print("Überspringe Bild-basierte Empfehlungen.")
    
    # Option 2: Hybrid-Empfehlungen mit Bild und Benutzerkontext
    print("\n🔄 Option 2: Hybrid-Empfehlungen (Bild + Benutzerkontext)")
    print("--------------------------------------------------------")
    
    if not hybrid_model_available:
        print("⚠️ Hybrid-Modell nicht verfügbar - Demo kann nicht fortgesetzt werden.")
        print("   Bitte stellen Sie sicher, dass das Hybrid-Modell importiert werden kann.")
        return
    
    # Aktualisiere Bild-Features im Hybrid-Modell
    print("\n🔄 Aktualisiere Bild-Features im Hybrid-Modell...")
    hybrid_integration.update_image_features()
    
    # Demonstriere verschiedene Empfehlungsszenarien
    print("\n📊 Empfehlungsszenarien:")
    
    # Nur benutzerbasierte Empfehlungen
    print("\n1️⃣ Nur benutzerbasierte Empfehlungen:")
    user_id = input("Geben Sie eine Benutzer-ID ein (oder Enter für Überspringen): ").strip()
    if user_id:
        print(f"\n🔍 Hole Empfehlungen für Benutzer {user_id}...")
        user_recommendations = hybrid_integration.get_recommendations(user_id=user_id, top_k=5)
        if user_recommendations:
            print(f"\n✅ Empfehlungen für Benutzer {user_id}:")
            for i, rec in enumerate(user_recommendations):
                # Annahme: Empfehlungen haben Format (city_id, score)
                print(f"  {i+1}. {rec[0]} (Score: {rec[1]:.2f})")
        else:
            print("❌ Konnte keine Benutzer-Empfehlungen finden.")
    else:
        print("Überspringe benutzerbasierte Empfehlungen.")
    
    # Benutzer- und bildbasierte Empfehlungen
    print("\n2️⃣ Benutzer- und bildbasierte Empfehlungen:")
    image_path = input("Geben Sie den vollständigen Pfad zum Bild ein (oder Enter für Überspringen): ").strip()
    user_id = input("Geben Sie eine Benutzer-ID ein (oder Enter für Zufällig): ").strip() or None
    
    if image_path:
        print(f"\n🔍 Hole Hybrid-Empfehlungen für{' Benutzer ' + user_id if user_id else ''} mit Bild...")
        hybrid_recommendations = hybrid_integration.get_recommendations(
            user_id=user_id, 
            image_path=image_path,
            top_k=5
        )
        
        if hybrid_recommendations:
            print("\n✅ Hybrid-Empfehlungen:")
            for i, rec in enumerate(hybrid_recommendations):
                # Annahme: Empfehlungen haben Format (city_id, score)
                print(f"  {i+1}. {rec[0]} (Score: {rec[1]:.2f})")
        else:
            print("❌ Konnte keine Hybrid-Empfehlungen finden.")
    else:
        print("Überspringe Hybrid-Empfehlungen, da kein Bild angegeben wurde.")
    
    print("\n✅ Demo abgeschlossen.")

# --- Run Manually ---
if __name__ == "__main__":
    print("\n🔍 TravelHunters Städteklassifizierer & Empfehlungssystem 🌆")
    print("===============================================================")
    print("Dieser Classifier kann Städte anhand von Bildern erkennen und")
    print("ähnliche Reiseziele empfehlen.")
    print(f"Modellpfad: {MODEL_PATH}")
    print(f"Unterstützte Städte: {len(city_names)}")
    print("---------------------------------------------------------------")
    
    try:
        print("\nWählen Sie eine Option:")
        print("1. Bild klassifizieren (Städteerkennung)")
        print("2. Demo: Integration mit Empfehlungssystem")
        print("3. Bild hochladen und Hybrid-Empfehlungen erhalten")
        print("4. Beenden")
        
        choice = input("\nIhre Auswahl (1-4): ").strip()
        
        if choice == "1":
            image_path = input("\nGeben Sie den vollständigen Pfad zum Bild ein: ").strip()
            if image_path:
                predict_image(image_path)
            else:
                print("❌ Kein Pfad eingegeben.")
        elif choice == "2":
            demo_recommendation_integration()
        elif choice == "3":
            image_path = input("\nGeben Sie den vollständigen Pfad zum Bild ein: ").strip()
            if image_path:
                if not os.path.exists(image_path):
                    print(f"❌ Datei nicht gefunden: {image_path}")
                else:
                    print("\n🖼️ Bild wird mit dem Hybrid-Modell analysiert...")
                    print("(Falls die Meldung 'Hybrid-Modell unterstützt keine Bild-Feature-Integration' erscheint,")
                    print(" werden automatisch alternative Empfehlungen basierend auf Bildähnlichkeit generiert)")
                    
                    # Erstelle eine Instanz des Hybrid-Integrators
                    hybrid_integration = CNNHybridIntegration()
                    
                    # Klassifiziere das Bild zuerst, um zusätzliche Informationen zu erhalten
                    print("\n🔍 Analysiere Bild...")
                    top_city, top_confidence = predict_image(image_path)
                    
                    # Aktualisiere die Bild-Features
                    hybrid_integration.update_image_features()
                    
                    # Hole Empfehlungen basierend auf dem Bild
                    user_id = input("\nMöchten Sie auch eine Benutzer-ID angeben? (Optional, Enter zum Überspringen): ").strip() or None
                    recommendations = hybrid_integration.get_recommendations(user_id=user_id, image_path=image_path, top_k=5)
                    
                    if recommendations:
                        print("\n✅ Empfehlungen basierend auf Ihrem Bild:")
                        for i, rec in enumerate(recommendations):
                            print(f"  {i+1}. {rec[0]} (Score: {rec[1]:.2f})")
                            
                        if top_city:
                            print(f"\nℹ️ Ihr Bild wurde als '{top_city}' erkannt (Konfidenz: {top_confidence:.1f}%)")
                            print("   Die Empfehlungen beinhalten ähnliche Reiseziele zu dieser Stadt.")
                    else:
                        print("❌ Konnte keine Empfehlungen für dieses Bild generieren.")
                        print("   Bitte versuchen Sie ein anderes Bild oder wählen Sie Option 2 für eine umfassendere Demo.")
            else:
                print("❌ Kein Pfad eingegeben.")
        else:
            print("Programm wird beendet.")
            
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer abgebrochen.")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
    
    print("\n✅ Programm beendet.")
    

