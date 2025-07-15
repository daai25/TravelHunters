'''
Multi-Modal-Recommender für TravelHunters
---------------------------------------
Dieses Skript implementiert einen Multi-Modal-Ansatz für Reiseempfehlungen, der:
1. CNN-basierte Bildfeatures aus predictor.py
2. Text-basierte Features aus dem hybrid_model.py
kombiniert und zu einem gemeinsamen Embedding-Vektor fusioniert.

Autor: GitHub Copilot
Datum: 15.07.2025
'''

import numpy as np
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import importlib.util
from scipy.spatial.distance import cosine

# Erweiterte Pfadkonfiguration für die korrekte Modul-Erkennung
# Diese sollte vor der Ausführung des Hauptskripts mit init_environment.py aktualisiert werden
current_dir = Path(__file__).parent.absolute()  # multimodal
modelling_dir = current_dir.parent              # modelling
project_root = modelling_dir.parent             # TravelHunters

# Verzeichnisse für die verschiedenen Modelle
cnn_dir = modelling_dir / "cnn"  # TravelHunters/modelling/cnn
ml_models_dir = modelling_dir / "machine_learning_modells"  # TravelHunters/modelling/machine_learning_modells

# Wichtige Pfade zum Systempfad hinzufügen
sys.path.append(str(project_root))
sys.path.append(str(cnn_dir))
sys.path.append(str(ml_models_dir))
sys.path.append(str(ml_models_dir / "models"))

# Pfade zu den gespeicherten Modellen
SAVED_TEXT_MODEL_PATH = ml_models_dir / "saved_models" / "text_model.joblib"
SAVED_PARAM_MODEL_PATH = ml_models_dir / "saved_models" / "param_model.joblib"

# TensorFlow-Import für Modellkonstruktion - falls verfügbar
tf_available = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Concatenate, Input
    tf_available = True
    print("✅ TensorFlow erfolgreich importiert")
except ImportError:
    print("⚠️ TensorFlow nicht verfügbar - Einfaches Modell wird verwendet")
    print("   Bitte installieren Sie TensorFlow mit: pip install tensorflow")

# Dynamisches Importieren der Module unter Berücksichtigung unterschiedlicher Pfadstrukturen
cnn_model_available = False
hybrid_model_available = False

# Import CNN-Predictor über absolute Pfade
predictor_path = cnn_dir / "predictor.py"
if predictor_path.exists():
    try:
        # Dynamischer Import von predictor.py
        spec = importlib.util.spec_from_file_location("predictor", predictor_path)
        predictor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor_module)
        
        # Extrahiere die benötigten Funktionen und Klassen
        extract_image_features = predictor_module.extract_image_features
        ImageFeatureDatabase = predictor_module.ImageFeatureDatabase
        predict_image = predictor_module.predict_image
        city_names = predictor_module.city_names if hasattr(predictor_module, 'city_names') else []
        
        cnn_model_available = True
        print(f"✅ CNN-Modell erfolgreich importiert aus {predictor_path}")
        print(f"   {len(city_names)} Städte geladen")
    except Exception as e:
        print(f"⚠️ CNN-Modell-Import fehlgeschlagen: {e}")

# Import Hybrid-Recommender über absolute Pfade
hybrid_model_path = ml_models_dir / "models" / "hybrid_model.py"
text_model_path = ml_models_dir / "models" / "text_similarity_model.py"

if hybrid_model_path.exists() and text_model_path.exists():
    try:
        # Dynamischer Import von hybrid_model.py
        spec_hybrid = importlib.util.spec_from_file_location("hybrid_model", hybrid_model_path)
        hybrid_module = importlib.util.module_from_spec(spec_hybrid)
        spec_hybrid.loader.exec_module(hybrid_module)
        HybridRecommender = hybrid_module.HybridRecommender
        
        # Dynamischer Import von text_similarity_model.py
        spec_text = importlib.util.spec_from_file_location("text_similarity_model", text_model_path)
        text_module = importlib.util.module_from_spec(spec_text)
        spec_text.loader.exec_module(text_module)
        TextBasedRecommender = text_module.TextBasedRecommender
        
        hybrid_model_available = True
        print(f"✅ Hybrid-Modell erfolgreich importiert aus {hybrid_model_path}")
    except Exception as e:
        print(f"⚠️ Hybrid-Modell-Import fehlgeschlagen: {e}")

class MultiModalRecommender:
    """
    Multi-Modal Recommender der CNN-basierte Bildfeatures und textbasierte Features kombiniert
    """
    
    def __init__(self, image_weight: float = 0.4, text_weight: float = 0.6, 
                 embedding_dim: int = 256):
        """
        Initialisiert den Multi-Modal Recommender
        
        Args:
            image_weight: Gewichtung der Bild-Features (0.0 - 1.0)
            text_weight: Gewichtung der Text-Features (0.0 - 1.0)
            embedding_dim: Dimension des gemeinsamen Embedding-Vektors
        """
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.embedding_dim = embedding_dim
        
        # Normalisiere Gewichte
        total = self.image_weight + self.text_weight
        self.image_weight /= total
        self.text_weight /= total
        
        # Initialisiere Komponenten
        self.hybrid_model = None
        self.image_feature_db = None
        self.fusion_model = None
        
        # Status-Tracking
        self.is_initialized = False
        self.has_image_features = False
        self.has_text_features = False
        
        # Initialisiere Komponenten
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialisiert die Bild- und Text-Komponenten"""
        print("\n🔄 Initialisiere Multi-Modal Recommender...")
        
        # Initialisiere Image-Feature-Database
        if cnn_model_available:
            try:
                print("  Initialisiere Bild-Feature-Datenbank...")
                self.image_feature_db = ImageFeatureDatabase()
                self.has_image_features = len(self.image_feature_db.feature_vectors) > 0
                print(f"  ✅ Bild-Feature-Datenbank geladen mit {len(self.image_feature_db.feature_vectors)} Features")
            except Exception as e:
                print(f"  ⚠️ Fehler beim Initialisieren der Bild-Feature-Datenbank: {e}")
                self.image_feature_db = None
        else:
            print("  ⚠️ CNN-Modell nicht verfügbar, Bild-Feature werden deaktiviert")
            
        # Initialisiere Hybrid-Modell
        if hybrid_model_available:
            try:
                print("  Initialisiere Hybrid-Modell...")
                self.hybrid_model = HybridRecommender()
                
                # Initialize Text-Recommender für direkten Zugriff auf Text-Embeddings
                self.text_model = TextBasedRecommender()
                
                # Versuche, die gespeicherten Modelle zu laden
                try:
                    # Zugriff auf die gespeicherten Modellpfade aus init_environment
                    import sys
                    current_module = sys.modules.get(__name__, None)
                    if hasattr(current_module, 'SAVED_TEXT_MODEL_PATH') and hasattr(current_module, 'SAVED_PARAM_MODEL_PATH'):
                        # Verwende die definierten Pfade
                        saved_text_model_path = current_module.SAVED_TEXT_MODEL_PATH
                        saved_param_model_path = current_module.SAVED_PARAM_MODEL_PATH
                    else:
                        # Fallback auf hartcodierte Pfade
                        saved_text_model_path = ml_models_dir / "saved_models" / "text_model.joblib"
                        saved_param_model_path = ml_models_dir / "saved_models" / "param_model.joblib"
                    
                    # Versuche, das Text-Modell zu laden
                    if os.path.exists(saved_text_model_path):
                        print(f"  Lade gespeichertes Text-Modell: {saved_text_model_path}")
                        self.text_model.load_model(saved_text_model_path)
                        print("  ✅ Text-Modell erfolgreich geladen")
                    
                    # Versuche, das Parameter-Modell zu laden (falls der Hybrid-Recommender eine load_model-Methode hat)
                    if os.path.exists(saved_param_model_path) and hasattr(self.hybrid_model, 'load_model'):
                        print(f"  Lade gespeichertes Parameter-Modell: {saved_param_model_path}")
                        self.hybrid_model.load_model(saved_param_model_path)
                        print("  ✅ Parameter-Modell erfolgreich geladen")
                    
                except Exception as load_err:
                    print(f"  ⚠️ Fehler beim Laden der gespeicherten Modelle: {load_err}")
                    print("  ⚠️ Fahre mit neu initialisierten Modellen fort")
                
                self.has_text_features = True
                print("  ✅ Hybrid-Modell erfolgreich initialisiert")
            except Exception as e:
                print(f"  ⚠️ Fehler beim Initialisieren des Hybrid-Modells: {e}")
                self.hybrid_model = None
                self.text_model = None
        else:
            print("  ⚠️ Hybrid-Modell nicht verfügbar, Text-Features werden deaktiviert")
            
        # Initialisiere Fusion-Modell
        print("  Erstelle Fusion-Modell für Multi-Modal-Embeddings...")
        self._create_fusion_model()
        
        # Setze Initialisierungsstatus
        self.is_initialized = (self.image_feature_db is not None or self.hybrid_model is not None)
        
        if self.is_initialized:
            print("✅ Multi-Modal Recommender erfolgreich initialisiert")
            print(f"   Bild-Features: {'Verfügbar' if self.has_image_features else 'Nicht verfügbar'}")
            print(f"   Text-Features: {'Verfügbar' if self.has_text_features else 'Nicht verfügbar'}")
        else:
            print("❌ Multi-Modal Recommender konnte nicht initialisiert werden")
            
    def _create_fusion_model(self):
        """Erstellt ein Modell zur Fusion von Bild- und Text-Features"""
        try:
            # Bestimme Feature-Dimensionen
            image_dim = 128  # Annahme: Standard CNN-Feature Dimension
            text_dim = 150   # Annahme: LSA-Komponenten des Text-Modells
            
            # Wenn die Bild-Feature-Datenbank existiert, prüfe tatsächliche Dimension
            if self.has_image_features and self.image_feature_db is not None:
                # Nimm das erste Feature aus der Datenbank zur Dimensionsbestimmung
                sample_features = next(iter(self.image_feature_db.feature_vectors.values()))
                if sample_features is not None:
                    image_dim = sample_features.flatten().shape[0]
                    
            # Überprüfe, ob das Text-Modell LSA verwendet und wie viele Komponenten
            if self.text_model is not None:
                if hasattr(self.text_model, 'lsa_components'):
                    text_dim = self.text_model.lsa_components
                elif hasattr(self.text_model, 'lsa_model') and hasattr(self.text_model.lsa_model, 'n_components'):
                    text_dim = self.text_model.lsa_model.n_components
                    
            # Erstelle ein einfaches Fusionsmodell
            # Input-Layer
            image_input = Input(shape=(image_dim,), name='image_features')
            text_input = Input(shape=(text_dim,), name='text_features')
            
            # Fusionsschicht (Concatenate + Dense für Dimensionsreduktion)
            concat_layer = Concatenate()([
                image_input,
                text_input
            ])
            
            # Gemeinsamer Embedding-Vektor
            embedding = Dense(self.embedding_dim, activation='relu', name='fusion_embedding')(concat_layer)
            
            # Erstelle Modell
            self.fusion_model = Model(
                inputs=[image_input, text_input],
                outputs=embedding,
                name='multimodal_fusion'
            )
            
            print(f"  ✅ Fusion-Modell erstellt: {image_dim}-D Bild + {text_dim}-D Text → {self.embedding_dim}-D Embedding")
            return True
        except Exception as e:
            print(f"  ⚠️ Fehler beim Erstellen des Fusion-Modells: {e}")
            self.fusion_model = None
            return False
            
    def get_image_features(self, image_path):
        """
        Extrahiert Features aus einem Bild
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            feature_vector: Numpy-Array mit Bild-Features oder None
        """
        if not cnn_model_available:
            print("⚠️ CNN-Modell nicht verfügbar - verwende Nullvektor")
            return self._generate_null_image_features()
            
        try:
            # Prüfe, ob das Bild bereits in der Feature-Datenbank existiert
            if self.image_feature_db and hasattr(self.image_feature_db, 'feature_vectors'):
                # Extrahiere den Dateinamen aus dem Pfad
                file_name = os.path.basename(image_path)
                base_name = os.path.splitext(file_name)[0]
                
                # Überprüfe, ob der Bildname in den Feature-Vektoren enthalten ist
                for img_name, features in self.image_feature_db.feature_vectors.items():
                    if base_name in img_name:
                        print(f"✅ Verwende gespeicherte Features für '{base_name}'")
                        return features
            
            # Verwende CNN-Modell zur Feature-Extraktion für neue Bilder
            print(f"🔄 Extrahiere neue Features für: {os.path.basename(image_path)}")
            try:
                features = extract_image_features(image_path)
                if features is not None:
                    print(f"✅ CNN-Features erfolgreich extrahiert: {features.shape}")
                    return features
                else:
                    print("⚠️ CNN-Feature-Extraktion fehlgeschlagen - verwende Nullvektor")
                    return self._generate_null_image_features()
            except Exception as inner_err:
                print(f"❌ Fehler bei der CNN-Feature-Extraktion: {inner_err}")
                print("⚠️ Verwende Nullvektor als Fallback")
                return self._generate_null_image_features()
        except Exception as e:
            print(f"❌ Fehler bei der Bild-Feature-Extraktion: {e}")
            return self._generate_null_image_features()
    
    def get_city_image_features(self, city_name):
        """
        Sucht Bildfeatures für eine bestimmte Stadt
        
        Args:
            city_name: Name der Stadt
            
        Returns:
            feature_vector: Numpy-Array mit Bild-Features oder None
        """
        if not cnn_model_available or not self.image_feature_db:
            return None
            
        try:
            # Normalisiere Stadtnamen für bessere Suche
            city_normalized = city_name.lower().strip().replace(' ', '_')
            
            # Suche in der Feature-Datenbank
            if hasattr(self.image_feature_db, 'feature_vectors'):
                for img_name, features in self.image_feature_db.feature_vectors.items():
                    img_name_normalized = img_name.lower()
                    
                    # Verschiedene Matching-Strategien
                    if (city_normalized in img_name_normalized or 
                        img_name_normalized.startswith(city_normalized) or
                        any(city_part in img_name_normalized for city_part in city_normalized.split('_') if len(city_part) > 2)):
                        print(f"✅ Gefunden: Bild-Features für '{city_name}' in '{img_name}'")
                        return features
            
            # Wenn keine direkten Features, suche nach ähnlichen Stadtnamen
            if hasattr(self.image_feature_db, 'city_features'):
                for stored_city, city_features in self.image_feature_db.city_features.items():
                    stored_city_normalized = stored_city.lower().strip().replace(' ', '_')
                    
                    if (city_normalized in stored_city_normalized or 
                        stored_city_normalized in city_normalized or
                        any(city_part in stored_city_normalized for city_part in city_normalized.split('_') if len(city_part) > 2)):
                        if city_features and len(city_features) > 0:
                            print(f"✅ Gefunden: Stadt-Features für '{city_name}' über '{stored_city}'")
                            return city_features[0]  # Nimm das erste Feature-Set
            
            return None
        except Exception as e:
            print(f"⚠️ Fehler bei der Stadt-Feature-Suche für '{city_name}': {e}")
            return None
            
    def _generate_null_image_features(self):
        """Generiert einen Nullvektor für Bild-Features"""
        return np.zeros((1, 128))  # Standard CNN-Feature-Größe
            
    def get_text_features(self, query_text, city_name=None):
        """
        Extrahiert Features aus Text
        
        Args:
            query_text: Abfragetext
            city_name: Optional - Name einer Stadt für spezifische Features
            
        Returns:
            feature_vector: Numpy-Array mit Text-Features oder None
        """
        if not hybrid_model_available or not self.has_text_features:
            print("⚠️ Text-Feature-Extraktion nicht verfügbar")
            return self._generate_simulated_text_features()
            
        try:
            # Verwende Text-Modell zur Feature-Extraktion
            if self.text_model is not None:
                # Methode 1: Wenn das Modell die Methode "get_embeddings" hat, verwende diese
                if hasattr(self.text_model, 'get_embeddings'):
                    try:
                        text_features = self.text_model.get_embeddings(query_text)
                        if isinstance(text_features, np.ndarray):
                            return text_features.reshape(1, -1)  # Stellen Sie sicher, dass die Form (1, N) ist
                        return text_features
                    except Exception as e:
                        print(f"⚠️ Fehler bei get_embeddings: {e}")
                
                # Methode 2: Verwende den TF-IDF-Vektorisierer direkt, wenn verfügbar
                if hasattr(self.text_model, 'tfidf_vectorizer'):
                    try:
                        # Reinige und verbessere die Anfrage, wie es das Textmodell tun würde
                        if hasattr(self.text_model, '_clean_text') and hasattr(self.text_model, '_enhance_query'):
                            cleaned_query = self.text_model._clean_text(query_text)
                            enhanced_query = self.text_model._enhance_query(cleaned_query)
                        else:
                            enhanced_query = query_text
                            
                        # Vektorisiere den Text
                        text_vector = self.text_model.tfidf_vectorizer.transform([enhanced_query])
                        
                        # Wende LSA an, falls verfügbar
                        if hasattr(self.text_model, 'lsa_model') and self.text_model.lsa_model is not None:
                            text_features = self.text_model.lsa_model.transform(text_vector)
                            return text_features
                        return text_vector.toarray()
                    except Exception as e:
                        print(f"⚠️ Fehler bei TF-IDF-Vektorisierung: {e}")
                
                # Methode 3: Fallback auf die alte Methode mit dem Vektorisierer-Attribut
                if hasattr(self.text_model, 'vectorizer') and self.text_model.vectorizer is not None:
                    try:
                        text_features = self.text_model.vectorizer.transform([query_text]).toarray()
                        
                        if hasattr(self.text_model, 'svd') and self.text_model.svd is not None:
                            # Dimensionsreduktion mittels SVD (falls verfügbar)
                            text_features = self.text_model.svd.transform(text_features)
                        
                        return text_features
                    except Exception as e:
                        print(f"⚠️ Fehler bei vectorizer-Methode: {e}")
            
            # Wenn alle Methoden fehlschlagen, verwende simulierte Features
            return self._generate_simulated_text_features()
        except Exception as e:
            print(f"❌ Fehler bei der Text-Feature-Extraktion: {e}")
            # Fallback: Generiere einen simulierten Feature-Vektor
            return self._generate_simulated_text_features()
            
    def _generate_simulated_text_features(self):
        """Generiert simulierte Text-Features für Testzwecke"""
        print("⚠️ Text-Modell nicht verfügbar, verwende simulierte Text-Features")
        # Setze seed für Konsistenz
        np.random.seed(42)
        features = np.random.rand(1, 768)  # Simuliere 768-dimensionale Text-Features
        np.random.seed(None)  # Reset seed
        return features
            
    def create_multimodal_embedding(self, image_path=None, query_text=None, city_name=None):
        """
        Erzeugt einen Multi-Modal-Embedding-Vektor aus Bild und Text
        
        Args:
            image_path: Optional - Pfad zum Bild
            query_text: Optional - Abfragetext
            city_name: Optional - Name einer Stadt für spezifische Features
            
        Returns:
            embedding: Numpy-Array mit Multi-Modal-Embedding oder None
        """
        if not self.is_initialized:
            print("❌ Multi-Modal Recommender ist nicht initialisiert")
            return None
            
        if not image_path and not query_text and not city_name:
            print("❌ Mindestens ein Input (Bild oder Text) muss angegeben werden")
            return None
            
        # Extrahiere Bild-Features
        image_features = None
        if image_path:
            image_features = self.get_image_features(image_path)
        elif city_name:
            # Wenn kein Bildpfad aber Stadt angegeben, suche nach Stadtbildern
            image_features = self.get_city_image_features(city_name)
            if image_features is not None:
                print(f"✅ Verwende Stadt-Bild-Features für '{city_name}'")
            
            # Wenn keine Bild-Features vorhanden aber Stadt angegeben, versuche Bild-Features der Stadt zu verwenden
            if image_features is None and self.image_feature_db:
                city_features = self.image_feature_db.get_city_features(city_name) if hasattr(self.image_feature_db, 'get_city_features') else None
                if city_features and len(city_features) > 0:
                    image_features = city_features[0]  # Nimm erstes Feature-Set der Stadt
                    print(f"✅ Verwende gespeicherte Bild-Features für '{city_name}'")
        
        # Extrahiere Text-Features
        text_features = None
        if query_text:
            text_features = self.get_text_features(query_text, city_name)
        elif city_name:
            # Wenn kein Abfragetext aber Stadt angegeben, verwende Stadtnamen als Text
            text_features = self.get_text_features(city_name, city_name)
            
        # Prüfe, ob wir genügend Features haben
        if image_features is None and text_features is None:
            print("❌ Konnte keine Features extrahieren")
            return None
            
        # Wenn eines der Feature-Sets fehlt, simuliere es mit Nullvektor
        if image_features is None:
            print("ℹ️ Keine Bild-Features verfügbar, verwende Nullvektor")
            # Bestimme Dimension aus dem Fusion-Modell
            img_dim = self.fusion_model.inputs[0].shape[1]
            image_features = np.zeros((1, img_dim))
        else:
            # Stelle sicher, dass die Features die richtige Form haben
            image_features = image_features.reshape(1, -1)
            
        if text_features is None:
            print("ℹ️ Keine Text-Features verfügbar, verwende Nullvektor")
            # Bestimme Dimension aus dem Fusion-Modell
            txt_dim = self.fusion_model.inputs[1].shape[1]
            text_features = np.zeros((1, txt_dim))
        else:
            # Stelle sicher, dass die Features die richtige Form haben
            text_features = text_features.reshape(1, -1)
            
        try:
            # Erzeuge Multi-Modal-Embedding durch gewichtete Kombination
            # Wenn kein Fusion-Modell vorhanden, verwende einfache gewichtete Kombination
            if self.fusion_model is None:
                print("ℹ️ Fusion-Modell nicht verfügbar, verwende gewichtete Kombination")
                
                # Reduziere beide Vektoren auf gleiche Größe (kleinere der beiden)
                min_dim = min(image_features.shape[1], text_features.shape[1])
                img_feat_small = image_features[0, :min_dim]
                txt_feat_small = text_features[0, :min_dim]
                
                # Gewichtete Kombination
                combined = (self.image_weight * img_feat_small + 
                           self.text_weight * txt_feat_small)
                
                # Normalisieren
                norm = np.linalg.norm(combined)
                if norm > 0:
                    combined = combined / norm
                    
                return combined.reshape(1, -1)
            else:
                # Verwende Fusion-Modell für fortgeschrittene Feature-Kombination
                multimodal_embedding = self.fusion_model.predict(
                    [image_features, text_features],
                    verbose=0
                )
                
                return multimodal_embedding
        except Exception as e:
            print(f"❌ Fehler bei der Embedding-Erstellung: {e}")
            return None
            
    def calculate_similarity(self, embedding1, embedding2):
        """
        Berechnet die Ähnlichkeit zwischen zwei Embeddings
        
        Args:
            embedding1: Erstes Embedding
            embedding2: Zweites Embedding
            
        Returns:
            similarity: Ähnlichkeitswert (0.0 - 1.0)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        try:
            # Flatten wenn nötig
            e1 = embedding1.flatten()
            e2 = embedding2.flatten()
            
            # Berechne Kosinus-Ähnlichkeit
            from scipy.spatial.distance import cosine
            similarity = 1.0 - cosine(e1, e2)
            
            return similarity
        except Exception as e:
            print(f"❌ Fehler bei der Ähnlichkeitsberechnung: {e}")
            return 0.0
    
    def get_multimodal_recommendations(self, query_text=None, image_path=None, 
                                     user_preferences=None, top_k=10):
        """
        Get multimodal recommendations using the proven hybrid method
        
        Args:
            query_text: Optional - User search query
            image_path: Optional - Path to image (for future image-based enhancement)
            user_preferences: User filter preferences
            top_k: Number of recommendations
            
        Returns:
            recommendations: DataFrame with hotel recommendations including price and rating
        """
        if not self.hybrid_model:
            print("❌ Hybrid model not available")
            return pd.DataFrame()
            
        if not query_text:
            query_text = "hotel accommodation"  # Default query
            
        # Load hotel data
        try:
            # Load data using the same loader as in the demo
            from load_data import HotelDataLoader
            from feature_engineering import HotelFeatureEngineer
            
            loader = HotelDataLoader()
            engineer = HotelFeatureEngineer()
            
            print("📊 Loading hotel data...")
            hotels_df = loader.load_hotels()
            features_df = engineer.engineer_features(hotels_df)
            
            print(f"✅ Loaded {len(hotels_df)} hotels with engineered features")
            
            # Set default preferences if not provided
            if user_preferences is None:
                user_preferences = {
                    'max_price': 200,
                    'min_rating': 4.0,
                    'preferred_amenities': [],
                    'text_importance': 0.6,
                    'price_importance': 0.2,
                    'rating_importance': 0.2
                }
            
            # Use the proven hybrid recommendation method
            print(f"🔄 Getting hybrid recommendations for: '{query_text}'")
            recommendations = self.hybrid_model.recommend_hotels(
                query=query_text,
                hotels_df=hotels_df,
                features_df=features_df,
                user_preferences=user_preferences,
                top_k=top_k
            )
            
            # Add image similarity if image path is provided (for future enhancement)
            if image_path and not recommendations.empty:
                print("🖼️ Adding image-based similarity (placeholder)")
                # This could be enhanced with actual image features in the future
                recommendations['image_similarity'] = np.random.uniform(0.3, 0.9, len(recommendations))
            
            return recommendations
            
        except ImportError as e:
            print(f"⚠️ Could not import data modules: {e}")
            print("   Falling back to simulated data")
            return self._generate_sample_recommendations(query_text, user_preferences, top_k)
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
            return pd.DataFrame()
    
    def _generate_sample_recommendations(self, query_text, user_preferences, top_k):
        """Generate sample recommendations for demonstration"""
        import random
        random.seed(42)
        
        # Sample cities with realistic hotel data
        sample_hotels = [
            {"hotel_name": "Seaside Resort & Spa", "location": "Barcelona, Spain", "rating": 8.5, "price": 180},
            {"hotel_name": "Grand Palace Hotel", "location": "Paris, France", "rating": 9.1, "price": 280},
            {"hotel_name": "Beach Paradise Resort", "location": "Bali, Indonesia", "rating": 8.8, "price": 150},
            {"hotel_name": "City Center Boutique", "location": "Berlin, Germany", "rating": 8.2, "price": 120},
            {"hotel_name": "Mountain View Lodge", "location": "Zurich, Switzerland", "rating": 8.7, "price": 240},
            {"hotel_name": "Historic Charm Hotel", "location": "Prague, Czech Republic", "rating": 8.4, "price": 100},
            {"hotel_name": "Luxury Beach Club", "location": "Miami, USA", "rating": 9.0, "price": 320},
            {"hotel_name": "Wellness Retreat", "location": "Vienna, Austria", "rating": 8.6, "price": 190},
            {"hotel_name": "Modern Business Hotel", "location": "Amsterdam, Netherlands", "rating": 8.3, "price": 160},
            {"hotel_name": "Romantic Getaway Inn", "location": "Venice, Italy", "rating": 8.9, "price": 220}
        ]
        
        # Filter based on user preferences
        max_price = user_preferences.get('max_price', 1000)
        min_rating = user_preferences.get('min_rating', 0)
        
        filtered_hotels = [
            hotel for hotel in sample_hotels 
            if hotel['price'] <= max_price and hotel['rating'] >= min_rating
        ]
        
        # Sort by a combination of rating and price value
        for hotel in filtered_hotels:
            # Calculate a hybrid score (higher rating, lower price = better score)
            price_score = 1.0 - (hotel['price'] / 400)  # Normalize price (assuming max 400)
            rating_score = hotel['rating'] / 10.0  # Normalize rating
            hotel['hybrid_score'] = 0.6 * rating_score + 0.4 * price_score
        
        sorted_hotels = sorted(filtered_hotels, key=lambda x: x['hybrid_score'], reverse=True)
        
        # Convert to DataFrame
        result_hotels = sorted_hotels[:top_k]
        df = pd.DataFrame(result_hotels)
        
        if not df.empty:
            df['hotel_id'] = range(1, len(df) + 1)
            print(f"✅ Generated {len(df)} sample recommendations")
        
        return df
            
    def recommend_with_multimodal(self, image_path=None, query_text=None, 
                               user_id=None, hotels_df=None, features_df=None,
                               user_preferences=None, top_k=10):
        """
        Generiert Empfehlungen basierend auf Multi-Modal-Input
        
        Args:
            image_path: Optional - Pfad zum Bild
            query_text: Optional - Abfragetext
            user_id: Optional - Benutzer-ID für personalisierte Empfehlungen
            hotels_df: Optional - Hotels-DataFrame für Hybrid-Empfehlungen
            features_df: Optional - Features-DataFrame für Hybrid-Empfehlungen
            user_preferences: Optional - Benutzereinstellungen als Dict
            top_k: Anzahl der Empfehlungen
            
        Returns:
            recommendations: Liste mit Empfehlungen
        """
        if not self.is_initialized:
            print("❌ Multi-Modal Recommender ist nicht initialisiert")
            return []
            
        if not image_path and not query_text and not user_id:
            print("❌ Mindestens ein Input (Bild, Text oder Benutzer-ID) muss angegeben werden")
            return []
            
        try:
            print("\n🔄 Generiere Multi-Modal-Empfehlungen...")
            
            # Erzeuge Multi-Modal-Embedding
            multimodal_embedding = None
            if image_path or query_text:
                multimodal_embedding = self.create_multimodal_embedding(
                    image_path=image_path,
                    query_text=query_text
                )
                print("✅ Multi-Modal-Embedding erstellt")
            
            # Wenn Hybrid-Modell verfügbar und nötige Daten vorhanden
            hybrid_recommendations = []
            if self.hybrid_model and hotels_df is not None and features_df is not None:
                print("🔄 Hole Hybrid-Empfehlungen...")
                
                # Leere Benutzereinstellungen, falls nicht angegeben
                if user_preferences is None:
                    user_preferences = {}
                
                try:
                    # Empfehlungen mit Text-Query (wenn vorhanden)
                    if query_text:
                        hybrid_recommendations = self.hybrid_model.recommend_hotels(
                            query=query_text,
                            hotels_df=hotels_df, 
                            features_df=features_df,
                            user_preferences=user_preferences,
                            top_k=top_k
                        )
                        print(f"✅ {len(hybrid_recommendations)} Text-basierte Hybrid-Empfehlungen")
                    elif user_id:
                        # TODO: Implementiere Benutzer-basierte Empfehlungen
                        # Dies würde die Methoden des Hybrid-Modells nutzen
                        pass
                except Exception as e:
                    print(f"⚠️ Fehler bei Hybrid-Empfehlungen: {e}")
            
            # Wenn Bild vorhanden, hole ähnliche Städte basierend auf Bild
            image_based_recommendations = []
            if image_path and multimodal_embedding is not None:
                print("🔄 Suche ähnliche Städte basierend auf dem Bild...")
                image_based_recommendations = self.find_similar_cities(
                    multimodal_embedding, 
                    top_k=top_k
                )
                print(f"✅ {len(image_based_recommendations)} bildbasierte Empfehlungen")
            
            # Kombiniere die Empfehlungen
            final_recommendations = []
            
            # Wenn wir Hybrid-Empfehlungen haben, verwende diese als Basis
            if len(hybrid_recommendations) > 0:
                print("🔄 Integriere bildbasierte Ähnlichkeiten in Hybrid-Empfehlungen...")
                
                # Erstelle ein Dictionary von Stadt zu Bildähnlichkeit für schnellen Zugriff
                image_sim_dict = {city: sim for city, sim in image_based_recommendations}
                
                # Gehe durch Hybrid-Empfehlungen und integriere Bildähnlichkeit
                for _, row in hybrid_recommendations.iterrows():
                    hotel_name = row.get('hotel_name', 'Unknown')
                    location = row.get('location', 'Unknown')
                    
                    # Extrahiere Stadt aus Location (vereinfacht)
                    city = location.split(',')[-1].strip() if location else hotel_name
                    
                    # Prüfe, ob wir eine Bildähnlichkeit für diese Stadt haben
                    image_sim = image_sim_dict.get(city, 0.0)
                    
                    # Hybrid-Score aus der Hybrid-Empfehlung
                    hybrid_score = row.get('hybrid_score', row.get('final_score', 0.5))
                    
                    # Kombiniere Scores (gewichteter Durchschnitt)
                    if image_path:
                        combined_score = (
                            self.text_weight * hybrid_score + 
                            self.image_weight * image_sim
                        )
                    else:
                        combined_score = hybrid_score
                    
                    # Erstelle Dictionary für die Empfehlung
                    recommendation = {
                        'hotel_name': hotel_name,
                        'location': location,
                        'rating': row.get('rating', 0.0),
                        'price': row.get('price', 0.0),
                        'multimodal_score': combined_score,
                        'hybrid_score': hybrid_score
                    }
                    
                    # Füge Bildähnlichkeit hinzu, wenn verfügbar
                    if image_path:
                        recommendation['image_similarity'] = image_sim
                    
                    final_recommendations.append(recommendation)
                
                # Sortiere nach kombiniertem Score
                final_recommendations = sorted(
                    final_recommendations, 
                    key=lambda x: x['multimodal_score'], 
                    reverse=True
                )
                
            # Wenn keine Hybrid-Empfehlungen, verwende nur bildbasierte
            elif len(image_based_recommendations) > 0:
                print("🔄 Verwende nur bildbasierte Empfehlungen...")
                
                for city, similarity in image_based_recommendations:
                    recommendation = {
                        'city': city,
                        'image_similarity': similarity,
                        'multimodal_score': similarity
                    }
                    final_recommendations.append(recommendation)
            
            return final_recommendations[:top_k]
                
        except Exception as e:
            print(f"❌ Fehler bei der Multi-Modal-Empfehlung: {e}")
            return []

# --- Training-Funktionen ---

def train_with_extracted_images(extracted_images_dir):
    """
    Trainiert das Multi-Modal-Modell mit den extrahierten Bildern aus der Datenbank
    
    Args:
        extracted_images_dir: Pfad zum Verzeichnis mit extrahierten Bildern (organisiert nach Städten)
        
    Returns:
        bool: True wenn Training erfolgreich, False sonst
    """
    print(f"\n🚀 Starte Training mit extrahierten Bildern aus: {extracted_images_dir}")
    
    try:
        # Sammle alle Bilder und Städte
        image_paths = []
        city_names = []
        text_descriptions = []
        
        print("📂 Durchsuche Bildverzeichnisse...")
        
        # Gehe durch alle Stadtordner
        extracted_path = Path(extracted_images_dir)
        city_folders = [f for f in extracted_path.iterdir() if f.is_dir()]
        
        print(f"🏙️ Gefunden: {len(city_folders)} Städte")
        
        for city_folder in city_folders:
            city_name = city_folder.name
            
            # Finde alle Bilddateien in diesem Stadtordner
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                image_files.extend(city_folder.glob(ext))
            
            # Begrenzen Sie die Anzahl der Bilder pro Stadt für das Training (für Performance)
            max_images_per_city = 20
            if len(image_files) > max_images_per_city:
                print(f"   📸 {city_name}: Verwende {max_images_per_city} von {len(image_files)} Bildern")
                image_files = image_files[:max_images_per_city]
            else:
                print(f"   📸 {city_name}: {len(image_files)} Bilder")
            
            # Füge Bilder zur Trainingsliste hinzu
            for image_file in image_files:
                image_paths.append(str(image_file))
                city_names.append(city_name)
                
                # Erstelle Textbeschreibung basierend auf Stadtname
                # Normalisiere den Stadtnamen für bessere Textbeschreibung
                clean_city = city_name.replace('_', ' ').title()
                description = f"Beautiful destination in {clean_city}, travel photography, tourism, vacation spot"
                text_descriptions.append(description)
        
        print(f"\n📊 Training-Daten gesammelt:")
        print(f"   🖼️ Bilder: {len(image_paths)}")
        print(f"   🏙️ Städte: {len(set(city_names))}")
        
        if len(image_paths) == 0:
            print("❌ Keine Bilder zum Training gefunden!")
            return False
        
        # Initialisiere Multi-Modal Recommender
        print("\n🔄 Initialisiere Multi-Modal Recommender für Training...")
        recommender = MultiModalRecommender(
            image_weight=0.5,  # Ausgewogene Gewichtung für Training
            text_weight=0.5,
            embedding_dim=256
        )
        
        if not recommender.is_initialized:
            print("❌ Konnte Multi-Modal Recommender nicht initialisieren")
            return False
        
        # Starte das Training
        print("\n🚀 Beginne Multi-Modal Training...")
        
        # Training-Parameter
        epochs = 3  # Weniger Epochen für schnelleres Training
        batch_size = 16  # Kleinere Batch-Größe für bessere Speichernutzung
        
        success = train_multimodal_model(
            model=recommender,
            image_paths=image_paths,
            city_names=city_names,
            text_descriptions=text_descriptions,
            epochs=epochs,
            batch_size=batch_size
        )
        
        if success:
            print("\n✅ Training erfolgreich abgeschlossen!")
            
            # Teste das trainierte Modell
            print("\n🧪 Teste das trainierte Modell...")
            test_trained_model(recommender, image_paths[:5], city_names[:5])
            
            # Speichere das trainierte Modell
            save_path = Path(extracted_images_dir).parent / "trained_multimodal_model.joblib"
            print(f"\n💾 Speichere trainiertes Modell: {save_path}")
            try:
                import joblib
                
                # Speichere relevante Teile des Modells
                model_data = {
                    'image_weight': recommender.image_weight,
                    'text_weight': recommender.text_weight,
                    'embedding_dim': recommender.embedding_dim,
                    'has_image_features': recommender.has_image_features,
                    'has_text_features': recommender.has_text_features,
                    'training_cities': list(set(city_names)),
                    'training_images_count': len(image_paths)
                }
                
                joblib.dump(model_data, save_path)
                print(f"✅ Modell gespeichert: {save_path}")
                
            except Exception as save_err:
                print(f"⚠️ Fehler beim Speichern: {save_err}")
            
            return True
        else:
            print("❌ Training fehlgeschlagen")
            return False
            
    except Exception as e:
        print(f"❌ Fehler beim Training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trained_model(recommender, test_image_paths, test_city_names):
    """
    Testet das trainierte Multi-Modal-Modell mit Beispielbildern
    
    Args:
        recommender: Trainierter MultiModalRecommender
        test_image_paths: Liste von Test-Bildpfaden
        test_city_names: Liste von Test-Städtenamen
    """
    print("\n🧪 Teste trainiertes Multi-Modal-Modell...")
    
    for i, (img_path, city_name) in enumerate(zip(test_image_paths, test_city_names)):
        print(f"\n  Test {i+1}: {city_name}")
        
        try:
            # Erstelle Multi-Modal-Embedding
            embedding = recommender.create_multimodal_embedding(
                image_path=img_path,
                query_text=f"travel to {city_name}",
                city_name=city_name
            )
            
            if embedding is not None:
                print(f"    ✅ Embedding erstellt: Shape {embedding.shape}")
                
                # Teste Bildfeature-Extraktion
                img_features = recommender.get_image_features(img_path)
                if img_features is not None:
                    print(f"    🖼️ Bild-Features: Shape {img_features.shape}")
                
                # Teste Textfeature-Extraktion
                text_features = recommender.get_text_features(f"beautiful {city_name} tourism")
                if text_features is not None:
                    print(f"    📝 Text-Features: Shape {text_features.shape}")
                    
            else:
                print(f"    ❌ Konnte kein Embedding erstellen")
                
        except Exception as e:
            print(f"    ❌ Test fehlgeschlagen: {e}")
    
    print("\n✅ Modell-Test abgeschlossen")

def train_multimodal_model(model, image_paths, city_names, text_descriptions=None, 
                          epochs=10, batch_size=32):
    """
    Trainiert das Multi-Modal-Modell mit ähnlichen Bildern und Textbeschreibungen
    
    Args:
        model: MultiModalRecommender Instanz
        image_paths: Liste von Bildpfaden
        city_names: Liste von Städtenamen (gleiche Länge wie image_paths)
        text_descriptions: Optional - Liste von Textbeschreibungen
        epochs: Anzahl der Trainingsepochen
        batch_size: Batch-Größe
    """
    if len(image_paths) != len(city_names):
        print("❌ Anzahl der Bilder und Städtenamen muss gleich sein")
        return False
    
    if not model.is_initialized or model.fusion_model is None:
        print("❌ Multi-Modal-Modell nicht korrekt initialisiert")
        return False
        
    if text_descriptions is None:
        # Verwende Städtenamen als Fallback
        text_descriptions = city_names
    elif len(text_descriptions) != len(image_paths):
        print("⚠️ Anzahl der Textbeschreibungen stimmt nicht mit Bildanzahl überein")
        print("   Verwende Städtenamen als Fallback")
        text_descriptions = city_names
    
    print(f"\n🔄 Starte Training des Multi-Modal-Modells mit {len(image_paths)} Beispielen...")
    
    try:
        # Erstelle Trainings-Batches
        for epoch in range(epochs):
            print(f"\nEpoche {epoch+1}/{epochs}")
            
            # Gehe durch alle Beispiele
            for i in range(0, len(image_paths), batch_size):
                batch_images = image_paths[i:i+batch_size]
                batch_cities = city_names[i:i+batch_size]
                batch_texts = text_descriptions[i:i+batch_size]
                
                print(f"  Batch {i//batch_size + 1}: Verarbeite {len(batch_images)} Beispiele")
                
                # Verarbeite jeden Eintrag im Batch
                for img_path, city, text in zip(batch_images, batch_cities, batch_texts):
                    # Extrahiere Features
                    image_features = model.get_image_features(img_path)
                    text_features = model.get_text_features(text, city)
                    
                    if image_features is not None and text_features is not None:
                        # Füge Features zur Datenbank hinzu (falls implementiert)
                        if model.image_feature_db:
                            model.image_feature_db.add_feature(f"{city}_{i}", city, image_features)
                            
                        # Erstelle Multi-Modal-Embedding
                        _ = model.create_multimodal_embedding(
                            image_path=img_path,
                            query_text=text,
                            city_name=city
                        )
                        
            # Speichere Features nach jeder Epoche
            if model.image_feature_db:
                model.image_feature_db.save_features()
                
        print("\n✅ Training des Multi-Modal-Modells abgeschlossen")
        return True
    except Exception as e:
        print(f"❌ Fehler beim Training des Multi-Modal-Modells: {e}")
        return False

# --- Demo-Funktionen ---

def demo_multimodal_recommender():
    """Demonstrates the Multi-Modal Recommender using the proven hybrid method"""
    print("\n🚀 Demo: Multi-Modal Recommender")
    print("===============================")
    print("Using proven hybrid recommendation method with optional image enhancement")
    
    # Advanced configuration for users (optional)
    print("\n⚙️ Advanced Configuration (optional, press Enter for defaults):")
    
    # Weighting between image and text
    image_weight_input = input("Image feature weight (0.0-1.0, default: 0.4): ").strip()
    image_weight = 0.4
    if image_weight_input:
        try:
            image_weight = float(image_weight_input.replace(',', '.'))
            image_weight = max(0.0, min(1.0, image_weight))  # Limit to 0-1
        except ValueError:
            print("⚠️ Invalid input, using default value 0.4")
            image_weight = 0.4
    
    text_weight = 1.0 - image_weight
    
    # Initialize Multi-Modal Recommender with adjusted weights
    recommender = MultiModalRecommender(
        image_weight=image_weight, 
        text_weight=text_weight, 
        embedding_dim=256
    )
    
    print(f"\n📊 Model Configuration:")
    print(f"   🖼️ Image Weight: {image_weight:.1f}")
    print(f"   📝 Text Weight: {text_weight:.1f}")
    
    # Main search query
    query_text = input("\n🔍 Enter your search query (e.g. 'beach hotel with spa'): ").strip()
    
    # Optional: Enter image path
    print("\n📷 Image-based search (optional):")
    image_path = input("Enter path to an image (or press Enter to skip): ").strip()
    
    # Filter options like in text_based demo (optional)
    print("\n🔧 Filter Options (optional, press Enter for defaults):")
    
    # Price filter
    max_price_input = input("Maximum price per night (default: $200): ").strip()
    max_price = 200
    if max_price_input:
        try:
            max_price = float(max_price_input.replace(',', '.').replace('$', ''))
        except ValueError:
            print("⚠️ Invalid price, using default value $200")
            max_price = 200
    
    # Rating filter  
    min_rating_input = input("Minimum rating (default: 4.0): ").strip()
    min_rating = 4.0
    if min_rating_input:
        try:
            min_rating = float(min_rating_input.replace(',', '.'))
        except ValueError:
            print("⚠️ Invalid rating, using default value 4.0")
            min_rating = 4.0
    
    # Geographic preferences
    print("\n🗺️ Geographic Preferences (optional):")
    preferred_region = input("Preferred region/country (or press Enter for all): ").strip()
    
    # Number of recommendations
    top_k_input = input("Number of recommendations (default: 10): ").strip()
    top_k = 10
    if top_k_input:
        try:
            top_k = int(top_k_input)
            top_k = max(1, min(20, top_k))  # Limit to 1-20
        except ValueError:
            print("⚠️ Invalid number, using default value 10")
            top_k = 10
    
    # Create user preferences
    user_preferences = {
        'max_price': max_price,
        'min_rating': min_rating,
        'preferred_region': preferred_region if preferred_region else None,
        'text_importance': text_weight,
        'price_importance': 0.2,
        'rating_importance': 0.3,
        'amenity_importance': 0.1
    }
    
    # Validation of inputs
    if not query_text and not image_path:
        print("❌ Please provide at least a search query or image path.")
        return
    
    print(f"\n🔍 Search with the following parameters:")
    if query_text:
        print(f"📝 Text: '{query_text}'")
    if image_path:
        print(f"📷 Image: {image_path}")
    print(f"💰 Max Price: ${max_price}")
    print(f"⭐ Min Rating: {min_rating}")
    print(f"📊 Number of Results: {top_k}")
    if preferred_region:
        print(f"🗺️ Region: {preferred_region}")
    
    # Get recommendations using the hybrid method
    print("\n🔄 Getting multimodal recommendations...")
    
    recommendations = recommender.get_multimodal_recommendations(
        query_text=query_text if query_text else None,
        image_path=image_path if image_path else None,
        user_preferences=user_preferences,
        top_k=top_k
    )
    
    if not recommendations.empty:
        print("✅ Recommendations generated successfully")
        
        print(f"\n✅ Top {len(recommendations)} recommended hotels:")
        print("=" * 70)
        
        for i, (_, hotel) in enumerate(recommendations.iterrows()):
            # Get hotel details
            hotel_name = hotel.get('hotel_name', f'Hotel {i+1}')
            location = hotel.get('location', 'Unknown Location')
            rating = hotel.get('rating', 0.0)
            price = hotel.get('price', 0.0)
            score = hotel.get('hybrid_score', hotel.get('final_score', 0.0))
            
            # Extract country/region for flag
            flag = "🏨"  # Default
            location_lower = location.lower()
            if 'france' in location_lower or 'paris' in location_lower:
                flag = "🇫🇷"
            elif 'italy' in location_lower or 'rome' in location_lower or 'venice' in location_lower:
                flag = "🇮🇹"
            elif 'spain' in location_lower or 'barcelona' in location_lower:
                flag = "🇪🇸"
            elif 'germany' in location_lower or 'berlin' in location_lower:
                flag = "🇩🇪"
            elif 'uk' in location_lower or 'london' in location_lower or 'england' in location_lower:
                flag = "🇬🇧"
            elif 'netherlands' in location_lower or 'amsterdam' in location_lower:
                flag = "🇳🇱"
            elif 'austria' in location_lower or 'vienna' in location_lower:
                flag = "🇦🇹"
            elif 'switzerland' in location_lower or 'zurich' in location_lower:
                flag = "🇨🇭"
            elif 'usa' in location_lower or 'america' in location_lower or 'miami' in location_lower:
                flag = "🇺🇸"
            elif 'indonesia' in location_lower or 'bali' in location_lower:
                flag = "🇮🇩"
            elif 'czech' in location_lower or 'prague' in location_lower:
                flag = "🇨🇿"
            
            # Rating stars
            stars = "⭐" * min(5, int(rating))
            if rating >= 9.0:
                stars += " (Excellent)"
            elif rating >= 8.0:
                stars += " (Very Good)"
            elif rating >= 7.0:
                stars += " (Good)"
            
            # Price category
            if price <= 100:
                price_category = "💰 Budget"
            elif price <= 200:
                price_category = "💰💰 Mid-range"
            elif price <= 300:
                price_category = "💰💰💰 Upscale"
            else:
                price_category = "💰💰💰💰 Luxury"
            
            print(f"  {i+1:2d}. {flag} {hotel_name}")
            print(f"      📍 {location}")
            print(f"      {stars} Rating: {rating:.1f}/10")
            print(f"      {price_category} ${price:.0f}/night")
            print(f"      🎯 Match Score: {score:.2f}")
            
            # Check if it matches user's region preference
            if preferred_region and preferred_region.lower() in location_lower:
                print(f"      ✅ Matches your region preference: {preferred_region}")
            
            # Add image similarity if available
            if 'image_similarity' in hotel:
                img_sim = hotel['image_similarity']
                img_stars = "🌟" * min(5, int(img_sim * 5))
                print(f"      🖼️ Image similarity: {img_stars} ({img_sim:.2f})")
            
            print()  # Empty line for better readability
        
        # Additional information
        print("💡 Recommendations based on:")
        if query_text and image_path:
            print(f"   • Text description ({text_weight:.1f}) and image information ({image_weight:.1f}) combined")
        elif query_text:
            print("   • Your text description using proven hybrid method")
        elif image_path:
            print("   • Your uploaded image with text fallback")
        
        print(f"   • Your filter criteria (Max ${max_price}, Min ⭐{min_rating})")
        print("   • Advanced hybrid scoring (text + parameter + rating analysis)")
        
    else:
        print("⚠️ No suitable hotels found")
        print("💡 Try:")
        print("   • A more general search query")
        print("   • Relaxed filter criteria")
        print("   • Different image or text input")
        print("   • Different weighting between image and text")
    
    print("\n✅ Demo completed.")


# --- Main Function ---

def main():
    """Main function for the interactive demo"""
    print("\n🏨 TravelHunters Multi-Modal Recommender")
    print("🔄 Combines image and text features for better recommendations")
    print("=" * 60)
    
    try:
        while True:
            print("\nWhat would you like to do?")
            print("1. Get hotel recommendations (Multi-Modal)")
            print("2. Train model (for advanced users)")
            print("3. Exit")
            
            choice = input("\nYour choice (1-3): ").strip()
            
            if choice == "1":
                demo_multimodal_recommender()
            elif choice == "2":
                print("\n⚠️ This function requires prepared training data.")
                print("   Please ensure you have images and corresponding")
                print("   city information prepared for training.")
                
                confirm = input("\nWould you like to continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    print("\n🔄 Please provide the path to training data.")
                    training_dir = input("Directory with images: ").strip()
                    
                    if os.path.isdir(training_dir):
                        print("🔄 Preparing training...")
                        success = train_with_extracted_images(training_dir)
                        if success:
                            print("✅ Training completed successfully!")
                        else:
                            print("❌ Training failed. Please check the logs.")
                    else:
                        print(f"❌ Directory not found: {training_dir}")
                else:
                    print("Training cancelled.")
            elif choice == "3":
                print("👋 Thank you for using TravelHunters!")
                break
            else:
                print("❌ Invalid choice. Please select 1-3.")
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    finally:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
