'''
Initialisierungsskript für den Multi-Modal-Recommender
----------------------------------------------------
Dieses Skript initialisiert die Umgebung für den Multi-Modal-Recommender
und stellt sicher, dass alle erforderlichen Pfade korrekt gesetzt sind.

Autor: GitHub Copilot
Datum: 15.07.2025
'''

import os
import sys
from pathlib import Path

# Bestimme die Verzeichnisstruktur basierend auf dem aktuellen Skript
CURRENT_DIR = Path(__file__).parent.absolute()  # multimodal
MODELLING_DIR = CURRENT_DIR.parent              # modelling
PROJECT_ROOT = MODELLING_DIR.parent             # TravelHunters

# Verzeichnisse für die verschiedenen Modelle
CNN_DIR = MODELLING_DIR / "cnn"  # TravelHunters/modelling/cnn
ML_MODELS_DIR = MODELLING_DIR / "machine_learning_modells"  # TravelHunters/modelling/machine_learning_modells

# Füge alle relevanten Verzeichnisse zum Python-Pfad hinzu
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(CNN_DIR))
sys.path.append(str(ML_MODELS_DIR))
sys.path.append(str(ML_MODELS_DIR / "models"))

# Überprüfe, ob alle erforderlichen Dateien existieren
def check_file_exists(path, description):
    if os.path.exists(path):
        print(f"✅ {description} gefunden: {path}")
        return True
    else:
        print(f"❌ {description} nicht gefunden: {path}")
        return False

# Überprüfe, ob TensorFlow installiert ist
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} ist installiert")
except ImportError:
    print("⚠️ TensorFlow ist nicht installiert. Bitte installieren Sie TensorFlow mit:")
    print("   pip install tensorflow")

# Überprüfe CNN-Modellkomponenten
cnn_predictor = CNN_DIR / "predictor.py"
cnn_model = CNN_DIR / "city_classifier_model.h5"

check_file_exists(cnn_predictor, "CNN Predictor")
check_file_exists(cnn_model, "CNN Modell")

# Überprüfe Hybrid-Modellkomponenten
hybrid_model = ML_MODELS_DIR / "models" / "hybrid_model.py"
text_model = ML_MODELS_DIR / "models" / "text_similarity_model.py"

# Prüfe alternative Pfade für Hybrid-Modelle
if not os.path.exists(hybrid_model):
    alternative_hybrid_model = PROJECT_ROOT / "models" / "hybrid_model.py"
    if os.path.exists(alternative_hybrid_model):
        hybrid_model = alternative_hybrid_model

if not os.path.exists(text_model):
    alternative_text_model = PROJECT_ROOT / "models" / "text_similarity_model.py"
    if os.path.exists(alternative_text_model):
        text_model = alternative_text_model

check_file_exists(hybrid_model, "Hybrid-Modell")
check_file_exists(text_model, "Text-Modell")

# Überprüfe gespeicherte Modelle
saved_text_model = ML_MODELS_DIR / "saved_models" / "text_model.joblib"
saved_param_model = ML_MODELS_DIR / "saved_models" / "param_model.joblib"

check_file_exists(saved_text_model, "Gespeichertes Text-Modell")
check_file_exists(saved_param_model, "Gespeichertes Parameter-Modell")

# Definiere Pfade für den Zugriff in anderen Skripten
SAVED_TEXT_MODEL_PATH = saved_text_model
SAVED_PARAM_MODEL_PATH = saved_param_model

# Erstelle Verzeichnisstruktur für Multi-Modal-Modell, falls nicht vorhanden
multimodal_dir = MODELLING_DIR / "multimodal"  # TravelHunters/modelling/multimodal
os.makedirs(multimodal_dir, exist_ok=True)
print(f"✅ Multi-Modal-Verzeichnis: {multimodal_dir}")

# Ausgabe der Umgebungsvariablen für Debugging
print("\n📊 Umgebungsvariablen:")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"MODELLING_DIR: {MODELLING_DIR}")
print(f"CNN_DIR: {CNN_DIR}")
print(f"ML_MODELS_DIR: {ML_MODELS_DIR}")

print("\n✅ Initialisierung abgeschlossen.")
