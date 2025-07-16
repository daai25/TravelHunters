# Bildmerkmalsextraktion für TravelHunters

Dieses Modul bietet Funktionen zum Extrahieren von Features aus Städtebildern mittels CNN und CLIP und deren Integration in das Hybrid-Empfehlungssystem von TravelHunters.

## Übersicht

Der Feature-Extraktor ist ein wichtiger Bestandteil des TravelHunters-Systems und ermöglicht es:

1. Städte anhand von Bildern zu klassifizieren
2. Feature-Vektoren aus Bildern zu extrahieren (via CNN oder CLIP)
3. Ähnliche Städte basierend auf Bildähnlichkeit zu empfehlen
4. Diese Bild-Features mit dem Hybrid-Empfehlungssystem zu integrieren

## Dateien

### CNN-basierte Feature-Extraktion:
- `predictor.py`: Hauptmodul mit CNN-Funktionen zur Städteklassifikation und Feature-Extraktion
- `compute_all_features.py`: Tool zum Berechnen von Feature-Vektoren für alle Städtebilder
- `demo_cnn_hybrid.py`: Demo-Skript für die Integration von CNN und Hybrid-Empfehlungssystem

### CLIP-basierte Feature-Extraktion (neu):
- `install_dependencies.py`: Installiert die notwendigen Bibliotheken (PyTorch, CLIP, Pillow)
- `prepare_test_images.py`: Kopiert Beispielbilder aus dem Datensatz für Testzwecke
- `test_feature_extraction.py`: Testet die Funktion zur Extraktion von Bildmerkmalen
- `image_similarity_demo.py`: Demonstriert Bildähnlichkeitssuche mit extrahierten Features
- `run_tests.sh`: Führt alle Tests in der richtigen Reihenfolge aus

## Anforderungen

### Für CNN-basierte Feature-Extraktion:
- TensorFlow >= 2.5.0
- NumPy
- Pillow (PIL)
- SciPy
- tqdm

### Für CLIP-basierte Feature-Extraktion:
- PyTorch
- OpenAI CLIP
- NumPy
- Pillow (PIL)
- Matplotlib
- Scikit-learn

Sie können die Abhängigkeiten automatisch installieren:

```bash
# Für CNN-basierte Features
pip install -r requirements.txt

# Für CLIP-basierte Features
python install_dependencies.py
```

## Verwendung

### 1. Städteklassifikation

Städte anhand eines Bildes identifizieren:

```python
from predictor import predict_image

# Bild klassifizieren
city, confidence = predict_image("pfad/zum/bild.jpg")
print(f"Erkannte Stadt: {city} (Konfidenz: {confidence:.2f})")
```

### 2. Feature-Extraktion

Features aus einem Bild extrahieren:

```python
from predictor import extract_image_features

# Features extrahieren
features = extract_image_features("pfad/zum/bild.jpg")
```

### 3. Ähnliche Städte empfehlen

Ähnliche Städte basierend auf einem Bild empfehlen:

```python
from predictor import recommend_similar_destinations

# Ähnliche Städte finden
similar_cities = recommend_similar_destinations("pfad/zum/bild.jpg", top_k=5)

# Ergebnisse ausgeben
for city, similarity in similar_cities:
    print(f"{city}: {similarity:.2f}")
```

### 4. Integration mit dem Hybrid-Empfehlungssystem

Verwendung der CNN-Features im Hybrid-Empfehlungssystem:

```python
from predictor import CNNHybridIntegration

# Integration initialisieren
integration = CNNHybridIntegration()

# Bild-Features im Hybrid-Modell aktualisieren
integration.update_image_features()

# Empfehlungen mit Benutzerprofil und Bild holen
recommendations = integration.get_recommendations(
    user_id="benutzer123",
    image_path="pfad/zum/bild.jpg",
    top_k=10
)
```

## Compute All Features

Mit dem Skript `compute_all_features.py` können Sie Feature-Vektoren für alle Städtebilder berechnen:

```bash
# Standard-Verzeichnis verwenden
python compute_all_features.py --default_dir

# Benutzerdefiniertes Verzeichnis
python compute_all_features.py --image_dir /pfad/zu/bildern
```

## Demo-Skript

Das Skript `demo_cnn_hybrid.py` demonstriert die Integration von CNN und Hybrid-Empfehlungssystem:

```bash
# Vollständige Demo
python demo_cnn_hybrid.py

# Nur CNN-Feature-Demo
python demo_cnn_hybrid.py --mode cnn

# Nur Hybrid-Integration-Demo
python demo_cnn_hybrid.py --mode hybrid
```

## Integration ins Hybrid-Modell

Der Feature-Extraktor integriert sich nahtlos mit dem Hybrid-Empfehlungssystem:

1. Extrahiert Features aus allen vorhandenen Städtebildern
2. Stellt diese Features dem Hybrid-Modell zur Verfügung
3. Ermöglicht Empfehlungen basierend auf Bildern und Benutzerprofilen

## CLIP-basierte Feature-Extraktion

Die neue CLIP-basierte Feature-Extraktion bietet:

1. **Verbesserte Feature-Qualität**: CLIP wurde auf 400 Millionen Bild-Text-Paaren trainiert und bietet semantisch reichere Features
2. **Robustheit**: Höhere Robustheit gegenüber verschiedenen Bildqualitäten und -formaten
3. **Zero-Shot-Fähigkeiten**: Kann auch Städtebilder erkennen, auf denen es nicht explizit trainiert wurde

### Erste Schritte mit CLIP

Um die CLIP-basierte Bildmerkmalsextraktion zu testen, führen Sie das Bash-Skript aus:

```bash
./run_tests.sh
```

Dies installiert alle notwendigen Abhängigkeiten, kopiert Beispielbilder und führt die Feature-Extraktion durch.

### Demo der Bildähnlichkeitssuche

Nach dem Ausführen der Tests können Sie die Bildähnlichkeitssuche demonstrieren:

```bash
python image_similarity_demo.py
```

Dies extrahiert Features aus allen Bildern im aktuellen Verzeichnis, berechnet Ähnlichkeiten und visualisiert die Feature-Vektoren.

## Fehlerbehandlung

Das System implementiert eine umfassende Fehlerbehandlung:

- Fallback-Mechanismen wenn das Modell nicht gefunden wird
- Robuste Bildverarbeitung mit verschiedenen Formaten
- Automatisches Dimensionsanpassen für die Modellverarbeitung
- Sicherheitsprüfungen für Datentypen und Wertebereiche

## Anforderungen des Auftrags

Diese Implementierung erfüllt die Anforderung "Default Task: Image Feature Embeddings for Items" aus dem Auftrag, indem:

1. Features aus Bildern mittels CNN extrahiert werden
2. Diese Features für Ähnlichkeitsberechnungen verwendet werden
3. Die Features in das Hybrid-Empfehlungssystem integriert werden

## Hinweise

- Das CNN-Modell muss im gleichen Verzeichnis wie `predictor.py` liegen oder der Pfad muss entsprechend angepasst werden
- Feature-Vektoren werden in einer Datei `image_features.npz` gespeichert
- Für beste Ergebnisse sollten alle Bilder in ähnlicher Qualität und Auflösung vorliegen
