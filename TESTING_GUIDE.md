# TravelHunters ML-System - Testanleitung

## 🎯 Ziel
Diese Anleitung zeigt dir, wie du das gesamte TravelHunters ML-Empfehlungssystem testen kannst, um sicherzustellen, dass alle Komponenten ordnungsgemäß funktionieren.

## 🚀 Schnelltest (Empfohlen)

### Schritt 1: Automatisierter Kompletttest
```bash
# 1. Navigiere zum modelling Verzeichnis
cd /pfad/zu/TravelHunters/modelling

# 2. Führe den automatisierten Test aus
python test_complete_system.py
```

**Was wird getestet:**
- ✅ Datenladen aus SQLite-Datenbank (8.000+ Hotels)
- ✅ Feature Engineering (30+ ML-Features)
- ✅ Parameter-basiertes Modell (Linear Regression)
- ✅ Text-basiertes Modell (TF-IDF + Cosine Similarity) 
- ✅ Hybrid-Modell (Kombination beider Ansätze)
- ✅ Evaluation-Framework (RMSE, R², Accuracy)
- ✅ Interaktive Demo-Schnittstelle

**Erwarteter erfolgreicher Output:**
```text
============================================================
🔍 TravelHunters ML System - Complete Test Suite
============================================================

✅ PASS Data Loading
✅ PASS Feature Engineering  
✅ PASS Parameter Model
✅ PASS Text Model
✅ PASS Hybrid Model
✅ PASS Evaluation Framework
✅ PASS Demo Interface

📊 Overall Results: 7/7 tests passed (100.0%)
🎉 All tests passed! System is ready for production.
```

### Schritt 2: Interaktive Demo starten
```bash
# Starte die Haupt-Anwendung
python demo.py
```

**Demo-Funktionen:**
1. **Daten-Übersicht** → Zeigt Statistiken der Hotel-Datenbank
2. **Hotel-Empfehlungen** → Teste alle drei Empfehlungsmodelle
3. **Modell-Evaluation** → Performance-Metriken anzeigen
4. **Beenden**

---

## 🔧 Einzelkomponenten testen

### Test 1: Datenbankladen
```bash
python -c "
from data_preparation.load_data import HotelDataLoader
loader = HotelDataLoader()
hotels = loader.load_hotels()
print(f'✅ {len(hotels)} Hotels geladen')
print(f'📊 Beispiel: {hotels.iloc[0][\"name\"]}')
print(f'💰 Preisbereich: €{hotels[\"price\"].min():.0f} - €{hotels[\"price\"].max():.0f}')
print(f'⭐ Bewertungen: {hotels[\"rating\"].min():.1f} - {hotels[\"rating\"].max():.1f}')
"
```

**Erwarteter Output:**
```text
✅ Loading hotel data from SQLite database...
✅ Loaded 8072 hotels from database
✅ 8072 Hotels geladen
📊 Beispiel: Renovated 2BD apartment in Parc Belleville  
💰 Preisbereich: €35 - €7493
⭐ Bewertungen: 2.0 - 10.0 (10-Punkte-Skala)
```

### Test 2: Feature Engineering
```bash
python -c "
from data_preparation.load_data import HotelDataLoader
from data_preparation.feature_engineering import HotelFeatureEngineer

loader = HotelDataLoader()
hotels = loader.load_hotels()

engineer = HotelFeatureEngineer()
features = engineer.engineer_numerical_features(hotels)

print(f'✅ {features.shape[1]} Features für {features.shape[0]} Hotels erstellt')
print(f'📊 Feature-Spalten: {list(features.columns)[:10]}...')

# Test Amenities-Extraktion
sample_desc = hotels.iloc[0]['description']
amenities = engineer.extract_amenities_from_description(sample_desc)
print(f'🏨 Extrahierte Amenities: {amenities}')
"
```

### Test 3: Parameter-basiertes Modell
```bash
python -c "
from data_preparation.load_data import HotelDataLoader
from data_preparation.feature_engineering import HotelFeatureEngineer
from models.parameter_model import ParameterBasedRecommender

# Daten laden und Features erstellen
loader = HotelDataLoader()
hotels = loader.load_hotels()
engineer = HotelFeatureEngineer()
features = engineer.engineer_numerical_features(hotels)

# Modell trainieren
model = ParameterBasedRecommender()
metrics = model.train(features, hotels['rating'])
print(f'✅ Modell trainiert - R²: {metrics[\"val_r2\"]:.3f}, RMSE: {metrics[\"val_rmse\"]:.3f}')

# Empfehlungen testen
user_prefs = {
    'max_price': 300,
    'min_rating': 6.0,  # 6.0/10 entspricht "gut"
    'required_amenities': ['wifi', 'breakfast']
}

recommendations = model.recommend_hotels(hotels, user_prefs)
print(f'✅ {len(recommendations)} Empfehlungen für Budget €300, Rating ≥6.0/10')

if len(recommendations) > 0:
    top = recommendations.iloc[0]
    print(f'🏆 Top-Empfehlung: {top[\"name\"]} (€{top[\"price\"]}/Nacht, {top[\"rating\"]}/10⭐)')
"
```

### Test 4: Text-basiertes Modell mit Preisfiltern
```bash
python -c "
from data_preparation.load_data import HotelDataLoader
from models.text_similarity_model import TextBasedRecommender

loader = HotelDataLoader()
hotels = loader.load_hotels()

model = TextBasedRecommender()
model.fit(hotels)
print(f'✅ Text-Modell trainiert für {len(hotels)} Hotels')

# Teste verschiedene Suchanfragen mit verschiedenen Budgets
test_cases = [
    ('relaxing in luxury hotel', 400),
    ('budget family hotel', 200), 
    ('business hotel with wifi', 300),
    ('romantic spa hotel', 500)
]

for query, budget in test_cases:
    print(f'\n🔍 \"{query}\" | Budget: €{budget}')
    user_prefs = {'max_price': budget}
    results = model.recommend_hotels(query, hotels, user_prefs, top_k=3)
    
    if len(results) > 0:
        for _, hotel in results.iterrows():
            print(f'  {hotel[\"name\"][:45]:45} | €{hotel[\"price\"]:6.0f} | {hotel[\"rating\"]}/10⭐')
    else:
        print('  ❌ Keine Hotels in dieser Preiskategorie gefunden')
"
```

**Erwarteter Output:**
```text
✅ Text-Modell trainiert für 8072 Hotels

🔍 "relaxing in luxury hotel" | Budget: €400
  The Editory Riverside Hotel, an Historic Hotel | €   258 | 8.4/10⭐
  The New Yorker Hotel, a Lotte Hotel            | €   346 | 7.7/10⭐

🔍 "business hotel with wifi" | Budget: €300
  Abstract Hotel                                 | €    93 | 8.2/10⭐
  LyLo Auckland Hotel                            | €   109 | 8.7/10⭐
  Travelodge Hotel Auckland Wynyard Quarter      | €   130 | 8.6/10⭐

🔍 "romantic spa hotel" | Budget: €500
  Elite Palace Hotel & Spa                       | €   249 | 8.2/10⭐
  The Houghton Hotel, Spa, Wellness & Golf       | €   342 | 9.0/10⭐
```

**Wichtige Erkenntnisse:**
- ✅ **Preisfilter funktionieren korrekt** - Unterschiedliche Budgets zeigen verschiedene Hotels
- ✅ **Keine Duplikate** - Jedes Hotel erscheint nur einmal
- ✅ **Relevante Treffer** - Hotels passen zur Suchanfrage ("spa" findet Spa-Hotels)
- ✅ **Automatische Filterung** - System filtert erst nach Preis, dann nach Relevanz

### Test 5: Hybrid-Modell
```bash
python -c "
from data_preparation.load_data import HotelDataLoader
from data_preparation.feature_engineering import HotelFeatureEngineer
from models.hybrid_model import HybridRecommender

# Daten vorbereiten
loader = HotelDataLoader()
hotels = loader.load_hotels()
engineer = HotelFeatureEngineer()
features = engineer.engineer_numerical_features(hotels)

# Hybrid-Modell trainieren
model = HybridRecommender()
model.train(hotels, features)
print('✅ Hybrid-Modell trainiert')

# Kombinierte Empfehlung testen
user_prefs = {
    'max_price': 400,
    'min_rating': 7.0,  # 7.0/10 entspricht "sehr gut"
    'text_importance': 0.6
}

query = 'Luxushotel mit Spa und Pool'
recommendations = model.recommend_hotels(query, hotels, features, user_prefs)

print(f'✅ {len(recommendations)} Hybrid-Empfehlungen')
if len(recommendations) > 0:
    top = recommendations.iloc[0]
    print(f'🏆 Top Hybrid-Empfehlung: {top[\"name\"]} (€{top[\"price\"]}, {top[\"rating\"]}/10⭐)')
"
```

---

## 🗃️ Datenbank-Verifikation

### Datenbank-Status prüfen
```bash
# Prüfe ob Datenbank-Datei existiert
ls -la ../data_acquisition/database/travelhunters.db

# Zeige Datenbank-Tabellen
sqlite3 ../data_acquisition/database/travelhunters.db ".tables"

# Anzahl Hotels anzeigen
sqlite3 ../data_acquisition/database/travelhunters.db "SELECT COUNT(*) FROM booking_worldwide;"

# Beispiel-Hotels anzeigen
sqlite3 ../data_acquisition/database/travelhunters.db "SELECT name, price, rating FROM booking_worldwide LIMIT 5;"
```

**Erwarteter Output:**
```text
-rw-r--r--  1 user  staff  12345678 Jul  8 10:30 travelhunters.db
booking_worldwide
unique_cities
8072
Hotel Saint-Louis Marais|480.0|4.4
Renaissance Paris Arc de Triomphe Hotel|803.0|4.2
Hotel des Grands Boulevards|420.0|4.3
Lula Tulum, a Small Luxury Hotel|520.0|5.0
Elite Palace Hotel & Spa|165.0|4.5
```

---

## ⚠️ Fehlerbehebung

### Problem: "Database file not found"
```bash
# Überprüfe Datenbank-Pfad
ls -la ../data_acquisition/database/

# Falls Datenbank fehlt, nutzt das System automatisch JSON/Mock-Daten
# Erwartete Fallback-Meldung:
# "❌ Error loading from database: Database file not found"
# "✅ Loading hotel data from: booking_worldwide.json"
```

### Problem: "No module named 'sklearn'"
```bash
# Installiere fehlende Python-Pakete
pip install scikit-learn pandas numpy

# Oder nutze requirements.txt
pip install -r requirements.txt
```

### Problem: "Empty recommendations"  
```bash
# Prüfe Datenqualität
python -c "
from data_preparation.load_data import HotelDataLoader
loader = HotelDataLoader()
summary = loader.get_data_summary()
print(f'Hotels: {summary[\"n_hotels\"]}')
print(f'Durchschnittliche Bewertung: {summary[\"avg_rating\"]:.2f}')
print(f'Preisbereich: €{summary[\"price_range\"][\"min\"]:.0f} - €{summary[\"price_range\"][\"max\"]:.0f}')
"
```

### Problem: Import-Fehler
```bash
# Prüfe Python-Version (3.8+ empfohlen)
python --version

# Prüfe ob alle Dateien vorhanden sind
ls -la models/*.py
ls -la data_preparation/*.py
```

---

## 📊 Performance-Benchmarks

**Erwartete Systemleistung:**
- **Datenladen**: ~2-3 Sekunden für 8.000+ Hotels
- **Feature Engineering**: ~1-2 Sekunden für 30+ Features
- **Parameter Model Training**: ~0.1-1 Sekunden
- **Text Model Training**: ~1-2 Sekunden
- **Empfehlungsabfrage**: ~0.5-1 Sekunden
- **Speicherverbrauch**: ~200-300 MB

**ML-Evaluation-Metriken:**
- **Parameter Model R²**: 0.8-1.0 (perfekte synthetische Daten)
- **Parameter Model RMSE**: 0.0-0.2 (auf 10-Punkte-Skala)
- **Text Model Cosine Similarity**: 0.5-0.9 für relevante Treffer
- **Prediction Accuracy (±0.5 Punkte)**: ~65-75% (auf 10-Punkte-Skala)

---

## 🎛️ Systemanforderungen

### Software-Anforderungen
```bash
# Python Version (3.8+ empfohlen)
python --version

# Erforderliche Python-Pakete
pip install pandas numpy scikit-learn
```

### Hardware-Empfehlungen
- **RAM**: Mindestens 4 GB (8 GB empfohlen)
- **CPU**: Multi-Core empfohlen für Training
- **Speicher**: ~500 MB für Datenbank + Models

### Benötigte Dateien
- ✅ `../data_acquisition/database/travelhunters.db` (Haupt-Datenquelle)
- ✅ `requirements.txt` (Python-Abhängigkeiten)
- ✅ `demo.py` (Interaktive Hauptanwendung)
- ✅ `test_complete_system.py` (Automatisierter Testsuite)
- ✅ `models/*.py` (ML-Modell-Implementierungen)
- ✅ `data_preparation/*.py` (Datenverarbeitung)

---

## 🚀 Produktionsbereitschaft

### Vollständiger Bereitstellungstest
```bash
# 1. Führe kompletten automatisierten Test durch
python test_complete_system.py

# 2. Falls alle Tests bestehen (6-7/7), starte Demo
python demo.py

# 3. Teste alle Demo-Funktionen:
#    - Datenübersicht anzeigen
#    - Empfehlungen für verschiedene Präferenzen abrufen  
#    - Modell-Performance überprüfen

# 4. System ist einsatzbereit! 🎉
```

### Erfolgs-Checkliste
- [ ] Automatisierte Tests bestehen (≥85% Erfolgsrate)
- [ ] Datenbank lädt korrekt (8.000+ Hotels) 
- [ ] Alle 3 ML-Modelle trainieren erfolgreich
- [ ] Demo-Interface startet ohne Fehler
- [ ] Empfehlungen werden generiert (<2 Sekunden)
- [ ] Performance-Metriken sind plausibel

---

## 📝 Fazit

Mit dieser umfassenden Testanleitung kannst du das gesamte TravelHunters ML-Empfehlungssystem systematisch verifizieren. Das System ist darauf ausgelegt, robust zu funktionieren, auch wenn einzelne Komponenten (wie die Datenbank) fehlen - es greift automatisch auf Fallback-Datenquellen zurück.

**Bei erfolgreichen Tests steht dir ein voll funktionsfähiges ML-System zur Verfügung, das:**
- Hotels aus einer echten Datenbank lädt
- Intelligente Empfehlungen basierend auf Benutzer-Präferenzen generiert
- Sowohl parameter-basierte als auch text-basierte Ansätze kombiniert
- Eine benutzerfreundliche Demo-Schnittstelle bietet
- Umfassende Evaluation und Debugging-Unterstützung includet

Viel Erfolg beim Testen! 🚀
