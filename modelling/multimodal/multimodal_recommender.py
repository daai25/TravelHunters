#!/usr/bin/env python3
"""
Multi-Modal Recommender System für TravelHunters
Kombiniert Text- und Bild-Features für Hotel-Empfehlungen
"""

import sys
import os
import numpy as np
import pandas as pd
import sqlite3
import re
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import joblib

# Pfad-Setup
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent  # Gehe zu TravelHunters root
modelling_dir = project_root / "modelling"
cnn_dir = modelling_dir / "cnn"
ml_models_dir = modelling_dir / "machine_learning_modells"

# Füge Pfade hinzu
sys.path.append(str(project_root))
sys.path.append(str(cnn_dir))
sys.path.append(str(ml_models_dir))
sys.path.append(str(ml_models_dir / "models"))


# Import der init_environment wenn verfügbar
try:
    import init_environment
    print("✅ init_environment erfolgreich geladen")
except ImportError:
    print("⚠️ init_environment nicht gefunden, verwende direkte Pfad-Konfiguration")


# Verwende die zentrale Bild-Prediction aus predictor.py
from predictor import predict_image


class MultiModalRecommender:
    """
    Multi-Modal Recommender, der CNN-basierte Bildfeatures und textbasierte Features kombiniert.
    Der CNN-Predictor ist modular und kann einfach ausgetauscht werden.
    """
    def __init__(self, image_weight: float = 0.35, text_weight: float = 0.65, 
                 embedding_dim: int = 256, adaptive_weighting: bool = True,
                 cnn_model_path: str = None):
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.embedding_dim = embedding_dim
        self.adaptive_weighting = adaptive_weighting
        self.is_initialized = False
        self.has_image_features = False
        self.has_text_features = True  # Text ist immer verfügbar
        # Textmodell laden
        text_model_path = str(ml_models_dir / "saved_models" / "text_model.joblib")
        try:
            self.text_model = joblib.load(text_model_path)
            print(f"✅ Textmodell geladen: {text_model_path}")
        except Exception as e:
            print(f"❌ Textmodell konnte nicht geladen werden: {e}")
            self.text_model = None
        print("✅ MultiModalRecommender initialisiert")
        print(f"   Adaptive Gewichtung: {'✅' if adaptive_weighting else '❌'}")
        self.is_initialized = True
    
    def get_automatic_recommendations(self, query_text=None, image_path=None, 
                                    max_price=200, min_rating=7.0, top_k=5):
        """
        Automatische Empfehlungen mit intelligenter Feature-Erkennung
        
        Args:
            query_text: Text-Suchanfrage
            image_path: Pfad zum Bild (optional)
            max_price: Maximaler Preis
            min_rating: Minimale Bewertung
            top_k: Anzahl der Empfehlungen
            
        Returns:
            recommendations: Liste mit Empfehlungen
        """
        # Automatische Feature-Erkennung
        has_image = bool(image_path and image_path.strip() and os.path.exists(image_path))
        has_text = bool(query_text and query_text.strip())
        
        print(f"🤖 Automatische Feature-Erkennung:")
        print(f"   📝 Text: {'✅ Erkannt' if has_text else '❌ Nicht vorhanden'}")
        print(f"   🖼️ Bild: {'✅ Erkannt' if has_image else '❌ Nicht vorhanden'}")
        
        # Adaptive Gewichtung basierend auf verfügbaren Features
        if self.adaptive_weighting:
            if has_text and not has_image:
                # Nur Text: 100% Text-Gewichtung
                text_weight = 1.0
                image_weight = 0.0
                print("📝 Nur Text-Features verfügbar - verwende 100% Text-Gewichtung")
            elif has_image and not has_text:
                # Nur Bild: 100% Bild-Gewichtung
                text_weight = 0.0
                image_weight = 1.0
                print("🖼️ Nur Bild-Features verfügbar - verwende 100% Bild-Gewichtung")
            elif has_text and has_image:
                # Beide Features: Optimale Multimodal-Gewichtung
                text_weight = 0.65  # Text ist meist präziser für Suchanfragen
                image_weight = 0.35  # Bild liefert emotionalen/visuellen Kontext
                print("🔄 Beide Features verfügbar - verwende Multimodal-Gewichtung (Text: 65%, Bild: 35%)")
            else:
                # Fallback: Standard-Gewichtung
                text_weight = self.text_weight
                image_weight = self.image_weight
                print("⚠️ Keine Features erkannt, verwende Standard-Gewichtung")
        else:
            # Verwende ursprünglich konfigurierte Gewichte
            text_weight = self.text_weight
            image_weight = self.image_weight
            print(f"⚙️ Fixe Gewichtung: Text {text_weight:.0%}, Bild {image_weight:.0%}")
        
        # Fallback wenn keine Features
        if not has_text and not has_image:
            print("⚠️ Keine Features erkannt, verwende Standard-Suchanfrage")
            query_text = "hotel accommodation"
            has_text = True
            text_weight = 1.0
            image_weight = 0.0
        
        # Zeige finale Gewichtung an
        print(f"🎯 Finale Gewichtung: Text {text_weight:.0%}, Bild {image_weight:.0%}")
        
        # Generiere Empfehlungen mit angepassten Gewichten
        return self._generate_recommendations_from_database(
            query_text=query_text,
            image_path=image_path if has_image else None,
            text_weight=text_weight,
            image_weight=image_weight,
            max_price=max_price,
            min_rating=min_rating,
            top_k=min(5, top_k)
        )
    
    def _generate_recommendations_from_database(self, query_text, image_path=None, 
                                              text_weight=1.0, image_weight=0.0,
                                              max_price=200, min_rating=7.0, top_k=5):
        """Generiere Empfehlungen aus der Hotel-Datenbank mit dynamischen Gewichten"""
        
        # Datenbank-Pfad
        db_path = project_root / "data_acquisition" / "database" / "travelhunters.db"
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Intelligente Datenbankabfrage - hole mehr Hotels für bessere Auswahl
            cursor.execute("""
                SELECT id, name, link, rating, price, location, image_url, latitude, longitude
                FROM booking_worldwide
                WHERE price <= ? AND rating >= ?
                ORDER BY rating DESC, price DESC
                LIMIT 500
            """, (max_price, max(min_rating - 1.0, 5.0)))  # Sehr flexible Mindestbewertung
            
            rows = cursor.fetchall()
            hotels = []
            
            for row in rows:
                hotel_id, name, link, rating, price, location, image_url, latitude, longitude = row
                
                # Rating verarbeiten
                rating_value = 0.0
                if rating:
                    if isinstance(rating, (int, float)):
                        rating_value = float(rating)
                    else:
                        rating_match = re.search(r'(\d+\.?\d*)', str(rating))
                        if rating_match:
                            rating_value = float(rating_match.group(1))
                
                # Preis verarbeiten
                price_value = 0.0
                if price is not None:
                    try:
                        price_value = float(price)
                    except (ValueError, TypeError):
                        pass
                
                # Text-Relevanz berechnen (nur wenn Text-Gewicht > 0)
                text_relevance = 0.0
                if text_weight > 0 and query_text:
                    if self.text_model is not None:
                        try:
                            # Annahme: Modell hat eine predict_proba- oder predict-Methode
                            # und akzeptiert als Input eine Liste von Strings
                            input_text = f"{query_text} [SEP] {name} {location}"
                            if hasattr(self.text_model, "predict_proba"):
                                proba = self.text_model.predict_proba([input_text])[0]
                                # Nehme Wahrscheinlichkeit für die relevanteste Klasse
                                text_relevance = float(np.max(proba))
                            elif hasattr(self.text_model, "predict"):
                                pred = self.text_model.predict([input_text])[0]
                                # Skaliere Vorhersage auf 0-1 falls nötig
                                text_relevance = float(pred)
                                if text_relevance > 1.0:
                                    text_relevance = 1.0
                                if text_relevance < 0.0:
                                    text_relevance = 0.0
                            else:
                                # Fallback: nutze alte Heuristik
                                text_relevance = self._calculate_text_relevance(query_text, name, location)
                        except Exception as e:
                            print(f"❌ Fehler bei Textmodell-Prediction: {e}")
                            text_relevance = self._calculate_text_relevance(query_text, name, location)
                    else:
                        text_relevance = self._calculate_text_relevance(query_text, name, location)
                
                # Bild-Relevanz berechnen (nur wenn Bild-Gewicht > 0)
                image_relevance = 0.0
                if image_weight > 0 and image_path:
                    # Nutze zentrale Bild-Prediction aus predictor.py
                    try:
                        _, confidence = predict_image(image_path)
                        if confidence is not None:
                            # confidence ist 0-100, normalisiere auf 0-1
                            image_relevance = float(confidence) / 100.0
                        else:
                            image_relevance = 0.0
                    except Exception as e:
                        print(f"❌ Fehler bei Bildbewertung mit predictor.py: {e}")
                        image_relevance = 0.0
                
                # Kombinierte Relevanz mit adaptiven Gewichten
                combined_relevance = (text_weight * text_relevance + 
                                    image_weight * image_relevance)
                
                # Andere Scores berechnen
                price_score = self._calculate_price_score(price_value, max_price)
                rating_score = self._calculate_rating_score(rating_value)
                location_score = self._calculate_location_score(location)
                
                # Finaler Score - vereinfacht für bessere Ergebnisse bei allen Budgets
                combined_score = (
                    0.5 * combined_relevance +   # Text/Bild-Relevanz (Hauptfaktor)
                    0.3 * rating_score +         # Bewertung wichtig
                    0.15 * price_score +         # Preis weniger wichtig, damit alle Budgets funktionieren
                    0.05 * location_score        # Location-Bonus
                )
                
                hotels.append({
                    'hotel_id': hotel_id,
                    'hotel_name': name,
                    'location': location,
                    'rating': rating_value,
                    'price': price_value,
                    'url': link,
                    'image_url': image_url,
                    'latitude': latitude,
                    'longitude': longitude,
                    'text_relevance': text_relevance,
                    'image_relevance': image_relevance,
                    'combined_relevance': combined_relevance,
                    'hybrid_score': combined_score
                })
            
            conn.close()
            
            # Sortiere nach Combined Score
            sorted_hotels = sorted(hotels, key=lambda x: x['hybrid_score'], reverse=True)
            # Diversität aktivieren: verschiedene Preisklassen und Locations
            diverse_hotels = self._ensure_diversity(sorted_hotels, top_k, max_price)
            return diverse_hotels
            
        except Exception as e:
            print(f"❌ Fehler beim Laden der Hotels: {e}")
            return []
    
    def _calculate_price_score(self, hotel_price, max_budget):
        """
        Vereinfachte Preis-Bewertung die bei allen Budgets funktioniert
        
        Args:
            hotel_price: Preis des Hotels
            max_budget: Maximales Budget
            
        Returns:
            price_score: Bewertung zwischen 0.0 und 1.0
        """
        if hotel_price <= 0 or max_budget <= 0:
            return 0.0
        
        if hotel_price > max_budget:
            return 0.0  # Über Budget = keine Bewertung
        
        # Einfache lineare Bewertung: höhere Preise = höhere Qualität erwartet
        # Aber trotzdem alle Preise im Budget akzeptieren
        price_ratio = hotel_price / max_budget
        
        if price_ratio >= 0.7:
            # 70-100% des Budgets = beste Bewertung
            return 0.9 + 0.1 * ((price_ratio - 0.7) / 0.3)
        elif price_ratio >= 0.4:
            # 40-70% des Budgets = sehr gute Bewertung
            return 0.7 + 0.2 * ((price_ratio - 0.4) / 0.3)
        elif price_ratio >= 0.2:
            # 20-40% des Budgets = gute Bewertung
            return 0.5 + 0.2 * ((price_ratio - 0.2) / 0.2)
        else:
            # 0-20% des Budgets = okay, aber möglicherweise niedrigere Qualität
            return 0.3 + 0.2 * (price_ratio / 0.2)
    
    def _calculate_text_relevance(self, query_text, hotel_name, location):
        """
        Erweiterte Text-Relevanz-Analyse mit semantischen Keywords
        
        Args:
            query_text: Suchanfrage
            hotel_name: Name des Hotels
            location: Standort des Hotels
            
        Returns:
            relevance_score: Bewertung zwischen 0.0 und 1.0
        """
        if not query_text:
            return 0.0
        
        # Text normalisieren
        query_lower = query_text.lower()
        hotel_text = f"{hotel_name} {location}".lower()
        query_words = query_lower.split()
        
        relevance_score = 0.0
        
        # Direkte Wort-Übereinstimmungen
        for word in query_words:
            if len(word) > 2:
                if word in hotel_text:
                    relevance_score += 0.25
        
        # Semantische Keywords für verschiedene Reisearten
        keyword_groups = {
            'family': ['family', 'kids', 'children', 'playground', 'pool', 'vacation'],
            'luxury': ['spa', 'suite', 'premium', 'luxury', 'boutique', 'resort', 'deluxe'],
            'business': ['business', 'center', 'conference', 'corporate', 'executive'],
            'beach': ['beach', 'ocean', 'sea', 'coastal', 'waterfront', 'bay'],
            'city': ['city', 'urban', 'downtown', 'center', 'central'],
            'romantic': ['romantic', 'honeymoon', 'couple', 'intimate', 'private']
        }
        
        # Bonus für semantische Übereinstimmungen
        for category, keywords in keyword_groups.items():
            query_matches = sum(1 for word in keywords if word in query_lower)
            hotel_matches = sum(1 for word in keywords if word in hotel_text)
            
            if query_matches > 0 and hotel_matches > 0:
                semantic_bonus = min(query_matches * hotel_matches * 0.15, 0.3)
                relevance_score += semantic_bonus
        
        return min(relevance_score, 1.0)
    
    # Die simulierte Bildrelevanz-Funktion entfällt, da jetzt der CNN-Predictor verwendet wird.
    
    def _calculate_rating_score(self, rating_value):
        """
        Intelligente Rating-Bewertung mit Bonus für Exzellenz
        
        Args:
            rating_value: Hotel-Bewertung
            
        Returns:
            rating_score: Bewertung zwischen 0.0 und 1.0
        """
        if rating_value <= 0:
            return 0.0
        
        # Basis-Score
        base_score = rating_value / 10.0
        
        # Bonus für sehr hohe Bewertungen
        if rating_value >= 9.5:
            return min(base_score + 0.15, 1.0)  # Exzellenz-Bonus
        elif rating_value >= 9.0:
            return min(base_score + 0.1, 1.0)   # Premium-Bonus
        elif rating_value >= 8.5:
            return min(base_score + 0.05, 1.0)  # Qualitäts-Bonus
        
        return base_score
    
    def _calculate_location_score(self, location):
        """
        Locations-Bonus für beliebte und hochwertige Destinationen
        
        Args:
            location: Standort des Hotels
            
        Returns:
            location_score: Bewertung zwischen 0.0 und 1.0
        """
        if not location:
            return 0.5  # Neutral wenn keine Location
        
        location_lower = location.lower()
        
        # Premium-Destinationen (höchster Bonus)
        premium_destinations = [
            'paris', 'london', 'new york', 'tokyo', 'singapore', 'zurich',
            'geneva', 'monaco', 'dubai', 'hong kong', 'sydney', 'san francisco'
        ]
        
        # Beliebte Städte (mittlerer Bonus)
        popular_cities = [
            'berlin', 'amsterdam', 'barcelona', 'rome', 'madrid', 'vienna',
            'prague', 'budapest', 'stockholm', 'copenhagen', 'oslo', 'helsinki'
        ]
        
        # Urlaubs-Destinationen (spezieller Bonus)
        vacation_spots = [
            'bali', 'santorini', 'mykonos', 'ibiza', 'maldives', 'seychelles',
            'caribbean', 'hawaii', 'tahiti', 'mauritius', 'phuket', 'goa'
        ]
        
        # Check für Premium-Destinationen
        for dest in premium_destinations:
            if dest in location_lower:
                return 0.9
        
        # Check für beliebte Städte
        for city in popular_cities:
            if city in location_lower:
                return 0.75
        
        # Check für Urlaubs-Destinationen
        for spot in vacation_spots:
            if spot in location_lower:
                return 0.8
        
        # Standard-Score für andere Locations
        return 0.5
    
    def _ensure_diversity(self, sorted_hotels, top_k, max_price):
        """
        Stellt sicher, dass die Empfehlungen vielfältig sind (verschiedene Preisklassen, Locations)
        
        Args:
            sorted_hotels: Nach Score sortierte Hotels
            top_k: Anzahl gewünschter Empfehlungen
            max_price: Maximales Budget
            
        Returns:
            diverse_recommendations: Diversifizierte Hotel-Liste
        """
        if len(sorted_hotels) <= top_k:
            return sorted_hotels
        
        # Definiere Preisklassen basierend auf Budget
        if max_price <= 200:
            price_ranges = [(0, 80), (80, 150), (150, max_price)]
        elif max_price <= 500:
            price_ranges = [(0, 150), (150, 350), (350, max_price)]
        elif max_price <= 1000:
            price_ranges = [(0, 300), (300, 600), (600, max_price)]
        else:
            price_ranges = [(0, 500), (500, 1000), (1000, max_price)]
        
        diverse_hotels = []
        used_locations = set()
        
        # Erste Runde: Beste Hotels aus jeder Preisklasse
        for price_min, price_max in price_ranges:
            for hotel in sorted_hotels:
                if (len(diverse_hotels) < top_k and 
                    price_min <= hotel['price'] <= price_max and
                    hotel not in diverse_hotels):
                    
                    # Prüfe Location-Diversität
                    location_key = hotel['location'].split(',')[0].strip().lower()
                    if location_key not in used_locations or len(diverse_hotels) < 2:
                        diverse_hotels.append(hotel)
                        used_locations.add(location_key)
                        break
        
        # Zweite Runde: Fülle auf mit besten verfügbaren Hotels
        for hotel in sorted_hotels:
            if len(diverse_hotels) >= top_k:
                break
            if hotel not in diverse_hotels:
                diverse_hotels.append(hotel)
        
        return diverse_hotels[:top_k]

def get_user_input_with_default(prompt, default_value, value_type=str):
    """Hilfsfunktion für Benutzereingaben mit Standardwerten"""
    try:
        user_input = input(f"{prompt} (Standard: {default_value}): ").strip()
        if not user_input:
            return default_value
        return value_type(user_input)
    except ValueError:
        print(f"⚠️ Ungültige Eingabe, verwende Standardwert: {default_value}")
        return default_value

def run_interactive_demo():
    """Interaktive Demo mit vollständigen Filteroptionen"""
    print("\n🚀 TravelHunters: Interaktive Multi-Modal Demo")
    print("=" * 55)
    
    try:
        # Erstelle MultiModalRecommender-Instanz
        print("⚙️ Initialisiere Multi-Modal Recommender...")
        
        recommender = MultiModalRecommender(
            adaptive_weighting=True,
            image_weight=0.35,
            text_weight=0.65,
            embedding_dim=256
        )
        
        print("🤖 Adaptive Gewichtung aktiviert - automatische Feature-Erkennung")
        
        # Zeige verfügbare Optionen
        print("\n📋 Verfügbare Suchoptionen:")
        print("   • Nur Text-Suche (automatisch 100% Text-Gewichtung)")
        print("   • Nur Bild-Suche (automatisch 100% Bild-Gewichtung)")  
        print("   • Multimodal (automatisch 65% Text, 35% Bild)")
        
        print("\n" + "=" * 55)
        print("📝 EINGABEOPTIONEN")
        print("=" * 55)
        
        # Suchanfrage
        query_text = input("\n🔍 Suchanfrage eingeben: ").strip()
        if not query_text:
            query_text = "hotel"
            print(f"   → Verwende Standard: '{query_text}'")
            
        # Bildpfad (optional)
        image_path = input("📷 Bildpfad (optional, Enter zum Überspringen): ").strip()
        if image_path and not os.path.exists(image_path):
            print("⚠️ Bildpfad nicht gefunden, überspringe Bild-Features")
            image_path = None
        
        print("\n" + "=" * 55)
        print("🎛️ FILTEREINSTELLUNGEN")
        print("=" * 55)
        
        # Maximaler Preis
        max_price = get_user_input_with_default(
            "\n💰 Maximaler Preis pro Nacht ($)", 
            300, 
            float
        )
        
        # Minimale Bewertung
        min_rating = get_user_input_with_default(
            "⭐ Minimale Bewertung (0-10)", 
            7.0, 
            float
        )
        
        # Anzahl Empfehlungen
        top_k = get_user_input_with_default(
            "📊 Anzahl Empfehlungen (max 5)", 
            5, 
            int
        )
        top_k = min(5, max(1, top_k))  # Begrenzen auf 1-5
        
        print("\n" + "=" * 55)
        print("🔄 EMPFEHLUNGSSUCHE")
        print("=" * 55)
        
        print(f"\n📋 Suchparameter:")
        print(f"   🔍 Anfrage: '{query_text}'")
        if image_path:
            print(f"   🖼️ Bild: {os.path.basename(image_path)}")
        print(f"   💰 Max. Preis: ${max_price}")
        print(f"   ⭐ Min. Bewertung: {min_rating}/10")
        print(f"   📊 Anzahl: {top_k}")
        
        try:
            print(f"\n🤖 Automatische Feature-Erkennung läuft...")
            recommendations = recommender.get_automatic_recommendations(
                query_text=query_text,
                image_path=image_path if image_path else None,
                max_price=max_price,
                min_rating=min_rating,
                top_k=top_k
            )
            
            if recommendations and len(recommendations) > 0:
                print(f"\n✅ {len(recommendations)} Empfehlungen gefunden:")
                print("=" * 55)
                
                for i, hotel in enumerate(recommendations, 1):
                    print(f"\n🏨 {i}. {hotel.get('hotel_name', 'Hotel')}")
                    print(f"   📍 Ort: {hotel.get('location', 'Unbekannt')}")
                    print(f"   ⭐ Bewertung: {hotel.get('rating', 'N/A')}/10")
                    print(f"   💰 Preis: ${hotel.get('price', 'N/A')} pro Nacht")
                    if hotel.get('url'):
                        print(f"   🔗 Link: {hotel['url'][:50]}...")
                    
                print("\n" + "=" * 55)
                print("✅ Suche abgeschlossen!")
                
            else:
                print("\n❌ Keine Empfehlungen gefunden")
                print("💡 Versuchen Sie weniger strenge Filter:")
                print(f"   • Höheren Maximalpreis (aktuell: ${max_price})")
                print(f"   • Niedrigere Mindestbewertung (aktuell: {min_rating})")
                
        except Exception as rec_error:
            print(f"\n❌ Fehler bei Empfehlungssuche: {rec_error}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_interactive_demo()
