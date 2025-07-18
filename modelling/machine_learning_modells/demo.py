"""
Demo Interface for TravelHunters Hotel Recommender System
Simple interactive demo to test the recommendation models
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir / "models"))
sys.path.append(str(current_dir / "data_preparation"))
sys.path.append(str(current_dir / "evaluation"))

from load_data import HotelDataLoader
from feature_engineering import HotelFeatureEngineer
from parameter_model import ParameterBasedRecommender
from text_similarity_model import TextBasedRecommender
from hybrid_model import HybridRecommender
from metrics import RecommenderEvaluator
try:
    from data_preparation.matrix_builder import RecommendationMatrixBuilder
except ImportError:
    RecommendationMatrixBuilder = None

class TravelHuntersDemo:
    """Interactive demo for the hotel recommendation system"""
    
    def __init__(self, enable_data_augmentation: bool = True, models_dir: str = None):
        self.loader = HotelDataLoader()
        self.engineer = HotelFeatureEngineer()
        self.evaluator = RecommenderEvaluator()
        self.matrix_builder = RecommendationMatrixBuilder() if RecommendationMatrixBuilder else None
        
        # Verzeichnis für gespeicherte Modelle
        if models_dir is None:
            self.models_dir = Path(current_dir) / "saved_models"
        else:
            self.models_dir = Path(models_dir)
        
        # Modell-Dateipfade
        self.param_model_path = self.models_dir / "param_model.joblib"
        self.text_model_path = self.models_dir / "text_model.joblib"
        self.hybrid_model_path = self.models_dir / "hybrid_model.joblib"
        
        # Data augmentation settings
        self.enable_data_augmentation = enable_data_augmentation
        
        # Models
        self.param_model = None
        self.text_model = None
        self.hybrid_model = None
        
        # Data
        self.hotels_df = None
        self.interactions_df = None
        self.features_df = None
        
        self.is_initialized = False
    
    def _safe_float_convert(self, input_str, default_value):
        """
        Konvertiert eine Eingabezeichenfolge sicher in einen Gleitkommawert.
        Behandelt leere Eingaben und ersetzt Kommas durch Punkte für Dezimalwerte.
        
        Args:
            input_str: Die zu konvertierende Eingabezeichenfolge
            default_value: Standardwert, wenn die Eingabe leer ist
            
        Returns:
            Der konvertierte Gleitkommawert
        """
        if not input_str.strip():
            return default_value
        # Ersetze Kommas durch Punkte für Dezimalwerte
        cleaned_input = input_str.strip().replace(',', '.')
        return float(cleaned_input)
    
    def initialize(self):
        """Initialize the demo with data and models"""
        print("🚀 Initializing TravelHunters Demo...")
        
        # Load data
        print("📊 Loading hotel data...")
        self.hotels_df = self.loader.load_hotels()
        
        # Apply data augmentation if enabled
        if self.enable_data_augmentation:
            print("🔄 Data augmentation enabled - generating more training data...")
            
            # Augment hotel data for more variety (optional)
            # self.hotels_df = self.loader.augment_hotel_features(self.hotels_df, augmentation_factor=2)
            
            # Load interactions with data augmentation to increase training data
            self.interactions_df = self.loader.load_user_interactions(augment_data=True, augmentation_factor=4)
        else:
            # Load standard interactions without augmentation
            self.interactions_df = self.loader.load_user_interactions(augment_data=False)
        
        if self.hotels_df.empty:
            print("❌ No hotel data found. Please check the database.")
            return False
        
        # Engineer features
        print("⚙️ Engineering features...")
        self.features_df, _ = self.engineer.prepare_parameter_features(self.hotels_df)
        
        # Apply feature noise for improved robustness if augmentation is enabled
        if self.enable_data_augmentation:
            self.features_df = self.engineer.add_feature_noise(self.features_df, noise_level=0.03)
        
        # Versuche gespeicherte Modelle zu laden oder trainiere neu
        if not self._load_models():
            print("🔄 Keine gespeicherten Modelle gefunden oder Laden fehlgeschlagen. Training beginnt...")
            self._train_models()
            self._save_models()
        else:
            print("✅ Gespeicherte Modelle erfolgreich geladen!")
        
        self.is_initialized = True
        print("✅ Demo initialized successfully!")
        return True
    
    def _ensure_model_dir_exists(self):
        """Stelle sicher, dass das Modell-Verzeichnis existiert"""
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Modell-Verzeichnis erstellt: {self.models_dir}")
    
    def _load_models(self):
        """Lade gespeicherte Modelle wenn vorhanden"""
        # Prüfe ob alle erforderlichen Modelldateien existieren
        param_model_exists = self.param_model_path.exists()
        text_model_exists = self.text_model_path.exists()
        
        if not param_model_exists or not text_model_exists:
            print("⚠️ Nicht alle Modelle gefunden:")
            if not param_model_exists:
                print(f"  • Parameter-Modell nicht gefunden: {self.param_model_path}")
            if not text_model_exists:
                print(f"  • Text-Modell nicht gefunden: {self.text_model_path}")
            return False
        
        print("📂 Lade gespeicherte Modelle...")
        
        # Erfolgs-Flag für das Laden
        param_model_loaded = False
        text_model_loaded = False
        
        # Parameter-Modell laden
        try:
            self.param_model = ParameterBasedRecommender()
            self.param_model.load_model(str(self.param_model_path))
            print("  ✓ Parameter-Modell geladen")
            param_model_loaded = True
        except Exception as e:
            print(f"  ❌ Fehler beim Laden des Parameter-Modells: {e}")
        
        # Text-Modell laden
        try:
            self.text_model = TextBasedRecommender()
            self.text_model.load_model(str(self.text_model_path))
            print("  ✓ Text-Modell geladen")
            text_model_loaded = True
        except Exception as e:
            print(f"  ❌ Fehler beim Laden des Text-Modells: {e}")
            print("  🔄 Versuche ein neues Text-Modell zu initialisieren...")
            self.text_model = TextBasedRecommender(max_features=500)
        
        # Wenn mindestens ein Modell geladen werden konnte, initialisiere das Hybrid-Modell
        if param_model_loaded or text_model_loaded:
            try:
                # Hybrid-Modell initialisieren - wir initialisieren neu statt zu laden
                self.hybrid_model = HybridRecommender()
                
                # Wenn Parameter-Modell geladen wurde, setze es im Hybrid-Modell
                if param_model_loaded:
                    self.hybrid_model.param_recommender = self.param_model
                
                # Wenn Text-Modell geladen wurde, setze es im Hybrid-Modell
                if text_model_loaded:
                    self.hybrid_model.text_recommender = self.text_model
                
                # Setze Trainingsstatus basierend auf geladenen Modellen
                self.hybrid_model.is_trained = param_model_loaded and text_model_loaded
                
                # Initialisiere den Scaler mit den geladenen Daten
                try:
                    print("  🔄 Initialisiere den Scaler für das Hybrid-Modell...")
                    if 'rating' in self.features_df.columns:
                        sample_data = self.features_df[['rating']].values
                        self.hybrid_model.scaler.fit(sample_data)
                        print("  ✓ Hybrid-Modell Scaler mit realen Daten initialisiert")
                    else:
                        # Fallback mit Dummy-Daten
                        import numpy as np
                        self.hybrid_model.scaler.fit(np.array([[5.0], [10.0]]))
                        print("  ✓ Hybrid-Modell Scaler mit Dummy-Daten initialisiert")
                except Exception as e:
                    print(f"  ⚠️ Warnung bei Scaler-Initialisierung: {e}")
                
                print("  ✓ Hybrid-Modell mit geladenen Komponenten initialisiert")
                
                # Rückgabe abhängig davon, ob beide Modelle geladen wurden
                return param_model_loaded and text_model_loaded
            
            except Exception as e:
                print(f"  ❌ Fehler beim Initialisieren des Hybrid-Modells: {e}")
                return False
        else:
            print("  ❌ Keines der Modelle konnte geladen werden.")
            return False
    
    def _save_models(self):
        """Speichere trainierte Modelle"""
        try:
            self._ensure_model_dir_exists()
            print("💾 Speichere trainierte Modelle...")
            
            # Parameter-Modell speichern
            try:
                self.param_model.save_model(str(self.param_model_path))
                print(f"  ✓ Parameter-Modell gespeichert: {self.param_model_path}")
            except Exception as e:
                print(f"  ⚠️ Fehler beim Speichern des Parameter-Modells: {e}")
            
            # Text-Modell speichern
            try:
                success = self.text_model.save_model(str(self.text_model_path))
                if success:
                    print(f"  ✓ Text-Modell gespeichert: {self.text_model_path}")
                else:
                    print(f"  ⚠️ Text-Modell konnte nicht vollständig gespeichert werden")
            except Exception as e:
                print(f"  ⚠️ Fehler beim Speichern des Text-Modells: {e}")
                print("  🔄 Versuche mit vereinfachter Methode zu speichern...")
                try:
                    # Speichere nur die wichtigsten Attribute als Fallback
                    import joblib
                    minimal_data = {
                        'is_fitted': True,
                        'max_features': self.text_model.max_features,
                        'use_lsa': self.text_model.use_lsa,
                        'lsa_components': self.text_model.lsa_components,
                        'hotel_texts': self.text_model.hotel_texts if hasattr(self.text_model, 'hotel_texts') else [],
                        'hotel_ids': self.text_model.hotel_ids if hasattr(self.text_model, 'hotel_ids') else []
                    }
                    joblib.dump(minimal_data, str(self.text_model_path))
                    print(f"  ✓ Minimales Text-Modell gespeichert: {self.text_model_path}")
                except Exception as e2:
                    print(f"  ❌ Auch das minimale Speichern ist fehlgeschlagen: {e2}")
            
            # Hybrid-Modell wird nicht gespeichert, da es die anderen Modelle verwendet
            print("  ℹ️ Hybrid-Modell verwendet die gespeicherten Basis-Modelle")
            
            return True
        
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Modelle: {e}")
            return False
    
    def _train_models(self):
        """Train all recommendation models"""
        # Sparse Matrix erstellen
        print("  Erstelle Benutzer-Item-Interaktionsmatrix...")
        try:
            interaction_matrix = self.matrix_builder.build_interaction_matrix(
                self.interactions_df, user_col='user_id', item_col='hotel_id', rating_col='rating'
            )
            
            # Leave-One-Out-Splitting anwenden
            print("  Wende Leave-One-Out-Splitting an...")
            train_matrix, test_matrix = self.matrix_builder.split_leave_one_out(
                interaction_matrix, test_size=0.2, random_state=42
            )
            
            # Zurück zu DataFrames konvertieren für die bestehenden Modelle
            train_df = self.matrix_builder.convert_to_dataframe(train_matrix)
            test_df = self.matrix_builder.convert_to_dataframe(test_matrix)
            
            # Daten für das Training verwenden
            print("  Verwende aufgeteilte Matrix-Daten für das Training...")
            training_interactions = train_df
            
        except Exception as e:
            print(f"  ⚠️ Konnte Matrix-Daten nicht erstellen: {e}")
            print("  ⚠️ Verwende stattdessen die ursprünglichen Interaktionsdaten...")
            training_interactions = self.interactions_df
        
        # Parameter model
        print("  Training parameter model...")
        self.param_model = ParameterBasedRecommender(model_type='ridge')
        X_param, y_param = self.param_model.prepare_training_data(self.features_df, training_interactions)
        self.param_model.train(X_param, y_param)
        
        # Text model
        print("  Training text model...")
        self.text_model = TextBasedRecommender(max_features=500)
        self.text_model.fit(self.features_df)
        
        # Hybrid model
        print("  Training hybrid model...")
        self.hybrid_model = HybridRecommender()
        self.hybrid_model.train(self.hotels_df, training_interactions, self.features_df)
    
    def show_data_summary(self):
        """Display summary of available data"""
        if not self.is_initialized:
            print("❌ Demo not initialized. Call initialize() first.")
            return
        
        summary = self.loader.get_data_summary()
        
        print("\n📊 TravelHunters Data Summary")
        print("=" * 40)
        print(f"🏨 Total Hotels: {summary['n_hotels']}")
        print(f"👥 Synthetic Users: {summary['n_users']}")
        print(f"⭐ Average Rating: {summary['avg_rating']:.1f}/10.0")
        print(f"💰 Price Range: ${summary['price_range']['min']:.0f} - ${summary['price_range']['max']:.0f}")
        
        # Enhanced global coverage analysis
        print(f"\n🌍 Global Coverage Analysis:")
        print("=" * 40)
        
        locations = self.hotels_df['location'].value_counts()
        print(f"🗺️  Total Locations: {len(locations)}")
        
        # Analyze by continent/region
        regions = {
            '🇪🇺 Europe': ['madrid', 'berlin', 'prague', 'stockholm', 'copenhagen', 'barcelona', 'lisbon', 'helsinki', 'oslo', 'amsterdam', 'london', 'vienna', 'paris', 'rome', 'milan', 'florence', 'venice', 'budapest', 'warsaw', 'athens', 'dublin', 'zurich', 'brussels', 'munich', 'frankfurt', 'hamburg', 'cologne', 'lyon', 'marseille', 'nice', 'turin', 'bologna', 'naples', 'seville', 'valencia', 'bilbao', 'porto', 'malaga', 'palma', 'santorini', 'mykonos', 'crete', 'rhodes', 'reykjavik', 'tallinn', 'riga', 'vilnius', 'bucharest', 'sofia', 'belgrade', 'zagreb', 'ljubljana'],
            '🌎 Americas': ['las vegas', 'miami', 'cancun', 'rio de janeiro', 'lima', 'vancouver', 'san francisco', 'new york', 'toronto', 'montreal', 'buenos aires', 'bogota', 'santiago', 'mexico city', 'chicago', 'los angeles', 'playa del carmen', 'tulum'],
            '🌏 Asia': ['phuket', 'hong kong', 'kuala lumpur', 'new delhi', 'bangkok', 'tokyo', 'osaka', 'kyoto', 'seoul', 'singapore', 'manila', 'jakarta', 'mumbai', 'chennai', 'bangalore', 'beijing', 'shanghai', 'taipei', 'ho chi minh'],
            '🌍 Africa': ['marrakech', 'cairo', 'casablanca', 'cape town', 'johannesburg', 'nairobi', 'tunis', 'algiers', 'addis ababa', 'dar es salaam', 'lagos', 'accra'],
            '🇦🇺 Oceania': ['perth', 'melbourne', 'brisbane', 'auckland', 'sydney', 'adelaide', 'wellington', 'christchurch', 'gold coast', 'cairns']
        }
        
        for region_name, keywords in regions.items():
            region_locations = [loc for loc in locations.index 
                              if any(keyword in loc.lower() for keyword in keywords)]
            total_hotels = sum(locations[loc] for loc in region_locations)
            print(f"{region_name}: {len(region_locations)} cities, {total_hotels} hotels")
        
        print(f"\n📍 Top Global Hotel Destinations:")
        for i, (location, count) in enumerate(locations.head(10).items(), 1):
            # Add flag emojis based on location
            flag = ""
            if any(x in location.lower() for x in ['madrid', 'barcelona', 'seville', 'valencia', 'santiago de compostela']):
                flag = "🇪🇸"
            elif any(x in location.lower() for x in ['berlin', 'munich', 'frankfurt', 'hamburg', 'cologne']):
                flag = "🇩🇪"
            elif any(x in location.lower() for x in ['prague']):
                flag = "🇨🇿"
            elif any(x in location.lower() for x in ['stockholm', 'gothenburg']):
                flag = "🇸🇪"
            elif any(x in location.lower() for x in ['copenhagen']):
                flag = "🇩🇰"
            elif any(x in location.lower() for x in ['lisbon', 'porto']):
                flag = "🇵🇹"
            elif any(x in location.lower() for x in ['helsinki']):
                flag = "🇫🇮"
            elif any(x in location.lower() for x in ['oslo']):
                flag = "🇳🇴"
            elif any(x in location.lower() for x in ['amsterdam']):
                flag = "🇳🇱"
            elif any(x in location.lower() for x in ['london']):
                flag = "🇬🇧"
            elif any(x in location.lower() for x in ['vienna']):
                flag = "🇦🇹"
            elif any(x in location.lower() for x in ['las vegas', 'miami', 'san francisco']):
                flag = "🇺🇸"
            elif any(x in location.lower() for x in ['phuket']):
                flag = "🇹🇭"
            elif any(x in location.lower() for x in ['cancun', 'playa del carmen', 'tulum']):
                flag = "🇲🇽"
            elif any(x in location.lower() for x in ['rio de janeiro']):
                flag = "🇧🇷"
            elif any(x in location.lower() for x in ['lima']):
                flag = "🇵🇪"
            elif any(x in location.lower() for x in ['perth', 'melbourne', 'brisbane']):
                flag = "🇦🇺"
            elif any(x in location.lower() for x in ['auckland']):
                flag = "🇳🇿"
            elif any(x in location.lower() for x in ['marrakech', 'casablanca']):
                flag = "🇲🇦"
            elif any(x in location.lower() for x in ['cairo']):
                flag = "🇪🇬"
            elif any(x in location.lower() for x in ['hong kong']):
                flag = "🇭🇰"
            elif any(x in location.lower() for x in ['kuala lumpur']):
                flag = "🇲🇾"
            elif any(x in location.lower() for x in ['new delhi']):
                flag = "🇮🇳"
            elif any(x in location.lower() for x in ['vancouver']):
                flag = "🇨🇦"
            elif any(x in location.lower() for x in ['mykonos', 'athens']):
                flag = "🇬🇷"
            
            print(f"  {i:2d}. {flag} {location}: {count} hotels")
    
    def interactive_recommendation(self):
        """Interactive recommendation session"""
        if not self.is_initialized:
            print("❌ Demo not initialized. Call initialize() first.")
            return
        
        print("\n🎯 Hotel Recommendation Demo")
        print("=" * 50)
        
        while True:
            print("\nWhat type of recommendation would you like?")
            print("1. Parameter-based (budget, rating, amenities)")
            print("2. Text-based (describe your ideal hotel)")
            print("3. Hybrid (combines both approaches)")
            print("4. Compare all models")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self._parameter_recommendation()
            elif choice == '2':
                self._text_recommendation()
            elif choice == '3':
                self._hybrid_recommendation()
            elif choice == '4':
                self._compare_models()
            elif choice == '5':
                print("👋 Thank you for using TravelHunters!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
    
    def _parameter_recommendation(self):
        """Parameter-based recommendation interface"""
        print("\n🔢 Parameter-Based Recommendation")
        print("-" * 40)
        
        try:
            # Get user preferences
            price_input = input("Maximum price per night ($): ")
            max_price = self._safe_float_convert(price_input, 200)
            
            rating_input = input("Minimum rating (1-10): ")
            min_rating = self._safe_float_convert(rating_input, 7.0)
            
            print("\nWhich amenities are important to you? (y/n)")
            print("(These are preferences only and will improve recommendations, but are not required)")
            amenities = []
            for amenity in ['wifi', 'pool', 'gym', 'parking', 'breakfast', 'spa']:
                answer = input(f"  {amenity.title()}: ").lower()
                if answer in ['y', 'yes', 'j', 'ja']:
                    amenities.append(amenity)
            
            # Create preferences
            user_prefs = {
                'max_price': max_price,
                'min_rating': min_rating,
                'preferred_amenities': amenities,
                'price_importance': 0.25,
                'rating_importance': 0.25,
                'model_importance': 0.35,  # ML model gets significant weight
                'amenity_importance': 0.15 if amenities else 0.0  # Only add weight if amenities are selected
            }
            
            # Get recommendations
            recommendations = self.param_model.recommend_hotels(
                self.features_df, user_prefs, top_k=5
            )
            
            self._display_recommendations(recommendations, "Parameter-Based")
            
        except ValueError as e:
            print(f"❌ Invalid input. Please enter valid numbers. Error: {e}")
            print("Debug: Bitte überprüfen Sie Ihre Eingaben. Verwenden Sie Punkte (.) statt Kommas (,) für Dezimalwerte.")
    
    def _text_recommendation(self):
        """Text-based recommendation interface"""
        print("\n📝 Text-Based Hotel Recommendations")
        print("-" * 60)
        
        print("Describe your ideal hotel in natural language (English or German).")
        print("\nExamples of effective search queries:")
        print(" • \"Luxury beach hotel with pool and excellent restaurant\"")
        print(" • \"Budget-friendly family hotel in city center with WiFi\"") 
        print(" • \"Quiet resort with spa, wellness and beautiful views\"")
        print(" • \"Modern business hotel with conference rooms and airport shuttle\"")
        print(" • \"Beach hotel with kids club and all-inclusive service\"")
        print(" • \"Luxury city center hotel with pool and fine dining\"")
        print("\n🌍 Geographic search examples:")
        print(" • \"Relaxing spa hotel in Europe\"")
        print(" • \"Beach resort in Asia with family activities\"")
        print(" • \"Business hotel in Madrid or Barcelona\"")
        print(" • \"Budget hotel in Berlin city center\"")
        print(" • \"Romantic hotel in Paris or Vienna\"")
        
        print("\nTips for better results:")
        print(" • Describe important features (location, amenities, atmosphere)")
        print(" • Specify the purpose of your stay (family, business, relaxation)")
        print(" • Include geographic preferences (Europe, Asia, specific cities)")
        print(" • The more specific your query, the better the recommendations")
        print(" • Our system recognizes both English and German queries")
        print(" • Even with typos (e.g. \"luxory\" instead of \"luxury\") relevant hotels will be found")
        
        query = input("\nDescribe your ideal hotel: ").strip()
        if not query:
            print("❌ Please enter a description.")
            return
        
        # Geographic filtering
        print("\n🗺️ Geographic preferences (optional):")
        region_filter = input("Preferred region (Europe/Asia/Americas/Africa/Oceania/All): ").strip().lower()
        city_filter = input("Specific city/country (or leave empty): ").strip().lower()
        
        try:
            print("\nFilter the results (optional):")
            price_input = input("Maximum price per night ($): ")
            max_price = self._safe_float_convert(price_input, 1000)
            
            rating_input = input("Minimum rating (1-10): ")
            min_rating = self._safe_float_convert(rating_input, 0)
            
            # Ask for extended preferences
            print("\nHow important are the following factors to you? (1-10)")
            text_input = input("  Match with your description: ")
            text_importance = self._safe_float_convert(text_input, 7) / 10
            
            price_input = input("  Price-value ratio: ")
            price_importance = self._safe_float_convert(price_input, 3) / 10
            
            rating_input = input("  Hotel ratings: ")
            rating_importance = self._safe_float_convert(rating_input, 3) / 10
            
            # ML predictions weight set automatically
            model_importance = 0.4  # Default value of 0.4
            
            # Normalize weights
            total_weight = text_importance + price_importance + rating_importance + model_importance
            user_prefs = {
                'max_price': max_price,
                'min_rating': min_rating,
                'text_importance': text_importance / total_weight,
                'price_importance': price_importance / total_weight,
                'rating_importance': rating_importance / total_weight,
                'model_importance': model_importance / total_weight
            }
            
            # Apply geographic filtering before recommendation
            filtered_hotels = self._apply_geographic_filter(self.hotels_df, region_filter, city_filter)
            
            if filtered_hotels.empty:
                print(f"\n❌ No hotels found in the specified region/city.")
                print("Try a broader search or different location.")
                return
                
            print(f"\n🌍 Geographic filter applied: {len(filtered_hotels)} hotels in selected region")
            
            print("\n🔍 Suche nach passenden Hotels...")
            print(f"Anfrage wird verarbeitet: '{query}'")
            print("Dies kann einen Moment dauern...")
            
            recommendations = self.text_model.recommend_hotels(
                query, filtered_hotels, user_prefs, top_k=5
            )
            
            print(f"\n🔍 Suchanfrage: '{query}'")
            keywords = self.text_model.get_query_keywords(query, top_k=8)
            print(f"📋 Erkannte Schlüsselwörter: {', '.join(keywords)}")
            
            if recommendations.empty:
                print("\n❌ Leider wurden keine Hotels gefunden, die Ihren Kriterien entsprechen.")
                print("Tipps:")
                print(" • Versuchen Sie eine allgemeinere Beschreibung")
                print(" • Erhöhen Sie den maximalen Preis oder senken Sie die Mindestbewertung")
                print(" • Erweitern Sie die geografische Suche")
                print(" • Verwenden Sie weniger spezifische Anforderungen")
                return
            
            self._display_recommendations(recommendations, f"Text-Based ({region_filter.title() if region_filter and region_filter != 'all' else 'Global'})")
            
            # Show extended explanation for the recommendations
            if not recommendations.empty:
                # Overall summary
                avg_similarity = recommendations['similarity_score'].mean()
                price_range = f"${recommendations['price'].min():.0f} - ${recommendations['price'].max():.0f}"
                rating_range = f"{recommendations['rating'].min():.1f} - {recommendations['rating'].max():.1f}"
                
                print("\n📊 Results Summary:")
                print(f"  • Hotels found: {len(recommendations)}")
                print(f"  • Price range: {price_range}")
                print(f"  • Rating range: {rating_range}/10.0")
                if region_filter and region_filter != 'all':
                    print(f"  • Region: {region_filter.title()}")
                if city_filter:
                    print(f"  • Location filter: {city_filter.title()}")
                
                # Show geographic distribution
                locations = recommendations['location'].value_counts()
                print(f"  • Cities: {', '.join(locations.head(3).index)}")
                
                # ...existing code...
        except ValueError as e:
            print(f"❌ Invalid input. Please enter valid numbers. Error: {e}")
            print("Debug: Bitte überprüfen Sie Ihre Eingaben. Verwenden Sie Punkte (.) statt Kommas (,) für Dezimalwerte.")
    
    def _apply_geographic_filter(self, hotels_df, region_filter, city_filter):
        """Apply geographic filtering to hotels"""
        filtered_df = hotels_df.copy()
        
        # Define regional keywords
        region_keywords = {
            'europe': ['madrid', 'berlin', 'prague', 'stockholm', 'copenhagen', 'barcelona', 'lisbon', 
                      'helsinki', 'oslo', 'amsterdam', 'london', 'vienna', 'paris', 'rome', 'milan',
                      'florence', 'venice', 'budapest', 'warsaw', 'athens', 'dublin', 'zurich', 
                      'brussels', 'munich', 'frankfurt', 'hamburg', 'cologne', 'lyon', 'marseille',
                      'nice', 'turin', 'bologna', 'naples', 'seville', 'valencia', 'bilbao', 'porto',
                      'malaga', 'palma', 'santorini', 'mykonos', 'crete', 'rhodes', 'reykjavik',
                      'tallinn', 'riga', 'vilnius', 'bucharest', 'sofia', 'belgrade', 'zagreb', 'ljubljana'],
            'asia': ['phuket', 'hong kong', 'kuala lumpur', 'new delhi', 'bangkok', 'tokyo', 'osaka',
                    'kyoto', 'seoul', 'singapore', 'manila', 'jakarta', 'mumbai', 'chennai', 'bangalore',
                    'beijing', 'shanghai', 'taipei', 'ho chi minh', 'hanoi', 'phnom penh', 'yangon',
                    'kathmandu', 'colombo', 'male', 'dhaka', 'karachi', 'islamabad', 'tehran', 'dubai',
                    'abu dhabi', 'doha', 'kuwait', 'riyadh', 'jeddah', 'muscat', 'baku', 'tbilisi'],
            'americas': ['las vegas', 'miami', 'cancun', 'rio de janeiro', 'lima', 'vancouver', 
                        'san francisco', 'new york', 'toronto', 'montreal', 'buenos aires', 'bogota',
                        'santiago', 'mexico city', 'chicago', 'los angeles', 'playa del carmen', 'tulum',
                        'quebec', 'ottawa', 'calgary', 'edmonton', 'winnipeg', 'halifax', 'boston',
                        'washington', 'philadelphia', 'atlanta', 'dallas', 'houston', 'phoenix',
                        'denver', 'seattle', 'portland', 'sao paulo', 'brasilia', 'salvador',
                        'recife', 'fortaleza', 'belo horizonte', 'curitiba', 'porto alegre'],
            'africa': ['marrakech', 'cairo', 'casablanca', 'cape town', 'johannesburg', 'nairobi',
                      'tunis', 'algiers', 'addis ababa', 'dar es salaam', 'lagos', 'accra', 'kampala',
                      'kigali', 'lusaka', 'harare', 'maputo', 'windhoek', 'gaborone', 'maseru',
                      'mbabane', 'antananarivo', 'port louis', 'victoria', 'moroni', 'djibouti',
                      'asmara', 'khartoum', 'juba', 'ndjamena', 'bangui', 'libreville', 'malabo'],
            'oceania': ['perth', 'melbourne', 'brisbane', 'auckland', 'sydney', 'adelaide', 'wellington',
                       'christchurch', 'gold coast', 'cairns', 'darwin', 'hobart', 'canberra',
                       'hamilton', 'tauranga', 'palmerston north', 'dunedin', 'invercargill',
                       'rotorua', 'napier', 'nelson', 'queenstown', 'whangarei', 'gisborne']
        }
        
        # Apply region filter
        if region_filter and region_filter in region_keywords:
            keywords = region_keywords[region_filter]
            mask = filtered_df['location'].str.lower().str.contains('|'.join(keywords), na=False)
            filtered_df = filtered_df[mask]
        
        # Apply city filter
        if city_filter:
            mask = filtered_df['location'].str.lower().str.contains(city_filter, na=False)
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def _hybrid_recommendation(self):
        """Hybrid recommendation interface"""
        print("\n🔄 Hybrid Recommendation")
        print("-" * 40)
        
        query = input("Describe your ideal hotel: ").strip()
        if not query:
            print("❌ Please enter a description.")
            return
        
        try:
            # Preiseingabe
            price_input = input("Maximum price per night ($): ")
            max_price = self._safe_float_convert(price_input, 200)
            
            # Bewertungseingabe
            rating_input = input("Minimum rating (1-10): ")
            min_rating = self._safe_float_convert(rating_input, 4.0)
            
            # Warnung, wenn die Mindestbewertung zu hoch ist
            if min_rating > 9.0:
                print("\n⚠️ Hinweis: Eine Mindestbewertung von {:.1f} ist sehr hoch und könnte zu wenigen oder keinen Ergebnissen führen.".format(min_rating))
                confirmation = input("Möchten Sie fortfahren? (j/n): ").lower()
                if confirmation != 'j' and confirmation != 'ja':
                    # Setze auf einen vernünftigeren Wert zurück
                    min_rating = 7.0
                    print(f"Mindestbewertung auf {min_rating:.1f} zurückgesetzt.")
            
            # Get weight preferences
            print("\nHow important are these factors? (1-10)")
            
            # Wichtigkeitseingaben mit verbesserter Konvertierung
            text_input = input("  Description match: ")
            text_importance = self._safe_float_convert(text_input, 7) / 10
            
            price_input = input("  Price considerations: ")
            price_importance = self._safe_float_convert(price_input, 3) / 10
            
            rating_input = input("  Hotel ratings: ")
            rating_importance = self._safe_float_convert(rating_input, 3) / 10
            
            # ML predictions weight set automatically
            model_importance = 0.4  # Default value of 0.4
            
            # Normalize weights
            total_weight = text_importance + price_importance + rating_importance + model_importance
            user_prefs = {
                'max_price': max_price,
                'min_rating': min_rating,
                'text_importance': text_importance / total_weight,
                'price_importance': price_importance / total_weight,
                'rating_importance': rating_importance / total_weight,
                'model_importance': model_importance / total_weight
            }
            
            # Set hybrid weights
            self.hybrid_model.set_weights(0.4, 0.6)  # Slightly favor text
            
            print("\n🔍 Suche nach passenden Hotels...")
            print(f"Query: '{query}'")
            print(f"Max. Preis: ${max_price:.0f}, Min. Bewertung: {min_rating:.1f}")
            print("Dies kann einen Moment dauern...")
            
            # Rufe die Empfehlungen mit Fehlerbehandlung ab
            try:
                recommendations = self.hybrid_model.recommend_hotels(
                    query, self.hotels_df, self.features_df, user_prefs, top_k=5
                )
            except Exception as e:
                print(f"\n❌ Fehler bei der Empfehlungsgenerierung: {e}")
                print("Versuche mit Standardeinstellungen...")
                
                # Versuche mit Standardeinstellungen
                default_prefs = {
                    'max_price': 1000,
                    'min_rating': 0,
                    'text_importance': 0.5,
                    'price_importance': 0.2,
                    'rating_importance': 0.2,
                    'model_importance': 0.1
                }
                
                try:
                    recommendations = self.hybrid_model.recommend_hotels(
                        query, self.hotels_df, self.features_df, default_prefs, top_k=5
                    )
                except Exception as e2:
                    print(f"\n❌ Auch mit Standardeinstellungen fehlgeschlagen: {e2}")
                    print("Bitte versuchen Sie es mit einem anderen Suchbegriff oder weniger strengen Filtern.")
                    return
            
            print(f"\n🔍 Search Query: '{query}'")
            
            if recommendations.empty:
                print("\n❌ Leider wurden keine Hotels gefunden, die Ihren Kriterien entsprechen.")
                print("Tipps:")
                print(" • Versuchen Sie eine allgemeinere Beschreibung")
                print(" • Erhöhen Sie den maximalen Preis oder senken Sie die Mindestbewertung")
                print(" • Verwenden Sie weniger spezifische Anforderungen")
                return
            
            self._display_recommendations(recommendations, "Hybrid", score_col='hybrid_score')
            
            # Show explanation for top recommendation
            if not recommendations.empty:
                try:
                    hotel_id = recommendations.iloc[0]['hotel_id']
                    explanation = self.hybrid_model.explain_recommendation(
                        hotel_id, query, self.hotels_df, self.features_df, user_prefs
                    )
                    self._display_explanation(explanation)
                except Exception as e:
                    print(f"\n⚠️ Konnte keine detaillierte Erklärung generieren: {e}")
            
        except ValueError as e:
            print(f"❌ Invalid input. Please enter valid numbers. Error: {e}")
            # Debug-Informationen ausgeben
            print("Debug: Bitte überprüfen Sie Ihre Eingaben. Verwenden Sie Punkte (.) statt Kommas (,) für Dezimalwerte.")
    
    def _compare_models(self):
        """Compare all three models"""
        print("\n⚖️ Model Comparison")
        print("-" * 40)
        
        query = input("Enter a search query: ").strip()
        if not query:
            print("❌ Please enter a query.")
            return
        
        # Standardwerte für Benutzereinstellungen
        max_price = 200
        min_rating = 4.0
        
        try:
            # Optional: Abrufen benutzerdefinierter Einstellungen
            print("\nFilter options (optional, press Enter for defaults):")
            price_input = input(f"Maximum price per night (default: ${max_price}): ")
            if price_input.strip():
                max_price = self._safe_float_convert(price_input, max_price)
                
            rating_input = input(f"Minimum rating (default: {min_rating}): ")
            if rating_input.strip():
                min_rating = self._safe_float_convert(rating_input, min_rating)
        
            user_prefs = {
                'max_price': max_price,
                'min_rating': min_rating,
                'text_importance': 0.5,
                'price_importance': 0.2,
                'rating_importance': 0.2,
                'model_importance': 0.1
            }
            
            print(f"\n🔍 Query: '{query}'")
            print(f"💰 Budget: ${user_prefs['max_price']}")
            print(f"⭐ Min Rating: {user_prefs['min_rating']}")
            
            # Get recommendations from all models
            param_recs = self.param_model.recommend_hotels(self.features_df, user_prefs, top_k=3)
            text_recs = self.text_model.recommend_hotels(query, self.hotels_df, user_prefs, top_k=3)
            hybrid_recs = self.hybrid_model.recommend_hotels(
                query, self.hotels_df, self.features_df, user_prefs, top_k=3
            )
            
            # Display side by side
            print("\n📊 Top 3 Recommendations from Each Model:")
            print("=" * 80)
            
            models = [
                ("Parameter", param_recs, 'final_score'),
                ("Text", text_recs, 'final_score'),
                ("Hybrid", hybrid_recs, 'hybrid_score')
            ]
            
            for i in range(3):
                print(f"\nRank {i+1}:")
                for model_name, recs, score_col in models:
                    if i < len(recs):
                        hotel = recs.iloc[i]
                        # Sichere Namenszuweisung mit Typprüfung
                        if 'hotel_name' in hotel and isinstance(hotel['hotel_name'], str):
                            name = hotel['hotel_name']
                        elif 'name' in hotel and isinstance(hotel['name'], str):
                            name = hotel['name']
                        else:
                            name = f"Hotel {hotel.get('hotel_id', i)}"
                            
                        # Sichere Wertextraktion mit Fallbacks
                        try:
                            score = float(hotel.get(score_col, 0.0))
                            price = float(hotel.get('price', 0.0))
                            rating = float(hotel.get('rating', 0.0))
                        except (TypeError, ValueError):
                            score = 0.0
                            price = 0.0
                            rating = 0.0
                            
                        # Formatiere Name für Ausgabe mit Längenprüfung
                        display_name = name[:25] if isinstance(name, str) else str(name)
                        print(f"  {model_name:8}: {display_name:25} (Score: {score:.3f}, ${price:.0f}, {rating:.1f}⭐)")
                    else:
                        print(f"  {model_name:8}: {'No recommendation':25}")
                        
        except ValueError as e:
            print(f"❌ Invalid input. Please enter valid numbers. Error: {e}")
            print("Debug: Bitte überprüfen Sie Ihre Eingaben. Verwenden Sie Punkte (.) statt Kommas (,) für Dezimalwerte.")
    
    def _display_recommendations(self, recommendations: pd.DataFrame, model_name: str, 
                               score_col: str = 'final_score'):
        """Display recommendations in a formatted way"""
        if recommendations is None or recommendations.empty:
            print(f"\n❌ No recommendations found with your criteria.")
            return
        
        try:
            # Überprüfe, ob die benötigte Score-Spalte vorhanden ist
            if score_col not in recommendations.columns:
                print(f"⚠️ Warning: Score column '{score_col}' not found in results")
                # Verwende eine alternative Score-Spalte wenn verfügbar
                alternative_scores = [col for col in recommendations.columns 
                                     if col.endswith('_score') or col.endswith('Score')]
                if alternative_scores:
                    score_col = alternative_scores[0]
                    print(f"Using '{score_col}' instead")
                else:
                    # Erstelle einen Dummy-Score basierend auf der Position
                    recommendations['dummy_score'] = [1.0 - (0.05 * i) for i in range(len(recommendations))]
                    score_col = 'dummy_score'
                    print("Using position-based dummy scores")
            
            # Entferne Duplikate vor der Anzeige
            unique_recommendations = recommendations.copy()
            
            if 'name' in unique_recommendations.columns and 'location' in unique_recommendations.columns:
                unique_recommendations = unique_recommendations.drop_duplicates(subset=['name', 'location'], keep='first')
            
                # Wenn immer noch weniger als gewünscht, versuche nur nach Namen zu deduplizieren
                if len(unique_recommendations) < len(recommendations) * 0.8:
                    unique_recommendations = recommendations.drop_duplicates(subset=['name'], keep='first')
            
            print(f"\n🏨 Top {len(unique_recommendations)} {model_name} Recommendations:")
            print("=" * 60)
            
            # Stelle sicher, dass name/hotel_name konsistent ist
            if 'hotel_name' not in unique_recommendations.columns and 'name' in unique_recommendations.columns:
                unique_recommendations['hotel_name'] = unique_recommendations['name']
            elif 'name' not in unique_recommendations.columns and 'hotel_name' in unique_recommendations.columns:
                unique_recommendations['name'] = unique_recommendations['hotel_name']
            
            for i, (_, hotel) in enumerate(unique_recommendations.iterrows(), 1):
                # Hole den Hotelnamen mit Fallbacks und Typüberprüfung
                if 'hotel_name' in hotel and isinstance(hotel['hotel_name'], str):
                    name = hotel['hotel_name']
                elif 'name' in hotel and isinstance(hotel['name'], str):
                    name = hotel['name']
                else:
                    # Fallback für nicht-String-Werte oder fehlende Namen
                    name = f"Hotel {hotel.get('hotel_id', i)}"
                
                # Hole den Score mit Fallbacks
                if score_col in hotel:
                    try:
                        score = float(hotel[score_col])
                    except (TypeError, ValueError):
                        score = 0.0
                else:
                    score = 0.0
                
                # Hole weitere Informationen mit Fallbacks
                price = hotel.get('price', 0)
                rating = hotel.get('rating', 0)
                location = hotel.get('location', 'Unknown Location')
                
                # Verwende abgesicherte String-Manipulation
                display_name = name[:50] if isinstance(name, str) else str(name)
                location_str = str(location) if pd.notna(location) else "Unknown Location"
                
                print(f"\n{i}. {display_name}")
                print(f"   📍 {location_str}")
                print(f"   💰 ${price:.0f}/night")
                print(f"   ⭐ {rating:.1f}/10.0")
                print(f"   🎯 Score: {score:.3f}")
                
                # Show model contribution if hybrid model
                if model_name.lower() == "hybrid" and 'param_contrib' in hotel and 'text_contrib' in hotel:
                    param_percent = hotel['param_contrib'] * 100
                    text_percent = hotel['text_contrib'] * 100
                    print(f"   📊 Model Contribution: Parameter {param_percent:.0f}%, Text {text_percent:.0f}%")
                
                # Show description if available
                if 'description' in hotel and pd.notna(hotel['description']):
                    desc = str(hotel['description'])[:100] + "..." if len(str(hotel['description'])) > 100 else str(hotel['description'])
                    print(f"   📝 {desc}")
                    
                if i >= 5:  # Beschränke auf maximal 5 Empfehlungen für bessere Übersicht
                    break
                    
        except Exception as e:
            print(f"⚠️ Fehler bei der Anzeige der Empfehlungen: {e}")
            # Einfache Fallback-Anzeige
            try:
                print("\nEmpfehlungen (einfaches Format):")
                for i, (_, hotel) in enumerate(recommendations.iterrows(), 1):
                    hotel_info = []
                    for key in ['hotel_name', 'name', 'location', 'price', 'rating']:
                        if key in hotel and pd.notna(hotel[key]):
                            hotel_info.append(f"{key}: {hotel[key]}")
                    print(f"{i}. {', '.join(hotel_info)}")
                    if i >= 5:
                        break
            except:
                print("❌ Konnte Empfehlungen nicht anzeigen.")
    
    def _display_explanation(self, explanation: dict):
        """Display recommendation explanation with improved error handling"""
        if not explanation or not isinstance(explanation, dict):
            print("\n⚠️ Keine Erklärung für diese Empfehlung verfügbar.")
            return
        
        try:
            # Überprüfe, ob wichtige Schlüssel vorhanden sind
            if 'error' in explanation:
                print(f"\n⚠️ Fehler bei der Erklärungsgenerierung: {explanation['error']}")
                return
            
            # Sichere Namensextraktion mit Typprüfung
            hotel_name = explanation.get('hotel_name', 'Dieses Hotel')
            if not isinstance(hotel_name, str) or not hotel_name.strip():
                hotel_name = f"Hotel {explanation.get('hotel_id', 'ID unbekannt')}"
                
            print(f"\n💡 Why we recommended {hotel_name}:")
            print("-" * 50)
            
            # Preise und Bewertungen anzeigen
            try:
                hotel_price = explanation.get('hotel_price', None)
                if hotel_price is not None and isinstance(hotel_price, (int, float)):
                    print(f"💰 Price: ${hotel_price:.0f}")
                
                hotel_rating = explanation.get('hotel_rating', None)
                if hotel_rating is not None and isinstance(hotel_rating, (int, float)):
                    print(f"⭐ Rating: {hotel_rating:.1f}/10.0")
            except Exception:
                # Fehler beim Anzeigen von Preis/Bewertung ignorieren
                pass
            
            if 'explanations' not in explanation or not explanation['explanations']:
                print("Keine detaillierten Erklärungen verfügbar.")
                print("Das Hotel wurde basierend auf Ihrer Anfrage ausgewählt, aber detaillierte Begründungen können nicht angezeigt werden.")
                return
            
            # Modellbeiträge anzeigen
            if 'model_contributions' in explanation and isinstance(explanation['model_contributions'], dict):
                print("\n📊 Relevanz der Modelle:")
                try:
                    param_pct = explanation['model_contributions'].get('parameter_model', 50.0)
                    text_pct = explanation['model_contributions'].get('text_model', 50.0)
                    
                    if isinstance(param_pct, (int, float)) and isinstance(text_pct, (int, float)):
                        print(f"  • Parameter-basiert: {param_pct:.1f}%")
                        print(f"  • Text-basiert: {text_pct:.1f}%")
                        
                        if 'model_contributions_note' in explanation:
                            print(f"  Hinweis: {explanation['model_contributions_note']}")
                except Exception:
                    print("  • Konnte prozentuale Verteilung nicht anzeigen.")
            
            # Einzelne Modell-Erklärungen
            for exp in explanation['explanations']:
                if not isinstance(exp, dict):
                    continue
                    
                model_type = exp.get('model', 'Unknown')
                if isinstance(model_type, str):
                    if model_type.lower() == 'parameter':
                        model_emoji = '🔢'
                        model_type = "Parameter"
                    elif model_type.lower() == 'text':
                        model_emoji = '📝'
                        model_type = "Text"
                    else:
                        model_emoji = '🔍'
                        model_type = model_type.title()
                else:
                    model_emoji = '🔍'
                    model_type = "Unbekanntes Modell"
                
                # Sichere Konvertierung des Scores
                score_display = "N/A"
                try:
                    score = exp.get('score', None)
                    if score is not None:
                        score = float(score)
                        # Für Text-Modell, das 0-1 Score hat
                        if model_type.lower() == 'text' and score <= 1.0:
                            score_display = f"{score:.3f}/1.0"
                        # Für Parameter-Modell, das 0-10 Score hat
                        else:
                            score_display = f"{score:.2f}/10.0"
                except (TypeError, ValueError):
                    pass
                
                print(f"\n{model_emoji} {model_type} Model (Score: {score_display}):")
                
                reasons = exp.get('reasons', [])
                if not reasons or not isinstance(reasons, list):
                    print("    • Keine Details verfügbar")
                    continue
                    
                for reason in reasons:
                    if isinstance(reason, str):
                        print(f"    • {reason}")
                    else:
                        print(f"    • {str(reason)}")
            
            # Sichere Zusammenfassung
            summary = explanation.get('summary', "Dieses Hotel entspricht am besten Ihren angegebenen Präferenzen.")
            if not isinstance(summary, str) or not summary.strip():
                summary = "Dieses Hotel wurde basierend auf der Kombination von Preis, Bewertung und Beschreibungsmatch ausgewählt."
                
            print(f"\n📋 Summary: {summary}")
            
        except Exception as e:
            print(f"\n⚠️ Fehler bei der Anzeige der Erklärung: {e}")
            print("Detaillierte Erklärung konnte nicht angezeigt werden.")
            # Fallback-Erklärung
            print("\n💡 Allgemeine Erklärung:")
            print("Dieses Hotel wurde ausgewählt, weil es Ihren Preis- und Bewertungskriterien entspricht")
            print("und eine gute Übereinstimmung mit Ihrer Beschreibung aufweist.")
    
    def run_evaluation(self):
        """Run model evaluation"""
        if not self.is_initialized:
            print("❌ Demo not initialized. Call initialize() first.")
            return
        
        print("\n📊 Model Evaluation")
        print("=" * 40)
        
        try:
            # Versuche, geteilte Matrix-Daten zu erstellen
            print("  Erstelle Benutzer-Item-Interaktionsmatrix für die Evaluation...")
            interaction_matrix = self.matrix_builder.build_interaction_matrix(
                self.interactions_df, user_col='user_id', item_col='hotel_id', rating_col='rating'
            )
            
            # Leave-One-Out-Splitting anwenden
            print("  Wende Leave-One-Out-Splitting an...")
            train_matrix, test_matrix = self.matrix_builder.split_leave_one_out(
                interaction_matrix, test_size=0.2, random_state=42
            )
            
            # Konvertiere zurück zu DataFrames
            train_df = self.matrix_builder.convert_to_dataframe(train_matrix)
            test_df = self.matrix_builder.convert_to_dataframe(test_matrix)
            print(f"  Verwende {len(test_df)} Test-Interaktionen für die Evaluation...")
            
            # Evaluate parameter model mit den Testdaten
            X_test, y_test = self.param_model.prepare_training_data(self.features_df, test_df)
            y_pred_param = self.param_model.model.predict(X_test)
            param_metrics = self.evaluator.evaluate_regression_model(y_test, y_pred_param, "Parameter Model")
            
            print("\n🏆 Evaluation mit Leave-One-Out-Splitting (echter Testdatensatz):")
            
            # Empfehlungsbasierte Metriken berechnen
            # Zuerst eine Empfehlungsliste für jeden Benutzer im Testdatensatz erstellen
            test_users = test_df['user_id'].unique()
            
            if len(test_users) > 0:
                # Wähle einen Beispiel-Benutzer zur Demonstration
                sample_user = test_users[0]
                print(f"\n📋 Empfehlungsmetriken für Beispiel-Benutzer (ID: {sample_user}):")
                
                # Standardvorlieben erstellen
                user_prefs = {
                    'max_price': 1000,  # Hohe Grenze, um die meisten Hotels einzuschließen
                    'min_rating': 0,    # Niedrige Grenze, um die meisten Hotels einzuschließen
                    'preferred_amenities': [],
                    'price_importance': 0.2,
                    'rating_importance': 0.3,
                    'model_importance': 0.5,  # ML model gets significant weight
                    'amenity_importance': 0.0  # No amenity preferences in evaluation
                }
                
                # Hole Empfehlungen von allen Modellen für diesen Benutzer
                param_recommendations = self.param_model.recommend_hotels(
                    self.features_df, user_prefs, top_k=20)
                
                # Text-Anfrage simulieren (allgemeine Hotelsuche)
                query = "good hotel with comfortable rooms"
                text_recommendations = self.text_model.recommend_hotels(
                    query, self.hotels_df, user_prefs, top_k=20)
                
                # Hybrid-Empfehlungen
                hybrid_recommendations = self.hybrid_model.recommend_hotels(
                    query, self.hotels_df, self.features_df, user_prefs, top_k=20)
                
                # Ground-Truth-Daten für diesen Benutzer aus dem Testset
                user_ground_truth = test_df[test_df['user_id'] == sample_user]
                
                # Berechne und zeige Ranking-Metriken für jedes Modell
                for model_name, recs in [
                    ("Parameter-Modell", param_recommendations),
                    ("Text-Modell", text_recommendations),
                    ("Hybrid-Modell", hybrid_recommendations)
                ]:
                    # Berechne Ranking-Metriken
                    ranking_metrics = self.evaluator.evaluate_ranking_quality(
                        recs, user_ground_truth, k_values=[5, 10]
                    )
                    
                    # Zeige Metriken
                    print(f"\n🔍 {model_name}:")
                    print(f"  Precision@5: {ranking_metrics['precision@5']:.3f}")
                    print(f"  Recall@5: {ranking_metrics['recall@5']:.3f}")
                    print(f"  F1@5: {ranking_metrics['f1@5']:.3f}")
                    print(f"  NDCG@5: {ranking_metrics['ndcg@5']:.3f}")
                    print(f"  Precision@10: {ranking_metrics['precision@10']:.3f}")
                    print(f"  Recall@10: {ranking_metrics['recall@10']:.3f}")
                    print(f"  F1@10: {ranking_metrics['f1@10']:.3f}")
                    print(f"  NDCG@10: {ranking_metrics['ndcg@10']:.3f}")
            
                # Vergleiche die Modellstärken und -schwächen
                print("\n📊 Stärken und Schwächen der Modelle:")
                print("  • Parameter-Modell: Gut für numerische Features, erfordert aber ausreichend Trainingsdaten.")
                print("  • Text-Modell: Stark bei textbasierten Anfragen, aber kann relevante numerische Attribute übersehen.")
                print("  • Hybrid-Modell: Kombiniert die Stärken beider Ansätze, benötigt aber mehr Rechenleistung.")
                
        except Exception as e:
            print(f"  ⚠️ Konnte Matrix-Testdaten nicht erstellen: {e}")
            print("  ⚠️ Verwende stattdessen die ursprünglichen Daten für die Evaluation...")
            
            # Fallback zur ursprünglichen Evaluationsmethode
            X_param, y_param = self.param_model.prepare_training_data(self.features_df, self.interactions_df)
            y_pred_param = self.param_model.model.predict(X_param)
            param_metrics = self.evaluator.evaluate_regression_model(y_param, y_pred_param, "Parameter Model")
            
            print("\n🏆 Standard-Evaluation (ohne separaten Testdatensatz):")
        
        print(f"\n🔢 Parameter Model Performance:")
        print(f"  RMSE: {param_metrics['rmse']:.3f}")
        print(f"  R²: {param_metrics['r2']:.3f}")
        print(f"  Accuracy (±0.5): {param_metrics['accuracy_0.5']:.1%}")
        
        # Feature importance
        importance = self.param_model.get_feature_importance()
        print(f"\n📊 Top 5 Important Features:")
        for _, row in importance.head(5).iterrows():
            if 'abs_coefficient' in row:
                print(f"  {row['feature']}: {row['abs_coefficient']:.3f}")
            else:
                print(f"  {row['feature']}: {row.get('importance', 'N/A')}")
    
    def analyze_models(self):
        """Ausführliche Analyse der Modellstärken und -schwächen"""
        if not self.is_initialized:
            print("❌ Demo not initialized. Call initialize() first.")
            return
        
        print("\n🔬 Ausführliche Modellanalyse")
        print("=" * 40)
        
        # Kategorien für die Analyse definieren
        categories = {
            'Budget-Hotels': {'max_price': 100, 'min_rating': 0},
            'Luxus-Hotels': {'max_price': 1000, 'min_rating': 8},
            'Preis-Leistungs-Hotels': {'max_price': 150, 'min_rating': 7},
            'Familien-Hotels': {'query': 'family hotel with pool kids activities', 'max_price': 200},
            'Business-Hotels': {'query': 'business hotel conference center wifi', 'max_price': 300},
            'Resort-Hotels': {'query': 'resort beach spa relaxation', 'max_price': 400}
        }
        
        print("\n📋 Modell-Performance nach Kategorien:")
        
        for category_name, filters in categories.items():
            print(f"\n▶️ Kategorie: {category_name}")
            
            # Grundlegende Benutzervorlieben
            user_prefs = {
                'max_price': filters.get('max_price', 1000),
                'min_rating': filters.get('min_rating', 0),
                'preferred_amenities': [],  # No amenities for category analysis
                'price_importance': 0.2,
                'rating_importance': 0.3,
                'text_importance': 0.3,
                'model_importance': 0.2,
                'amenity_importance': 0.0
            }
            
            # Abfrage für textbasierte Modelle
            query = filters.get('query', 'good hotel')
            
            # Hole Empfehlungen von allen Modellen
            try:
                param_recs = self.param_model.recommend_hotels(self.features_df, user_prefs, top_k=5)
                
                text_recs = self.text_model.recommend_hotels(
                    query, self.hotels_df, user_prefs, top_k=5
                )
                
                hybrid_recs = self.hybrid_model.recommend_hotels(
                    query, self.hotels_df, self.features_df, user_prefs, top_k=5
                )
                
                # Vergleich zwischen den Modellen - Überschneidungen
                param_hotels = set(param_recs['hotel_id'])
                text_hotels = set(text_recs['hotel_id']) 
                hybrid_hotels = set(hybrid_recs['hotel_id'])
                
                overlap_param_text = len(param_hotels.intersection(text_hotels))
                overlap_param_hybrid = len(param_hotels.intersection(hybrid_hotels))
                overlap_text_hybrid = len(text_hotels.intersection(hybrid_hotels))
                
                print(f"  📊 Überschneidungen zwischen Modellen:")
                print(f"    • Parameter & Text: {overlap_param_text} Hotels")
                print(f"    • Parameter & Hybrid: {overlap_param_hybrid} Hotels")
                print(f"    • Text & Hybrid: {overlap_text_hybrid} Hotels")
                
                # Durchschnittspreise und -bewertungen
                for name, recs in [
                    ("Parameter", param_recs), 
                    ("Text", text_recs), 
                    ("Hybrid", hybrid_recs)
                ]:
                    avg_price = recs['price'].mean() if 'price' in recs.columns else 0
                    avg_rating = recs['rating'].mean() if 'rating' in recs.columns else 0
                    print(f"  📊 {name}-Modell - Durchschnittswerte:")
                    print(f"    • Preis: ${avg_price:.2f}")
                    print(f"    • Bewertung: {avg_rating:.1f}/10.0")
                
            except Exception as e:
                print(f"  ❌ Fehler bei der Analyse für '{category_name}': {e}")
        
        print("\n📝 Zusammenfassung der Modellstärken und -schwächen:")
        print("""
🔹 Parameter-Modell:
  ✅ Stärken:
    • Berücksichtigt quantitative Faktoren wie Preis und Bewertung effektiv
    • Reagiert gut auf harte Filter (z.B. Preislimits, Mindestbewertung)
    • Schneller als die anderen Modelle
  ❌ Schwächen:
    • Kann subtile Benutzerpräferenzen nicht erfassen
    • Berücksichtigt keine Textbeschreibungen oder semantische Ähnlichkeiten
    • Erfordert numerische Features für alle relevanten Eigenschaften

🔹 Text-Modell:
  ✅ Stärken:
    • Versteht natürliche Sprachbeschreibungen und Nuancen
    • Kann semantisch ähnliche Hotels finden
    • Funktioniert gut für spezifische Anfragen (z.B. "Strandhotel mit Spa")
  ❌ Schwächen:
    • Kann wichtige numerische Faktoren übersehen
    • Stärker abhängig von der Qualität der Hotelbeschreibungen
    • Langsamer als das Parameter-Modell

🔹 Hybrid-Modell:
  ✅ Stärken:
    • Kombiniert die Vorteile beider Ansätze
    • Ausgewogene Berücksichtigung von numerischen und Textfaktoren
    • Am besten für allgemeine Benutzeranforderungen geeignet
  ❌ Schwächen:
    • Am rechenintensivsten
    • Komplexere Konfiguration und Abstimmung erforderlich
    • Ergebnisse können schwieriger zu erklären sein
        """)

def main():
    """Main demo function"""
    # Enable data augmentation to increase training data size
    demo = TravelHuntersDemo(enable_data_augmentation=True)
    
    print("🏨 Welcome to TravelHunters Recommendation Demo!")
    print("🔄 Enhanced with Data Augmentation for Better Performance")
    print("=" * 60)
    
    if not demo.initialize():
        print("❌ Fehler bei der Initialisierung der Demo. Beende Programm.")
        return
    
    # Flag für neues Training während der Demo-Laufzeit
    new_training_done = False
    
    try:
        while True:
            print("\nWhat would you like to do?")
            print("1. View data summary")
            print("2. Get hotel recommendations")
            print("3. Run model evaluation")
            print("4. Analyze model strengths & weaknesses")
            print("5. Modelle neu trainieren und speichern")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                demo.show_data_summary()
            elif choice == '2':
                demo.interactive_recommendation()
            elif choice == '3':
                demo.run_evaluation()
            elif choice == '4':
                demo.analyze_models()
            elif choice == '5':
                print("\n🔄 Starte Neutraining der Modelle...")
                demo._train_models()
                success = demo._save_models()
                if success:
                    print("✅ Modelle wurden neu trainiert und gespeichert!")
                    new_training_done = True
                else:
                    print("⚠️ Modelle wurden trainiert, aber es gab Probleme beim Speichern.")
            elif choice == '6':
                print("👋 Thank you for using TravelHunters!")
                
                # Beim Beenden automatisch speichern, wenn ein neues Training gemacht wurde
                if new_training_done:
                    print("Modelle wurden bereits während der Sitzung gespeichert.")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Programm durch Benutzer unterbrochen")
        
    except Exception as e:
        print(f"\n\n❌ Unerwarteter Fehler: {e}")
        
    finally:
        print("\n👋 Auf Wiedersehen!")

if __name__ == "__main__":
    main()
