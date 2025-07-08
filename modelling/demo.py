"""
Demo Interface for TravelHunters Hotel Recommender System
Simple interactive demo to test the recommendation models
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file            
            print("\n🔍 Searching for matching hotels...")
            print(f"Processing query: '{query}'")
            print("This may take a moment...")
            
            recommendations = self.text_model.recommend_hotels(
                query, self.hotels_df, user_prefs, top_k=5
            )
            
            print(f"\n🔍 Search query: '{query}'")
            keywords = self.text_model.get_query_keywords(query, top_k=8)
            print(f"📋 Recognized keywords: {', '.join(keywords)}")
            
            if recommendations.empty:
                print("\n❌ Unfortunately, no hotels were found that match your criteria.")
                print("Tips:")
                print(" • Try a more general description")
                print(" • Increase the maximum price or lower the minimum rating")
                print(" • Use fewer specific requirements")
                returnappend(str(current_dir / "models"))
sys.path.append(str(current_dir / "data_preparation"))
sys.path.append(str(current_dir / "evaluation"))

from load_data import HotelDataLoader
from feature_engineering import HotelFeatureEngineer
from parameter_model import ParameterBasedRecommender
from text_similarity_model import TextBasedRecommender
from hybrid_model import HybridRecommender
from metrics import RecommenderEvaluator
from data_preparation.matrix_builder import RecommendationMatrixBuilder  # Neue Klasse importieren

class TravelHuntersDemo:
    """Interactive demo for the hotel recommendation system"""
    
    def __init__(self):
        self.loader = HotelDataLoader()
        self.engineer = HotelFeatureEngineer()
        self.evaluator = RecommenderEvaluator()
        self.matrix_builder = RecommendationMatrixBuilder()  # Neue Instanz hinzufügen
        
        # Models
        self.param_model = None
        self.text_model = None
        self.hybrid_model = None
        
        # Data
        self.hotels_df = None
        self.interactions_df = None
        self.features_df = None
        
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the demo with data and models"""
        print("🚀 Initializing TravelHunters Demo...")
        
        # Load data
        print("📊 Loading hotel data...")
        self.hotels_df = self.loader.load_hotels()
        self.interactions_df = self.loader.load_user_interactions()
        
        if self.hotels_df.empty:
            print("❌ No hotel data found. Please check the database.")
            return False
        
        # Engineer features
        print("⚙️ Engineering features...")
        self.features_df, _ = self.engineer.prepare_parameter_features(self.hotels_df)
        
        # Initialize models
        print("🤖 Training models...")
        self._train_models()
        
        self.is_initialized = True
        print("✅ Demo initialized successfully!")
        return True
    
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
        print(f"📍 Sample Locations:")
        
        sample_locations = self.hotels_df['location'].value_counts().head(5)
        for location, count in sample_locations.items():
            print(f"    - {location}: {count} hotels")
    
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
            max_price = float(input("Maximum price per night ($): ") or "200")
            min_rating = float(input("Minimum rating (1-10): ") or "7.0")
            
            print("\nWhich amenities are important to you? (y/n)")
            amenities = []
            for amenity in ['wifi', 'pool', 'gym', 'parking', 'breakfast', 'spa']:
                answer = input(f"  {amenity.title()}: ").lower()
                if answer in ['y', 'yes']:
                    amenities.append(amenity)
            
            # Create preferences
            user_prefs = {
                'max_price': max_price,
                'min_rating': min_rating,
                'required_amenities': amenities,
                'price_importance': 0.4,
                'rating_importance': 0.4,
                'model_importance': 0.2
            }
            
            # Get recommendations
            recommendations = self.param_model.recommend_hotels(
                self.features_df, user_prefs, top_k=5
            )
            
            self._display_recommendations(recommendations, "Parameter-Based")
            
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
    
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
        
        print("\nTips for better results:")
        print(" • Describe important features (location, amenities, atmosphere)")
        print(" • Specify the purpose of your stay (family, business, relaxation)")
        print(" • The more specific your query, the better the recommendations")
        print(" • Our system recognizes both English and German queries")
        print(" • Even with typos (e.g. \"luxory\" instead of \"luxury\") relevant hotels will be found")
        
        query = input("\nDescribe your ideal hotel: ").strip()
        if not query:
            print("❌ Please enter a description.")
            return
        
        try:
            print("\nFilter the results (optional):")
            max_price = float(input("Maximum price per night ($): ") or "1000")
            min_rating = float(input("Minimum rating (1-10): ") or "0")
            
            # Ask for extended preferences
            print("\nHow important are the following factors to you? (1-10)")
            text_importance = float(input("  Match with your description: ") or "7") / 10
            price_importance = float(input("  Price-value ratio: ") or "5") / 10
            rating_importance = float(input("  Hotel ratings: ") or "6") / 10
            
            # Normalize weights
            total_weight = text_importance + price_importance + rating_importance
            user_prefs = {
                'max_price': max_price,
                'min_rating': min_rating,
                'text_importance': text_importance / total_weight,
                'price_importance': price_importance / total_weight,
                'rating_importance': rating_importance / total_weight
            }
            
            print("\n🔍 Suche nach passenden Hotels...")
            print(f"Anfrage wird verarbeitet: '{query}'")
            print("Dies kann einen Moment dauern...")
            
            recommendations = self.text_model.recommend_hotels(
                query, self.hotels_df, user_prefs, top_k=5
            )
            
            print(f"\n� Suchanfrage: '{query}'")
            keywords = self.text_model.get_query_keywords(query, top_k=8)
            print(f"📋 Erkannte Schlüsselwörter: {', '.join(keywords)}")
            
            if recommendations.empty:
                print("\n❌ Leider wurden keine Hotels gefunden, die Ihren Kriterien entsprechen.")
                print("Tipps:")
                print(" • Versuchen Sie eine allgemeinere Beschreibung")
                print(" • Erhöhen Sie den maximalen Preis oder senken Sie die Mindestbewertung")
                print(" • Verwenden Sie weniger spezifische Anforderungen")
                return
            
            self._display_recommendations(recommendations, "Text-Based")
            
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
                
                # Explanations for the weighting factors
                if 'text_contribution' in recommendations.columns:
                    top_hotel = recommendations.iloc[0]
                    hotel_name = top_hotel.get('name', 'Unknown')
                    
                    print(f"\n💡 Why we recommend \"{hotel_name}\":")
                    text_match = top_hotel.get('text_contribution', 0)
                    price_factor = top_hotel.get('price_contribution', 0)
                    rating_factor = top_hotel.get('rating_contribution', 0)
                    
                    # Erklärungen mit Prozentangaben
                    total_contribution = text_match + price_factor + rating_factor
                    if total_contribution > 0:
                        text_percent = (text_match / total_contribution) * 100
                        price_percent = (price_factor / total_contribution) * 100
                        rating_percent = (rating_factor / total_contribution) * 100
                        
                        # Erkannte Schlüsselwörter aus dem ersten Hotel extrahieren, falls verfügbar
                        hotel_text = top_hotel.get('hotel_text', '')
                        key_features = []
                        
                        # Wichtigste Merkmale extrahieren
                        feature_checks = [
                            ('pool', ['pool', 'swimming pool', 'schwimmbad']),
                            ('strand/meer', ['beach', 'sea', 'ocean', 'strand', 'meer']),
                            ('kinder/familie', ['family', 'children', 'kids', 'familie', 'kinder']),
                            ('spa/wellness', ['spa', 'wellness', 'massage']),
                            ('wifi/internet', ['wifi', 'wlan', 'internet']),
                            ('frühstück', ['breakfast', 'frühstück', 'buffet']),
                            ('zentrale lage', ['downtown', 'central', 'city centre', 'zentrum', 'innenstadt']),
                            ('luxus', ['luxury', 'luxurious', 'luxus', 'premium']),
                            ('günstig', ['affordable', 'budget', 'günstig', 'preiswert'])
                        ]
                        
                        # Überprüfe, welche Features im Hoteltext vorhanden sind
                        if hotel_text:
                            hotel_text_lower = hotel_text.lower()
                            for feature_name, terms in feature_checks:
                                if any(term in hotel_text_lower for term in terms):
                                    key_features.append(feature_name)
                        
                        print(f"  • Match with your description: {text_percent:.1f}%")
                        print(f"  • Price-value ratio: {price_percent:.1f}%")
                        print(f"  • Hotel rating: {rating_percent:.1f}%")
                        
                        # Show recognized features
                        if key_features:
                            print(f"  • Recognized features: {', '.join(key_features)}")
                            
                        # Price segment and quality category
                        price_val = top_hotel.get('price', 0)
                        if price_val < 100:
                            price_category = "Budget/Affordable"
                        elif price_val < 200:
                            price_category = "Mid-range"
                        elif price_val < 300:
                            price_category = "Upscale"
                        else:
                            price_category = "Luxury"
                            
                        rating_val = top_hotel.get('rating', 0)
                        if rating_val >= 9.0:
                            quality_category = "Outstanding"
                        elif rating_val >= 8.0:
                            quality_category = "Very good"
                        elif rating_val >= 7.0:
                            quality_category = "Good"
                        else:
                            quality_category = "Average"
                            
                        print(f"  • Price category: {price_category} (${price_val:.0f})")
                        print(f"  • Quality category: {quality_category} ({rating_val:.1f}/10.0)")
                    
                    # Explain specific recommendation factors
                    location = top_hotel.get('location', '')
                    price = top_hotel.get('price', 0)
                    rating = top_hotel.get('rating', 0)
                    
                    print("\n🔍 Key factors:")
                    if price <= max_price * 0.7:
                        print(f"  • Great price: ${price:.0f} per night (below your budget of ${max_price:.0f})")
                    if rating >= 8.5:
                        print(f"  • Excellent rating: {rating:.1f}/10.0")
                    if location and any(keyword.lower() in location.lower() for keyword in keywords):
                        print(f"  • Perfect location: {location}")
                        
                    # Additional recommendations
                    print("\n💬 For even better results:")
                    print("  • Try more precise keywords like \"beach\", \"pool\", \"center\"")
                    print("  • Specify if you're traveling with family, alone, or for business")
                    print("  • Combine text search with parameter-based model for more precise filters")
            
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
    
    def _hybrid_recommendation(self):
        """Hybrid recommendation interface"""
        print("\n🔄 Hybrid Recommendation")
        print("-" * 40)
        
        query = input("Describe your ideal hotel: ").strip()
        if not query:
            print("❌ Please enter a description.")
            return
        
        try:
            max_price = float(input("Maximum price per night ($): ") or "200")
            min_rating = float(input("Minimum rating (1-10): ") or "4.0")
            
            # Get weight preferences
            print("\nHow important are these factors? (1-10)")
            text_importance = float(input("  Description match: ") or "7") / 10
            price_importance = float(input("  Price considerations: ") or "5") / 10
            rating_importance = float(input("  Hotel ratings: ") or "8") / 10
            
            # Normalize weights
            total_weight = text_importance + price_importance + rating_importance
            user_prefs = {
                'max_price': max_price,
                'min_rating': min_rating,
                'text_importance': text_importance / total_weight,
                'price_importance': price_importance / total_weight,
                'rating_importance': rating_importance / total_weight
            }
            
            # Set hybrid weights
            self.hybrid_model.set_weights(0.4, 0.6)  # Slightly favor text
            
            recommendations = self.hybrid_model.recommend_hotels(
                query, self.hotels_df, self.features_df, user_prefs, 
                top_k=5, combination_method='weighted_sum'
            )
            
            print(f"\n🔍 Search Query: '{query}'")
            self._display_recommendations(recommendations, "Hybrid", score_col='hybrid_score')
            
            # Show explanation for top recommendation
            if not recommendations.empty:
                hotel_id = recommendations.iloc[0]['hotel_id']
                explanation = self.hybrid_model.explain_recommendation(
                    hotel_id, query, self.hotels_df, self.features_df, user_prefs
                )
                self._display_explanation(explanation)
            
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
    
    def _compare_models(self):
        """Compare all three models"""
        print("\n⚖️ Model Comparison")
        print("-" * 40)
        
        query = input("Enter a search query: ").strip()
        if not query:
            print("❌ Please enter a query.")
            return
        
        user_prefs = {
            'max_price': 200,
            'min_rating': 4.0,
            'text_importance': 0.6,
            'price_importance': 0.2,
            'rating_importance': 0.2
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
                    name = hotel.get('hotel_name', hotel.get('name', 'Unknown'))
                    score = hotel[score_col]
                    price = hotel['price']
                    rating = hotel['rating']
                    print(f"  {model_name:8}: {name[:25]:25} (Score: {score:.3f}, ${price:.0f}, {rating:.1f}⭐)")
                else:
                    print(f"  {model_name:8}: {'No recommendation':25}")
    
    def _display_recommendations(self, recommendations: pd.DataFrame, model_name: str, 
                               score_col: str = 'final_score'):
        """Display recommendations in a formatted way"""
        if recommendations.empty:
            print(f"\n❌ No recommendations found with your criteria.")
            return
        
        print(f"\n🏨 Top {len(recommendations)} {model_name} Recommendations:")
        print("=" * 60)
        
        for i, (_, hotel) in enumerate(recommendations.iterrows(), 1):
            name = hotel.get('hotel_name', hotel.get('name', 'Unknown Hotel'))
            score = hotel[score_col]
            price = hotel['price']
            rating = hotel['rating']
            location = hotel.get('location', 'Unknown Location')
            
            print(f"\n{i}. {name}")
            print(f"   📍 {location}")
            print(f"   💰 ${price:.0f}/night")
            print(f"   ⭐ {rating:.1f}/10.0")
            print(f"   🎯 Score: {score:.3f}")
            
            # Show description if available
            if 'description' in hotel and pd.notna(hotel['description']):
                desc = str(hotel['description'])[:100] + "..." if len(str(hotel['description'])) > 100 else str(hotel['description'])
                print(f"   📝 {desc}")
    
    def _display_explanation(self, explanation: dict):
        """Display recommendation explanation"""
        print(f"\n💡 Why we recommended {explanation['hotel_name']}:")
        print("-" * 50)
        
        for exp in explanation['explanations']:
            print(f"\n🔍 {exp['model'].title()} Model (Score: {exp['score']:.3f}):")
            for reason in exp['reasons']:
                print(f"    • {reason}")
        
        print(f"\n📋 Summary: {explanation['summary']}")
    
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
                    'required_amenities': [],
                    'price_importance': 0.3,
                    'rating_importance': 0.5,
                    'model_importance': 0.2
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
                'price_importance': 0.3,
                'rating_importance': 0.4,
                'text_importance': 0.3
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
    demo = TravelHuntersDemo()
    
    print("🏨 Welcome to TravelHunters Recommendation Demo!")
    print("=" * 50)
    
    if not demo.initialize():
        return
    
    while True:
        print("\nWhat would you like to do?")
        print("1. View data summary")
        print("2. Get hotel recommendations")
        print("3. Run model evaluation")
        print("4. Analyze model strengths & weaknesses")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            demo.show_data_summary()
        elif choice == '2':
            demo.interactive_recommendation()
        elif choice == '3':
            demo.run_evaluation()
        elif choice == '4':
            demo.analyze_models()
        elif choice == '5':
            print("👋 Thank you for using TravelHunters!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
