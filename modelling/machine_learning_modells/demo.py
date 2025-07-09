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
    
    def __init__(self, enable_data_augmentation: bool = True):
        self.loader = HotelDataLoader()
        self.engineer = HotelFeatureEngineer()
        self.evaluator = RecommenderEvaluator()
        self.matrix_builder = RecommendationMatrixBuilder() if RecommendationMatrixBuilder else None
        
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
            max_price = float(input("Maximum price per night ($): ") or "200")
            min_rating = float(input("Minimum rating (1-10): ") or "7.0")
            
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
            max_price = float(input("Maximum price per night ($): ") or "1000")
            min_rating = float(input("Minimum rating (1-10): ") or "0")
            
            # Ask for extended preferences
            print("\nHow important are the following factors to you? (1-10)")
            text_importance = float(input("  Match with your description: ") or "7") / 10
            price_importance = float(input("  Price-value ratio: ") or "3") / 10
            rating_importance = float(input("  Hotel ratings: ") or "3") / 10
            model_importance = float(input("  ML predictions: ") or "4") / 10
            
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
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
            
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
            max_price = float(input("Maximum price per night ($): ") or "200")
            min_rating = float(input("Minimum rating (1-10): ") or "4.0")
            
            # Get weight preferences
            print("\nHow important are these factors? (1-10)")
            text_importance = float(input("  Description match: ") or "7") / 10
            price_importance = float(input("  Price considerations: ") or "3") / 10
            rating_importance = float(input("  Hotel ratings: ") or "3") / 10
            model_importance = float(input("  ML predictions: ") or "4") / 10
            
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
            
            recommendations = self.hybrid_model.recommend_hotels(
                query, self.hotels_df, self.features_df, user_prefs, top_k=5
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
        
        # Entferne Duplikate vor der Anzeige
        unique_recommendations = recommendations.drop_duplicates(subset=['name', 'location'], keep='first')
        
        # Wenn immer noch weniger als gewünscht, versuche nur nach Namen zu deduplizieren
        if len(unique_recommendations) < len(recommendations) * 0.8:
            unique_recommendations = recommendations.drop_duplicates(subset=['name'], keep='first')
        
        print(f"\n🏨 Top {len(unique_recommendations)} {model_name} Recommendations:")
        print("=" * 60)
        
        for i, (_, hotel) in enumerate(unique_recommendations.iterrows(), 1):
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
                
            if i >= 5:  # Beschränke auf maximal 5 Empfehlungen für bessere Übersicht
                break
    
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
