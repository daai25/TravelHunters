"""
Demo Interface for TravelHunters Hotel Recommender System
Simple interactive demo to test the recommendation models
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "models"))
sys.path.append(str(current_dir / "data_preparation"))
sys.path.append(str(current_dir / "evaluation"))

from load_data import HotelDataLoader
from feature_engineering import HotelFeatureEngineer
from parameter_model import ParameterBasedRecommender
from text_similarity_model import TextBasedRecommender
from hybrid_model import HybridRecommender
from metrics import RecommenderEvaluator

class TravelHuntersDemo:
    """Interactive demo for the hotel recommendation system"""
    
    def __init__(self):
        self.loader = HotelDataLoader()
        self.engineer = HotelFeatureEngineer()
        self.evaluator = RecommenderEvaluator()
        
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
        # Parameter model
        print("  Training parameter model...")
        self.param_model = ParameterBasedRecommender(model_type='ridge')
        X_param, y_param = self.param_model.prepare_training_data(self.features_df, self.interactions_df)
        self.param_model.train(X_param, y_param)
        
        # Text model
        print("  Training text model...")
        self.text_model = TextBasedRecommender(max_features=500)
        self.text_model.fit(self.features_df)
        
        # Hybrid model
        print("  Training hybrid model...")
        self.hybrid_model = HybridRecommender()
        self.hybrid_model.train(self.hotels_df, self.interactions_df, self.features_df)
    
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
        print("\n📝 Text-Based Recommendation")
        print("-" * 40)
        
        query = input("Describe your ideal hotel: ").strip()
        if not query:
            print("❌ Please enter a description.")
            return
        
        try:
            max_price = float(input("Maximum price per night ($, optional): ") or "1000")
            min_rating = float(input("Minimum rating (1-10, optional): ") or "0")
            
            user_prefs = {
                'max_price': max_price,
                'min_rating': min_rating,
                'text_importance': 0.7,
                'price_importance': 0.2,
                'rating_importance': 0.1
            }
            
            recommendations = self.text_model.recommend_hotels(
                query, self.hotels_df, user_prefs, top_k=5
            )
            
            print(f"\n🔍 Search Query: '{query}'")
            keywords = self.text_model.get_query_keywords(query, top_k=5)
            print(f"📋 Key Terms: {', '.join(keywords)}")
            
            self._display_recommendations(recommendations, "Text-Based")
            
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
        
        # Evaluate parameter model
        X_param, y_param = self.param_model.prepare_training_data(self.features_df, self.interactions_df)
        y_pred_param = self.param_model.model.predict(X_param)
        param_metrics = self.evaluator.evaluate_regression_model(y_param, y_pred_param, "Parameter Model")
        
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
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            demo.show_data_summary()
        elif choice == '2':
            demo.interactive_recommendation()
        elif choice == '3':
            demo.run_evaluation()
        elif choice == '4':
            print("👋 Thank you for using TravelHunters!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()
