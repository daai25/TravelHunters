#!/usr/bin/env python3
"""
Quick test for hybrid model duplicate removal
"""

import sys
import os
sys.path.append('/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/modelling/data_preparation')
sys.path.append('/Users/leonakryeziu/PycharmProjects/SummerSchool/TravelHunters/modelling/models')

from load_data import HotelDataLoader
from feature_engineering import HotelFeatureEngineer
from text_similarity_model import TextBasedRecommender
from parameter_model import ParameterBasedRecommender

def test_hybrid_duplicates():
    """Test that hybrid recommendations don't contain duplicates"""
    print("🔍 Testing Hybrid Model Duplicate Removal...")
    
    # Load data
    loader = HotelDataLoader()
    hotels_df = loader.load_hotels()
    print(f"✅ Loaded {len(hotels_df)} hotels")
    
    # Engineer features
    engineer = HotelFeatureEngineer()
    features_df, _ = engineer.prepare_parameter_features(hotels_df)
    print(f"✅ Engineered features for {len(features_df)} hotels")
    
    # Create models
    param_model = ParameterBasedRecommender()
    text_model = TextBasedRecommender(max_features=500)
    
    # Generate mock interactions
    interactions_df = loader.load_user_interactions()
    
    # Train parameter model
    X_param, y_param = param_model.prepare_training_data(features_df, interactions_df)
    param_model.train(X_param, y_param)
    print("✅ Parameter model trained")
    
    # Train text model
    text_model.fit(hotels_df)
    print("✅ Text model trained")
    
    # Test preferences
    user_preferences = {
        'max_price': 300,
        'min_rating': 7.0,
        'required_amenities': ['wifi', 'pool'],
        'text_importance': 0.6,
        'price_importance': 0.2,
        'rating_importance': 0.2
    }
    
    # Get parameter recommendations
    param_recs = param_model.recommend_hotels(features_df, user_preferences, top_k=10)
    print(f"📊 Parameter model: {len(param_recs)} recommendations")
    if len(param_recs) > 0:
        print("Parameter recommendations (first 3):")
        for i, (_, hotel) in enumerate(param_recs.head(3).iterrows()):
            print(f"  {i+1}. {hotel.get('hotel_name', hotel.get('name', 'Unknown'))[:50]} - €{hotel.get('price', 0):.0f}")
    
    # Get text recommendations
    query = "luxury spa hotel with pool"
    text_recs = text_model.recommend_hotels(query, hotels_df, user_preferences, top_k=10)
    print(f"📝 Text model: {len(text_recs)} recommendations")
    if len(text_recs) > 0:
        print("Text recommendations (first 3):")
        for i, (_, hotel) in enumerate(text_recs.head(3).iterrows()):
            print(f"  {i+1}. {hotel.get('name', 'Unknown')[:50]} - €{hotel.get('price', 0):.0f}")
    
    # Manual hybrid combination (weighted sum)
    from hybrid_model import HybridRecommender
    hybrid = HybridRecommender()
    
    # Simulate the weighted sum combination
    if len(param_recs) > 0 and len(text_recs) > 0:
        hybrid_result = hybrid._weighted_sum_combination(param_recs, text_recs, top_k=10)
        print(f"🔄 Hybrid model: {len(hybrid_result)} recommendations")
        
        # Check for duplicates by name
        hotel_names = hybrid_result['hotel_name'].tolist()
        unique_names = list(set(hotel_names))
        
        print(f"🔍 Total recommendations: {len(hotel_names)}")
        print(f"🔍 Unique hotel names: {len(unique_names)}")
        
        if len(hotel_names) == len(unique_names):
            print("✅ No duplicates found in hybrid recommendations!")
        else:
            print("❌ Duplicates found in hybrid recommendations!")
            duplicates = [name for name in hotel_names if hotel_names.count(name) > 1]
            print(f"Duplicate names: {set(duplicates)}")
        
        print("\nHybrid recommendations:")
        for i, (_, hotel) in enumerate(hybrid_result.head(5).iterrows()):
            print(f"  {i+1}. {hotel.get('hotel_name', 'Unknown')[:50]} - €{hotel.get('price', 0):.0f} - Score: {hotel.get('hybrid_score', 0):.3f}")
    
    else:
        print("❌ Not enough recommendations from individual models to test hybrid")

if __name__ == "__main__":
    test_hybrid_duplicates()
