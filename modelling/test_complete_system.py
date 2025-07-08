#!/usr/bin/env python3
"""
Complete System Test for TravelHunters ML Recommendation System
This script tests all components and provides a comprehensive overview.
"""

import sys
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """Print a test step"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def test_data_loading():
    """Test data loading from database"""
    print_step("1", "Testing Data Loading from Database")
    
    try:
        from data_preparation.load_data import HotelDataLoader
        
        loader = HotelDataLoader()
        start_time = time.time()
        hotels_df = loader.load_hotels()
        load_time = time.time() - start_time
        
        print(f"✅ Data loading successful!")
        print(f"📊 Hotels loaded: {len(hotels_df):,}")
        print(f"⏱️ Loading time: {load_time:.2f} seconds")
        
        if not hotels_df.empty:
            print(f"🏨 Sample hotel: {hotels_df.iloc[0]['name']}")
            print(f"💰 Price range: ${hotels_df['price'].min():.0f} - ${hotels_df['price'].max():.0f}")
            print(f"⭐ Rating range: {hotels_df['rating'].min():.1f} - {hotels_df['rating'].max():.1f}")
            print(f"📍 Locations: {hotels_df['location'].nunique()} different cities")
        
        return True, hotels_df
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False, None

def test_feature_engineering(hotels_df):
    """Test feature engineering"""
    print_step("2", "Testing Feature Engineering")
    
    try:
        from data_preparation.feature_engineering import HotelFeatureEngineer
        
        engineer = HotelFeatureEngineer()
        start_time = time.time()
        features_df = engineer.engineer_numerical_features(hotels_df)
        feature_time = time.time() - start_time
        
        print(f"✅ Feature engineering successful!")
        print(f"📊 Features created: {len(features_df.columns)} columns")
        print(f"⏱️ Processing time: {feature_time:.2f} seconds")
        
        # Show engineered features
        engineered_features = [col for col in features_df.columns if col not in hotels_df.columns]
        print(f"🔧 New features: {engineered_features[:5]}...")
        
        return True, features_df
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False, None

def test_parameter_model(hotels_df, features_df):
    """Test parameter-based model"""
    print_step("3", "Testing Parameter-Based Model")
    
    try:
        from models.parameter_model import ParameterBasedRecommender
        
        model = ParameterBasedRecommender()
        
        # Train model with available data
        start_time = time.time()
        model.train(features_df, hotels_df['rating'])
        train_time = time.time() - start_time
        
        print(f"✅ Parameter model training successful!")
        print(f"📊 Training features: {features_df.shape[1]} features")
        print(f"🎯 Training samples: {features_df.shape[0]} hotels")
        print(f"⏱️ Training time: {train_time:.2f} seconds")
        
        # Test recommendation
        user_prefs = {'max_price': 200, 'min_rating': 4.0, 'required_amenities': ['wifi']}
        recommendations = model.recommend_hotels(hotels_df, user_prefs)
        
        print(f"🎯 Sample recommendation test:")
        print(f"   Query: Hotels under $200, rating ≥4.0, with WiFi")
        print(f"   Results: {len(recommendations)} hotels found")
        
        if not recommendations.empty:
            top_hotel = recommendations.iloc[0]
            print(f"   Top result: {top_hotel['name']} (${top_hotel['price']}/night, {top_hotel['rating']}⭐)")
        return True, model
        
    except Exception as e:
        print(f"❌ Parameter model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_text_model(hotels_df):
    """Test text-based model"""
    print_step("4", "Testing Text-Based Model")
    
    try:
        from models.text_similarity_model import TextBasedRecommender
        
        model = TextBasedRecommender()
        
        # Fit model
        start_time = time.time()
        model.fit(hotels_df)
        fit_time = time.time() - start_time
        
        print(f"✅ Text model fitting successful!")
        print(f"📊 Hotels processed: {len(model.hotel_ids):,}")
        print(f"📝 TF-IDF features: {model.tfidf_matrix.shape[1] if model.tfidf_matrix is not None else 0}")
        print(f"⏱️ Fitting time: {fit_time:.2f} seconds")
        
        # Test text search
        test_queries = [
            "luxury spa hotel",
            "budget family hotel with pool",
            "business hotel near city center"
        ]
        
        print(f"🔍 Sample text searches:")
        for query in test_queries:
            results = model.recommend_hotels(query, hotels_df, {})
            if not results.empty:
                top_result = results.iloc[0]
                score = results.iloc[0].get('similarity_score', 'N/A')
                print(f"   '{query}' → {top_result['name']} (Score: {score})")
            else:
                print(f"   '{query}' → No results")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Text model test failed: {e}")
        return False, None

def test_hybrid_model(hotels_df, features_df):
    """Test hybrid model"""
    print_step("5", "Testing Hybrid Model")
    
    try:
        from models.hybrid_model import HybridRecommender
        
        model = HybridRecommender()
        
        # Train model
        start_time = time.time()
        # Create mock interactions for hybrid model - using proper column names
        name_col = 'name' if 'name' in hotels_df.columns else 'hotel_name'
        mock_interactions = hotels_df[[name_col, 'rating']].rename(columns={name_col: 'hotel_id', 'rating': 'user_rating'})
        model.train(hotels_df, mock_interactions, features_df)
        train_time = time.time() - start_time
        
        print(f"✅ Hybrid model training successful!")
        print(f"⏱️ Training time: {train_time:.2f} seconds")
        print(f"⚖️ Parameter weight: {model.param_weight:.1f}")
        print(f"📝 Text weight: {model.text_weight:.1f}")
        
        # Test hybrid recommendation
        user_prefs = {'max_price': 300, 'min_rating': 4.2, 'text_importance': 0.6}
        query = "luxury spa resort with pool"
        
        recommendations = model.recommend_hotels(query, hotels_df, features_df, user_prefs)
        
        print(f"🎯 Hybrid recommendation test:")
        print(f"   Query: '{query}'")
        print(f"   Preferences: ≤$300, ≥4.2⭐")
        print(f"   Results: {len(recommendations)} hotels found")
        
        if not recommendations.empty:
            top_hotel = recommendations.iloc[0]
            print(f"   Top result: {top_hotel['name']} (${top_hotel['price']}/night, {top_hotel['rating']}⭐)")
        
        return True, model
        
    except ImportError as e:
        print(f"❌ Hybrid model import failed: {e}")
        print("   Note: Hybrid model may depend on parameter/text models")
        return False, None
    except Exception as e:
        print(f"❌ Hybrid model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_evaluation_metrics():
    """Test evaluation framework"""
    print_step("6", "Testing Evaluation Framework")
    
    try:
        from evaluation.metrics import RecommenderEvaluator
        import numpy as np
        
        evaluator = RecommenderEvaluator()
        
        # Create mock evaluation data
        y_true = np.random.uniform(3, 5, 100)
        y_pred = y_true + np.random.normal(0, 0.5, 100)
        
        # Test regression metrics
        metrics = evaluator.evaluate_regression_model(y_true, y_pred, "Test Model")
        
        print(f"✅ Evaluation framework working!")
        print(f"📊 Metrics calculated:")
        print(f"   RMSE: {metrics['rmse']:.3f}")
        print(f"   R²: {metrics['r2']:.3f}")
        print(f"   MAE: {metrics['mae']:.3f}")
        print(f"   Accuracy (±0.5): {metrics['accuracy_0.5']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def test_demo_interface():
    """Test demo interface initialization"""
    print_step("7", "Testing Demo Interface")
    
    try:
        from demo import TravelHuntersDemo
        
        demo = TravelHuntersDemo()
        
        # Test initialization (but don't run full demo)
        print("✅ Demo class created successfully!")
        print("📱 Interactive demo available via: python demo.py")
        print("🎮 Demo features:")
        print("   - Data summary display")
        print("   - Hotel recommendations")
        print("   - Model evaluation")
        print("   - Model comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo interface test failed: {e}")
        return False

def run_complete_system_test():
    """Run complete system test"""
    print_header("TravelHunters ML System - Complete Test Suite")
    
    start_time = time.time()
    results = []
    
    # Test 1: Data Loading
    success, hotels_df = test_data_loading()
    results.append(("Data Loading", success))
    if not success or hotels_df is None:
        print("\n❌ Cannot proceed without data. Please check database setup.")
        return
    
    # Test 2: Feature Engineering
    success, features_df = test_feature_engineering(hotels_df)
    results.append(("Feature Engineering", success))
    if not success:
        features_df = hotels_df  # Use original data as fallback
    
    # Test 3: Parameter Model
    success, param_model = test_parameter_model(hotels_df, features_df)
    results.append(("Parameter Model", success))
    
    # Test 4: Text Model
    success, text_model = test_text_model(hotels_df)
    results.append(("Text Model", success))
    
    # Test 5: Hybrid Model
    success, hybrid_model = test_hybrid_model(hotels_df, features_df)
    results.append(("Hybrid Model", success))
    
    # Test 6: Evaluation
    success = test_evaluation_metrics()
    results.append(("Evaluation Framework", success))
    
    # Test 7: Demo Interface
    success = test_demo_interface()
    results.append(("Demo Interface", success))
    
    # Summary
    total_time = time.time() - start_time
    print_header("Test Results Summary")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall Results:")
    print(f"   Tests passed: {passed}/{len(results)}")
    print(f"   Success rate: {passed/len(results)*100:.1f}%")
    print(f"   Total time: {total_time:.1f} seconds")
    
    if passed == len(results):
        print(f"\n🎉 All tests passed! System is ready for use.")
        print(f"   Run 'python demo.py' to start the interactive demo.")
    else:
        print(f"\n⚠️ Some tests failed. Check the error messages above.")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting TravelHunters ML System Test...")
    print("This will test all components of the recommendation system.")
    
    try:
        results = run_complete_system_test()
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user.")
    except Exception as e:
        print(f"\n\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
