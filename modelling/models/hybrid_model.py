"""
Hybrid Hotel Recommender System
Combines parameter-based and text-based recommendations for improved performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('../models')
sys.path.append('../data_preparation')

from parameter_model import ParameterBasedRecommender
from text_similarity_model import TextBasedRecommender

class HybridRecommender:
    """Hybrid recommender combining parameter-based and text-based approaches"""
    
    def __init__(self, param_model_type: str = 'ridge', text_max_features: int = 1000):
        """
        Initialize hybrid recommender
        
        Args:
            param_model_type: Type of parameter model ('ridge', 'linear', 'random_forest')
            text_max_features: Maximum features for text model
        """
        self.param_recommender = ParameterBasedRecommender(model_type=param_model_type)
        self.text_recommender = TextBasedRecommender(max_features=text_max_features)
        
        self.is_trained = False
        self.weights = {
            'parameter': 0.5,
            'text': 0.5
        }
    
    def train(self, hotels_df: pd.DataFrame, interactions_df: pd.DataFrame, 
             features_df: pd.DataFrame) -> Dict:
        """
        Train both parameter and text models
        
        Args:
            hotels_df: Original hotel data
            interactions_df: User interactions
            features_df: Engineered features
            
        Returns:
            Training metrics for both models
        """
        print("🔄 Training hybrid recommender...")
        
        # Train parameter model
        print("  Training parameter model...")
        X_param, y_param = self.param_recommender.prepare_training_data(features_df, interactions_df)
        param_metrics = self.param_recommender.train(X_param, y_param)
        
        # Train text model
        print("  Training text model...")
        self.text_recommender.fit(features_df)
        
        self.is_trained = True
        
        metrics = {
            'parameter_model': param_metrics,
            'text_model': {'status': 'fitted', 'n_hotels': len(features_df)},
            'hybrid_status': 'trained'
        }
        
        print("✅ Hybrid model training completed!")
        return metrics
    
    def set_weights(self, parameter_weight: float, text_weight: float):
        """
        Set weights for combining models
        
        Args:
            parameter_weight: Weight for parameter-based model
            text_weight: Weight for text-based model
        """
        total_weight = parameter_weight + text_weight
        self.weights = {
            'parameter': parameter_weight / total_weight,
            'text': text_weight / total_weight
        }
        print(f"📊 Weights updated: Parameter={self.weights['parameter']:.2f}, Text={self.weights['text']:.2f}")
    
    def recommend_hotels(self, query: str, hotels_df: pd.DataFrame, features_df: pd.DataFrame,
                        user_preferences: Dict, top_k: int = 10, 
                        combination_method: str = 'weighted_sum') -> pd.DataFrame:
        """
        Generate hybrid recommendations
        
        Args:
            query: User text query
            hotels_df: Original hotel data
            features_df: Engineered features
            user_preferences: User preferences
            top_k: Number of recommendations
            combination_method: 'weighted_sum', 'rank_fusion', or 'cascade'
            
        Returns:
            Hybrid recommendations
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making recommendations")
        
        # Get parameter-based recommendations
        param_recs = self.param_recommender.recommend_hotels(
            features_df, user_preferences, top_k=top_k*2  # Get more for fusion
        )
        
        # Get text-based recommendations
        text_recs = self.text_recommender.recommend_hotels(
            query, hotels_df, user_preferences, top_k=top_k*2
        )
        
        # Combine recommendations based on method
        if combination_method == 'weighted_sum':
            hybrid_recs = self._weighted_sum_combination(param_recs, text_recs, top_k)
        elif combination_method == 'rank_fusion':
            hybrid_recs = self._rank_fusion_combination(param_recs, text_recs, top_k)
        elif combination_method == 'cascade':
            hybrid_recs = self._cascade_combination(param_recs, text_recs, user_preferences, top_k)
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
        
        return hybrid_recs
    
    def _weighted_sum_combination(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame, 
                                 top_k: int) -> pd.DataFrame:
        """Combine using weighted sum of scores"""
        # Normalize scores to 0-1 range
        param_scores = param_recs.set_index('hotel_id')['final_score']
        text_scores = text_recs.set_index('hotel_id')['final_score']
        
        if len(param_scores) > 1:
            param_scores = (param_scores - param_scores.min()) / (param_scores.max() - param_scores.min())
        if len(text_scores) > 1:
            text_scores = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min())
        
        # Get all unique hotel IDs
        all_hotel_ids = set(param_recs['hotel_id']) | set(text_recs['hotel_id'])
        
        # Calculate hybrid scores
        hybrid_scores = []
        for hotel_id in all_hotel_ids:
            param_score = param_scores.get(hotel_id, 0) * self.weights['parameter']
            text_score = text_scores.get(hotel_id, 0) * self.weights['text']
            hybrid_score = param_score + text_score
            
            # Get hotel details
            param_hotel = param_recs[param_recs['hotel_id'] == hotel_id]
            text_hotel = text_recs[text_recs['hotel_id'] == hotel_id]
            
            if not param_hotel.empty:
                hotel_info = param_hotel.iloc[0]
            elif not text_hotel.empty:
                hotel_info = text_hotel.iloc[0]
            else:
                continue
            
            hybrid_scores.append({
                'hotel_id': hotel_id,
                'hotel_name': hotel_info.get('hotel_name', hotel_info.get('name', 'Unknown')),
                'hybrid_score': hybrid_score,
                'param_score': param_scores.get(hotel_id, 0),
                'text_score': text_scores.get(hotel_id, 0),
                'price': hotel_info.get('price', 0),
                'rating': hotel_info.get('rating', 0),
                'location': hotel_info.get('location', 'Unknown')
            })
        
        # Convert to DataFrame and sort
        hybrid_df = pd.DataFrame(hybrid_scores)
        hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False).head(top_k)
        
        return hybrid_df
    
    def _rank_fusion_combination(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame, 
                                top_k: int) -> pd.DataFrame:
        """Combine using rank fusion (Reciprocal Rank Fusion)"""
        # Create rank dictionaries
        param_ranks = {hotel_id: rank+1 for rank, hotel_id in enumerate(param_recs['hotel_id'])}
        text_ranks = {hotel_id: rank+1 for rank, hotel_id in enumerate(text_recs['hotel_id'])}
        
        # Get all unique hotel IDs
        all_hotel_ids = set(param_recs['hotel_id']) | set(text_recs['hotel_id'])
        
        # Calculate RRF scores
        k = 60  # RRF parameter
        rrr_scores = []
        
        for hotel_id in all_hotel_ids:
            param_rank = param_ranks.get(hotel_id, len(param_recs) + 1)
            text_rank = text_ranks.get(hotel_id, len(text_recs) + 1)
            
            # RRF formula: 1/(k + rank)
            rrr_score = (
                self.weights['parameter'] * (1 / (k + param_rank)) +
                self.weights['text'] * (1 / (k + text_rank))
            )
            
            # Get hotel details
            param_hotel = param_recs[param_recs['hotel_id'] == hotel_id]
            text_hotel = text_recs[text_recs['hotel_id'] == hotel_id]
            
            if not param_hotel.empty:
                hotel_info = param_hotel.iloc[0]
            elif not text_hotel.empty:
                hotel_info = text_hotel.iloc[0]
            else:
                continue
            
            rrr_scores.append({
                'hotel_id': hotel_id,
                'hotel_name': hotel_info.get('hotel_name', hotel_info.get('name', 'Unknown')),
                'rrr_score': rrr_score,
                'param_rank': param_rank,
                'text_rank': text_rank,
                'price': hotel_info.get('price', 0),
                'rating': hotel_info.get('rating', 0),
                'location': hotel_info.get('location', 'Unknown')
            })
        
        # Convert to DataFrame and sort
        hybrid_df = pd.DataFrame(rrr_scores)
        hybrid_df = hybrid_df.sort_values('rrr_score', ascending=False).head(top_k)
        
        return hybrid_df
    
    def _cascade_combination(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame,
                           user_preferences: Dict, top_k: int) -> pd.DataFrame:
        """Combine using cascade approach (parameter model first, then text refinement)"""
        # Start with parameter-based recommendations
        base_recs = param_recs.head(top_k * 2)  # Get more candidates
        
        # If user provides a text query with high text importance, boost text scores
        text_importance = user_preferences.get('text_importance', 0.3)
        
        if text_importance > 0.5:  # Text is important
            # Re-rank using text similarity
            text_hotel_ids = set(text_recs['hotel_id'])
            
            cascaded_recs = []
            for _, hotel in base_recs.iterrows():
                hotel_id = hotel['hotel_id']
                base_score = hotel['final_score']
                
                # Boost if also in text recommendations
                if hotel_id in text_hotel_ids:
                    text_hotel = text_recs[text_recs['hotel_id'] == hotel_id]
                    if not text_hotel.empty:
                        text_boost = text_hotel.iloc[0]['final_score'] * 0.3  # 30% boost
                        final_score = base_score + text_boost
                    else:
                        final_score = base_score
                else:
                    final_score = base_score * 0.8  # Slight penalty for not being in text results
                
                cascaded_recs.append({
                    'hotel_id': hotel_id,
                    'hotel_name': hotel.get('hotel_name', 'Unknown'),
                    'cascade_score': final_score,
                    'base_score': base_score,
                    'in_text_recs': hotel_id in text_hotel_ids,
                    'price': hotel.get('price', 0),
                    'rating': hotel.get('rating', 0),
                    'location': hotel.get('location', 'Unknown')
                })
            
            # Sort by cascade score
            cascade_df = pd.DataFrame(cascaded_recs)
            cascade_df = cascade_df.sort_values('cascade_score', ascending=False).head(top_k)
            
        else:
            # Text is not important, just use parameter recommendations
            cascade_df = base_recs.head(top_k).copy()
            cascade_df['cascade_score'] = cascade_df['final_score']
        
        return cascade_df
    
    def explain_recommendation(self, hotel_id: int, query: str, hotels_df: pd.DataFrame,
                              features_df: pd.DataFrame, user_preferences: Dict) -> Dict:
        """
        Provide explanation for why a hotel was recommended
        
        Args:
            hotel_id: ID of the hotel to explain
            query: User query
            hotels_df: Hotel data
            features_df: Features data
            user_preferences: User preferences
            
        Returns:
            Explanation dictionary
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before generating explanations")
        
        explanation = {
            'hotel_id': hotel_id,
            'query': query,
            'explanations': []
        }
        
        # Get hotel details
        hotel = hotels_df[hotels_df['id'] == hotel_id]
        if hotel.empty:
            return {'error': 'Hotel not found'}
        
        hotel_info = hotel.iloc[0]
        explanation['hotel_name'] = hotel_info['name']
        
        # Parameter-based explanation
        param_predictions = self.param_recommender.predict_hotel_scores(features_df)
        hotel_param_score = param_predictions[param_predictions['hotel_id'] == hotel_id]
        
        if not hotel_param_score.empty:
            param_score = hotel_param_score.iloc[0]['predicted_score']
            explanation['explanations'].append({
                'model': 'parameter',
                'score': param_score,
                'reasons': [
                    f"Price: ${hotel_info['price']:.0f} (fits budget: {hotel_info['price'] <= user_preferences.get('max_price', 1000)})",
                    f"Rating: {hotel_info['rating']:.1f}/5.0 (meets minimum: {hotel_info['rating'] >= user_preferences.get('min_rating', 0)})",
                    f"Predicted satisfaction: {param_score:.2f}/5.0"
                ]
            })
        
        # Text-based explanation
        text_results = self.text_recommender.search_hotels(query, top_k=50)
        hotel_text_result = text_results[text_results['hotel_id'] == hotel_id]
        
        if not hotel_text_result.empty:
            similarity_score = hotel_text_result.iloc[0]['similarity_score']
            query_keywords = self.text_recommender.get_query_keywords(query, top_k=5)
            
            explanation['explanations'].append({
                'model': 'text',
                'score': similarity_score,
                'reasons': [
                    f"Text similarity to query: {similarity_score:.3f}",
                    f"Matching keywords: {', '.join(query_keywords[:3])}",
                    f"Description matches your search for: '{query}'"
                ]
            })
        
        # Overall explanation
        explanation['summary'] = f"This hotel was recommended because it combines good parameter fit (price, rating) with relevant text description matching your search."
        
        return explanation
    
    def save_models(self, param_path: str, text_path: str):
        """Save both models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        self.param_recommender.save_model(param_path)
        self.text_recommender.save_model(text_path)
        print(f"✅ Hybrid models saved to {param_path} and {text_path}")
    
    def load_models(self, param_path: str, text_path: str):
        """Load both models"""
        self.param_recommender.load_model(param_path)
        self.text_recommender.load_model(text_path)
        self.is_trained = True
        print(f"✅ Hybrid models loaded from {param_path} and {text_path}")

if __name__ == "__main__":
    # Test hybrid recommender
    from load_data import HotelDataLoader
    from feature_engineering import HotelFeatureEngineer
    
    # Load data
    loader = HotelDataLoader()
    hotels_df = loader.load_hotels()
    interactions_df = loader.load_user_interactions()
    
    if not hotels_df.empty and not interactions_df.empty:
        # Engineer features
        engineer = HotelFeatureEngineer()
        features_df, _ = engineer.prepare_parameter_features(hotels_df)
        
        # Create and train hybrid model
        hybrid = HybridRecommender()
        metrics = hybrid.train(hotels_df, interactions_df, features_df)
        
        # Test different combination methods
        query = "luxury spa resort with pool and gym"
        user_prefs = {
            'max_price': 300,
            'min_rating': 4.0,
            'text_importance': 0.6,
            'price_importance': 0.2,
            'rating_importance': 0.2
        }
        
        print(f"\n🔍 Query: '{query}'")
        
        # Test weighted sum
        hybrid.set_weights(0.4, 0.6)  # More weight on text
        recs_weighted = hybrid.recommend_hotels(
            query, hotels_df, features_df, user_prefs, top_k=5, 
            combination_method='weighted_sum'
        )
        print(f"\n📊 Weighted Sum Recommendations:")
        print(recs_weighted[['hotel_name', 'hybrid_score', 'price', 'rating']])
        
        # Test rank fusion
        recs_rank = hybrid.recommend_hotels(
            query, hotels_df, features_df, user_prefs, top_k=5,
            combination_method='rank_fusion'
        )
        print(f"\n🔄 Rank Fusion Recommendations:")
        print(recs_rank[['hotel_name', 'rrr_score', 'price', 'rating']])
        
        # Test explanation
        if not recs_weighted.empty:
            hotel_id = recs_weighted.iloc[0]['hotel_id']
            explanation = hybrid.explain_recommendation(
                hotel_id, query, hotels_df, features_df, user_prefs
            )
            print(f"\n💡 Explanation for {explanation['hotel_name']}:")
            for exp in explanation['explanations']:
                print(f"  {exp['model']}: {exp['score']:.3f}")
                for reason in exp['reasons']:
                    print(f"    - {reason}")
