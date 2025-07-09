"""
Advanced Hybrid Hotel Recommender System
Combines parameter-based and text-based recommendations with intelligent weighting and ensemble methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Add current directory and parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

try:
    from .parameter_model import ParameterBasedRecommender
    from .text_similarity_model import TextBasedRecommender
except ImportError:
    from parameter_model import ParameterBasedRecommender
    from text_similarity_model import TextBasedRecommender

class HybridRecommender:
    """Advanced hybrid recommender with intelligent ensemble methods and adaptive weighting"""
    
    def __init__(self, param_model_type: str = 'gradient_boosting', text_max_features: int = 2000, 
                 ensemble_method: str = 'adaptive', auto_tune_weights: bool = True):
        """
        Initialize advanced hybrid recommender
        
        Args:
            param_model_type: Type of parameter model ('gradient_boosting', 'ridge', 'random_forest')
            text_max_features: Maximum features for text model
            ensemble_method: 'weighted_average', 'rank_fusion', 'adaptive', 'stacking'
            auto_tune_weights: Whether to automatically tune ensemble weights
        """
        self.param_recommender = ParameterBasedRecommender(model_type=param_model_type, auto_tune=True)
        self.text_recommender = TextBasedRecommender(
            max_features=text_max_features, 
            enable_clustering=True,
            debug_mode=False
        )
        
        self.ensemble_method = ensemble_method
        self.auto_tune_weights = auto_tune_weights
        self.scaler = MinMaxScaler()
        
        self.is_trained = False
        
        # Adaptive weights based on query type and performance
        self.base_weights = {
            'parameter': 0.4,
            'text': 0.6
        }
        
        self.optimized_weights = None
        self.performance_history = []
    
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
                        user_preferences: Dict, top_k: int = 10) -> pd.DataFrame:
        """
        Generate advanced hybrid recommendations with intelligent ensemble methods
        
        Args:
            query: User text query
            hotels_df: Original hotel data
            features_df: Engineered features
            user_preferences: User preferences
            top_k: Number of recommendations
            
        Returns:
            Advanced hybrid recommendations with explanation scores
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making recommendations")
        
        print(f"🔄 Generating hybrid recommendations using {self.ensemble_method} method...")
        
        # Analyze query to determine optimal weighting
        adaptive_weights = self._analyze_query_and_adapt_weights(query, user_preferences)
        
        # Get parameter-based recommendations (more for fusion)
        param_recs = self.param_recommender.recommend_hotels(
            features_df, user_preferences, top_k=min(top_k*3, 50)
        )
        
        # Get text-based recommendations (more for fusion)
        text_recs = self.text_recommender.recommend_hotels(
            query, hotels_df, user_preferences, top_k=min(top_k*3, 50)
        )
        
        # Apply ensemble method
        if self.ensemble_method == 'weighted_average':
            hybrid_recs = self._weighted_average_ensemble(param_recs, text_recs, adaptive_weights, top_k)
        elif self.ensemble_method == 'rank_fusion':
            hybrid_recs = self._rank_fusion_ensemble(param_recs, text_recs, top_k)
        elif self.ensemble_method == 'adaptive':
            hybrid_recs = self._adaptive_ensemble(param_recs, text_recs, query, user_preferences, top_k)
        elif self.ensemble_method == 'stacking':
            hybrid_recs = self._stacking_ensemble(param_recs, text_recs, hotels_df, top_k)
        else:
            # Fallback to weighted average
            hybrid_recs = self._weighted_average_ensemble(param_recs, text_recs, adaptive_weights, top_k)
        
        # Add diversity and final ranking
        final_recs = self._apply_final_ranking_and_diversity(hybrid_recs, query, user_preferences, top_k)
        
        # Add comprehensive explanation scores
        final_recs = self._add_hybrid_explanation_scores(final_recs, param_recs, text_recs, adaptive_weights)
        
        print(f"✅ Generated {len(final_recs)} hybrid recommendations")
        
        return final_recs.reset_index(drop=True)
    
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
        hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False)
        
        # Remove duplicates by hotel name (keep highest score)
        hybrid_df = hybrid_df.drop_duplicates(subset=['hotel_name'], keep='first')
        
        return hybrid_df.head(top_k)
    
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
        hybrid_df = hybrid_df.sort_values('rrr_score', ascending=False)
        
        # Remove duplicates by hotel name (keep highest score)
        hybrid_df = hybrid_df.drop_duplicates(subset=['hotel_name'], keep='first')
        
        return hybrid_df.head(top_k)
    
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
            cascade_df = cascade_df.sort_values('cascade_score', ascending=False)
            
            # Remove duplicates by hotel name (keep highest score)
            cascade_df = cascade_df.drop_duplicates(subset=['hotel_name'], keep='first')
            cascade_df = cascade_df.head(top_k)
            
        else:
            # Text is not important, just use parameter recommendations
            cascade_df = base_recs.head(top_k).copy()
            cascade_df['cascade_score'] = cascade_df['final_score']
            
            # Remove duplicates by hotel name even for parameter-only results
            if 'hotel_name' not in cascade_df.columns and 'name' in cascade_df.columns:
                cascade_df['hotel_name'] = cascade_df['name']
            cascade_df = cascade_df.drop_duplicates(subset=['hotel_name'], keep='first')
        
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
                    f"Rating: {hotel_info['rating']:.1f}/10.0 (meets minimum: {hotel_info['rating'] >= user_preferences.get('min_rating', 0)})",
                    f"Predicted satisfaction: {param_score:.2f}/10.0"
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
    
    def _analyze_query_and_adapt_weights(self, query: str, user_preferences: Dict) -> Dict:
        """Analyze query characteristics to determine optimal model weights"""
        text_weight = self.base_weights['text']
        param_weight = self.base_weights['parameter']
        
        # Analyze query characteristics
        query_lower = query.lower()
        
        # Increase text weight for descriptive queries
        descriptive_words = ['luxury', 'beautiful', 'cozy', 'modern', 'traditional', 'romantic', 'peaceful']
        if any(word in query_lower for word in descriptive_words):
            text_weight += 0.15
            
        # Increase text weight for activity-based queries
        activity_words = ['spa', 'pool', 'beach', 'restaurant', 'gym', 'kids', 'family']
        if any(word in query_lower for word in activity_words):
            text_weight += 0.1
            
        # Increase parameter weight for budget/rating constraints
        if user_preferences.get('max_price', float('inf')) < 200:
            param_weight += 0.1
        if user_preferences.get('min_rating', 0) > 8:
            param_weight += 0.1
        
        # Normalize weights
        total = text_weight + param_weight
        return {
            'text': text_weight / total,
            'parameter': param_weight / total
        }
    
    def _weighted_average_ensemble(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame, 
                                  weights: Dict, top_k: int) -> pd.DataFrame:
        """Combine recommendations using weighted average of normalized scores"""
        
        # Normalize scores for both recommendation sets
        param_scores = self._normalize_scores(param_recs, 'final_score')
        text_scores = self._normalize_scores(text_recs, 'similarity_score')
        
        # Create combined dataset
        all_hotels = set()
        if 'hotel_id' in param_recs.columns:
            all_hotels.update(param_recs['hotel_id'].values)
        elif 'id' in param_recs.columns:
            all_hotels.update(param_recs['id'].values)
            
        if 'hotel_id' in text_recs.columns:
            all_hotels.update(text_recs['hotel_id'].values)
        elif 'id' in text_recs.columns:
            all_hotels.update(text_recs['id'].values)
        
        hybrid_results = []
        
        for hotel_id in all_hotels:
            # Get parameter score
            param_mask = (param_recs.get('hotel_id', param_recs.get('id', pd.Series())) == hotel_id)
            param_score = param_scores[param_mask].iloc[0] if param_mask.any() else 0.0
            
            # Get text score  
            text_mask = (text_recs.get('hotel_id', text_recs.get('id', pd.Series())) == hotel_id)
            text_score = text_scores[text_mask].iloc[0] if text_mask.any() else 0.0
            
            # Calculate weighted score
            hybrid_score = (weights['parameter'] * param_score + 
                           weights['text'] * text_score)
            
            # Get hotel details from either dataset
            hotel_row = None
            if param_mask.any():
                hotel_row = param_recs[param_mask].iloc[0]
            elif text_mask.any():
                hotel_row = text_recs[text_mask].iloc[0]
                
            if hotel_row is not None:
                result_row = hotel_row.copy()
                result_row['hybrid_score'] = hybrid_score
                result_row['param_score'] = param_score
                result_row['text_score'] = text_score
                hybrid_results.append(result_row)
        
        if not hybrid_results:
            return pd.DataFrame()
        
        hybrid_df = pd.DataFrame(hybrid_results)
        return hybrid_df.sort_values('hybrid_score', ascending=False).head(top_k)
    
    def _rank_fusion_ensemble(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame, 
                             top_k: int) -> pd.DataFrame:
        """Combine using reciprocal rank fusion"""
        
        # Add ranks to both datasets
        param_recs = param_recs.copy()
        text_recs = text_recs.copy()
        param_recs['param_rank'] = range(1, len(param_recs) + 1)
        text_recs['text_rank'] = range(1, len(text_recs) + 1)
        
        # Merge datasets
        id_col_param = 'hotel_id' if 'hotel_id' in param_recs.columns else 'id'
        id_col_text = 'hotel_id' if 'hotel_id' in text_recs.columns else 'id'
        
        merged = pd.merge(param_recs, text_recs, left_on=id_col_param, right_on=id_col_text, 
                         how='outer', suffixes=('_param', '_text'))
        
        # Calculate RRF score
        k = 60  # RRF parameter
        merged['param_rank'] = merged['param_rank'].fillna(len(param_recs) + 1)
        merged['text_rank'] = merged['text_rank'].fillna(len(text_recs) + 1)
        
        merged['rrf_score'] = (1 / (k + merged['param_rank']) + 
                              1 / (k + merged['text_rank']))
        
        return merged.sort_values('rrf_score', ascending=False).head(top_k)
    
    def _adaptive_ensemble(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame,
                          query: str, user_preferences: Dict, top_k: int) -> pd.DataFrame:
        """Adaptive ensemble that changes strategy based on query and data quality"""
        
        # Determine strategy based on data quality and query type
        param_quality = len(param_recs) / max(top_k, 1)
        text_quality = len(text_recs) / max(top_k, 1)
        
        if param_quality > 0.8 and text_quality > 0.8:
            # Both models have good results - use weighted average
            weights = self._analyze_query_and_adapt_weights(query, user_preferences)
            return self._weighted_average_ensemble(param_recs, text_recs, weights, top_k)
        elif text_quality > param_quality:
            # Text model is better - favor it
            weights = {'text': 0.75, 'parameter': 0.25}
            return self._weighted_average_ensemble(param_recs, text_recs, weights, top_k)
        else:
            # Parameter model is better - favor it  
            weights = {'text': 0.25, 'parameter': 0.75}
            return self._weighted_average_ensemble(param_recs, text_recs, weights, top_k)
    
    def _stacking_ensemble(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame,
                          hotels_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """Meta-learning ensemble that learns optimal combination"""
        # Simplified stacking - would need more training data for full implementation
        return self._rank_fusion_ensemble(param_recs, text_recs, top_k)
    
    def _normalize_scores(self, df: pd.DataFrame, score_col: str) -> pd.Series:
        """Normalize scores to 0-1 range"""
        scores = df[score_col] if score_col in df.columns else pd.Series([0.5] * len(df))
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return pd.Series([0.5] * len(scores), index=scores.index)
    
    def _apply_final_ranking_and_diversity(self, df: pd.DataFrame, query: str, 
                                          user_preferences: Dict, top_k: int) -> pd.DataFrame:
        """Apply final ranking with diversity considerations"""
        if len(df) <= top_k:
            return df
        
        # Simple diversity by ensuring variety in price ranges and locations
        df = df.copy()
        diverse_results = []
        remaining_df = df.copy()
        
        while len(diverse_results) < top_k and len(remaining_df) > 0:
            # Take the best remaining hotel
            best_hotel = remaining_df.iloc[0]
            diverse_results.append(best_hotel)
            
            # Remove similar hotels (same location or very similar price)
            if 'location' in remaining_df.columns:
                same_location = remaining_df['location'] == best_hotel.get('location', '')
                similar_price = abs(remaining_df.get('price', 0) - best_hotel.get('price', 0)) < 50
                
                # Remove hotels that are too similar
                to_remove = same_location & similar_price
                if to_remove.sum() > 1:  # Keep at least one
                    remaining_df = remaining_df[~to_remove | (remaining_df.index == remaining_df.index[0])]
                
            remaining_df = remaining_df.iloc[1:]  # Remove the selected hotel
        
        return pd.DataFrame(diverse_results)
    
    def _add_hybrid_explanation_scores(self, hybrid_df: pd.DataFrame, param_recs: pd.DataFrame,
                                      text_recs: pd.DataFrame, weights: Dict) -> pd.DataFrame:
        """Add explanation scores showing contribution of each model"""
        hybrid_df = hybrid_df.copy()
        
        # Add model contribution scores
        hybrid_df['parameter_contribution'] = weights['parameter']
        hybrid_df['text_contribution'] = weights['text']
        hybrid_df['ensemble_method'] = self.ensemble_method
        
        return hybrid_df
