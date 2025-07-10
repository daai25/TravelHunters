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
        
        # Kopieren der Benutzereinstellungen für mögliche Entspannungen
        original_preferences = user_preferences.copy()
        
        # Analyze query to determine optimal weighting
        adaptive_weights = self._analyze_query_and_adapt_weights(query, user_preferences)
        
        # Get parameter-based recommendations (more for fusion)
        try:
            print("🎯 Generating parameter-based recommendations...")
            param_recs = self.param_recommender.recommend_hotels(
                features_df, user_preferences, top_k=min(top_k*3, 50)
            )
            if param_recs.empty:
                print("⚠️ No hotels match your parameter criteria. Using relaxed filters.")
                # Erstelle eine Kopie der Benutzereinstellungen mit entspannteren Filtern
                relaxed_prefs = user_preferences.copy()
                relaxed_prefs['min_rating'] = max(0, min(relaxed_prefs.get('min_rating', 7.0) - 2.0, 6.0))  # Reduziere um 2 Punkte oder auf max. 6.0
                relaxed_prefs['max_price'] = relaxed_prefs.get('max_price', 200) * 1.5  # Erhöhe das Budget um 50%
                param_recs = self.param_recommender.recommend_hotels(
                    features_df, relaxed_prefs, top_k=min(top_k*3, 50)
                )
                
                # Wenn immer noch leer, versuche mit minimalen Filtern
                if param_recs.empty:
                    print("⚠️ Noch immer keine Ergebnisse. Versuche mit minimalen Filtern.")
                    minimal_prefs = user_preferences.copy()
                    minimal_prefs['min_rating'] = 1.0  # Minimale Bewertung
                    minimal_prefs['max_price'] = 10000  # Sehr hohes Budget
                    param_recs = self.param_recommender.recommend_hotels(
                        features_df, minimal_prefs, top_k=min(top_k*3, 50)
                    )
        except Exception as e:
            print(f"⚠️ Error in parameter recommendations: {e}")
            param_recs = pd.DataFrame()  # Leeres DataFrame als Fallback
        
        # Get text-based recommendations (more for fusion)
        try:
            print("🔍 Generating text-based recommendations...")
            text_recs = self.text_recommender.recommend_hotels(
                query, hotels_df, user_preferences, top_k=min(top_k*3, 50)
            )
            
            # Wenn keine Text-Empfehlungen, versuche mit entspannten Filtern
            if text_recs.empty:
                print("⚠️ Keine Hotels mit Text-Matching gefunden. Entspanne die Filter.")
                relaxed_prefs = user_preferences.copy()
                relaxed_prefs['min_rating'] = max(0, min(relaxed_prefs.get('min_rating', 7.0) - 2.0, 6.0))  # Reduziere um 2 Punkte
                relaxed_prefs['max_price'] = relaxed_prefs.get('max_price', 200) * 1.5  # Erhöhe das Budget um 50%
                text_recs = self.text_recommender.recommend_hotels(
                    query, hotels_df, relaxed_prefs, top_k=min(top_k*3, 50)
                )
        except Exception as e:
            print(f"⚠️ Error in text recommendations: {e}")
            text_recs = pd.DataFrame()  # Leeres DataFrame als Fallback
        
        # Überprüfen, ob wir Ergebnisse haben
        if param_recs.empty and text_recs.empty:
            print("❌ Keine Hotels gefunden, die Ihren Kriterien entsprechen. Versuche mit minimalen Filtern.")
            
            # Letzte Chance mit absolut minimalen Filtern für beide Modelle
            try:
                minimal_prefs = {'min_rating': 0.0, 'max_price': 10000}
                param_recs = self.param_recommender.recommend_hotels(
                    features_df, minimal_prefs, top_k=min(top_k*3, 50)
                )
                text_recs = self.text_recommender.recommend_hotels(
                    query, hotels_df, minimal_prefs, top_k=min(top_k*3, 50)
                )
                
                # Wenn immer noch keine Ergebnisse, geben ein leeres DataFrame zurück
                if param_recs.empty and text_recs.empty:
                    print("❌ Leider konnten keine Hotels gefunden werden, selbst mit minimalen Filtern.")
                    return pd.DataFrame()
            except Exception as e:
                print(f"❌ Fehler bei der letzten Empfehlungssuche: {e}")
                return pd.DataFrame()  # Leeres DataFrame zurückgeben
        
        # Wenn nur ein Modell Ergebnisse liefert, verwenden wir diese direkt
        if param_recs.empty and not text_recs.empty:
            print("ℹ️ Nur textbasierte Empfehlungen verfügbar.")
            final_results = text_recs.head(top_k).copy()
            final_results['hybrid_score'] = final_results['final_score']  # Kopiere den Score für konsistente Spaltenbezeichnung
            return final_results
        
        if not param_recs.empty and text_recs.empty:
            print("ℹ️ Nur parameterbasierte Empfehlungen verfügbar.")
            final_results = param_recs.head(top_k).copy()
            final_results['hybrid_score'] = final_results['final_score']  # Kopiere den Score für konsistente Spaltenbezeichnung
            return final_results
            
        # Apply ensemble method
        print(f"🔄 Combining recommendations using {self.ensemble_method} method...")
        
        # Überprüfe vor der Anwendung des Ensemble-Verfahrens
        if len(param_recs) == 0 or len(text_recs) == 0:
            print("⚠️ Eines der Modelle hat keine Ergebnisse. Verwende nur das andere Modell.")
            if len(param_recs) > 0:
                return param_recs.head(top_k)
            else:
                return text_recs.head(top_k)
        
        # Alles gut, wende das Ensemble-Verfahren an
        try:
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
        except Exception as e:
            print(f"⚠️ Fehler beim Kombinieren der Empfehlungen: {e}")
            # Fallback: Einfach die besten Empfehlungen aus beiden Modellen nehmen
            hybrid_recs = pd.concat([param_recs.head(top_k // 2), text_recs.head(top_k // 2)])
            hybrid_recs = hybrid_recs.sort_values('final_score', ascending=False).head(top_k)
        
        # Überprüfe, ob das Ensemble-Verfahren Ergebnisse geliefert hat
        if hybrid_recs.empty:
            print("⚠️ Das Ensemble-Verfahren hat keine Ergebnisse geliefert. Verwende beide Modelle direkt.")
            hybrid_recs = pd.concat([param_recs.head(top_k // 2), text_recs.head(top_k // 2)])
            hybrid_recs = hybrid_recs.drop_duplicates(subset=['hotel_id'], keep='first')
            hybrid_recs['hybrid_score'] = hybrid_recs['final_score']  # Verwende den ursprünglichen Score
        
        # Add diversity and final ranking
        try:
            final_recs = self._apply_final_ranking_and_diversity(hybrid_recs, query, user_preferences, top_k)
        except Exception as e:
            print(f"⚠️ Fehler beim finalen Ranking: {e}")
            final_recs = hybrid_recs.head(top_k)  # Fallback zur einfachen Auswahl
        
        # Add comprehensive explanation scores
        try:
            final_recs = self._add_hybrid_explanation_scores(final_recs, param_recs, text_recs, adaptive_weights)
        except Exception as e:
            print(f"⚠️ Fehler beim Hinzufügen der Erklärungen: {e}")
            # Nicht kritisch, wir können ohne Erklärungen weitermachen
        
        # Stelle sicher, dass ein Score für das Hybrid-Modell existiert
        if 'hybrid_score' not in final_recs.columns:
            final_recs['hybrid_score'] = final_recs['final_score']
        
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
        print("🔄 Applying rank fusion ensemble")
        
        # Überprüfe, ob beide DataFrames Daten enthalten
        if param_recs.empty and text_recs.empty:
            print("❌ Beide Empfehlungsmodelle liefern keine Ergebnisse.")
            return pd.DataFrame()
        
        # Wenn nur ein Modell Ergebnisse liefert, verwenden wir diese direkt
        if param_recs.empty:
            print("ℹ️ Nur textbasierte Empfehlungen verfügbar für Rank Fusion.")
            result = text_recs.copy()
            result['hybrid_score'] = result['final_score']
            return result.head(top_k)
            
        if text_recs.empty:
            print("ℹ️ Nur parameterbasierte Empfehlungen verfügbar für Rank Fusion.")
            result = param_recs.copy()
            result['hybrid_score'] = result['final_score']
            return result.head(top_k)
        
        try:
            # Create rank dictionaries
            param_ranks = {hotel_id: rank+1 for rank, hotel_id in enumerate(param_recs['hotel_id'])}
            text_ranks = {hotel_id: rank+1 for rank, hotel_id in enumerate(text_recs['hotel_id'])}
            
            # Get all unique hotel IDs
            all_hotel_ids = set(param_recs['hotel_id']) | set(text_recs['hotel_id'])
            
            if not all_hotel_ids:
                print("⚠️ Keine Hotel-IDs gefunden für Rank Fusion.")
                return pd.DataFrame()
            
            # Calculate RRF scores
            k = 60  # RRF parameter
            rrr_scores = []
            
            # Standardgewichte verwenden
            weights = self.base_weights
            
            for hotel_id in all_hotel_ids:
                param_rank = param_ranks.get(hotel_id, len(param_recs) + 1)
                text_rank = text_ranks.get(hotel_id, len(text_recs) + 1)
                
                # RRF formula: 1/(k + rank)
                rrr_score = (
                    weights['parameter'] * (1 / (k + param_rank)) +
                    weights['text'] * (1 / (k + text_rank))
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
                
                hotel_entry = {
                    'hotel_id': hotel_id,
                    'hotel_name': hotel_info.get('hotel_name', hotel_info.get('name', 'Unknown')),
                    'hybrid_score': rrr_score,  # Direkter Name für die Konsistenz
                    'rrr_score': rrr_score,
                    'param_rank': param_rank,
                    'text_rank': text_rank,
                    'price': hotel_info.get('price', 0),
                    'rating': hotel_info.get('rating', 0),
                    'location': hotel_info.get('location', 'Unknown')
                }
                
                # Zusätzliche Spalten kopieren, falls vorhanden
                for key in ['description', 'amenities', 'similarity_score', 'final_score']:
                    if key in hotel_info:
                        hotel_entry[key] = hotel_info[key]
                
                rrr_scores.append(hotel_entry)
            
            # Convert to DataFrame and sort
            hybrid_df = pd.DataFrame(rrr_scores)
            
            # Überprüfe, ob das DataFrame leer ist
            if hybrid_df.empty:
                print("⚠️ Keine gemeinsamen Hotels zwischen den Modellen gefunden.")
                # Fallback: Einfach die besten Empfehlungen aus beiden Modellen nehmen
                combined_df = pd.concat([param_recs.head(top_k // 2), text_recs.head(top_k // 2)])
                combined_df['hybrid_score'] = combined_df['final_score']
                combined_df = combined_df.sort_values('hybrid_score', ascending=False)
                return combined_df.head(top_k)
            
            hybrid_df = hybrid_df.sort_values('rrr_score', ascending=False)
            
            # Remove duplicates by hotel name (keep highest score)
            if 'hotel_name' in hybrid_df.columns:
                hybrid_df = hybrid_df.drop_duplicates(subset=['hotel_name'], keep='first')
            
            return hybrid_df.head(top_k)
            
        except Exception as e:
            print(f"⚠️ Fehler beim Rank Fusion Ensemble: {e}")
            # Fallback: Einfach die besten Empfehlungen aus beiden Modellen nehmen
            try:
                combined_df = pd.concat([param_recs.head(max(1, top_k // 2)), text_recs.head(max(1, top_k // 2))])
                combined_df['hybrid_score'] = combined_df['final_score']
                combined_df = combined_df.sort_values('hybrid_score', ascending=False)
                return combined_df.head(top_k).drop_duplicates(subset=['hotel_id'], keep='first')
            except Exception as inner_e:
                print(f"⚠️ Auch Fallback fehlgeschlagen: {inner_e}")
                if not param_recs.empty:
                    return param_recs.head(top_k)
                elif not text_recs.empty:
                    return text_recs.head(top_k)
                else:
                    return pd.DataFrame()
    
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
        # Find the hotel in the predictions by index or by ID column
        if 'id' in param_predictions.columns:
            hotel_param_score = param_predictions[param_predictions['id'] == hotel_id]
        else:
            # If no ID column, find by index matching the hotel's position
            hotel_index = hotel.index[0]
            if hotel_index < len(param_predictions):
                hotel_param_score = param_predictions.iloc[[hotel_index]]
            else:
                hotel_param_score = pd.DataFrame()
        
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
        # Find the hotel in the text results by ID or index
        if 'hotel_id' in text_results.columns:
            hotel_text_result = text_results[text_results['hotel_id'] == hotel_id]
        elif 'id' in text_results.columns:
            hotel_text_result = text_results[text_results['id'] == hotel_id]
        else:
            # If no ID column, find by index matching the hotel's position
            hotel_index = hotel.index[0]
            if hotel_index < len(text_results):
                hotel_text_result = text_results.iloc[[hotel_index]]
            else:
                hotel_text_result = pd.DataFrame()
        
        if not hotel_text_result.empty:
            similarity_score = hotel_text_result.iloc[0]['similarity_score']
            query_keywords = self.text_recommender.get_query_keywords(query, top_k=5);
            
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
        """
        Analyze query and user preferences to determine optimal weighting
        
        Args:
            query: User's text query
            user_preferences: User preference dictionary
            
        Returns:
            Adjusted weights dictionary
        """
        # Default weights
        weights = self.base_weights.copy()
        
        try:
            # Check if query is empty or too short
            if not query or len(query) < 5:
                # Leere oder zu kurze Anfrage - stärker auf Parameter-Modell verlassen
                weights['parameter'] = 0.75
                weights['text'] = 0.25
                return weights
            
            # Analyze query
            query_tokens = query.lower().split()
            query_length = len(query)
            
            # Check for price indicators
            price_terms = ['cheap', 'budget', 'affordable', 'expensive', 'luxury',
                          'günstig', 'billig', 'preiswert', 'teuer', 'luxuriös']
            has_price_terms = any(term in query_tokens for term in price_terms)
            
            # Check for rating/quality indicators
            quality_terms = ['best', 'top', 'high quality', 'excellent', 'outstanding', 'poor', 'bad',
                           'beste', 'gut', 'ausgezeichnet', 'hervorragend', 'schlecht']
            has_quality_terms = any(term in query_tokens for term in quality_terms)
            
            # Check for specific features that text model is better at
            text_specific_terms = ['view', 'atmosphere', 'style', 'modern', 'traditional', 'cozy',
                                  'aussicht', 'atmosphäre', 'stil', 'modern', 'traditionell', 'gemütlich']
            has_text_specifics = any(term in query.lower() for term in text_specific_terms)
            
            # Check for detailed description (suggests text model should be weighted more)
            is_detailed_query = query_length > 30
            
            # Check for single-word query (parameter model might be better)
            is_simple_query = len(query_tokens) <= 2
            
            # Adjust weights based on analysis
            param_adjustment = 0.0
            text_adjustment = 0.0
            
            # Specific terms adjustments
            if has_price_terms or has_quality_terms:
                param_adjustment += 0.1
                text_adjustment -= 0.1
            
            if has_text_specifics:
                text_adjustment += 0.15
                param_adjustment -= 0.15
            
            # Query complexity adjustments
            if is_detailed_query:
                text_adjustment += 0.1
                param_adjustment -= 0.1
            
            if is_simple_query:
                param_adjustment += 0.05
                text_adjustment -= 0.05
            
            # User preference adjustments
            if user_preferences:
                if user_preferences.get('text_importance', 0) > 0.6:
                    text_adjustment += 0.1
                    param_adjustment -= 0.1
                
                if user_preferences.get('price_importance', 0) > 0.6 or user_preferences.get('rating_importance', 0) > 0.6:
                    param_adjustment += 0.1
                    text_adjustment -= 0.1
            
            # Apply adjustments
            weights['parameter'] = max(0.1, min(0.9, weights['parameter'] + param_adjustment))
            weights['text'] = max(0.1, min(0.9, weights['text'] + text_adjustment))
            
            # Ensure weights sum to 1
            total = weights['parameter'] + weights['text']
            weights['parameter'] /= total
            weights['text'] /= total
            
        except Exception as e:
            print(f"⚠️ Fehler bei der Anpassung der Gewichte: {e}")
            # Im Fehlerfall die Standardgewichte zurückgeben
            weights = self.base_weights.copy()
        
        print(f"Query analysis: Parameter weight: {weights['parameter']:.2f}, Text weight: {weights['text']:.2f}")
        return weights
    
    def _weighted_average_ensemble(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame,
                                weights: Dict, top_k: int) -> pd.DataFrame:
        """Combine recommendations using weighted average of scores"""
        print(f"🔄 Applying weighted average ensemble with weights: param={weights['parameter']:.2f}, text={weights['text']:.2f}")
        
        # Überprüfe, ob beide DataFrames Daten enthalten
        if param_recs.empty and text_recs.empty:
            print("❌ Beide Empfehlungsmodelle liefern keine Ergebnisse.")
            return pd.DataFrame()
        
        # Wenn nur ein Modell Ergebnisse liefert, verwenden wir diese direkt
        if param_recs.empty:
            print("ℹ️ Nur textbasierte Empfehlungen verfügbar für Ensemble.")
            result = text_recs.copy()
            result['hybrid_score'] = result['final_score']
            return result.head(top_k)
            
        if text_recs.empty:
            print("ℹ️ Nur parameterbasierte Empfehlungen verfügbar für Ensemble.")
            result = param_recs.copy()
            result['hybrid_score'] = result['final_score']
            return result.head(top_k)
        
        try:
            # Normalize scores to 0-1 range
            param_scores = param_recs.set_index('hotel_id')['final_score']
            text_scores = text_recs.set_index('hotel_id')['final_score']
            
            # Sichere Normalisierung (mit Überprüfung auf leere DataFrames und Division durch Null)
            if len(param_scores) > 1:
                param_min = param_scores.min()
                param_max = param_scores.max()
                if abs(param_max - param_min) > 0.0001:  # Vermeidet Division durch 0
                    param_scores = (param_scores - param_min) / (param_max - param_min)
                else:
                    param_scores = pd.Series(0.5, index=param_scores.index)  # Alle auf 0.5 setzen wenn alle gleich sind
            
            if len(text_scores) > 1:
                text_min = text_scores.min()
                text_max = text_scores.max()
                if abs(text_max - text_min) > 0.0001:  # Vermeidet Division durch 0
                    text_scores = (text_scores - text_min) / (text_max - text_min)
                else:
                    text_scores = pd.Series(0.5, index=text_scores.index)  # Alle auf 0.5 setzen wenn alle gleich sind
            
            # Get all unique hotel IDs
            all_hotel_ids = set(param_recs['hotel_id']) | set(text_recs['hotel_id'])
            
            # Calculate hybrid scores
            hybrid_scores = []
            for hotel_id in all_hotel_ids:
                param_score = param_scores.get(hotel_id, 0) * weights['parameter']
                text_score = text_scores.get(hotel_id, 0) * weights['text']
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
                
                # Zusätzliche Spalten kopieren, falls vorhanden
                for key in ['description', 'amenities', 'similarity_score']:
                    if key in hotel_info:
                        hybrid_scores[-1][key] = hotel_info[key]
            
            # Convert to DataFrame and sort
            hybrid_df = pd.DataFrame(hybrid_scores)
            
            # Überprüfe, ob das DataFrame leer ist
            if hybrid_df.empty:
                print("⚠️ Keine gemeinsamen Hotels zwischen den Modellen gefunden.")
                # Fallback: Einfach die besten Empfehlungen aus beiden Modellen nehmen
                combined_df = pd.concat([param_recs.head(top_k // 2), text_recs.head(top_k // 2)])
                combined_df['hybrid_score'] = combined_df['final_score']
                combined_df = combined_df.sort_values('hybrid_score', ascending=False)
                return combined_df.head(top_k)
            
            hybrid_df = hybrid_df.sort_values('hybrid_score', ascending=False)
            
            # Remove duplicates by hotel name (keep highest score)
            if 'hotel_name' in hybrid_df.columns:
                hybrid_df = hybrid_df.drop_duplicates(subset=['hotel_name'], keep='first')
            
            return hybrid_df.head(top_k)
            
        except Exception as e:
            print(f"⚠️ Fehler beim Weighted Average Ensemble: {e}")
            # Fallback: Einfach die besten Empfehlungen aus beiden Modellen nehmen
            try:
                combined_df = pd.concat([param_recs.head(max(1, top_k // 2)), text_recs.head(max(1, top_k // 2))])
                combined_df['hybrid_score'] = combined_df['final_score']
                combined_df = combined_df.sort_values('hybrid_score', ascending=False)
                return combined_df.head(top_k).drop_duplicates(subset=['hotel_id'], keep='first')
            except Exception as inner_e:
                print(f"⚠️ Auch Fallback fehlgeschlagen: {inner_e}")
                if not param_recs.empty:
                    return param_recs.head(top_k)
                elif not text_recs.empty:
                    return text_recs.head(top_k)
                else:
                    return pd.DataFrame()
    
    def _rank_fusion_ensemble(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame, 
                             top_k: int) -> pd.DataFrame:
        """Combine using reciprocal rank fusion"""
        print("🔄 Applying rank fusion ensemble")
        
        # Überprüfe, ob beide DataFrames Daten enthalten
        if param_recs.empty and text_recs.empty:
            print("❌ Beide Empfehlungsmodelle liefern keine Ergebnisse.")
            return pd.DataFrame()
        
        # Wenn nur ein Modell Ergebnisse liefert, verwenden wir diese direkt
        if param_recs.empty:
            print("ℹ️ Nur textbasierte Empfehlungen verfügbar für Rank Fusion.")
            result = text_recs.copy()
            result['hybrid_score'] = result['final_score']
            return result.head(top_k)
            
        if text_recs.empty:
            print("ℹ️ Nur parameterbasierte Empfehlungen verfügbar für Rank Fusion.")
            result = param_recs.copy()
            result['hybrid_score'] = result['final_score']
            return result.head(top_k)
        
        try:
            # Create rank dictionaries
            param_ranks = {hotel_id: rank+1 for rank, hotel_id in enumerate(param_recs['hotel_id'])}
            text_ranks = {hotel_id: rank+1 for rank, hotel_id in enumerate(text_recs['hotel_id'])}
            
            # Get all unique hotel IDs
            all_hotel_ids = set(param_recs['hotel_id']) | set(text_recs['hotel_id'])
            
            if not all_hotel_ids:
                print("⚠️ Keine Hotel-IDs gefunden für Rank Fusion.")
                return pd.DataFrame()
            
            # Calculate RRF scores
            k = 60  # RRF parameter
            rrr_scores = []
            
            # Standardgewichte verwenden
            weights = self.base_weights
            
            for hotel_id in all_hotel_ids:
                param_rank = param_ranks.get(hotel_id, len(param_recs) + 1)
                text_rank = text_ranks.get(hotel_id, len(text_recs) + 1)
                
                # RRF formula: 1/(k + rank)
                rrr_score = (
                    weights['parameter'] * (1 / (k + param_rank)) +
                    weights['text'] * (1 / (k + text_rank))
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
                
                hotel_entry = {
                    'hotel_id': hotel_id,
                    'hotel_name': hotel_info.get('hotel_name', hotel_info.get('name', 'Unknown')),
                    'hybrid_score': rrr_score,  # Direkter Name für die Konsistenz
                    'rrr_score': rrr_score,
                    'param_rank': param_rank,
                    'text_rank': text_rank,
                    'price': hotel_info.get('price', 0),
                    'rating': hotel_info.get('rating', 0),
                    'location': hotel_info.get('location', 'Unknown')
                }
                
                # Zusätzliche Spalten kopieren, falls vorhanden
                for key in ['description', 'amenities', 'similarity_score', 'final_score']:
                    if key in hotel_info:
                        hotel_entry[key] = hotel_info[key]
                
                rrr_scores.append(hotel_entry)
            
            # Convert to DataFrame and sort
            hybrid_df = pd.DataFrame(rrr_scores)
            
            # Überprüfe, ob das DataFrame leer ist
            if hybrid_df.empty:
                print("⚠️ Keine gemeinsamen Hotels zwischen den Modellen gefunden.")
                # Fallback: Einfach die besten Empfehlungen aus beiden Modellen nehmen
                combined_df = pd.concat([param_recs.head(top_k // 2), text_recs.head(top_k // 2)])
                combined_df['hybrid_score'] = combined_df['final_score']
                combined_df = combined_df.sort_values('hybrid_score', ascending=False)
                return combined_df.head(top_k)
            
            hybrid_df = hybrid_df.sort_values('rrr_score', ascending=False)
            
            # Remove duplicates by hotel name (keep highest score)
            if 'hotel_name' in hybrid_df.columns:
                hybrid_df = hybrid_df.drop_duplicates(subset=['hotel_name'], keep='first')
            
            return hybrid_df.head(top_k)
            
        except Exception as e:
            print(f"⚠️ Fehler beim Rank Fusion Ensemble: {e}")
            # Fallback: Einfach die besten Empfehlungen aus beiden Modellen nehmen
            try:
                combined_df = pd.concat([param_recs.head(max(1, top_k // 2)), text_recs.head(max(1, top_k // 2))])
                combined_df['hybrid_score'] = combined_df['final_score']
                combined_df = combined_df.sort_values('hybrid_score', ascending=False)
                return combined_df.head(top_k).drop_duplicates(subset=['hotel_id'], keep='first')
            except Exception as inner_e:
                print(f"⚠️ Auch Fallback fehlgeschlagen: {inner_e}")
                if not param_recs.empty:
                    return param_recs.head(top_k)
                elif not text_recs.empty:
                    return text_recs.head(top_k)
                else:
                    return pd.DataFrame()
    
    def _adaptive_ensemble(self, param_recs: pd.DataFrame, text_recs: pd.DataFrame,
                          query: str, user_preferences: Dict, top_k: int) -> pd.DataFrame:
        """Adaptive ensemble based on query type and user preferences"""
        print("🔄 Applying adaptive ensemble")
        
        # Überprüfe, ob beide DataFrames Daten enthalten
        if param_recs.empty and text_recs.empty:
            print("❌ Beide Empfehlungsmodelle liefern keine Ergebnisse für adaptives Ensemble.")
            return pd.DataFrame()
        
        # Wenn nur ein Modell Ergebnisse liefert, verwenden wir diese direkt
        if param_recs.empty:
            print("ℹ️ Nur textbasierte Empfehlungen verfügbar für adaptives Ensemble.")
            result = text_recs.copy()
            result['hybrid_score'] = result['final_score']
            return result.head(top_k)
            
        if text_recs.empty:
            print("ℹ️ Nur parameterbasierte Empfehlungen verfügbar für adaptives Ensemble.")
            result = param_recs.copy()
            result['hybrid_score'] = result['final_score']
            return result.head(top_k)
        
        try:
            # Analyze query and preferences
            query_tokens = query.lower().split()
            
            # Default weights
            param_weight = 0.5
            text_weight = 0.5
            
            # Check query type
            has_price_terms = any(word in query_tokens for word in ['cheap', 'budget', 'affordable', 'inexpensive', 'günstig', 'billig', 'preiswert'])
            has_quality_terms = any(word in query_tokens for word in ['luxury', 'premium', 'high-end', 'best', 'top', 'luxus', 'premium', 'hochwertig', 'beste'])
            has_location_terms = any(word in query_tokens for word in ['city', 'center', 'downtown', 'beach', 'mountains', 'lake', 'stadt', 'zentrum', 'strand', 'berg', 'see'])
            has_amenity_terms = any(word in query_tokens for word in ['pool', 'wifi', 'spa', 'restaurant', 'gym', 'breakfast', 'frühstück', 'schwimmbad', 'wlan'])
            has_specific_details = len(query) > 30  # Long query suggests specific needs
            
            # Adjust weights based on query type
            if has_price_terms or user_preferences.get('price_importance', 0) > 0.4:
                param_weight += 0.15
                text_weight -= 0.15
                
            if has_quality_terms:
                param_weight += 0.1
                text_weight -= 0.1
                
            if has_location_terms or has_amenity_terms:
                text_weight += 0.2
                param_weight -= 0.2
                
            if has_specific_details:
                text_weight += 0.15
                param_weight -= 0.15
            
            # Normalize weights
            total = param_weight + text_weight
            adaptive_weights = {
                'parameter': param_weight / total,
                'text': text_weight / total
            }
            
            print(f"Adaptive weights: Parameter={adaptive_weights['parameter']:.2f}, Text={adaptive_weights['text']:.2f}")
            
            # Use weighted average with adaptive weights
            return self._weighted_average_ensemble(param_recs, text_recs, adaptive_weights, top_k)
        
        except Exception as e:
            print(f"⚠️ Fehler beim adaptiven Ensemble: {e}")
            # Fallback zum weighted_average_ensemble mit Standardgewichten
            try:
                return self._weighted_average_ensemble(param_recs, text_recs, self.base_weights, top_k)
            except Exception as inner_e:
                print(f"⚠️ Auch Fallback fehlgeschlagen: {inner_e}")
                # Zweiter Fallback: Einfach die besten Empfehlungen aus beiden Modellen nehmen
                try:
                    combined_df = pd.concat([param_recs.head(max(1, top_k // 2)), text_recs.head(max(1, top_k // 2))])
                    combined_df['hybrid_score'] = combined_df['final_score']
                    combined_df = combined_df.sort_values('hybrid_score', ascending=False)
                    return combined_df.head(top_k).drop_duplicates(subset=['hotel_id'], keep='first')
                except:
                    if not param_recs.empty:
                        return param_recs.head(top_k)
                    elif not text_recs.empty:
                        return text_recs.head(top_k)
                    else:
                        return pd.DataFrame()
    
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
    
    def _apply_final_ranking_and_diversity(self, recs_df: pd.DataFrame, query: str, 
                                          user_preferences: Dict, top_k: int) -> pd.DataFrame:
        """Apply final ranking adjustments and add diversity"""
        if recs_df.empty:
            print("⚠️ Keine Empfehlungen für finales Ranking vorhanden.")
            return pd.DataFrame()
        
        try:
            results = recs_df.copy()
            
            # Überprüfe, ob genügend Empfehlungen für ein sinnvolles Diversity-Ranking vorhanden sind
            if len(results) <= 2:
                return results  # Bei nur 1-2 Hotels kein Diversity-Ranking nötig
                
            # Ensure we have a hybrid_score column
            if 'hybrid_score' not in results.columns:
                if 'rrr_score' in results.columns:
                    results['hybrid_score'] = results['rrr_score']
                elif 'final_score' in results.columns:
                    results['hybrid_score'] = results['final_score']
                else:
                    # Create a dummy score based on position
                    results['hybrid_score'] = [1.0 - (0.05 * i) for i in range(len(results))]
            
            # Check if we have location info to cluster
            if 'location' in results.columns and len(set(results['location'])) > 1:
                # Add diversity by penalizing similar locations
                locations = results['location'].values
                unique_locations = set(locations)
                
                location_counts = {}
                for loc in locations:
                    location_counts[loc] = location_counts.get(loc, 0) + 1
                
                # Apply diversity adjustment - reduce score for over-represented locations
                for i, row in results.iterrows():
                    loc = row['location']
                    count = location_counts.get(loc, 0)
                    if count > 1:
                        # Gradually reduce score for hotels from same location
                        diversity_penalty = 0.05 * (count - 1)  # 5% penalty per additional hotel
                        results.at[i, 'hybrid_score'] = results.at[i, 'hybrid_score'] * (1 - diversity_penalty)
            
            # Add price-quality ratio enhancement
            if 'price' in results.columns and 'rating' in results.columns:
                # Calculate price-quality ratio
                results['price_quality_ratio'] = results['rating'] / (results['price'] + 1)  # +1 to avoid div by zero
                
                # Normalize ratio
                if len(results) > 1:
                    min_ratio = results['price_quality_ratio'].min()
                    max_ratio = results['price_quality_ratio'].max()
                    
                    if abs(max_ratio - min_ratio) > 0.0001:
                        results['price_quality_bonus'] = (results['price_quality_ratio'] - min_ratio) / (max_ratio - min_ratio) * 0.1
                        
                        # Apply small bonus to final score
                        results['hybrid_score'] = results['hybrid_score'] + results['price_quality_bonus']
            
            # Final sorting
            final_results = results.sort_values('hybrid_score', ascending=False).head(top_k)
            return final_results
        
        except Exception as e:
            print(f"⚠️ Fehler beim finalen Ranking: {e}")
            # Im Fehlerfall einfach die ursprünglichen Empfehlungen zurückgeben
            try:
                if 'hybrid_score' not in recs_df.columns and 'final_score' in recs_df.columns:
                    recs_df['hybrid_score'] = recs_df['final_score']
                return recs_df.head(top_k)
            except:
                print("⚠️ Konnte auch keine ursprünglichen Empfehlungen zurückgeben.")
                return pd.DataFrame()
    
    def _add_hybrid_explanation_scores(self, hybrid_recs: pd.DataFrame, param_recs: pd.DataFrame, 
                                 text_recs: pd.DataFrame, weights: Dict) -> pd.DataFrame:
        """Add comprehensive explanation scores to hybrid recommendations"""
        if hybrid_recs.empty:
            return hybrid_recs  # Nichts zu tun bei leeren Empfehlungen
            
        try:
            results = hybrid_recs.copy()
            
            # Stelle sicher, dass wir die notwendigen Spalten haben
            hotel_ids = set(results['hotel_id'])
            
            # Create dictionaries for fast lookups
            param_dict = {}
            if not param_recs.empty:
                for _, row in param_recs.iterrows():
                    if 'hotel_id' in row and 'final_score' in row:
                        param_dict[row['hotel_id']] = row['final_score']
            
            text_dict = {}
            similarity_dict = {}
            if not text_recs.empty:
                for _, row in text_recs.iterrows():
                    if 'hotel_id' in row:
                        if 'final_score' in row:
                            text_dict[row['hotel_id']] = row['final_score']
                        if 'similarity_score' in row:
                            similarity_dict[row['hotel_id']] = row['similarity_score']
            
            # Add explanation scores
            for i, row in results.iterrows():
                hotel_id = row['hotel_id']
                
                # Add component scores
                results.at[i, 'parameter_score'] = param_dict.get(hotel_id, 0)
                results.at[i, 'text_score'] = text_dict.get(hotel_id, 0)
                results.at[i, 'text_similarity'] = similarity_dict.get(hotel_id, 0)
                
                # Calculate contribution percentages
                hybrid_score = results.at[i, 'hybrid_score']
                
                if hybrid_score > 0:
                    # Parameter model contribution
                    param_contrib = (param_dict.get(hotel_id, 0) * weights['parameter']) / hybrid_score
                    param_contrib = max(0, min(param_contrib, 1))  # Ensure between 0 and 1
                    
                    # Text model contribution
                    text_contrib = (text_dict.get(hotel_id, 0) * weights['text']) / hybrid_score
                    text_contrib = max(0, min(text_contrib, 1))  # Ensure between 0 and 1
                    
                    # Normalize to 100%
                    total = param_contrib + text_contrib
                    if total > 0:
                        param_contrib = param_contrib / total
                        text_contrib = text_contrib / total
                    else:
                        param_contrib = 0.5
                        text_contrib = 0.5
                else:
                    param_contrib = 0.5
                    text_contrib = 0.5
                
                # Store contribution percentages
                results.at[i, 'param_contrib'] = param_contrib
                results.at[i, 'text_contrib'] = text_contrib
            
            return results
            
        except Exception as e:
            print(f"⚠️ Fehler beim Hinzufügen von Erklärungswerten: {e}")
            return hybrid_recs  # Im Fehlerfall die ursprünglichen Empfehlungen zurückgeben
