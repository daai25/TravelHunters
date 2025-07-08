"""
Text-Based Hotel Recommender using NLP Similarity
Recommends hotels based on text similarity to user queries
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import re
from typing import List, Dict, Tuple, Optional
import joblib

class TextBasedRecommender:
    """Text-based hotel recommender using TF-IDF and cosine similarity"""
    
    def __init__(self, max_features: int = 1000, use_lsa: bool = True, lsa_components: int = 100):
        """
        Initialize the text-based recommender
        
        Args:
            max_features: Maximum number of TF-IDF features
            use_lsa: Whether to use Latent Semantic Analysis (SVD)
            lsa_components: Number of LSA components
        """
        self.max_features = max_features
        self.use_lsa = use_lsa
        self.lsa_components = lsa_components
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            lowercase=True,
            strip_accents='unicode'
        )
        
        # LSA for dimensionality reduction
        self.lsa_model = TruncatedSVD(n_components=lsa_components, random_state=42) if use_lsa else None
        
        # Store processed data
        self.hotel_texts = []
        self.hotel_ids = []
        self.tfidf_matrix = None
        self.lsa_matrix = None
        self.is_fitted = False
        
    def prepare_hotel_texts(self, hotels_df: pd.DataFrame) -> List[str]:
        """
        Prepare text representations of hotels
        
        Args:
            hotels_df: Hotels dataframe with text features
            
        Returns:
            List of text representations
        """
        hotel_texts = []
        
        for idx, hotel in hotels_df.iterrows():
            text_parts = []
            
            # Hotel name (high importance)
            if pd.notna(hotel['name']):
                name = str(hotel['name']).strip()
                text_parts.extend([name] * 3)  # Repeat for emphasis
            
            # Location
            if pd.notna(hotel['location']):
                location = str(hotel['location']).strip()
                text_parts.append(location)
            
            # Description (most important)
            if pd.notna(hotel['description']):
                description = str(hotel['description']).strip()
                # Clean and expand description
                description = self._clean_text(description)
                text_parts.extend([description] * 2)  # Repeat for emphasis
            
            # Amenities
            if pd.notna(hotel['amenities']):
                amenities = self._process_amenities_text(hotel['amenities'])
                text_parts.append(amenities)
            
            # Price category
            if pd.notna(hotel.get('price')):
                price_category = self._categorize_price(hotel['price'])
                text_parts.append(price_category)
            
            # Rating category
            if pd.notna(hotel.get('rating')):
                rating_category = self._categorize_rating(hotel['rating'])
                text_parts.append(rating_category)
            
            # Hotel type inference from name
            hotel_type = self._infer_hotel_type(str(hotel['name']) if pd.notna(hotel['name']) else "")
            if hotel_type:
                text_parts.append(hotel_type)
            
            # Combine all text parts
            combined_text = ' '.join(text_parts)
            hotel_texts.append(self._clean_text(combined_text))
        
        return hotel_texts
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _process_amenities_text(self, amenities: str) -> str:
        """Process amenities into clean text"""
        try:
            import json
            # Try parsing as JSON
            if amenities.startswith('[') or amenities.startswith('{'):
                amenity_list = json.loads(amenities)
                if isinstance(amenity_list, list):
                    return ' '.join(amenity_list)
                else:
                    return str(amenity_list)
            else:
                # Clean string representation
                amenities = amenities.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                return amenities
        except:
            # Fallback to string processing
            amenities = str(amenities).replace('[', '').replace(']', '').replace('"', '').replace("'", "")
            return amenities
    
    def _categorize_price(self, price: float) -> str:
        """Categorize price into text description"""
        if price <= 50:
            return "budget cheap affordable economy"
        elif price <= 100:
            return "mid-range moderate standard"
        elif price <= 200:
            return "premium upscale quality"
        else:
            return "luxury expensive high-end exclusive"
    
    def _categorize_rating(self, rating: float) -> str:
        """Categorize rating into text description"""
        if rating <= 3.5:
            return "basic decent"
        elif rating <= 4.0:
            return "good quality recommended"
        elif rating <= 4.5:
            return "very good excellent superior"
        else:
            return "outstanding exceptional perfect top-rated"
    
    def _infer_hotel_type(self, name: str) -> str:
        """Infer hotel type from name"""
        name_lower = name.lower()
        
        hotel_types = {
            'resort': 'resort vacation leisure spa',
            'inn': 'inn cozy intimate boutique',
            'suite': 'suite apartment extended-stay family',
            'hotel': 'hotel standard accommodation',
            'motel': 'motel budget roadside',
            'hostel': 'hostel budget backpacker social',
            'b&b': 'bed-breakfast intimate personal',
            'lodge': 'lodge rustic nature outdoor',
            'villa': 'villa luxury private exclusive'
        }
        
        for keyword, description in hotel_types.items():
            if keyword in name_lower:
                return description
        
        return ""
    
    def fit(self, hotels_df: pd.DataFrame):
        """
        Fit the text-based model on hotel data
        
        Args:
            hotels_df: Hotels dataframe
        """
        # Prepare hotel texts
        self.hotel_texts = self.prepare_hotel_texts(hotels_df)
        self.hotel_ids = hotels_df['id'].tolist()
        
        # Fit TF-IDF vectorizer
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.hotel_texts)
        
        # Apply LSA if enabled
        if self.use_lsa and self.lsa_model:
            self.lsa_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)
        
        self.is_fitted = True
        
        print(f"✅ Text model fitted successfully!")
        print(f"Hotels processed: {len(self.hotel_texts)}")
        print(f"TF-IDF features: {self.tfidf_matrix.shape[1]}")
        if self.use_lsa:
            print(f"LSA components: {self.lsa_matrix.shape[1]}")
    
    def search_hotels(self, query: str, top_k: int = 10, similarity_threshold: float = 0.1) -> pd.DataFrame:
        """
        Search hotels based on text query
        
        Args:
            query: User search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            DataFrame with hotel recommendations and similarity scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before searching")
        
        # Clean and vectorize query
        cleaned_query = self._clean_text(query)
        query_vector = self.tfidf_vectorizer.transform([cleaned_query])
        
        # Use LSA if available
        if self.use_lsa and self.lsa_model:
            query_lsa = self.lsa_model.transform(query_vector)
            similarity_scores = cosine_similarity(query_lsa, self.lsa_matrix).flatten()
        else:
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'hotel_id': self.hotel_ids,
            'similarity_score': similarity_scores,
            'hotel_text': self.hotel_texts
        })
        
        # Filter by threshold and sort
        results_df = results_df[results_df['similarity_score'] >= similarity_threshold]
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        return results_df.head(top_k)
    
    def recommend_hotels(self, query: str, hotels_df: pd.DataFrame, user_preferences: Dict = None, 
                        top_k: int = 10) -> pd.DataFrame:
        """
        Recommend hotels based on text query and preferences
        
        Args:
            query: User search query
            hotels_df: Original hotels dataframe
            user_preferences: Additional preference filters
            top_k: Number of recommendations
            
        Returns:
            DataFrame with detailed hotel recommendations
        """
        # Get text-based similarities
        text_results = self.search_hotels(query, top_k=top_k*2)  # Get more for filtering
        
        # Merge with hotel details
        detailed_results = text_results.merge(
            hotels_df[['id', 'name', 'location', 'price', 'rating', 'description']], 
            left_on='hotel_id', 
            right_on='id', 
            how='inner'
        )
        
        # Apply user preference filters if provided
        if user_preferences:
            detailed_results = self._apply_text_filters(detailed_results, user_preferences)
        
        # Apply preference weighting
        if user_preferences:
            detailed_results = self._apply_text_weighting(detailed_results, user_preferences)
        else:
            detailed_results['final_score'] = detailed_results['similarity_score']
        
        # Sort and return top-k
        final_results = detailed_results.sort_values('final_score', ascending=False).head(top_k)
        
        return final_results[['hotel_id', 'name', 'final_score', 'similarity_score', 
                             'price', 'rating', 'location', 'description']]
    
    def _apply_text_filters(self, results_df: pd.DataFrame, preferences: Dict) -> pd.DataFrame:
        """Apply preference filters to text search results"""
        filtered_df = results_df.copy()
        
        # Price filter
        if 'max_price' in preferences:
            filtered_df = filtered_df[filtered_df['price'] <= preferences['max_price']]
        
        # Rating filter
        if 'min_rating' in preferences:
            filtered_df = filtered_df[filtered_df['rating'] >= preferences['min_rating']]
        
        return filtered_df
    
    def _apply_text_weighting(self, results_df: pd.DataFrame, preferences: Dict) -> pd.DataFrame:
        """Apply preference weighting to combine text similarity with other factors"""
        weighted_df = results_df.copy()
        
        # Default weights
        text_weight = preferences.get('text_importance', 0.6)
        price_weight = preferences.get('price_importance', 0.2)
        rating_weight = preferences.get('rating_importance', 0.2)
        
        # Normalize scores
        if len(weighted_df) > 1:
            # Text similarity is already 0-1
            weighted_df['text_score'] = weighted_df['similarity_score']
            
            # Price score (lower price = higher score)
            max_price = weighted_df['price'].max()
            min_price = weighted_df['price'].min()
            if max_price > min_price:
                weighted_df['price_score'] = 1 - (weighted_df['price'] - min_price) / (max_price - min_price)
            else:
                weighted_df['price_score'] = 1.0
            
            # Rating score
            weighted_df['rating_score'] = weighted_df['rating'] / 5.0
        else:
            weighted_df['text_score'] = weighted_df['similarity_score']
            weighted_df['price_score'] = 1.0
            weighted_df['rating_score'] = weighted_df['rating'] / 5.0
        
        # Calculate final weighted score
        weighted_df['final_score'] = (
            text_weight * weighted_df['text_score'] +
            price_weight * weighted_df['price_score'] +
            rating_weight * weighted_df['rating_score']
        )
        
        return weighted_df
    
    def get_query_keywords(self, query: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from query based on TF-IDF"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        cleaned_query = self._clean_text(query)
        query_vector = self.tfidf_vectorizer.transform([cleaned_query])
        
        # Get feature names and scores
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        scores = query_vector.toarray()[0]
        
        # Get top keywords
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores = [(word, score) for word, score in keyword_scores if score > 0]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, score in keyword_scores[:top_k]]
    
    def save_model(self, filepath: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'lsa_model': self.lsa_model,
            'hotel_texts': self.hotel_texts,
            'hotel_ids': self.hotel_ids,
            'tfidf_matrix': self.tfidf_matrix,
            'lsa_matrix': self.lsa_matrix,
            'max_features': self.max_features,
            'use_lsa': self.use_lsa,
            'lsa_components': self.lsa_components
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Text model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a fitted model"""
        model_data = joblib.load(filepath)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.lsa_model = model_data['lsa_model']
        self.hotel_texts = model_data['hotel_texts']
        self.hotel_ids = model_data['hotel_ids']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.lsa_matrix = model_data['lsa_matrix']
        self.max_features = model_data['max_features']
        self.use_lsa = model_data['use_lsa']
        self.lsa_components = model_data['lsa_components']
        self.is_fitted = True
        
        print(f"✅ Text model loaded from {filepath}")

if __name__ == "__main__":
    # Test the text-based recommender
    import sys
    sys.path.append('../data_preparation')
    from load_data import HotelDataLoader
    from feature_engineering import HotelFeatureEngineer
    
    # Load data
    loader = HotelDataLoader()
    hotels_df = loader.load_hotels()
    
    if not hotels_df.empty:
        # Engineer features for text processing
        engineer = HotelFeatureEngineer()
        features_df, _ = engineer.prepare_parameter_features(hotels_df)
        
        # Create and fit text model
        text_recommender = TextBasedRecommender(max_features=500, use_lsa=True)
        text_recommender.fit(features_df)
        
        # Test text search
        test_queries = [
            "cheap family friendly hotel with pool",
            "luxury spa resort with breakfast",
            "budget accommodation near city center",
            "business hotel with wifi and gym"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Get keywords
            keywords = text_recommender.get_query_keywords(query, top_k=5)
            print(f"Keywords: {keywords}")
            
            # Get recommendations
            user_prefs = {
                'max_price': 200,
                'min_rating': 3.5,
                'text_importance': 0.7,
                'price_importance': 0.2,
                'rating_importance': 0.1
            }
            
            recommendations = text_recommender.recommend_hotels(
                query, features_df, user_prefs, top_k=3
            )
            
            print(f"Top 3 recommendations:")
            for _, hotel in recommendations.iterrows():
                print(f"  - {hotel['name']} (Score: {hotel['final_score']:.3f}, "
                      f"Price: ${hotel['price']:.0f}, Rating: {hotel['rating']:.1f})")
