"""
Feature Engineering for TravelHunters Hotel Recommendation
Prepares features for both parameter-based and text-based models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json
from typing import List, Dict, Tuple

class HotelFeatureEngineer:
    """Feature engineering for hotel recommendation models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
    def engineer_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create numerical features for parameter-based model"""
        features_df = df.copy()
        
        # Basic numerical features
        features_df['price_log'] = np.log1p(features_df['price'])
        features_df['rating_normalized'] = features_df['rating'] / 5.0
        features_df['review_count_log'] = np.log1p(features_df['review_count'])
        
        # Distance features
        if 'distance_to_center' in features_df.columns:
            features_df['distance_km'] = pd.to_numeric(features_df['distance_to_center'], errors='coerce')
            features_df['is_city_center'] = (features_df['distance_km'] <= 1.0).astype(int)
        else:
            features_df['distance_km'] = 0
            features_df['is_city_center'] = 0
        
        # Price categories
        features_df['price_category'] = pd.cut(
            features_df['price'], 
            bins=[0, 50, 100, 200, float('inf')], 
            labels=['budget', 'mid', 'premium', 'luxury']
        )
        
        # Rating categories
        features_df['rating_category'] = pd.cut(
            features_df['rating'],
            bins=[0, 3.5, 4.0, 4.5, 5.0],
            labels=['poor', 'good', 'very_good', 'excellent']
        )
        
        # Amenities processing
        features_df = self._process_amenities(features_df)
        
        return features_df
    
    def _process_amenities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process amenities into binary features"""
        common_amenities = [
            'wifi', 'pool', 'gym', 'parking', 'breakfast', 
            'spa', 'restaurant', 'bar', 'room_service', 'air_conditioning'
        ]
        
        # Initialize amenity columns
        for amenity in common_amenities:
            df[f'has_{amenity}'] = 0
        
        # Process amenities column (assuming it's JSON or comma-separated)
        for idx, amenities in df['amenities'].fillna('').items():
            if amenities:
                try:
                    # Try parsing as JSON
                    if amenities.startswith('[') or amenities.startswith('{'):
                        amenity_list = json.loads(amenities)
                        if isinstance(amenity_list, list):
                            amenity_text = ' '.join(amenity_list).lower()
                        else:
                            amenity_text = str(amenity_list).lower()
                    else:
                        amenity_text = str(amenities).lower()
                    
                    # Check for common amenities
                    for amenity in common_amenities:
                        if amenity.replace('_', ' ') in amenity_text or amenity in amenity_text:
                            df.loc[idx, f'has_{amenity}'] = 1
                            
                except (json.JSONDecodeError, TypeError):
                    # Fallback to string processing
                    amenity_text = str(amenities).lower()
                    for amenity in common_amenities:
                        if amenity.replace('_', ' ') in amenity_text:
                            df.loc[idx, f'has_{amenity}'] = 1
        
        return df
    
    def engineer_text_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Create text features for NLP-based model"""
        # Combine relevant text fields
        text_features = []
        
        for idx, row in df.iterrows():
            text_parts = []
            
            # Hotel name
            if pd.notna(row['name']):
                text_parts.append(str(row['name']))
            
            # Description
            if pd.notna(row['description']):
                text_parts.append(str(row['description']))
            
            # Location
            if pd.notna(row['location']):
                text_parts.append(str(row['location']))
            
            # Amenities as text
            if pd.notna(row['amenities']):
                amenities_text = str(row['amenities']).replace('[', '').replace(']', '').replace('"', '')
                text_parts.append(amenities_text)
            
            # Price category as text
            if 'price_category' in row:
                text_parts.append(f"price_{row['price_category']}")
            
            # Rating category as text
            if 'rating_category' in row:
                text_parts.append(f"rating_{row['rating_category']}")
            
            combined_text = ' '.join(text_parts)
            text_features.append(self._clean_text(combined_text))
        
        # Create TF-IDF features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        
        return tfidf_matrix, self.tfidf_vectorizer
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_cols: List[str], fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        scaled_df = df.copy()
        
        if fit:
            scaled_df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols].fillna(0))
        else:
            scaled_df[numerical_cols] = self.scaler.transform(df[numerical_cols].fillna(0))
        
        return scaled_df
    
    def prepare_parameter_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for parameter-based model"""
        # Engineer features
        features_df = self.engineer_numerical_features(df)
        
        # Define feature columns for the model
        numerical_features = [
            'price_log', 'rating_normalized', 'review_count_log', 'distance_km', 'is_city_center'
        ]
        
        amenity_features = [col for col in features_df.columns if col.startswith('has_')]
        
        all_features = numerical_features + amenity_features
        
        # Handle missing values
        for col in all_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0)
        
        # Scale numerical features
        features_df = self.scale_numerical_features(features_df, numerical_features)
        
        return features_df, all_features
    
    def create_user_preference_vector(self, preferences: Dict) -> np.ndarray:
        """Create feature vector from user preferences"""
        # Default preferences
        default_prefs = {
            'max_price': 200,
            'min_rating': 4.0,
            'max_distance': 5.0,
            'amenities': [],
            'price_importance': 0.3,
            'rating_importance': 0.4,
            'distance_importance': 0.2,
            'amenities_importance': 0.1
        }
        
        # Update with user preferences
        prefs = {**default_prefs, **preferences}
        
        # Create preference vector (this will be used to weight features)
        pref_vector = {
            'price_weight': 1.0 - (prefs['max_price'] / 500.0),  # Lower price preference = higher weight
            'rating_weight': prefs['min_rating'] / 5.0,
            'distance_weight': 1.0 - (prefs['max_distance'] / 20.0),
            'importance_weights': [
                prefs['price_importance'],
                prefs['rating_importance'], 
                prefs['distance_importance'],
                prefs['amenities_importance']
            ]
        }
        
        return pref_vector

    def add_feature_noise(self, features_df: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
        """
        Add small amounts of noise to numerical features to increase training robustness
        
        Args:
            features_df: DataFrame with engineered features
            noise_level: Standard deviation of noise relative to feature values
            
        Returns:
            DataFrame with noise added to numerical features
        """
        print(f"🔄 Adding feature noise for robustness (noise level: {noise_level})...")
        
        augmented_df = features_df.copy()
        
        # Numerical features to add noise to
        numerical_features = [
            'price', 'rating', 'review_count', 'distance_km',
            'price_log', 'rating_normalized', 'review_count_log'
        ]
        
        for feature in numerical_features:
            if feature in augmented_df.columns:
                # Calculate noise based on feature values
                feature_values = augmented_df[feature]
                noise_std = feature_values.std() * noise_level
                
                # Add gaussian noise
                noise = np.random.normal(0, noise_std, len(augmented_df))
                augmented_df[feature] = feature_values + noise
                
                # Ensure values stay within reasonable bounds
                if feature == 'rating':
                    augmented_df[feature] = np.clip(augmented_df[feature], 1.0, 10.0)
                elif feature == 'rating_normalized':
                    augmented_df[feature] = np.clip(augmented_df[feature], 0.0, 2.0)
                elif feature == 'price':
                    augmented_df[feature] = np.maximum(augmented_df[feature], 10.0)  # Minimum price
                elif feature == 'review_count':
                    augmented_df[feature] = np.maximum(augmented_df[feature], 1.0)   # Minimum reviews
                elif feature == 'distance_km':
                    augmented_df[feature] = np.maximum(augmented_df[feature], 0.0)   # Non-negative distance
        
        # Recalculate some derived features that might be affected
        if 'rating_normalized' in augmented_df.columns:
            augmented_df['rating_normalized'] = augmented_df['rating'] / 10.0  # Recalculate based on new rating
        
        if 'is_city_center' in augmented_df.columns and 'distance_km' in augmented_df.columns:
            augmented_df['is_city_center'] = (augmented_df['distance_km'] <= 1.0).astype(int)
        
        print(f"✅ Feature noise added to {len(numerical_features)} numerical features")
        
        return augmented_df

if __name__ == "__main__":
    # Test feature engineering
    from load_data import HotelDataLoader
    
    loader = HotelDataLoader()
    hotels_df = loader.load_hotels()
    
    if not hotels_df.empty:
        engineer = HotelFeatureEngineer()
        
        # Test numerical features
        features_df, feature_cols = engineer.prepare_parameter_features(hotels_df)
        print(f"✅ Created {len(feature_cols)} numerical features")
        print(f"Features: {feature_cols}")
        
        # Test text features
        tfidf_matrix, vectorizer = engineer.engineer_text_features(features_df)
        print(f"✅ Created TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Test user preferences
        user_prefs = {
            'max_price': 150,
            'min_rating': 4.5,
            'amenities': ['wifi', 'pool']
        }
        pref_vector = engineer.create_user_preference_vector(user_prefs)
        print(f"✅ User preference vector created")
        
        print(f"\n📊 Feature Summary:")
        print(f"Hotels processed: {len(features_df)}")
        print(f"Numerical features: {len([c for c in feature_cols if not c.startswith('has_')])}")
        print(f"Amenity features: {len([c for c in feature_cols if c.startswith('has_')])}")
        print(f"Text features (TF-IDF): {tfidf_matrix.shape[1]}")
