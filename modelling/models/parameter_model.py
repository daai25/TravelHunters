"""
Parameter-Based Hotel Recommender using Linear Regression
Recommends hotels based on numerical parameters and user preferences
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Dict, Tuple, Optional
import joblib

class ParameterBasedRecommender:
    """Parameter-based hotel recommender using regression models"""
    
    def __init__(self, model_type: str = 'ridge'):
        """
        Initialize the recommender
        
        Args:
            model_type: 'linear', 'ridge', or 'random_forest'
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.feature_columns = []
        self.is_trained = False
        
    def _create_model(self):
        """Create the regression model"""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_training_data(self, hotels_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from hotel features and user interactions
        
        Args:
            hotels_df: Hotel features dataframe
            interactions_df: User-hotel interactions dataframe
            
        Returns:
            X: Feature matrix
            y: Target ratings
        """
        # Merge hotels with interactions
        # Ensure compatible data types for merging
        if 'hotel_id' in interactions_df.columns and 'id' in hotels_df.columns:
            # Convert both to string to ensure compatibility
            interactions_df = interactions_df.copy()
            hotels_df = hotels_df.copy()
            interactions_df['hotel_id'] = interactions_df['hotel_id'].astype(str)
            hotels_df['id'] = hotels_df['id'].astype(str)
        
        merged_df = interactions_df.merge(hotels_df, left_on='hotel_id', right_on='id', how='inner')
        
        # Use interaction ratings as target (not hotel ratings from the hotel data)
        y = merged_df['rating_x'].values if 'rating_x' in merged_df.columns else merged_df['rating'].values
        
        # Select feature columns (excluding non-feature columns)
        exclude_cols = [
            'id', 'name', 'location', 'image_url', 'description', 'link',
            'amenities', 'latitude', 'longitude', 'user_id', 
            'hotel_id', 'rating', 'rating_x', 'rating_y', 'interaction_type', 
            'price_category', 'rating_category'
        ]
        
        feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = merged_df[feature_cols].fillna(0).values
        
        print(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Features: {feature_cols[:10]}...")  # Show first 10 features
        
        return X, y
    
    def train(self, X, y, validation_split: float = 0.2) -> Dict:
        """
        Train the parameter-based model
        
        Args:
            X: Feature matrix (DataFrame or numpy array)
            y: Target ratings (Series or numpy array)
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics
        """
        # Handle DataFrame input - select only numeric columns
        if isinstance(X, pd.DataFrame):
            # Select only numeric columns for training
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_columns].fillna(0)
            self.feature_columns = list(numeric_columns)
            X = X_numeric.values
            print(f"ℹ️  Using {len(self.feature_columns)} numeric features for training")
        
        # Handle Series input for y
        if isinstance(y, pd.Series):
            y = y.values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Validate
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'val_mse': mean_squared_error(y_val, y_pred_val),
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': r2_score(y_val, y_pred_val),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val))
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        print(f"✅ Model trained successfully!")
        print(f"Validation R²: {metrics['val_r2']:.3f}")
        print(f"Validation RMSE: {metrics['val_rmse']:.3f}")
        print(f"CV R² (mean ± std): {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
        
        return metrics
    
    def predict_hotel_scores(self, hotels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict scores for all hotels
        
        Args:
            hotels_df: Hotels DataFrame (original or with engineered features)
            
        Returns:
            DataFrame with hotel IDs and predicted scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Check if we have the required feature columns
        available_columns = list(hotels_df.columns)
        missing_features = [col for col in self.feature_columns if col not in available_columns]
        
        if missing_features:
            # If features are missing, use feature engineering to create them
            print(f"ℹ️  Creating missing features: {missing_features[:5]}...")
            from data_preparation.feature_engineering import HotelFeatureEngineer
            engineer = HotelFeatureEngineer()
            hotels_with_features = engineer.engineer_numerical_features(hotels_df)
        else:
            hotels_with_features = hotels_df
        
        # Prepare features
        X = hotels_with_features[self.feature_columns].fillna(0).values
        
        # Predict scores
        scores = self.model.predict(X)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'hotel_id': hotels_df['id'],
            'predicted_score': scores
        })
        
        return results_df
    
    def recommend_hotels(self, hotels_df: pd.DataFrame, user_preferences: Dict, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend top-k hotels based on user preferences
        
        Args:
            hotels_df: Hotels with engineered features
            user_preferences: User preference dictionary
            top_k: Number of recommendations
            
        Returns:
            Top-k recommended hotels
        """
        # Get base predictions
        predictions_df = self.predict_hotel_scores(hotels_df)
        
        # Apply user preference filters
        filtered_df = self._apply_user_filters(predictions_df, hotels_df, user_preferences)
        
        # Apply preference weighting
        weighted_df = self._apply_preference_weighting(filtered_df, user_preferences)
        
        # Sort by final score and return top-k
        recommendations = weighted_df.sort_values('final_score', ascending=False)
        
        # Remove duplicates by hotel name (keep highest score)
        if 'name' in recommendations.columns:
            # Add hotel_name column for consistency with other models
            recommendations['hotel_name'] = recommendations['name']
            recommendations = recommendations.drop_duplicates(subset=['hotel_name'], keep='first')
        
        # Return top-k after deduplication
        recommendations = recommendations.head(top_k)
        
        # Return with available columns
        available_columns = ['hotel_id', 'predicted_score', 'final_score']
        if 'hotel_name' in recommendations.columns:
            available_columns.append('hotel_name')
        if 'name' in recommendations.columns:
            available_columns.append('name')
        if 'price' in recommendations.columns:
            available_columns.append('price')
        if 'rating' in recommendations.columns:
            available_columns.append('rating')
        if 'location' in recommendations.columns:
            available_columns.append('location')
        
        return recommendations[available_columns]
    
    def _apply_user_filters(self, predictions_df: pd.DataFrame, hotels_df: pd.DataFrame,
                           preferences: Dict) -> pd.DataFrame:
        """Apply hard filters based on user preferences"""
        # Join predictions with hotel data for filtering
        filtered_df = predictions_df.merge(
            hotels_df[['id', 'name', 'price', 'rating', 'location', 'amenities']],
            left_on='hotel_id', right_on='id', how='left'
        )
        
        # Price filter
        if 'max_price' in preferences:
            max_price = preferences['max_price']
            filtered_df = filtered_df[filtered_df['price'] <= max_price]
        
        # Rating filter
        if 'min_rating' in preferences:
            min_rating = preferences['min_rating']
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        
        # Distance filter (if available)
        if 'max_distance' in preferences and 'distance_km' in hotels_df.columns:
            max_distance = preferences['max_distance']
            hotel_distances = hotels_df.set_index('id')['distance_km']
            valid_hotels = hotel_distances[hotel_distances <= max_distance].index
            filtered_df = filtered_df[filtered_df['hotel_id'].isin(valid_hotels)]
        
        # Amenities filter
        if 'required_amenities' in preferences:
            required_amenities = preferences['required_amenities']
            if required_amenities:
                hotels_with_amenities = hotels_df.set_index('id')
                for amenity in required_amenities:
                    amenity_col = f'has_{amenity}'
                    if amenity_col in hotels_with_amenities.columns:
                        valid_hotels = hotels_with_amenities[hotels_with_amenities[amenity_col] == 1].index
                        filtered_df = filtered_df[filtered_df['hotel_id'].isin(valid_hotels)]
        
        return filtered_df
    
    def _apply_preference_weighting(self, filtered_df: pd.DataFrame, preferences: Dict) -> pd.DataFrame:
        """Apply preference weighting to scores"""
        weighted_df = filtered_df.copy()
        
        # Default weights
        price_weight = preferences.get('price_importance', 0.3)
        rating_weight = preferences.get('rating_importance', 0.4)
        model_weight = preferences.get('model_importance', 0.3)
        
        # Normalize scores to 0-1 range
        if len(weighted_df) > 1:
            # Price score (lower price = higher score)
            max_price = weighted_df['price'].max()
            min_price = weighted_df['price'].min()
            if max_price > min_price:
                weighted_df['price_score'] = 1 - (weighted_df['price'] - min_price) / (max_price - min_price)
            else:
                weighted_df['price_score'] = 1.0
            
            # Rating score (normalized to 0-1) - using 10-point scale
            weighted_df['rating_score'] = weighted_df['rating'] / 10.0
            
            # Model score (normalized to 0-1)
            max_pred = weighted_df['predicted_score'].max()
            min_pred = weighted_df['predicted_score'].min()
            if max_pred > min_pred:
                weighted_df['model_score'] = (weighted_df['predicted_score'] - min_pred) / (max_pred - min_pred)
            else:
                weighted_df['model_score'] = 0.5
        else:
            weighted_df['price_score'] = 1.0
            weighted_df['rating_score'] = weighted_df['rating'] / 10.0
            weighted_df['model_score'] = 0.5
        
        # Calculate final weighted score
        weighted_df['final_score'] = (
            price_weight * weighted_df['price_score'] +
            rating_weight * weighted_df['rating_score'] +
            model_weight * weighted_df['model_score']
        )
        
        return weighted_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (for tree-based models)"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': self.model.coef_,
                'abs_coefficient': np.abs(self.model.coef_)
            }).sort_values('abs_coefficient', ascending=False)
        else:
            importance_df = pd.DataFrame({'feature': self.feature_columns})
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = True
        print(f"✅ Model loaded from {filepath}")

if __name__ == "__main__":
    # Test the parameter-based recommender
    import sys
    sys.path.append('../data_preparation')
    from load_data import HotelDataLoader
    from feature_engineering import HotelFeatureEngineer
    
    # Load data
    loader = HotelDataLoader()
    hotels_df = loader.load_hotels()
    interactions_df = loader.load_user_interactions()
    
    if not hotels_df.empty and not interactions_df.empty:
        # Engineer features
        engineer = HotelFeatureEngineer()
        features_df, feature_cols = engineer.prepare_parameter_features(hotels_df)
        
        # Create and train model
        recommender = ParameterBasedRecommender(model_type='ridge')
        X, y = recommender.prepare_training_data(features_df, interactions_df)
        metrics = recommender.train(X, y)
        
        # Test recommendations
        user_prefs = {
            'max_price': 150,
            'min_rating': 4.0,
            'price_importance': 0.4,
            'rating_importance': 0.4,
            'model_importance': 0.2
        }
        
        recommendations = recommender.recommend_hotels(features_df, user_prefs, top_k=5)
        
        print(f"\n🏨 Top 5 Hotel Recommendations:")
        print(recommendations[['hotel_name', 'final_score', 'price', 'rating', 'location']])
        
        # Feature importance
        importance = recommender.get_feature_importance()
        print(f"\n📊 Top Feature Importance:")
        print(importance.head(10))
