"""
Parameter-Based Hotel Recommender using Advanced Regression Models
Recommends hotels based on numerical parameters and user preferences with improved feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from typing import List, Dict, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

class ParameterBasedRecommender:
    """Advanced parameter-based hotel recommender using multiple regression models with feature engineering"""
    
    def __init__(self, model_type: str = 'gradient_boosting', auto_tune: bool = True):
        """
        Initialize the recommender with enhanced capabilities
        
        Args:
            model_type: 'linear', 'ridge', 'elastic_net', 'random_forest', 'gradient_boosting'
            auto_tune: Whether to automatically tune hyperparameters
        """
        self.model_type = model_type
        self.auto_tune = auto_tune
        self.model = self._create_model()
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = None
        self.feature_columns = []
        self.is_trained = False
        self.feature_importance_ = None
        self.best_params_ = None
        
    def _create_model(self):
        """Create the regression model with improved configurations"""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            if self.auto_tune:
                return Ridge(random_state=42)  # Will be tuned later
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'elastic_net':
            if self.auto_tune:
                return ElasticNet(random_state=42, max_iter=2000)
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        elif self.model_type == 'random_forest':
            if self.auto_tune:
                return RandomForestRegressor(random_state=42, n_jobs=-1)
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            if self.auto_tune:
                return GradientBoostingRegressor(random_state=42)
            return GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
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
        Train the parameter-based model with advanced feature engineering and hyperparameter tuning
        
        Args:
            X: Feature matrix (DataFrame or numpy array)
            y: Target ratings (Series or numpy array)
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics
        """
        print(f"\n🤖 Training {self.model_type} model with advanced features...")
        
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
        
        # Remove any remaining NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=None
        )
        
        # Feature scaling (important for Ridge, ElasticNet)
        print("🔧 Applying feature scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Feature selection (select top K features)
        if len(self.feature_columns) > 50:  # Only if we have many features
            print("🎯 Applying feature selection...")
            k_features = min(50, len(self.feature_columns))  # Select top 50 features
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_val_scaled = self.feature_selector.transform(X_val_scaled)
            print(f"   Selected {k_features} best features")
        
        # Hyperparameter tuning if enabled
        if self.auto_tune:
            print("⚙️ Tuning hyperparameters...")
            self._tune_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Train the model
        print("🏋️ Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance_ = np.abs(self.model.coef_)
        
        self.is_trained = True
        
        metrics = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'best_params': self.best_params_
        }
        
        print(f"📊 Training completed!")
        print(f"   Train RMSE: {train_rmse:.3f} | Val RMSE: {val_rmse:.3f}")
        print(f"   Train R²: {train_r2:.3f} | Val R²: {val_r2:.3f}")
        
        return metrics
    
    def _tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Tune hyperparameters using GridSearchCV"""
        param_grids = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'elastic_net': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            }
        }
        
        if self.model_type in param_grids:
            grid_search = GridSearchCV(
                self.model, 
                param_grids[self.model_type],
                cv=3,  # 3-fold CV for speed
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            print(f"   Best parameters: {self.best_params_}")
        
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
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        scores = self.model.predict(X_scaled)
        
        # Create results dataframe with all hotel information
        results_df = hotels_with_features.copy()
        results_df['predicted_score'] = scores
        
        return results_df
    
    def recommend_hotels(self, hotels_df: pd.DataFrame, user_preferences: Dict, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend top-k hotels based on user preferences with advanced filtering and ranking
        
        Args:
            hotels_df: Hotels dataframe with engineered features
            user_preferences: User preference dictionary
            top_k: Number of recommendations
            
        Returns:
            Top-k recommended hotels with enhanced scoring
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        print(f"🎯 Generating parameter-based recommendations...")
        
        # Get base predictions
        predictions_df = self.predict_hotel_scores(hotels_df)
        
        # Apply user preference filters FIRST
        filtered_df = self._apply_user_filters(predictions_df, user_preferences)
        
        if filtered_df.empty:
            print("❌ No hotels match your criteria. Try relaxing your filters.")
            return pd.DataFrame()
        
        print(f"🔍 After filtering: {len(filtered_df)} hotels remain")
        
        # Apply sophisticated preference weighting
        weighted_df = self._apply_preference_weighting(filtered_df, user_preferences)
        
        # Add diversity scoring to avoid too similar hotels
        if len(weighted_df) > top_k:
            weighted_df = self._add_diversity_scoring(weighted_df, top_k)
        
        # Sort by final score and return top-k
        recommendations = weighted_df.sort_values('final_score', ascending=False).head(top_k)
        
        # Remove duplicates by hotel name (keep highest score)
        if 'name' in recommendations.columns:
            recommendations = recommendations.drop_duplicates(subset=['name'], keep='first')
        
        # Add explanation scores for transparency
        if len(recommendations) > 0:
            recommendations = self._add_explanation_scores(recommendations, user_preferences)
        
        print(f"✅ Generated {len(recommendations)} parameter-based recommendations")
        
        return recommendations.reset_index(drop=True)
    
    def _add_diversity_scoring(self, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """Add diversity scoring to avoid recommending too similar hotels"""
        if len(df) <= top_k:
            return df
        
        # Simple diversity based on location and price range
        df = df.copy()
        df['diversity_score'] = 0.0
        
        # Only consider top candidates for diversity (to avoid massive penalties)
        # Sort by final score first and only apply diversity to top candidates
        df_sorted = df.sort_values('final_score', ascending=False)
        top_candidates = df_sorted.head(top_k * 3)  # Consider 3x more candidates
        
        # Penalize hotels in the same location (much smaller penalty)
        location_counts = top_candidates['location'].value_counts()
        for location, count in location_counts.items():
            if count > 1:
                mask = top_candidates['location'] == location
                # Much smaller penalty: 0.01 instead of 0.1
                penalty = 0.01 * (count - 1)
                top_candidates.loc[mask, 'diversity_score'] -= penalty
        
        # Penalize hotels in the same price range (much smaller penalty)
        top_candidates['price_range'] = pd.cut(top_candidates['price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        price_counts = top_candidates['price_range'].value_counts()
        for price_range, count in price_counts.items():
            if count > 2:
                mask = top_candidates['price_range'] == price_range
                # Much smaller penalty: 0.005 instead of 0.05
                penalty = 0.005 * (count - 2)
                top_candidates.loc[mask, 'diversity_score'] -= penalty
        
        # Apply diversity bonus to final score
        top_candidates['final_score'] += top_candidates['diversity_score']
        
        # Update the original dataframe with the modified scores
        df.loc[top_candidates.index, 'final_score'] = top_candidates['final_score']
        
        return df
    
    def _add_explanation_scores(self, df: pd.DataFrame, user_preferences: Dict) -> pd.DataFrame:
        """Add explanation scores to show why hotels were recommended"""
        df = df.copy()
        
        # Calculate contribution of different factors
        if 'ml_score' in df.columns:
            df['model_contribution'] = df['ml_score']
        else:
            df['model_contribution'] = 0.5
        
        # Price contribution (based on actual price score)
        if 'price_score' in df.columns:
            df['price_contribution'] = df['price_score']
        else:
            max_price = user_preferences.get('max_price', df['price'].max())
            df['price_contribution'] = 1 - (df['price'] / max_price).clip(0, 1)
        
        # Rating contribution (based on actual rating score)
        if 'rating_score' in df.columns:
            df['rating_contribution'] = df['rating_score']
        else:
            min_rating = user_preferences.get('min_rating', 0)
            df['rating_contribution'] = (df['rating'] - min_rating) / (10 - min_rating) if min_rating < 10 else 0.5
        
        # Create explanation string
        df['explanation'] = df.apply(
            lambda row: f"Price: {row['price_contribution']:.2f}, "
                       f"Rating: {row['rating_contribution']:.2f}, "
                       f"ML Model: {row['model_contribution']:.2f}",
            axis=1
        )
        
        return df
        
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
    
    def _apply_user_filters(self, predictions_df: pd.DataFrame, preferences: Dict) -> pd.DataFrame:
        """Apply hard filters based on user preferences"""
        # Now predictions_df contains all hotel data including predicted_score
        filtered_df = predictions_df.copy()
        
        print(f"🔍 Starting with {len(filtered_df)} hotels")
        
        # Price filter
        if 'max_price' in preferences:
            max_price = preferences['max_price']
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['price'] <= max_price]
            print(f"🔍 Price filter (≤${max_price}): {before_count} → {len(filtered_df)} hotels")
        
        # Rating filter
        if 'min_rating' in preferences:
            min_rating = preferences['min_rating']
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
            print(f"🔍 Rating filter (≥{min_rating}): {before_count} → {len(filtered_df)} hotels")
        
        # Distance filter (if available)
        if 'max_distance' in preferences and 'distance_km' in filtered_df.columns:
            max_distance = preferences['max_distance']
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['distance_km'] <= max_distance]
            print(f"🔍 Distance filter (≤{max_distance}km): {before_count} → {len(filtered_df)} hotels")
        
        # Note: Amenities are now handled as preferences in scoring, not as hard filters
        
        return filtered_df
    
    def _apply_preference_weighting(self, filtered_df: pd.DataFrame, preferences: Dict) -> pd.DataFrame:
        """Apply preference weighting to scores using ML model predictions"""
        weighted_df = filtered_df.copy()
        
        # Get importance weights from preferences
        price_weight = preferences.get('price_importance', 0.3)
        rating_weight = preferences.get('rating_importance', 0.3)
        model_weight = preferences.get('model_importance', 0.4)  # ML model gets significant weight
        amenity_weight = preferences.get('amenity_importance', 0.1)  # New amenity weight
        
        # Normalize weights if amenities are included
        total_weight = price_weight + rating_weight + model_weight + amenity_weight
        price_weight = price_weight / total_weight
        rating_weight = rating_weight / total_weight
        model_weight = model_weight / total_weight
        amenity_weight = amenity_weight / total_weight
        
        # Calculate individual scores
        if len(weighted_df) > 1:
            user_max_price = preferences.get('max_price', weighted_df['price'].max())
            
            # Price scoring - encourage full budget utilization
            weighted_df['price_score'] = weighted_df['price'].apply(
                lambda p: self._balanced_price_score(p, user_max_price)
            )
            
            # Rating score (normalized to 0-1)
            weighted_df['rating_score'] = weighted_df['rating'] / 10.0
            
            # ML model score - use trained model predictions
            weighted_df['ml_score'] = self._get_ml_predictions(weighted_df)
            
            # Amenity score - bonus for preferred amenities
            weighted_df['amenity_score'] = self._calculate_amenity_score(weighted_df, preferences)
            
            # Final score - combining all components
            weighted_df['final_score'] = (
                price_weight * weighted_df['price_score'] +
                rating_weight * weighted_df['rating_score'] +
                model_weight * weighted_df['ml_score'] +
                amenity_weight * weighted_df['amenity_score']
            )
        else:
            weighted_df['price_score'] = 1.0
            weighted_df['rating_score'] = weighted_df['rating'] / 10.0
            weighted_df['ml_score'] = 0.5  # Default ML score
            weighted_df['amenity_score'] = 0.5  # Default amenity score
            weighted_df['final_score'] = (
                price_weight * weighted_df['price_score'] +
                rating_weight * weighted_df['rating_score'] +
                model_weight * weighted_df['ml_score'] +
                amenity_weight * weighted_df['amenity_score']
            )
        
        return weighted_df
    
    def _get_ml_predictions(self, filtered_df: pd.DataFrame) -> pd.Series:
        """
        Get ML model predictions for hotels
        
        Args:
            filtered_df: Filtered hotels DataFrame
            
        Returns:
            Series of normalized ML predictions (0-1 scale)
        """
        if not self.is_trained:
            print("⚠️ Model not trained, using default ML scores")
            return pd.Series([0.5] * len(filtered_df), index=filtered_df.index)
        
        # Use the predicted_score column that was already calculated
        if 'predicted_score' in filtered_df.columns:
            predictions = filtered_df['predicted_score'].values
        else:
            print("⚠️ No predicted_score column found, using default ML scores")
            return pd.Series([0.5] * len(filtered_df), index=filtered_df.index)
        
        # Normalize predictions to 0-1 scale
        if len(predictions) > 1:
            min_pred = predictions.min()
            max_pred = predictions.max()
            if max_pred > min_pred:
                normalized_preds = (predictions - min_pred) / (max_pred - min_pred)
            else:
                normalized_preds = np.ones_like(predictions) * 0.5
        else:
            normalized_preds = np.array([0.5])
        
        return pd.Series(normalized_preds, index=filtered_df.index)
    
    def _calculate_amenity_score(self, filtered_df: pd.DataFrame, preferences: Dict) -> pd.Series:
        """
        Calculate amenity score based on preferred amenities
        
        Args:
            filtered_df: Filtered hotels DataFrame
            preferences: User preferences including preferred_amenities
            
        Returns:
            Series of amenity scores (0-1 scale)
        """
        amenity_scores = pd.Series([0.5] * len(filtered_df), index=filtered_df.index)
        
        preferred_amenities = preferences.get('preferred_amenities', [])
        if not preferred_amenities:
            return amenity_scores
        
        # Calculate score based on how many preferred amenities the hotel has
        for i, (idx, hotel) in enumerate(filtered_df.iterrows()):
            matching_amenities = 0
            total_amenities = len(preferred_amenities)
            
            for amenity in preferred_amenities:
                amenity_col = f'has_{amenity}'
                if amenity_col in hotel.index and hotel[amenity_col] == 1:
                    matching_amenities += 1
            
            if total_amenities > 0:
                # Score from 0.3 (no matches) to 1.0 (all matches)
                base_score = 0.3
                bonus = 0.7 * (matching_amenities / total_amenities)
                amenity_scores.iloc[i] = base_score + bonus
            else:
                amenity_scores.iloc[i] = 0.5  # Neutral score if no preferences
        
        return amenity_scores

    def _balanced_price_score(self, price: float, max_budget: float) -> float:
        """
        Balanced price scoring that allows hotels across the full budget range
        
        Args:
            price: Hotel price
            max_budget: User's maximum budget
            
        Returns:
            Price score between 0 and 1
        """
        if price > max_budget:
            return 0.0
        
        # Use a more balanced approach that doesn't heavily favor expensive hotels
        price_ratio = price / max_budget
        
        # Sigmoid-like function that gives reasonable scores across the full range
        # but still provides some preference for using more of the budget
        if price_ratio <= 0.5:
            # For lower half of budget: score from 0.7 to 0.85
            return 0.7 + (price_ratio / 0.5) * 0.15
        else:
            # For upper half of budget: score from 0.85 to 1.0
            return 0.85 + ((price_ratio - 0.5) / 0.5) * 0.15
    
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
    
    try:
        from data_preparation.load_data import HotelDataLoader
        from data_preparation.feature_engineering import HotelFeatureEngineer
    except ImportError:
        print("⚠️ Could not import data modules. Please run from the modelling directory.")
        sys.exit(1)
    
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
            'max_price': 250,
            'min_rating': 4.0,
            'price_importance': 0.3,
            'rating_importance': 0.3,
            'model_importance': 0.4  # ML model gets significant weight
        }
        
        recommendations = recommender.recommend_hotels(features_df, user_prefs, top_k=5)
        
        print(f"\n🏨 Top 5 Hotel Recommendations:")
        print(recommendations[['hotel_name', 'final_score', 'price', 'rating', 'location']])
        
        # Feature importance
        importance = recommender.get_feature_importance()
        print(f"\n📊 Top Feature Importance:")
        print(importance.head(10))
