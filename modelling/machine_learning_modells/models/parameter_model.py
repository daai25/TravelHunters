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
            
        # Sicherstellen, dass der Scaler richtig initialisiert ist
        if not hasattr(self.scaler, 'n_samples_seen_') or self.scaler.n_samples_seen_ is None:
            try:
                if 'rating' in hotels_df.columns:
                    self.scaler.fit(hotels_df[['rating']].values)
                else:
                    import numpy as np
                    self.scaler.fit(np.array([[5.0], [10.0]]))
            except Exception as e:
                print(f"⚠️ Warnung bei Scaler-Initialisierung: {e}")
                # Als Fallback neu erstellen
                try:
                    from sklearn.preprocessing import RobustScaler
                    import numpy as np
                    self.scaler = RobustScaler()
                    self.scaler.fit(np.array([[5.0], [10.0]]))
                except Exception as e2:
                    print(f"❌ Fehler bei Scaler-Erstellung: {e2}")
        
        print(f"🎯 Generating parameter-based recommendations...")
        
        try:
            # Get base predictions
            predictions_df = self.predict_hotel_scores(hotels_df)
            
            # Apply user preference filters FIRST
            filtered_df = self._apply_user_filters(predictions_df, user_preferences)
            
            if filtered_df.empty:
                print("❌ No hotels match your criteria. Trying relaxed filters.")
                
                # Automatisch Filter entspannen und erneut versuchen
                relaxed_preferences = user_preferences.copy()
                
                # Stufenweise Entspannung der Filter
                relaxation_steps = [
                    {"min_rating_change": -1.0, "max_price_multiplier": 1.25, "message": "Etwas entspannte Filter anwenden..."},
                    {"min_rating_change": -2.0, "max_price_multiplier": 1.5, "message": "Stark entspannte Filter anwenden..."},
                    {"min_rating_change": -5.0, "max_price_multiplier": 2.0, "message": "Minimale Filter anwenden..."},
                ]
                
                for step in relaxation_steps:
                    print(f"⚠️ {step['message']}")
                    
                    # Bewertungsfilter entspannen
                    if 'min_rating' in relaxed_preferences:
                        original_rating = relaxed_preferences['min_rating']
                        relaxed_preferences['min_rating'] = max(0.0, original_rating + step['min_rating_change'])
                    
                    # Preisfilter entspannen
                    if 'max_price' in relaxed_preferences:
                        original_price = relaxed_preferences['max_price']
                        relaxed_preferences['max_price'] = original_price * step['max_price_multiplier']
                    
                    # Erneut mit entspannten Filtern versuchen
                    filtered_df = self._apply_user_filters(predictions_df, relaxed_preferences)
                    
                    if not filtered_df.empty and len(filtered_df) >= 3:
                        print(f"✅ Mit entspannten Filtern wurden {len(filtered_df)} Hotels gefunden.")
                        break
                
                # Wenn immer noch leer, keine Filter anwenden
                if filtered_df.empty:
                    print("⚠️ Auch mit minimalen Filtern keine Hotels gefunden. Verwende alle verfügbaren Hotels.")
                    filtered_df = predictions_df
            
            print(f"🔍 After filtering: {len(filtered_df)} hotels remain")
            
            # Apply sophisticated preference weighting
            weighted_df = self._apply_preference_weighting(filtered_df, user_preferences)
            
            # Add diversity scoring to avoid too similar hotels
            if len(weighted_df) > top_k:
                try:
                    weighted_df = self._add_diversity_scoring(weighted_df, top_k)
                except Exception as e:
                    print(f"⚠️ Fehler beim Diversity-Scoring: {e}")
                    # Nicht kritisch, können ohne Diversity weitermachen
            
            # Sortieren und Top-K zurückgeben
            if 'final_score' in weighted_df.columns:
                sorted_df = weighted_df.sort_values('final_score', ascending=False)
            else:
                # Fallback wenn final_score nicht existiert
                sorted_df = weighted_df.sort_values('predicted_score', ascending=False)
                sorted_df['final_score'] = sorted_df['predicted_score']
            
            # Stelle sicher, dass wir Ergebnisse haben
            if sorted_df.empty:
                print("❌ Keine Hotels gefunden, die Ihren Kriterien entsprechen.")
                return pd.DataFrame()
            
            results = sorted_df.head(top_k)
            print(f"✅ Generated {len(results)} parameter-based recommendations")
            
            # Kopiere wichtige Spalten falls sie fehlen
            for col_name in ['hotel_name', 'name']:
                if col_name in results.columns:
                    if 'hotel_name' not in results.columns:
                        results['hotel_name'] = results[col_name]
                    elif 'name' not in results.columns:
                        results['name'] = results[col_name]
                    break
            
            # Stelle sicher, dass hotel_id als Spalte existiert
            if 'hotel_id' not in results.columns and 'id' in results.columns:
                results['hotel_id'] = results['id']
            
            # Entferne Duplikate vor der Rückgabe
            if 'name' in results.columns:
                results = results.drop_duplicates(subset=['name'], keep='first')
            elif 'hotel_name' in results.columns:
                results = results.drop_duplicates(subset=['hotel_name'], keep='first')
            
            return results
            
        except Exception as e:
            print(f"❌ Error in parameter-based recommendations: {e}")
            # Im Fehlerfall ein leeres DataFrame zurückgeben
            return pd.DataFrame()
    
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
    
    def _apply_user_filters(self, predictions_df: pd.DataFrame, user_preferences: Dict) -> pd.DataFrame:
        """
        Apply user preference filters with improved error handling and more flexible filtering
        
        Args:
            predictions_df: DataFrame with hotel predictions
            user_preferences: User preference dictionary
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = predictions_df.copy()
        
        try:
            # Price filter
            if 'max_price' in user_preferences and pd.notna(user_preferences['max_price']):
                max_price = float(user_preferences['max_price'])
                price_mask = filtered_df['price'] <= max_price
                filtered_df = filtered_df[price_mask]
                print(f"  After price filter (≤${max_price}): {len(filtered_df)} hotels")
                
                # Wenn zu wenige Hotels übrig sind, Filter entspannen
                if len(filtered_df) < 5 and len(predictions_df) > 5:
                    relaxed_max_price = max_price * 1.3  # 30% höheres Budget
                    price_mask = predictions_df['price'] <= relaxed_max_price
                    filtered_df = predictions_df[price_mask]
                    print(f"  Relaxed price filter (≤${relaxed_max_price:.2f}): {len(filtered_df)} hotels")
            
            # Rating filter (if available)
            if 'min_rating' in user_preferences and pd.notna(user_preferences['min_rating']):
                min_rating = float(user_preferences['min_rating'])
                
                # Prüfe, ob die Bewertungen in der richtigen Skala sind
                # Manche Systeme verwenden 1-5 Sterne, andere 1-10
                # Skalieren wenn nötig
                max_rating_in_data = filtered_df['rating'].max()
                
                # Wenn das Maximum unter 5.5 ist, nutzen wir wahrscheinlich eine 5-Punkte-Skala
                if max_rating_in_data <= 5.5 and min_rating > 5:
                    adjusted_min_rating = min_rating / 2  # Konvertiere von 10er auf 5er Skala
                    print(f"  Adjusting rating scale from 1-10 to 1-5. Min rating: {min_rating} → {adjusted_min_rating}")
                    min_rating = adjusted_min_rating
                
                rating_mask = filtered_df['rating'] >= min_rating
                filtered_df_with_rating = filtered_df[rating_mask]
                print(f"  After rating filter (≥{min_rating}): {len(filtered_df_with_rating)} hotels")
                
                # Entspanne den Filter schrittweise, wenn zu wenige Hotels übrig bleiben
                if len(filtered_df_with_rating) < 5:
                    relaxation_steps = [0.5, 1.0, 1.5, 2.0]  # Entspanne in 0.5-Punkt-Schritten
                    
                    for step in relaxation_steps:
                        relaxed_min_rating = max(0, min_rating - step)
                        rating_mask = filtered_df['rating'] >= relaxed_min_rating
                        relaxed_df = filtered_df[rating_mask]
                        
                        if len(relaxed_df) >= 5 or step == relaxation_steps[-1]:
                            filtered_df = relaxed_df
                            print(f"  Relaxed rating filter (≥{relaxed_min_rating}): {len(filtered_df)} hotels")
                            break
                else:
                    filtered_df = filtered_df_with_rating
            
            # Warnungen ausgeben, wenn nach der Filterung keine Hotels übrig bleiben
            if filtered_df.empty:
                print("⚠️ No hotels match all filters. Trying with minimal filters.")
                
                # Versuche nur mit einem minimalen Preis-Filter
                if 'max_price' in user_preferences and pd.notna(user_preferences['max_price']):
                    max_price = float(user_preferences['max_price']) * 1.5  # 50% höheres Budget
                    price_mask = predictions_df['price'] <= max_price
                    filtered_df = predictions_df[price_mask]
                    
                    if filtered_df.empty:
                        # Wenn immer noch leer, ignoriere auch den Preisfilter
                        filtered_df = predictions_df.copy()
                
                print(f"  After minimal filtering: {len(filtered_df)} hotels")
            
            return filtered_df
            
        except Exception as e:
            print(f"⚠️ Error in filter application: {e}")
            return predictions_df  # Return unfiltered on error
    
    def _apply_preference_weighting(self, filtered_df: pd.DataFrame, user_preferences: Dict) -> pd.DataFrame:
        """
        Apply sophisticated preference weighting with error handling
        
        Args:
            filtered_df: DataFrame with filtered hotels
            user_preferences: User preference dictionary
            
        Returns:
            DataFrame with preference-weighted scores
        """
        weighted_df = filtered_df.copy()
        
        try:
            # Start with the ML predicted score
            if 'predicted_score' in weighted_df.columns:
                weighted_df['final_score'] = weighted_df['predicted_score'].copy()
            else:
                # Fallback, wenn keine Vorhersagen vorhanden sind
                print("⚠️ No predicted scores found, using ratings as base scores")
                weighted_df['final_score'] = weighted_df['rating'] / 10.0  # Normalisieren auf 0-1 Skala
            
            # Extract weights from user preferences with defaults
            model_importance = user_preferences.get('model_importance', 0.4)
            price_importance = user_preferences.get('price_importance', 0.25)
            rating_importance = user_preferences.get('rating_importance', 0.25)
            amenity_importance = user_preferences.get('amenity_importance', 0.1)
            
            # Normalize weights
            total_weight = model_importance + price_importance + rating_importance + amenity_importance
            if total_weight == 0:
                # Fallback Gewichte wenn alle 0 sind
                model_importance = 0.4
                price_importance = 0.3
                rating_importance = 0.3
                amenity_importance = 0.0
                total_weight = 1.0
            
            model_weight = model_importance / total_weight
            price_weight = price_importance / total_weight
            rating_weight = rating_importance / total_weight
            amenity_weight = amenity_importance / total_weight
            
            # Sichere Normalisierung von numerischen Werten
            def safe_normalize(series):
                if len(series) <= 1 or series.max() == series.min():
                    return pd.Series(0.5, index=series.index)  # Default bei nur einem Wert
                else:
                    return (series - series.min()) / (series.max() - series.min())
            
            # Preiswichtung - niedrigere Preise bekommen höhere Scores
            if 'price' in weighted_df.columns and price_importance > 0:
                price_scores = 1 - safe_normalize(weighted_df['price'])
                weighted_df['price_score'] = price_scores
                weighted_df['final_score'] = (
                    weighted_df['final_score'] * model_weight + 
                    weighted_df['price_score'] * price_weight
                )
            
            # Bewertungswichtung
            if 'rating' in weighted_df.columns and rating_importance > 0:
                # Normalize ratings to 0-1
                max_rating = weighted_df['rating'].max()
                if max_rating > 0:
                    rating_scale = 10 if max_rating > 5 else 5  # Detect scale
                    rating_scores = weighted_df['rating'] / rating_scale
                    weighted_df['rating_score'] = rating_scores
                    
                    # Cubic function to emphasize high ratings (optional)
                    weighted_df['rating_score'] = weighted_df['rating_score'] ** 2
                    
                    weighted_df['final_score'] = (
                        weighted_df['final_score'] * (model_weight + price_weight) / (model_weight + price_weight + rating_weight) + 
                        weighted_df['rating_score'] * rating_weight / (model_weight + price_weight + rating_weight)
                    )
            
            # Amenity weighting
            if 'amenity_importance' in user_preferences and amenity_importance > 0:
                preferred_amenities = user_preferences.get('preferred_amenities', [])
                
                if preferred_amenities and 'amenities' in weighted_df.columns:
                    # Calculate match score for amenities
                    amenity_scores = []
                    
                    for _, hotel in weighted_df.iterrows():
                        # Safely extract amenities
                        if pd.isna(hotel['amenities']):
                            amenity_scores.append(0)
                            continue
                            
                        hotel_amenities = str(hotel['amenities']).lower()
                        
                        # Count matching amenities
                        match_count = sum(1 for amenity in preferred_amenities 
                                        if amenity.lower() in hotel_amenities)
                        
                        # Calculate score as proportion of matches
                        score = match_count / len(preferred_amenities) if preferred_amenities else 0
                        amenity_scores.append(score)
                    
                    weighted_df['amenity_score'] = amenity_scores
                    
                    if sum(amenity_scores) > 0:  # Only adjust if we have matches
                        weighted_df['final_score'] = (
                            weighted_df['final_score'] * (1 - amenity_weight) + 
                            weighted_df['amenity_score'] * amenity_weight
                        )
            
            # Apply logarithmic function to final scores to reduce extreme differences
            if 'final_score' in weighted_df.columns:
                min_score = weighted_df['final_score'].min()
                weighted_df['final_score'] = weighted_df['final_score'] - min_score + 0.01
                weighted_df['final_score'] = 0.1 + 0.9 * (weighted_df['final_score'] / weighted_df['final_score'].max())
            
            return weighted_df
            
        except Exception as e:
            print(f"⚠️ Error in preference weighting: {e}")
            # Fallback bei Fehler
            if 'predicted_score' in weighted_df.columns:
                weighted_df['final_score'] = weighted_df['predicted_score']
            elif 'rating' in weighted_df.columns:
                weighted_df['final_score'] = weighted_df['rating'] / 10.0
            else:
                weighted_df['final_score'] = 0.5  # Default wenn keine bessere Option verfügbar
            
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
        
        # Ensure the scaler is fit before saving
        if not hasattr(self.scaler, 'n_samples_seen_') or self.scaler.n_samples_seen_ is None:
            try:
                import numpy as np
                self.scaler.fit(np.array([[5.0], [10.0]]))
                print("⚠️ Initialized scaler with dummy data before saving")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize scaler: {e}")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'scaler': self.scaler,  # Speichere auch den Skalierer
            'feature_selector': self.feature_selector,
            'best_params_': self.best_params_,
            'feature_importance_': self.feature_importance_
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        
        # Lade auch den Skalierer, wenn vorhanden
        if 'scaler' in model_data:
            self.scaler = model_data['scaler']
            print("  ✓ Scaler loaded from saved model")
        else:
            print("  ⚠️ No scaler found in saved model, using default")
            
        # Lade weitere gespeicherte Attribute wenn vorhanden
        if 'feature_selector' in model_data:
            self.feature_selector = model_data['feature_selector']
        if 'best_params_' in model_data:
            self.best_params_ = model_data['best_params_']
        if 'feature_importance_' in model_data:
            self.feature_importance_ = model_data['feature_importance_']
            
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
