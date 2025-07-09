"""
Model Evaluation Utilities for TravelHunters Recommender Systems
Provides metrics and evaluation functions for both parameter-based and text-based models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class RecommenderEvaluator:
    """Evaluation utilities for recommendation models"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str = "Model") -> Dict:
        """
        Evaluate regression-based recommender (parameter model)
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'n_samples': len(y_true)
        }
        
        # Additional metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        # Accuracy within thresholds
        metrics['accuracy_0.5'] = np.mean(np.abs(residuals) <= 0.5)
        metrics['accuracy_1.0'] = np.mean(np.abs(residuals) <= 1.0)
        
        self.results[model_name] = metrics
        
        return metrics
    
    def evaluate_ranking_quality(self, recommendations_df: pd.DataFrame, 
                                ground_truth_df: pd.DataFrame, 
                                k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate ranking quality for recommendation lists
        
        Args:
            recommendations_df: Recommended hotels with scores
            ground_truth_df: Ground truth ratings/preferences
            k_values: Values of k for Precision@k and Recall@k
            
        Returns:
            Dictionary of ranking metrics
        """
        metrics = {}
        
        # For each k value
        for k in k_values:
            precision_k = self._precision_at_k(recommendations_df, ground_truth_df, k)
            recall_k = self._recall_at_k(recommendations_df, ground_truth_df, k)
            
            metrics[f'precision@{k}'] = precision_k
            metrics[f'recall@{k}'] = recall_k
            
            if precision_k + recall_k > 0:
                metrics[f'f1@{k}'] = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                metrics[f'f1@{k}'] = 0.0
        
        # NDCG (Normalized Discounted Cumulative Gain)
        for k in k_values:
            ndcg_k = self._ndcg_at_k(recommendations_df, ground_truth_df, k)
            metrics[f'ndcg@{k}'] = ndcg_k
        
        return metrics
    
    def _precision_at_k(self, recommendations_df: pd.DataFrame, 
                       ground_truth_df: pd.DataFrame, k: int) -> float:
        """Calculate Precision@k"""
        if len(recommendations_df) == 0:
            return 0.0
        
        # Get top-k recommendations
        top_k = recommendations_df.head(k)
        
        # Define relevance threshold (e.g., rating >= 4.0)
        relevance_threshold = 4.0
        
        # Check how many top-k items are relevant
        relevant_items = 0
        for _, rec in top_k.iterrows():
            hotel_id = rec['hotel_id']
            # Check if this hotel has high ratings in ground truth
            hotel_ratings = ground_truth_df[ground_truth_df['hotel_id'] == hotel_id]
            if not hotel_ratings.empty:
                avg_rating = hotel_ratings['rating'].mean()
                if avg_rating >= relevance_threshold:
                    relevant_items += 1
        
        return relevant_items / min(k, len(top_k))
    
    def _recall_at_k(self, recommendations_df: pd.DataFrame, 
                    ground_truth_df: pd.DataFrame, k: int) -> float:
        """Calculate Recall@k"""
        if len(ground_truth_df) == 0:
            return 0.0
        
        # Get top-k recommendations
        top_k = recommendations_df.head(k)
        
        # Define relevance threshold
        relevance_threshold = 4.0
        
        # Get all relevant items from ground truth
        relevant_hotels = ground_truth_df[ground_truth_df['rating'] >= relevance_threshold]['hotel_id'].unique()
        
        if len(relevant_hotels) == 0:
            return 0.0
        
        # Check how many relevant items are in top-k
        recommended_hotels = set(top_k['hotel_id'])
        relevant_recommended = len(set(relevant_hotels) & recommended_hotels)
        
        return relevant_recommended / len(relevant_hotels)
    
    def _ndcg_at_k(self, recommendations_df: pd.DataFrame, 
                  ground_truth_df: pd.DataFrame, k: int) -> float:
        """Calculate NDCG@k (Normalized Discounted Cumulative Gain)"""
        if len(recommendations_df) == 0:
            return 0.0
        
        # Get top-k recommendations
        top_k = recommendations_df.head(k)
        
        # Calculate DCG
        dcg = 0.0
        for i, (_, rec) in enumerate(top_k.iterrows()):
            hotel_id = rec['hotel_id']
            # Get relevance score (rating from ground truth)
            hotel_ratings = ground_truth_df[ground_truth_df['hotel_id'] == hotel_id]
            if not hotel_ratings.empty:
                relevance = hotel_ratings['rating'].mean()
            else:
                relevance = 0.0
            
            # DCG formula: rel_i / log2(i + 2)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG (Ideal DCG)
        all_relevances = []
        for hotel_id in ground_truth_df['hotel_id'].unique():
            hotel_ratings = ground_truth_df[ground_truth_df['hotel_id'] == hotel_id]
            if not hotel_ratings.empty:
                all_relevances.append(hotel_ratings['rating'].mean())
        
        all_relevances.sort(reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(all_relevances[:k]):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / np.log2(i + 2)
        
        # NDCG = DCG / IDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            model_results: Dictionary of {model_name: metrics_dict}
            
        Returns:
            Comparison dataframe
        """
        comparison_df = pd.DataFrame(model_results).T
        return comparison_df
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5, scoring: str = 'r2') -> Dict:
        """
        Cross-validate a model
        
        Args:
            model: Scikit-learn model
            X: Features
            y: Target
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            CV results
        """
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        
        cv_results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
        
        return cv_results
    
    def evaluate_recommendation_diversity(self, recommendations_list: List[pd.DataFrame]) -> Dict:
        """
        Evaluate diversity of recommendations across different queries/users
        
        Args:
            recommendations_list: List of recommendation dataframes
            
        Returns:
            Diversity metrics
        """
        if not recommendations_list:
            return {}
        
        # Collect all recommended hotel IDs
        all_recommendations = []
        for rec_df in recommendations_list:
            all_recommendations.extend(rec_df['hotel_id'].tolist())
        
        # Calculate diversity metrics
        unique_hotels = len(set(all_recommendations))
        total_recommendations = len(all_recommendations)
        
        # Intra-list diversity (average uniqueness within each recommendation list)
        intra_diversities = []
        for rec_df in recommendations_list:
            if len(rec_df) > 0:
                unique_in_list = len(set(rec_df['hotel_id']))
                diversity = unique_in_list / len(rec_df)
                intra_diversities.append(diversity)
        
        diversity_metrics = {
            'catalog_coverage': unique_hotels / total_recommendations if total_recommendations > 0 else 0,
            'avg_intra_list_diversity': np.mean(intra_diversities) if intra_diversities else 0,
            'total_unique_hotels': unique_hotels,
            'total_recommendations': total_recommendations
        }
        
        return diversity_metrics
    
    def plot_evaluation_results(self, metrics_dict: Dict, save_path: str = None):
        """
        Plot evaluation results
        
        Args:
            metrics_dict: Dictionary of metrics
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # Plot 1: R² and RMSE comparison
        if 'r2' in metrics_dict and 'rmse' in metrics_dict:
            ax1 = axes[0, 0]
            metrics = ['r2', 'rmse', 'mae']
            values = [metrics_dict.get(m, 0) for m in metrics]
            ax1.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax1.set_title('Regression Metrics')
            ax1.set_ylabel('Score')
        
        # Plot 2: Accuracy at thresholds
        if 'accuracy_0.5' in metrics_dict:
            ax2 = axes[0, 1]
            thresholds = ['0.5', '1.0']
            accuracies = [metrics_dict.get(f'accuracy_{t}', 0) for t in thresholds]
            ax2.bar(thresholds, accuracies, color='orange')
            ax2.set_title('Prediction Accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Threshold')
        
        # Plot 3: Precision@k and Recall@k
        precision_keys = [k for k in metrics_dict.keys() if k.startswith('precision@')]
        recall_keys = [k for k in metrics_dict.keys() if k.startswith('recall@')]
        
        if precision_keys and recall_keys:
            ax3 = axes[1, 0]
            k_values = [k.split('@')[1] for k in precision_keys]
            precisions = [metrics_dict[k] for k in precision_keys]
            recalls = [metrics_dict[k] for k in recall_keys]
            
            x = np.arange(len(k_values))
            width = 0.35
            
            ax3.bar(x - width/2, precisions, width, label='Precision', color='lightblue')
            ax3.bar(x + width/2, recalls, width, label='Recall', color='lightcoral')
            ax3.set_xlabel('k')
            ax3.set_ylabel('Score')
            ax3.set_title('Precision@k and Recall@k')
            ax3.set_xticks(x)
            ax3.set_xticklabels(k_values)
            ax3.legend()
        
        # Plot 4: NDCG@k
        ndcg_keys = [k for k in metrics_dict.keys() if k.startswith('ndcg@')]
        if ndcg_keys:
            ax4 = axes[1, 1]
            k_values = [k.split('@')[1] for k in ndcg_keys]
            ndcg_values = [metrics_dict[k] for k in ndcg_keys]
            
            ax4.plot(k_values, ndcg_values, marker='o', color='green', linewidth=2, markersize=8)
            ax4.set_xlabel('k')
            ax4.set_ylabel('NDCG@k')
            ax4.set_title('NDCG@k')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Evaluation plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, model_results: Dict, save_path: str = None) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            model_results: Dictionary of model evaluation results
            save_path: Path to save the report
            
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("# TravelHunters Recommender System Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        for model_name, metrics in model_results.items():
            report_lines.append(f"## {model_name}")
            report_lines.append("-" * 40)
            
            # Regression metrics
            if 'rmse' in metrics:
                report_lines.append("### Regression Performance:")
                report_lines.append(f"  - RMSE: {metrics.get('rmse', 0):.3f}")
                report_lines.append(f"  - MAE:  {metrics.get('mae', 0):.3f}")
                report_lines.append(f"  - R²:   {metrics.get('r2', 0):.3f}")
                report_lines.append("")
            
            # Ranking metrics
            precision_keys = [k for k in metrics.keys() if k.startswith('precision@')]
            if precision_keys:
                report_lines.append("### Ranking Performance:")
                for k in precision_keys:
                    k_val = k.split('@')[1]
                    precision = metrics[k]
                    recall = metrics.get(f'recall@{k_val}', 0)
                    ndcg = metrics.get(f'ndcg@{k_val}', 0)
                    report_lines.append(f"  - k={k_val}: Precision={precision:.3f}, Recall={recall:.3f}, NDCG={ndcg:.3f}")
                report_lines.append("")
            
            # Diversity metrics
            if 'catalog_coverage' in metrics:
                report_lines.append("### Diversity Metrics:")
                report_lines.append(f"  - Catalog Coverage: {metrics['catalog_coverage']:.3f}")
                report_lines.append(f"  - Avg Intra-list Diversity: {metrics['avg_intra_list_diversity']:.3f}")
                report_lines.append("")
        
        # Summary and recommendations
        report_lines.append("## Summary & Recommendations")
        report_lines.append("-" * 40)
        report_lines.append("Based on the evaluation results:")
        report_lines.append("1. Consider the trade-off between accuracy and diversity")
        report_lines.append("2. Monitor performance across different user segments")
        report_lines.append("3. Regularly retrain models with new user feedback")
        report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✅ Evaluation report saved to {save_path}")
        
        return report

if __name__ == "__main__":
    # Test evaluation utilities
    evaluator = RecommenderEvaluator()
    
    # Mock data for testing
    y_true = np.array([4.2, 3.8, 4.5, 3.9, 4.1])
    y_pred = np.array([4.0, 3.9, 4.3, 4.0, 4.2])
    
    # Test regression evaluation
    reg_metrics = evaluator.evaluate_regression_model(y_true, y_pred, "Test Model")
    print("Regression Metrics:")
    for metric, value in reg_metrics.items():
        print(f"  {metric}: {value}")
    
    # Mock recommendations for ranking evaluation
    recommendations = pd.DataFrame({
        'hotel_id': [1, 2, 3, 4, 5],
        'score': [0.9, 0.8, 0.7, 0.6, 0.5]
    })
    
    ground_truth = pd.DataFrame({
        'hotel_id': [1, 2, 3, 6, 7],
        'rating': [4.5, 4.2, 3.8, 4.0, 4.3]
    })
    
    ranking_metrics = evaluator.evaluate_ranking_quality(recommendations, ground_truth)
    print(f"\nRanking Metrics:")
    for metric, value in ranking_metrics.items():
        print(f"  {metric}: {value:.3f}")
