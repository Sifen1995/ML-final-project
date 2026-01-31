import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import setup_logger

logger = setup_logger(__name__, "model_evaluator.log")


class ModelEvaluator:
    """Handles evaluation and scoring of student grade prediction models."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions.
        
        Args:
            y_true: Actual grades
            y_pred: Predicted grades
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Calculate regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        max_err = max_error(y_true, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Calculate prediction accuracy within tolerance
        tolerance_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100  # Within 1 grade point
        tolerance_2 = np.mean(np.abs(y_true - y_pred) <= 2) * 100  # Within 2 grade points
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_var,
            'max_error': max_err,
            'mape': mape,
            'accuracy_within_1pt': tolerance_1,
            'accuracy_within_2pt': tolerance_2
        }
        
        # Calculate residuals
        residuals = y_true - y_pred
        metrics['residual_mean'] = residuals.mean()
        metrics['residual_std'] = residuals.std()
        
        # Log metrics
        logger.info("\n" + "="*60)
        logger.info("EVALUATION METRICS")
        logger.info("="*60)
        logger.info(f"R² Score:           {metrics['r2']:.4f}")
        logger.info(f"RMSE:               {metrics['rmse']:.4f}")
        logger.info(f"MAE:                {metrics['mae']:.4f}")
        logger.info(f"Explained Variance: {metrics['explained_variance']:.4f}")
        logger.info(f"Max Error:          {metrics['max_error']:.4f}")
        logger.info(f"Within 1 point:     {metrics['accuracy_within_1pt']:.1f}%")
        logger.info(f"Within 2 points:    {metrics['accuracy_within_2pt']:.1f}%")
        logger.info("="*60 + "\n")
        
        return metrics
    
    def evaluate_by_grade_range(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate model performance by grade range.
        
        Args:
            y_true: Actual grades
            y_pred: Predicted grades
        
        Returns:
            DataFrame with metrics by grade range
        """
        results = []
        
        # Define grade ranges
        ranges = [
            ('Failing (0-9)', 0, 10),
            ('Passing (10-13)', 10, 14),
            ('Good (14-16)', 14, 17),
            ('Excellent (17-20)', 17, 21)
        ]
        
        for range_name, low, high in ranges:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                y_true_range = y_true[mask]
                y_pred_range = y_pred[mask]
                
                results.append({
                    'Grade Range': range_name,
                    'Count': mask.sum(),
                    'MAE': mean_absolute_error(y_true_range, y_pred_range),
                    'RMSE': np.sqrt(mean_squared_error(y_true_range, y_pred_range)),
                    'R²': r2_score(y_true_range, y_pred_range) if len(y_true_range) > 1 else np.nan
                })
        
        return pd.DataFrame(results)
    
    def generate_prediction_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        passing_grade: int = 10
    ) -> str:
        """
        Generate a detailed prediction report.
        
        Args:
            y_true: Actual grades
            y_pred: Predicted grades
            passing_grade: Threshold for passing (default: 10)
        
        Returns:
            Formatted report string
        """
        # Convert to binary pass/fail for confusion-like analysis
        actual_pass = y_true >= passing_grade
        predicted_pass = y_pred >= passing_grade
        
        # Calculate pass/fail prediction accuracy
        tp = ((actual_pass) & (predicted_pass)).sum()
        tn = ((~actual_pass) & (~predicted_pass)).sum()
        fp = ((~actual_pass) & (predicted_pass)).sum()
        fn = ((actual_pass) & (~predicted_pass)).sum()
        
        total = len(y_true)
        accuracy = (tp + tn) / total
        
        report = f"""
================================================================================
                     STUDENT GRADE PREDICTION REPORT
================================================================================

REGRESSION METRICS:
------------------
Mean Absolute Error (MAE):     {mean_absolute_error(y_true, y_pred):.3f} grade points
Root Mean Square Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.3f} grade points
R² Score:                      {r2_score(y_true, y_pred):.4f}

PREDICTION ACCURACY:
-------------------
Predictions within 1 point:    {np.mean(np.abs(y_true - y_pred) <= 1) * 100:.1f}%
Predictions within 2 points:   {np.mean(np.abs(y_true - y_pred) <= 2) * 100:.1f}%
Predictions within 3 points:   {np.mean(np.abs(y_true - y_pred) <= 3) * 100:.1f}%

PASS/FAIL PREDICTION (Threshold: {passing_grade}):
-----------------------------------------
Total Students:                {total}
Correctly Predicted Pass:      {tp} ({tp/total*100:.1f}%)
Correctly Predicted Fail:      {tn} ({tn/total*100:.1f}%)
False Pass Predictions:        {fp} ({fp/total*100:.1f}%)
False Fail Predictions:        {fn} ({fn/total*100:.1f}%)
Overall Pass/Fail Accuracy:    {accuracy*100:.1f}%

GRADE DISTRIBUTION:
------------------
Actual Mean Grade:             {y_true.mean():.2f}
Predicted Mean Grade:          {np.mean(y_pred):.2f}
Actual Std Deviation:          {y_true.std():.2f}
Predicted Std Deviation:       {np.std(y_pred):.2f}

================================================================================
"""
        logger.info(report)
        return report
    
    def plot_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot actual vs predicted grades.
        
        Args:
            y_true: Actual grades
            y_pred: Predicted grades
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Scatter plot: Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='none')
        axes[0].plot([0, 20], [0, 20], 'r--', label='Perfect Prediction')
        axes[0].set_xlabel('Actual Grade')
        axes[0].set_ylabel('Predicted Grade')
        axes[0].set_title('Actual vs Predicted Grades')
        axes[0].set_xlim(0, 20)
        axes[0].set_ylim(0, 20)
        axes[0].legend()
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Grade')
        axes[1].set_ylabel('Residual (Actual - Predicted)')
        axes[1].set_title('Residual Plot')
        
        # Residual distribution
        axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(x=0, color='r', linestyle='--')
        axes[2].set_xlabel('Residual')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residual Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance.
        
        Args:
            feature_importance_df: DataFrame with feature importance
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance for Grade Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_grade_distribution(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot grade distribution comparison.
        
        Args:
            y_true: Actual grades
            y_pred: Predicted grades
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram comparison
        bins = np.arange(0, 22, 1)
        axes[0].hist(y_true, bins=bins, alpha=0.5, label='Actual', edgecolor='black')
        axes[0].hist(y_pred, bins=bins, alpha=0.5, label='Predicted', edgecolor='black')
        axes[0].set_xlabel('Grade')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Grade Distribution: Actual vs Predicted')
        axes[0].legend()
        
        # Box plot comparison
        data = pd.DataFrame({
            'Actual': y_true.values,
            'Predicted': y_pred
        })
        data.boxplot(ax=axes[1])
        axes[1].set_ylabel('Grade')
        axes[1].set_title('Grade Comparison: Actual vs Predicted')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grade distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
