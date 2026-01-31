import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import joblib

from src.logger import setup_logger
from src.model_trainer import ModelTrainer
from src.feature_engineer import prepare_features

logger = setup_logger(__name__, "predictor.log")


class GradePredictor:
    """Handles predictions for student grade prediction."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model (loaded if provided)
        """
        self.model_trainer = None
        self.dataset_type = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
        """
        logger.info(f"Loading model from {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Infer dataset type from filename
        filename = model_path.stem
        if 'math' in filename.lower():
            self.dataset_type = 'math'
        elif 'por' in filename.lower() or 'portuguese' in filename.lower():
            self.dataset_type = 'portuguese'
        else:
            logger.warning("Could not infer dataset type from filename")
        
        # Load model
        self.model_trainer = ModelTrainer()
        self.model_trainer.load_model(model_path)
        
        logger.info(f"Model loaded: {self.model_trainer.model_type}")
    
    def predict(
        self,
        X: pd.DataFrame,
        return_details: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            return_details: Whether to return detailed predictions
        
        Returns:
            Predictions or dict with predictions and performance level
        """
        if self.model_trainer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        logger.info(f"Making predictions on {len(X)} samples")
        
        # Validate features
        self._validate_features(X)
        
        # Get predictions
        y_pred = self.model_trainer.predict(X)
        
        # Clip predictions to valid range
        y_pred = np.clip(y_pred, 0, 20)
        
        logger.info(
            f"Predictions complete: "
            f"Mean predicted grade: {y_pred.mean():.2f}"
        )
        
        if return_details:
            return {
                'predicted_grade': y_pred,
                'performance_level': np.array([self._get_performance_level(g) for g in y_pred]),
                'pass_status': (y_pred >= 10).astype(int)
            }
        
        return y_pred
    
    def predict_single(
        self,
        student_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict grade for a single student.
        
        Args:
            student_data: Dictionary with student features
        
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        X = pd.DataFrame([student_data])
        
        # Make prediction
        result = self.predict(X, return_details=True)
        
        predicted_grade = float(result['predicted_grade'][0])
        
        prediction_result = {
            'predicted_grade': round(predicted_grade, 1),
            'predicted_grade_int': int(round(predicted_grade)),
            'performance_level': result['performance_level'][0],
            'pass_status': 'Pass' if predicted_grade >= 10 else 'Fail',
            'confidence_range': self._get_confidence_range(predicted_grade)
        }
        
        logger.info(
            f"Single prediction: grade={prediction_result['predicted_grade']}, "
            f"level={prediction_result['performance_level']}"
        )
        
        return prediction_result
    
    def predict_batch(
        self,
        students: pd.DataFrame,
        include_details: bool = True
    ) -> pd.DataFrame:
        """
        Predict grades for a batch of students.
        
        Args:
            students: DataFrame with student features
            include_details: Whether to include detailed predictions
        
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Processing batch of {len(students)} students")
        
        results = self.predict(students, return_details=True)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'predicted_grade': results['predicted_grade'],
            'predicted_grade_int': np.round(results['predicted_grade']).astype(int)
        })
        
        if include_details:
            results_df['performance_level'] = results['performance_level']
            results_df['pass_status'] = np.where(
                results['predicted_grade'] >= 10, 'Pass', 'Fail'
            )
            results_df['confidence_range'] = [
                self._get_confidence_range(g) for g in results['predicted_grade']
            ]
        
        return results_df
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate that input features match trained model.
        
        Args:
            X: Input features
        
        Raises:
            ValueError: If features don't match
        """
        expected_features = set(self.model_trainer.feature_names)
        actual_features = set(X.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                X[feature] = 0
        
        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")
        
        # Ensure correct column order
        X = X[self.model_trainer.feature_names]
    
    @staticmethod
    def _get_performance_level(grade: float) -> str:
        """
        Convert grade to performance level.
        
        Args:
            grade: Predicted grade
        
        Returns:
            Performance level string
        """
        if grade < 10:
            return 'Failing'
        elif grade < 14:
            return 'Passing'
        elif grade < 17:
            return 'Good'
        else:
            return 'Excellent'
    
    @staticmethod
    def _get_confidence_range(grade: float, margin: float = 1.5) -> str:
        """
        Get confidence range for prediction.
        
        Args:
            grade: Predicted grade
            margin: Error margin (default ±1.5 points)
        
        Returns:
            String with confidence range
        """
        lower = max(0, grade - margin)
        upper = min(20, grade + margin)
        return f"{lower:.1f} - {upper:.1f}"
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature contributions for predictions.
        
        Args:
            X: Features to explain
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature contributions
        """
        if self.model_trainer is None:
            raise RuntimeError("No model loaded")
        
        # Get feature importance
        feature_importance = self.model_trainer.get_feature_importance()
        
        if feature_importance is None:
            logger.warning("Feature importance not available for this model")
            return pd.DataFrame()
        
        # Get top features
        top_features = feature_importance.head(top_n)
        
        logger.info(f"Top {top_n} features for grade prediction:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return top_features


def load_predictor(
    model_type: str,
    dataset_type: str,
    models_dir: Optional[Path] = None
) -> GradePredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        model_type: Type of model ('linear_regression', 'random_forest', 'xgboost')
        dataset_type: Type of dataset ('math', 'portuguese')
        models_dir: Directory containing models
    
    Returns:
        Loaded GradePredictor
    """
    from src.config import model_config
    
    if models_dir is None:
        models_dir = model_config.model_save_dir
    
    # Try standard naming convention
    model_path = models_dir / f"{model_type}_{dataset_type}_model.joblib"
    
    if model_path.exists():
        logger.info(f"Loading model: {model_path}")
        return GradePredictor(model_path)
    
    # Try alternative naming
    alt_path = models_dir / f"{dataset_type}_{model_type}_model.joblib"
    if alt_path.exists():
        logger.info(f"Loading model: {alt_path}")
        return GradePredictor(alt_path)
    
    raise FileNotFoundError(
        f"Model not found. Searched for:\n"
        f"  - {model_path}\n"
        f"  - {alt_path}\n"
        f"Please train a model first using the notebooks or dashboard."
    )
