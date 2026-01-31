import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path

from src.logger import setup_logger
from src.config import model_config

logger = setup_logger(__name__, "model_trainer.log")


class ModelTrainer:
    """Handles training of various student grade prediction models."""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on type."""
        if self.model_type == "linear_regression":
            self.model = LinearRegression(**model_config.lr_params)
            logger.info("Initialized Linear Regression model")
            
        elif self.model_type == "ridge":
            self.model = Ridge(**model_config.ridge_params)
            logger.info("Initialized Ridge Regression model")
            
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(**model_config.rf_params)
            logger.info("Initialized Random Forest Regressor model")
            
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(**model_config.xgb_params)
            logger.info("Initialized XGBoost Regressor model")
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        validate: bool = True
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            validate: Whether to perform cross-validation
        
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info(f"Model training complete")
        
        # Get training score (R² for regression)
        train_score = self.model.score(X_train, y_train)
        metrics = {"train_r2": train_score}
        
        logger.info(f"Training R²: {train_score:.4f}")
        
        # Perform cross-validation if requested
        if validate:
            cv_scores = self._cross_validate(X_train, y_train)
            metrics.update(cv_scores)
        
        return metrics
    
    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Dictionary of CV metrics
        """
        logger.info(f"Performing {model_config.cv_folds}-fold cross-validation")
        
        # Cross-validation with R² scoring
        cv_r2_scores = cross_val_score(
            self.model,
            X, y,
            cv=model_config.cv_folds,
            scoring='r2',
            n_jobs=-1
        )
        
        # Cross-validation with negative MAE (sklearn convention)
        cv_mae_scores = cross_val_score(
            self.model,
            X, y,
            cv=model_config.cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        # Cross-validation with negative RMSE
        cv_rmse_scores = cross_val_score(
            self.model,
            X, y,
            cv=model_config.cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        metrics = {
            "cv_mean_r2": cv_r2_scores.mean(),
            "cv_std_r2": cv_r2_scores.std(),
            "cv_mean_mae": -cv_mae_scores.mean(),  # Convert back to positive
            "cv_std_mae": cv_mae_scores.std(),
            "cv_mean_rmse": -cv_rmse_scores.mean(),  # Convert back to positive
            "cv_std_rmse": cv_rmse_scores.std(),
            "cv_r2_scores": cv_r2_scores.tolist()
        }
        
        logger.info(
            f"CV R²: {metrics['cv_mean_r2']:.4f} "
            f"(+/- {metrics['cv_std_r2']:.4f})"
        )
        logger.info(f"CV MAE: {metrics['cv_mean_mae']:.4f}")
        logger.info(f"CV RMSE: {metrics['cv_mean_rmse']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict
        
        Returns:
            Predicted grades
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        
        # Clip predictions to valid grade range (0-20)
        predictions = np.clip(predictions, 0, 20)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance, or None if not available
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        if self.model_type == "linear_regression" or self.model_type == "ridge":
            # Use coefficients for linear models
            importance = np.abs(self.model.coef_)
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_model(
        self,
        filepath: Optional[Path] = None,
        dataset_name: str = "model"
    ) -> Path:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
            dataset_name: Name of dataset for filename
        
        Returns:
            Path where model was saved
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        if filepath is None:
            filename = f"{self.model_type}_{dataset_name}_model.joblib"
            filepath = model_config.model_save_dir / filename
        
        logger.info(f"Saving model to {filepath}")
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved successfully")
        
        return filepath
    
    def load_model(self, filepath: Path) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
        """
        logger.info(f"Loading model from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded successfully: {self.model_type}")


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dataset_name: str,
    model_types: Optional[list] = None
) -> Dict[str, ModelTrainer]:
    """
    Train multiple models on the same dataset.
    
    Args:
        X_train: Training features
        y_train: Training target
        dataset_name: Name of dataset for saving
        model_types: List of model types to train
    
    Returns:
        Dictionary of trained ModelTrainer objects
    """
    if model_types is None:
        model_types = model_config.model_types
    
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type} for {dataset_name}")
        logger.info(f"{'='*60}\n")
        
        try:
            trainer = ModelTrainer(model_type=model_type)
            metrics = trainer.train(X_train, y_train)
            
            # Save model
            trainer.save_model(dataset_name=dataset_name)
            
            trained_models[model_type] = trainer
            
            logger.info(f"✓ {model_type} trained successfully")
            
        except Exception as e:
            logger.error(f"✗ Error training {model_type}: {str(e)}")
            continue
    
    return trained_models
