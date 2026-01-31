import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler

from src.logger import setup_logger
from src.config import model_config

logger = setup_logger(__name__, "preprocessor.log")


class DataPreprocessor:
    """Handles data preprocessing including scaling for student grade prediction."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = None
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessor and transform training data.
        
        Args:
            X: Input features
            y: Target variable
        
        Returns:
            Tuple of (transformed features, target)
        """
        logger.info("Fitting and transforming training data")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Scale features
        X_scaled = self._scale_features(X, fit=True)
        
        self.fitted = True
        return X_scaled, y
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test/new data using fitted preprocessor.
        
        Args:
            X: Input features
        
        Returns:
            Transformed features
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming data")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Scale features
        X_scaled = self._scale_features(X, fit=False)
        
        return X_scaled
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            X: Input features
        
        Returns:
            DataFrame with missing values handled
        """
        X = X.copy()
        
        missing_counts = X.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Found missing values in columns: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Fill numeric columns with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X[col].isnull().any():
                    X[col].fillna(X[col].median(), inplace=True)
            
            # Fill remaining with mode or 0
            X.fillna(0, inplace=True)
            
            logger.info("Missing values handled")
        
        return X
    
    def _scale_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X: Input features
            fit: Whether to fit the scaler
        
        Returns:
            Scaled DataFrame
        """
        X = X.copy()
        
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("Fitted and scaled features")
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("Scaled features using fitted scaler")
        
        return X_scaled
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        return self.feature_names


def preprocess_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DataPreprocessor]:
    """
    Complete preprocessing pipeline.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
    
    Returns:
        Tuple of (X_train_processed, X_test_processed, y_train, y_test, preprocessor)
    """
    preprocessor = DataPreprocessor()
    
    # Fit and transform training data
    X_train_processed, y_train = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    logger.info(f"Preprocessing complete: {X_train_processed.shape[1]} features")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
