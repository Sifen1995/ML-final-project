"""
Tests for preprocessor module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.preprocessor import StudentPreprocessor


class TestStudentPreprocessor:
    """Test cases for StudentPreprocessor."""
    
    def test_preprocessor_initialization(self):
        """Test StudentPreprocessor initialization."""
        preprocessor = StudentPreprocessor()
        assert preprocessor is not None
        assert preprocessor.scaler is not None
    
    def test_fit_transform(self, sample_numeric_features):
        """Test fit_transform on numeric features."""
        X, y = sample_numeric_features
        preprocessor = StudentPreprocessor()
        
        X_scaled = preprocessor.fit_transform(X)
        
        assert X_scaled.shape == X.shape
        # Scaled data should have mean ~0 and std ~1
        assert np.abs(X_scaled.mean()) < 0.1
        assert np.abs(X_scaled.std() - 1) < 0.1
    
    def test_transform(self, sample_numeric_features):
        """Test transform after fitting."""
        X, y = sample_numeric_features
        preprocessor = StudentPreprocessor()
        
        preprocessor.fit(X)
        X_scaled = preprocessor.transform(X)
        
        assert X_scaled.shape == X.shape
    
    def test_transform_without_fit_raises(self, sample_numeric_features):
        """Test transform without fit raises error."""
        X, y = sample_numeric_features
        preprocessor = StudentPreprocessor()
        
        with pytest.raises(Exception):
            preprocessor.transform(X)
    
    def test_inverse_transform(self, sample_numeric_features):
        """Test inverse transform recovers original data."""
        X, y = sample_numeric_features
        preprocessor = StudentPreprocessor()
        
        X_scaled = preprocessor.fit_transform(X)
        X_recovered = preprocessor.inverse_transform(X_scaled)
        
        np.testing.assert_array_almost_equal(X, X_recovered, decimal=5)
    
    def test_fit_on_dataframe(self, sample_student_data):
        """Test fitting on DataFrame."""
        preprocessor = StudentPreprocessor()
        df = sample_student_data.copy()
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].drop(columns=['G3'])
        
        X_scaled = preprocessor.fit_transform(X)
        
        assert X_scaled.shape == X.shape


class TestPreprocessorPipeline:
    """Test cases for preprocessing pipeline."""
    
    def test_handle_categorical_encoding(self, sample_student_data):
        """Test categorical variable encoding."""
        preprocessor = StudentPreprocessor()
        df = sample_student_data.copy()
        
        encoded = preprocessor.encode_categorical(df)
        
        # All columns should be numeric after encoding
        for col in encoded.columns:
            assert encoded[col].dtype in [np.int64, np.int32, np.float64]
    
    def test_handle_missing_values(self, sample_student_data):
        """Test missing value handling."""
        preprocessor = StudentPreprocessor()
        df = sample_student_data.copy()
        
        # Introduce missing values
        df.loc[0, 'age'] = np.nan
        df.loc[1, 'absences'] = np.nan
        
        cleaned = preprocessor.handle_missing(df)
        
        assert cleaned.isna().sum().sum() == 0
    
    def test_full_preprocessing_pipeline(self, sample_student_data):
        """Test full preprocessing pipeline."""
        preprocessor = StudentPreprocessor()
        df = sample_student_data.copy()
        
        # Run full pipeline
        X, y = preprocessor.preprocess(df, target_column='G3')
        
        assert 'G3' not in X.columns if hasattr(X, 'columns') else True
        assert len(X) == len(y)
    
    def test_save_and_load_scaler(self, sample_numeric_features, temp_model_dir):
        """Test saving and loading scaler."""
        X, y = sample_numeric_features
        preprocessor = StudentPreprocessor()
        
        preprocessor.fit(X)
        scaler_path = temp_model_dir / "scaler.joblib"
        preprocessor.save_scaler(str(scaler_path))
        
        # Load in new preprocessor
        new_preprocessor = StudentPreprocessor()
        new_preprocessor.load_scaler(str(scaler_path))
        
        # Should produce same results
        X_scaled1 = preprocessor.transform(X)
        X_scaled2 = new_preprocessor.transform(X)
        
        np.testing.assert_array_almost_equal(X_scaled1, X_scaled2)


class TestDataSplitting:
    """Test cases for data splitting."""
    
    def test_train_test_split(self, sample_numeric_features):
        """Test train-test splitting."""
        X, y = sample_numeric_features
        preprocessor = StudentPreprocessor()
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_split_ratio(self, sample_numeric_features):
        """Test split ratio is approximately correct."""
        X, y = sample_numeric_features
        preprocessor = StudentPreprocessor()
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X, y, test_size=0.3, random_state=42
        )
        
        expected_test_ratio = 0.3
        actual_test_ratio = len(X_test) / len(X)
        
        assert abs(actual_test_ratio - expected_test_ratio) < 0.05
