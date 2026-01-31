"""
Tests for data loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import StudentDataLoader


class TestStudentDataLoader:
    """Test cases for StudentDataLoader."""
    
    def test_loader_initialization(self):
        """Test StudentDataLoader initialization."""
        loader = StudentDataLoader()
        assert loader is not None
    
    def test_load_from_dataframe(self, sample_student_data):
        """Test loading from DataFrame."""
        loader = StudentDataLoader()
        result = loader.load_from_dataframe(sample_student_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_student_data)
    
    def test_validate_columns(self, sample_student_data):
        """Test column validation."""
        loader = StudentDataLoader()
        
        # Should not raise for valid data
        loader._validate_dataframe(sample_student_data)
    
    def test_validate_missing_target(self, sample_student_data):
        """Test validation fails for missing target."""
        loader = StudentDataLoader()
        df = sample_student_data.drop(columns=['G3'])
        
        with pytest.raises(ValueError):
            loader._validate_dataframe(df)
    
    def test_get_feature_columns(self, sample_student_data):
        """Test feature column extraction."""
        loader = StudentDataLoader()
        features = loader.get_feature_columns(sample_student_data)
        
        assert 'G3' not in features
        assert 'age' in features
        assert 'studytime' in features
    
    def test_split_features_target(self, sample_student_data):
        """Test splitting features and target."""
        loader = StudentDataLoader()
        X, y = loader.split_features_target(sample_student_data)
        
        assert 'G3' not in X.columns
        assert len(X) == len(y)
        assert y.name == 'G3'
    
    def test_load_csv_file(self, temp_data_dir):
        """Test loading from CSV file."""
        loader = StudentDataLoader()
        file_path = temp_data_dir / "student_mat.csv"
        
        df = loader.load_csv(str(file_path))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        loader = StudentDataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent_file.csv")


class TestDataValidation:
    """Test cases for data validation."""
    
    def test_validate_grade_range(self, sample_student_data):
        """Test grade values are in valid range."""
        loader = StudentDataLoader()
        df = sample_student_data.copy()
        
        # All grades should be between 0 and 20
        for col in ['G1', 'G2', 'G3']:
            if col in df.columns:
                assert df[col].min() >= 0
                assert df[col].max() <= 20
    
    def test_handle_missing_values(self, sample_student_data):
        """Test handling of missing values."""
        loader = StudentDataLoader()
        df = sample_student_data.copy()
        df.loc[0, 'age'] = np.nan
        
        cleaned = loader.handle_missing_values(df)
        
        assert cleaned['age'].isna().sum() == 0
    
    def test_encode_categorical(self, sample_student_data):
        """Test categorical encoding."""
        loader = StudentDataLoader()
        df = sample_student_data.copy()
        
        encoded = loader.encode_categorical(df)
        
        # Check that categorical columns are numeric
        assert encoded['school'].dtype in [np.int64, np.int32, np.float64]
        assert encoded['sex'].dtype in [np.int64, np.int32, np.float64]
