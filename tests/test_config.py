"""
Tests for configuration module.
"""

import pytest
from src.config import DataConfig, ModelConfig, StudentFeatures


class TestDataConfig:
    """Test cases for DataConfig."""
    
    def test_data_config_attributes(self):
        """Test DataConfig has required attributes."""
        assert hasattr(DataConfig, 'STUDENT_DATA_PATH')
        assert hasattr(DataConfig, 'TARGET_COLUMN')
        assert hasattr(DataConfig, 'TEST_SIZE')
        assert hasattr(DataConfig, 'RANDOM_STATE')
    
    def test_target_column_is_g3(self):
        """Test target column is G3 (final grade)."""
        assert DataConfig.TARGET_COLUMN == 'G3'
    
    def test_test_size_is_valid(self):
        """Test test size is between 0 and 1."""
        assert 0 < DataConfig.TEST_SIZE < 1
    
    def test_random_state_is_integer(self):
        """Test random state is an integer."""
        assert isinstance(DataConfig.RANDOM_STATE, int)


class TestModelConfig:
    """Test cases for ModelConfig."""
    
    def test_model_config_attributes(self):
        """Test ModelConfig has required attributes."""
        assert hasattr(ModelConfig, 'MODELS')
        assert hasattr(ModelConfig, 'CV_FOLDS')
    
    def test_models_is_dict(self):
        """Test MODELS is a dictionary."""
        assert isinstance(ModelConfig.MODELS, dict)
    
    def test_models_contains_required_models(self):
        """Test required regression models are defined."""
        required_models = ['linear_regression', 'random_forest']
        for model in required_models:
            assert model in ModelConfig.MODELS
    
    def test_cv_folds_is_positive(self):
        """Test CV folds is positive integer."""
        assert ModelConfig.CV_FOLDS > 0


class TestStudentFeatures:
    """Test cases for StudentFeatures."""
    
    def test_student_features_attributes(self):
        """Test StudentFeatures has required lists."""
        assert hasattr(StudentFeatures, 'DEMOGRAPHIC')
        assert hasattr(StudentFeatures, 'ACADEMIC')
        assert hasattr(StudentFeatures, 'SOCIAL')
    
    def test_demographic_features(self):
        """Test demographic features list."""
        assert 'age' in StudentFeatures.DEMOGRAPHIC
        assert 'sex' in StudentFeatures.DEMOGRAPHIC
    
    def test_academic_features(self):
        """Test academic features list."""
        assert 'studytime' in StudentFeatures.ACADEMIC
        assert 'failures' in StudentFeatures.ACADEMIC
    
    def test_all_features_combined(self):
        """Test all features can be combined."""
        all_features = (
            StudentFeatures.DEMOGRAPHIC + 
            StudentFeatures.ACADEMIC + 
            StudentFeatures.SOCIAL
        )
        assert len(all_features) > 0
        assert len(set(all_features)) == len(all_features)  # No duplicates
