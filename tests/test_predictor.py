"""
Tests for predictor module.
"""

import pytest
import numpy as np
import pandas as pd

from src.predictor import GradePredictor


class TestGradePredictor:
    """Test cases for GradePredictor."""
    
    def test_predictor_initialization(self):
        """Test GradePredictor initialization."""
        predictor = GradePredictor()
        assert predictor is not None
    
    def test_get_performance_level_excellent(self):
        """Test excellent performance level classification."""
        predictor = GradePredictor()
        
        level = predictor.get_performance_level(18)
        assert level == 'Excellent'
        
        level = predictor.get_performance_level(16)
        assert level == 'Excellent'
    
    def test_get_performance_level_good(self):
        """Test good performance level classification."""
        predictor = GradePredictor()
        
        level = predictor.get_performance_level(15)
        assert level == 'Good'
        
        level = predictor.get_performance_level(14)
        assert level == 'Good'
    
    def test_get_performance_level_average(self):
        """Test average performance level classification."""
        predictor = GradePredictor()
        
        level = predictor.get_performance_level(13)
        assert level == 'Average'
        
        level = predictor.get_performance_level(12)
        assert level == 'Average'
    
    def test_get_performance_level_poor(self):
        """Test poor performance level classification."""
        predictor = GradePredictor()
        
        level = predictor.get_performance_level(11)
        assert level == 'Poor'
        
        level = predictor.get_performance_level(10)
        assert level == 'Poor'
    
    def test_get_performance_level_failing(self):
        """Test failing performance level classification."""
        predictor = GradePredictor()
        
        level = predictor.get_performance_level(9)
        assert level == 'Failing'
        
        level = predictor.get_performance_level(0)
        assert level == 'Failing'
    
    def test_clip_prediction_high(self):
        """Test clipping predictions above 20."""
        predictor = GradePredictor()
        
        clipped = predictor.clip_prediction(25)
        assert clipped == 20
    
    def test_clip_prediction_low(self):
        """Test clipping predictions below 0."""
        predictor = GradePredictor()
        
        clipped = predictor.clip_prediction(-5)
        assert clipped == 0
    
    def test_clip_prediction_in_range(self):
        """Test predictions in valid range unchanged."""
        predictor = GradePredictor()
        
        clipped = predictor.clip_prediction(15)
        assert clipped == 15


class TestPredictionFormatting:
    """Test cases for prediction formatting."""
    
    def test_format_single_prediction(self):
        """Test formatting single prediction."""
        predictor = GradePredictor()
        
        result = predictor.format_prediction(15.5)
        
        assert 'grade' in result
        assert 'level' in result
        assert result['grade'] == 15.5
        assert result['level'] == 'Good'
    
    def test_format_batch_predictions(self):
        """Test formatting batch predictions."""
        predictor = GradePredictor()
        predictions = np.array([18, 14, 10, 5])
        
        results = predictor.format_batch_predictions(predictions)
        
        assert len(results) == 4
        assert results[0]['level'] == 'Excellent'
        assert results[1]['level'] == 'Good'
        assert results[2]['level'] == 'Poor'
        assert results[3]['level'] == 'Failing'
    
    def test_prediction_summary(self):
        """Test prediction summary statistics."""
        predictor = GradePredictor()
        predictions = np.array([18, 14, 12, 10, 8])
        
        summary = predictor.get_prediction_summary(predictions)
        
        assert 'mean' in summary
        assert 'median' in summary
        assert 'min' in summary
        assert 'max' in summary
        assert 'std' in summary
    
    def test_level_distribution(self):
        """Test performance level distribution."""
        predictor = GradePredictor()
        predictions = np.array([18, 18, 14, 14, 12, 10, 5])
        
        distribution = predictor.get_level_distribution(predictions)
        
        assert 'Excellent' in distribution
        assert distribution['Excellent'] == 2
        assert distribution['Good'] == 2


class TestPredictorWithModel:
    """Test cases for predictor with model integration."""
    
    def test_predict_with_mock_model(self, sample_numeric_features):
        """Test prediction with mock model."""
        X, y = sample_numeric_features
        predictor = GradePredictor()
        
        # Create simple mock model
        class MockModel:
            def predict(self, X):
                return np.random.uniform(0, 20, len(X))
        
        predictor.model = MockModel()
        predictions = predictor.predict(X)
        
        assert len(predictions) == len(X)
        assert all(0 <= p <= 20 for p in predictions)
    
    def test_predict_proba_not_applicable(self):
        """Test that predict_proba raises error for regression."""
        predictor = GradePredictor()
        
        with pytest.raises(NotImplementedError):
            predictor.predict_proba(np.array([[1, 2, 3]]))
    
    def test_get_recommendations(self):
        """Test getting improvement recommendations."""
        predictor = GradePredictor()
        
        # Student at risk
        student_data = {
            'studytime': 1,
            'failures': 2,
            'absences': 30,
            'G1': 8,
            'G2': 7
        }
        
        recommendations = predictor.get_recommendations(student_data, predicted_grade=6)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
