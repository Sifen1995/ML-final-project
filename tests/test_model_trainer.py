"""
Tests for model trainer module.
"""

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from src.model_trainer import GradeModelTrainer


class TestGradeModelTrainer:
    """Test cases for GradeModelTrainer."""
    
    def test_trainer_initialization(self):
        """Test GradeModelTrainer initialization."""
        trainer = GradeModelTrainer()
        assert trainer is not None
    
    def test_get_model_linear_regression(self):
        """Test getting linear regression model."""
        trainer = GradeModelTrainer()
        model = trainer.get_model('linear_regression')
        
        assert isinstance(model, LinearRegression)
    
    def test_get_model_ridge(self):
        """Test getting ridge regression model."""
        trainer = GradeModelTrainer()
        model = trainer.get_model('ridge')
        
        assert isinstance(model, Ridge)
    
    def test_get_model_random_forest(self):
        """Test getting random forest regressor."""
        trainer = GradeModelTrainer()
        model = trainer.get_model('random_forest')
        
        assert isinstance(model, RandomForestRegressor)
    
    def test_get_model_invalid_raises(self):
        """Test invalid model name raises error."""
        trainer = GradeModelTrainer()
        
        with pytest.raises(ValueError):
            trainer.get_model('invalid_model')
    
    def test_train_model(self, sample_numeric_features):
        """Test training a model."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        model = trainer.train(X, y, model_name='linear_regression')
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_train_and_predict(self, sample_numeric_features):
        """Test training and making predictions."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        model = trainer.train(X, y, model_name='ridge')
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_cross_validation(self, sample_numeric_features):
        """Test cross-validation scoring."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        scores = trainer.cross_validate(X, y, model_name='linear_regression', cv=3)
        
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)


class TestModelTraining:
    """Test cases for model training functionality."""
    
    def test_train_all_models(self, sample_numeric_features):
        """Test training all available models."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        results = trainer.train_all_models(X, y)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for model_name, result in results.items():
            assert 'model' in result
            assert 'score' in result
    
    def test_get_best_model(self, sample_numeric_features):
        """Test getting best performing model."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        trainer.train_all_models(X, y)
        best_model, best_name, best_score = trainer.get_best_model()
        
        assert best_model is not None
        assert isinstance(best_name, str)
        assert isinstance(best_score, float)
    
    def test_model_predictions_in_range(self, sample_numeric_features):
        """Test predictions are in reasonable range."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        model = trainer.train(X, y, model_name='random_forest')
        predictions = model.predict(X)
        
        # Predictions should be somewhat close to actual values
        assert predictions.min() >= -5  # Allow some margin
        assert predictions.max() <= 25  # Allow some margin above 20
    
    def test_save_and_load_model(self, sample_numeric_features, temp_model_dir):
        """Test saving and loading models."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        model = trainer.train(X, y, model_name='ridge')
        model_path = temp_model_dir / "model.joblib"
        trainer.save_model(model, str(model_path))
        
        loaded_model = trainer.load_model(str(model_path))
        
        # Should produce same predictions
        pred1 = model.predict(X)
        pred2 = loaded_model.predict(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)


class TestHyperparameterTuning:
    """Test cases for hyperparameter tuning."""
    
    def test_grid_search(self, sample_numeric_features):
        """Test grid search for hyperparameters."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        best_model, best_params = trainer.grid_search(
            X, y, model_name='ridge', param_grid=param_grid, cv=3
        )
        
        assert best_model is not None
        assert 'alpha' in best_params
    
    def test_random_search(self, sample_numeric_features):
        """Test random search for hyperparameters."""
        X, y = sample_numeric_features
        trainer = GradeModelTrainer()
        
        param_dist = {'n_estimators': [10, 50, 100]}
        best_model, best_params = trainer.random_search(
            X, y, model_name='random_forest', 
            param_distributions=param_dist, 
            n_iter=2, cv=3
        )
        
        assert best_model is not None
