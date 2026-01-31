"""
Pytest fixtures for Student Grade Prediction tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_student_data():
    """Create sample student data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'school': np.random.choice(['GP', 'MS'], n_samples),
        'sex': np.random.choice(['F', 'M'], n_samples),
        'age': np.random.randint(15, 22, n_samples),
        'address': np.random.choice(['U', 'R'], n_samples),
        'famsize': np.random.choice(['LE3', 'GT3'], n_samples),
        'Pstatus': np.random.choice(['T', 'A'], n_samples),
        'Medu': np.random.randint(0, 5, n_samples),
        'Fedu': np.random.randint(0, 5, n_samples),
        'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
        'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
        'reason': np.random.choice(['home', 'reputation', 'course', 'other'], n_samples),
        'guardian': np.random.choice(['mother', 'father', 'other'], n_samples),
        'traveltime': np.random.randint(1, 5, n_samples),
        'studytime': np.random.randint(1, 5, n_samples),
        'failures': np.random.randint(0, 4, n_samples),
        'schoolsup': np.random.choice(['yes', 'no'], n_samples),
        'famsup': np.random.choice(['yes', 'no'], n_samples),
        'paid': np.random.choice(['yes', 'no'], n_samples),
        'activities': np.random.choice(['yes', 'no'], n_samples),
        'nursery': np.random.choice(['yes', 'no'], n_samples),
        'higher': np.random.choice(['yes', 'no'], n_samples),
        'internet': np.random.choice(['yes', 'no'], n_samples),
        'romantic': np.random.choice(['yes', 'no'], n_samples),
        'famrel': np.random.randint(1, 6, n_samples),
        'freetime': np.random.randint(1, 6, n_samples),
        'goout': np.random.randint(1, 6, n_samples),
        'Dalc': np.random.randint(1, 6, n_samples),
        'Walc': np.random.randint(1, 6, n_samples),
        'health': np.random.randint(1, 6, n_samples),
        'absences': np.random.randint(0, 50, n_samples),
        'G1': np.random.randint(0, 21, n_samples),
        'G2': np.random.randint(0, 21, n_samples),
        'G3': np.random.randint(0, 21, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_numeric_features():
    """Create sample numeric features for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 21, n_samples).astype(float)
    
    return X, y


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory with sample files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample CSV
    sample_data = {
        'school': ['GP', 'MS', 'GP'],
        'sex': ['F', 'M', 'F'],
        'age': [18, 17, 16],
        'Medu': [4, 3, 4],
        'Fedu': [4, 3, 2],
        'studytime': [2, 3, 1],
        'failures': [0, 0, 1],
        'absences': [6, 4, 10],
        'G1': [15, 12, 10],
        'G2': [14, 13, 10],
        'G3': [15, 13, 10]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(data_dir / "student_mat.csv", sep=';', index=False)
    
    return data_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
