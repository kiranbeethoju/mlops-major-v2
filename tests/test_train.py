"""
Unit tests for the training pipeline.
"""

import pytest
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Import functions to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_california_housing_data, calculate_r2_score, calculate_mse
from train import train_model


class TestDatasetLoading:
    """Test dataset loading functionality."""
    
    def test_load_california_housing_data(self):
        """Test that dataset loading returns correct data types and shapes."""
        X_train, X_test, y_train, y_test, scaler = load_california_housing_data()
        
        # Check data types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check shapes
        assert len(X_train.shape) == 2
        assert len(X_test.shape) == 2
        assert len(y_train.shape) == 1
        assert len(y_test.shape) == 1
        
        # Check that training and test sets have same number of features
        assert X_train.shape[1] == X_test.shape[1]
        
        # Check that data is not empty
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        
        print("Dataset loading test passed")


class TestModelCreation:
    """Test model creation and validation."""
    
    def test_linear_regression_instance(self):
        """Test that LinearRegression model can be created."""
        model = LinearRegression()
        assert isinstance(model, LinearRegression)
        print("LinearRegression instance test passed")
    
    def test_model_training(self):
        """Test that model can be trained and has coefficients."""
        # Load data
        X_train, X_test, y_train, y_test, scaler = load_california_housing_data()
        
        # Create and train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Check that model has been trained (coefficients exist)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        # Check coefficient shape matches feature count
        assert len(model.coef_) == X_train.shape[1]
        
        print("Model training test passed")


class TestModelPerformance:
    """Test model performance metrics."""
    
    def test_r2_score_calculation(self):
        """Test R² score calculation."""
        # Create sample data
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        r2 = calculate_r2_score(y_true, y_pred)
        
        # R² should be between 0 and 1 for this case
        assert 0 <= r2 <= 1
        assert isinstance(r2, float)
        
        print("R² score calculation test passed")
    
    def test_mse_calculation(self):
        """Test MSE calculation."""
        # Create sample data
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        mse = calculate_mse(y_true, y_pred)
        
        # MSE should be positive
        assert mse > 0
        assert isinstance(mse, float)
        
        print("MSE calculation test passed")
    
    def test_model_performance_threshold(self):
        """Test that model performance exceeds minimum threshold."""
        # Train model
        model, X_test, y_test, scaler = train_model()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate R² score
        r2 = calculate_r2_score(y_test, y_pred)
        
        # R² should be above 0.5 (reasonable threshold for linear regression)
        assert r2 > 0.5, f"R² score {r2:.4f} is below threshold 0.5"
        
        print(f"Model performance threshold test passed (R² = {r2:.4f})")


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_model_saving(self):
        """Test that model can be saved and loaded."""
        # Train model
        model, X_test, y_test, scaler = train_model()
        
        # Check that model files exist
        assert os.path.exists('model.joblib')
        assert os.path.exists('scaler.joblib')
        
        # Load model
        loaded_model = joblib.load('model.joblib')
        loaded_scaler = joblib.load('scaler.joblib')
        
        # Check that loaded model is same type
        assert isinstance(loaded_model, LinearRegression)
        
        # Check that predictions are same
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        
        print("Model persistence test passed")


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 