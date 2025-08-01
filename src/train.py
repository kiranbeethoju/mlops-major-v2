"""
Model training script for Linear Regression on California Housing dataset.
"""

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from utils import load_california_housing_data, calculate_r2_score, calculate_mse


def train_model():
    """
    Train Linear Regression model on California Housing dataset.
    
    Returns:
        tuple: (model, X_test, y_test, scaler) - Trained model and test data
    """
    print("Loading California Housing dataset...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_california_housing_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create and train Linear Regression model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = calculate_r2_score(y_train, y_train_pred)
    test_r2 = calculate_r2_score(y_test, y_test_pred)
    train_mse = calculate_mse(y_train, y_train_pred)
    test_mse = calculate_mse(y_test, y_test_pred)
    
    # Print results
    print("\n=== Model Performance ===")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    print("Model training completed successfully!")
    
    return model, X_test, y_test, scaler


if __name__ == "__main__":
    train_model() 