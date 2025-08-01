"""
Utility functions for the MLOps Linear Regression pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_california_housing_data():
    """
    Load California Housing dataset from sklearn.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and test data
    """
    # Load the California Housing dataset
    california = fetch_california_housing()
    X = california.data
    y = california.target
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score manually.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        float: R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        float: MSE
    """
    return np.mean((y_true - y_pred) ** 2) 