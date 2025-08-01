"""
Prediction script for model verification in Docker container.
"""

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import calculate_r2_score


def load_model_and_data():
    """
    Load trained model and prepare test data.
    
    Returns:
        tuple: (model, scaler, X_test, y_test) - Model and test data
    """
    print("Loading trained model and scaler...")
    
    # Load model and scaler
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    print("Loading California Housing dataset...")
    
    # Load fresh data for testing
    california = fetch_california_housing()
    X = california.data
    y = california.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale test data using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    return model, scaler, X_test_scaled, y_test


def run_predictions():
    """
    Run predictions on test data and print sample outputs.
    """
    print("=== Model Prediction Verification ===")
    
    # Load model and data
    model, scaler, X_test, y_test = load_model_and_data()
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of test samples: {len(y_test)}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Calculate performance metrics
    r2_score = calculate_r2_score(y_test, predictions)
    mse = np.mean((y_test - predictions) ** 2)
    
    print(f"\n=== Performance Metrics ===")
    print(f"R² Score: {r2_score:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Print sample predictions
    print(f"\n=== Sample Predictions ===")
    print("Sample | Actual | Predicted | Difference")
    print("-" * 45)
    
    for i in range(min(10, len(predictions))):
        actual = y_test[i]
        predicted = predictions[i]
        difference = abs(actual - predicted)
        print(f"{i+1:6d} | {actual:6.3f} | {predicted:9.3f} | {difference:9.3f}")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Mean Actual Value: {np.mean(y_test):.3f}")
    print(f"Mean Predicted Value: {np.mean(predictions):.3f}")
    print(f"Standard Deviation of Predictions: {np.std(predictions):.3f}")
    print(f"Min Prediction: {np.min(predictions):.3f}")
    print(f"Max Prediction: {np.max(predictions):.3f}")
    
    # Verify model is working correctly
    if r2_score > 0.5:
        print("\nModel verification successful!")
        print("Predictions are reasonable and model is working correctly.")
        return True
    else:
        print("\nModel verification failed!")
        print("R² score is too low, model may not be working correctly.")
        return False


def test_quantized_model():
    """
    Test predictions with quantized model parameters.
    """
    print("\n=== Testing Quantized Model ===")
    
    try:
        # Load quantized parameters
        quantized_params = joblib.load('quant_params.joblib')
        
        # Load test data
        model, scaler, X_test, y_test = load_model_and_data()
        
        # De-quantize parameters
        from quantize import dequantize_from_8bit
        
        dequantized_coefficients = dequantize_from_8bit(
            quantized_params['coefficients'],
            quantized_params['coef_min'],
            quantized_params['coef_max']
        )
        dequantized_intercept = dequantize_from_8bit(
            quantized_params['intercept'],
            quantized_params['intercept_min'],
            quantized_params['intercept_max']
        )
        
        # Make predictions with de-quantized weights
        quantized_predictions = np.dot(X_test, dequantized_coefficients) + dequantized_intercept
        
        # Calculate performance
        quantized_r2 = calculate_r2_score(y_test, quantized_predictions)
        
        print(f"Quantized Model R² Score: {quantized_r2:.4f}")
        
        if quantized_r2 > 0.5:
            print("Quantized model verification successful!")
        else:
            print("Quantized model verification failed!")
            
    except FileNotFoundError:
        print("Quantized parameters not found, skipping quantized model test.")


if __name__ == "__main__":
    # Run main prediction verification
    success = run_predictions()
    
    # Test quantized model if available
    test_quantized_model()
    
    if success:
        print("\nAll prediction tests completed successfully!")
        exit(0)
    else:
        print("\nPrediction tests failed!")
        exit(1) 