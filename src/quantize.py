"""
Manual quantization script for Linear Regression model parameters.
"""

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from utils import load_california_housing_data, calculate_r2_score


def quantize_to_8bit(value, min_val, max_val):
    """
    Quantize a value to 8-bit unsigned integer.
    
    Args:
        value: Value to quantize
        min_val: Minimum value in the range
        max_val: Maximum value in the range
        
    Returns:
        int: Quantized 8-bit value (0-255)
    """
    # Scale to 0-255 range
    scaled = ((value - min_val) / (max_val - min_val)) * 255
    # Round to nearest integer and clip to valid range
    quantized = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    return quantized


def dequantize_from_8bit(quantized_value, min_val, max_val):
    """
    De-quantize an 8-bit value back to original scale.
    
    Args:
        quantized_value: Quantized 8-bit value (0-255)
        min_val: Minimum value in the original range
        max_val: Maximum value in the original range
        
    Returns:
        float: De-quantized value
    """
    # Scale back to original range
    dequantized = (quantized_value / 255.0) * (max_val - min_val) + min_val
    return dequantized


def quantize_model_parameters():
    """
    Quantize Linear Regression model parameters to 8-bit integers.
    """
    print("Loading trained model...")
    
    # Load trained model
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Extract model parameters
    coefficients = model.coef_
    intercept = model.intercept_
    
    print(f"Original coefficients shape: {coefficients.shape}")
    print(f"Original intercept: {intercept}")
    
    # Find min and max values for quantization
    coef_min, coef_max = coefficients.min(), coefficients.max()
    intercept_min, intercept_max = intercept, intercept  # Single value
    
    print(f"\nCoefficients range: [{coef_min:.6f}, {coef_max:.6f}]")
    print(f"Intercept: {intercept:.6f}")
    
    # Quantize coefficients
    print("\nQuantizing coefficients...")
    quantized_coefficients = quantize_to_8bit(coefficients, coef_min, coef_max)
    
    # Quantize intercept (handle single value - use a small range)
    intercept_range = 0.1
    quantized_intercept = quantize_to_8bit(intercept, intercept - intercept_range, intercept + intercept_range)
    
    print(f"Quantized coefficients shape: {quantized_coefficients.shape}")
    print(f"Quantized intercept: {quantized_intercept}")
    
    # De-quantize parameters
    print("\nDe-quantizing parameters...")
    dequantized_coefficients = dequantize_from_8bit(quantized_coefficients, coef_min, coef_max)
    dequantized_intercept = dequantize_from_8bit(quantized_intercept, intercept - intercept_range, intercept + intercept_range)
    
    print(f"De-quantized coefficients shape: {dequantized_coefficients.shape}")
    print(f"De-quantized intercept: {dequantized_intercept}")
    
    # Calculate quantization error
    coef_error = np.mean(np.abs(coefficients - dequantized_coefficients))
    intercept_error = abs(intercept - dequantized_intercept)
    
    print(f"\nQuantization Error:")
    print(f"Coefficients MAE: {coef_error:.8f}")
    print(f"Intercept Error: {intercept_error:.8f}")
    
    # Save quantized parameters
    print("\nSaving quantized parameters...")
    quantized_params = {
        'coefficients': quantized_coefficients,
        'intercept': quantized_intercept,
        'coef_min': coef_min,
        'coef_max': coef_max,
        'intercept_min': intercept - intercept_range,
        'intercept_max': intercept + intercept_range
    }
    
    joblib.dump(quantized_params, 'quant_params.joblib')
    joblib.dump({
        'coefficients': coefficients,
        'intercept': intercept
    }, 'unquant_params.joblib')
    
    print("Quantized parameters saved successfully!")
    
    return quantized_params, dequantized_coefficients, dequantized_intercept


def test_quantized_inference():
    """
    Test inference with de-quantized weights.
    """
    print("\n=== Testing Quantized Inference ===")
    
    # Load test data
    X_train, X_test, y_train, y_test, scaler = load_california_housing_data()
    
    # Load original model for comparison
    original_model = joblib.load('model.joblib')
    
    # Load quantized parameters
    quantized_params = joblib.load('quant_params.joblib')
    
    # De-quantize parameters
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
    
    # Create predictions with original model
    original_predictions = original_model.predict(X_test)
    
    # Create predictions with de-quantized weights
    dequantized_predictions = np.dot(X_test, dequantized_coefficients) + dequantized_intercept
    
    # Calculate R² scores
    original_r2 = calculate_r2_score(y_test, original_predictions)
    dequantized_r2 = calculate_r2_score(y_test, dequantized_predictions)
    
    print(f"Original model R² score: {original_r2:.4f}")
    print(f"De-quantized model R² score: {dequantized_r2:.4f}")
    
    # Calculate prediction difference
    prediction_diff = np.mean(np.abs(original_predictions - dequantized_predictions))
    print(f"Average prediction difference: {prediction_diff:.6f}")
    
    # Check if performance degradation is acceptable (within 1%)
    performance_degradation = abs(original_r2 - dequantized_r2)
    print(f"Performance degradation: {performance_degradation:.6f}")
    
    if performance_degradation < 0.01:
        print("Quantization successful - performance degradation is acceptable")
    else:
        print("Warning - significant performance degradation detected")
    
    return original_r2, dequantized_r2


if __name__ == "__main__":
    # Run quantization
    quantized_params, dequantized_coefficients, dequantized_intercept = quantize_model_parameters()
    
    # Test quantized inference
    original_r2, dequantized_r2 = test_quantized_inference()
    
    print("\nQuantization process completed successfully!") 