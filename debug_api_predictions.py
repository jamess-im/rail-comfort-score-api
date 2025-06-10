#!/usr/bin/env python3
"""
Debug script to check why API predictions are biased toward "Very Busy"
"""

import joblib
import numpy as np
from datetime import datetime

def debug_feature_construction():
    print("=== DEBUGGING API FEATURE CONSTRUCTION ===")
    
    # Load model artifacts directly
    model = joblib.load('models/xgboost_comfort_classifier.joblib')
    feature_list = joblib.load('models/feature_list.joblib')
    try:
        scaler = joblib.load('models/feature_scaler.joblib')
    except:
        scaler = None
    
    print(f"Model expects {len(feature_list)} features")
    print(f"First 10 features: {feature_list[:10]}")
    
    # Create a simple test feature vector that should predict "Very Quiet"
    # Early morning (3 AM), weekend, low historical values
    print(f"\nTesting with early morning (3 AM) features that should predict Very Quiet...")
    
    # Create feature vector manually with values that should predict "Very Quiet"
    test_features = np.zeros(len(feature_list))
    
    # Set time features for 3 AM
    if 'hour_of_day' in feature_list:
        test_features[feature_list.index('hour_of_day')] = 3  # 3 AM
    if 'is_weekend' in feature_list:
        test_features[feature_list.index('is_weekend')] = 1   # Weekend
    if 'is_peak_hour' in feature_list:
        test_features[feature_list.index('is_peak_hour')] = 0  # Not peak
    
    # Set low occupancy-related features
    occupancy_features = ['occupancy_percentage_std', 'occupancy_percentage_first', 'total_occupancy']
    for feat in occupancy_features:
        if feat in feature_list:
            test_features[feature_list.index(feat)] = 5.0  # Very low occupancy
    
    print(f"Manual feature vector shape: {test_features.shape}")
    print(f"Key features set:")
    for feat in ['hour_of_day', 'is_weekend', 'is_peak_hour']:
        if feat in feature_list:
            idx = feature_list.index(feat)
            print(f"  {feat}: {test_features[idx]}")
    
    # Make prediction
    try:
        proba = model.predict_proba(test_features.reshape(1, -1))[0]
        predicted_class = np.argmax(proba)
        class_names = ["Very Busy", "Busy", "Moderate", "Quiet", "Very Quiet"]
        
        print(f"\nPrediction probabilities:")
        for i, (name, prob) in enumerate(zip(class_names, proba)):
            print(f"  {name}: {prob:.4f}")
        
        print(f"\nPredicted: {class_names[predicted_class]}")
        
        if predicted_class in [0, 1]:  # Very Busy or Busy
            print("❌ PROBLEM: Model predicting Busy/Very Busy for 3 AM!")
            print("This suggests an issue with model training or feature construction.")
        else:
            print("✅ Good: Model correctly predicts quiet times for 3 AM")
            
    except Exception as e:
        print(f"ERROR in prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_construction()