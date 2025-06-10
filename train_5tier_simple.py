#!/usr/bin/env python3
"""
Simple 5-tier model training without hyperparameter tuning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_selection import feature_selection_pipeline
from src.xgboost_model_training import split_data, train_xgboost_model, evaluate_model, save_model_and_artifacts
import warnings
warnings.filterwarnings('ignore')

def train_5tier_model_simple():
    """Train 5-tier model without hyperparameter tuning."""
    print("=== SIMPLE 5-TIER MODEL TRAINING ===")
    
    # Get features and target
    X, y, feature_info, df = feature_selection_pipeline()
    print(f"Features: {X.shape}")
    print(f"Target distribution:")
    for i, tier in enumerate(['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet']):
        count = (y == i).sum()
        print(f"  {tier}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model (without hyperparameter tuning)
    model = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_train, X_test, y_train, y_test, feature_info['feature_list'])
    
    # Save model and artifacts
    save_model_and_artifacts(
        model=model,
        feature_info=feature_info,
        evaluation_results=evaluation_results,
        best_params=model.get_params(),  # Use default params
        scaler=None
    )
    
    print("=== 5-TIER MODEL TRAINING COMPLETE ===")
    print("Model artifacts saved to models/ directory")

if __name__ == "__main__":
    train_5tier_model_simple()