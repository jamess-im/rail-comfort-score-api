#!/usr/bin/env python3
"""
Quick training script for 5-tier model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.target_definition_simple_5tier import target_variable_pipeline_five_tier
from src.feature_selection import select_model_features
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def main():
    print('=== QUICK 5-TIER MODEL TRAINING ===')
    
    print('Loading data...')
    df, target_encoder, encoders, thresholds = target_variable_pipeline_five_tier()
    X, y, feature_list = select_model_features(df)

    print(f'Training on {X.shape[0]} samples with {X.shape[1]} features')
    print(f'Target distribution: {np.bincount(y)}')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Calculate class weights  
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[i] for i in y_train])

    print('Training XGBoost model...')
    model = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=5, 
        random_state=42, 
        eval_metric='mlogloss', 
        verbosity=0,
        n_estimators=100
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    test_acc = model.score(X_test, y_test)
    print(f'Test accuracy: {test_acc:.3f}')

    print('Saving model artifacts...')
    joblib.dump(model, 'models/xgboost_comfort_classifier.joblib')
    joblib.dump(target_encoder, 'models/target_encoder.joblib') 
    joblib.dump(encoders, 'models/feature_encoders.joblib')
    joblib.dump(feature_list, 'models/feature_list.joblib')

    metadata = {
        'model_type': 'XGBoost Classifier (5-Tier)',
        'feature_count': len(feature_list),
        'target_classes': ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'],
        'test_accuracy': test_acc,
        'feature_list': feature_list
    }
    joblib.dump(metadata, 'models/model_metadata.joblib')

    print('=== 5-TIER MODEL TRAINING COMPLETE ===')
    print(f'Model saved with {len(feature_list)} features')
    print(f'Target classes: {metadata["target_classes"]}')

if __name__ == "__main__":
    main()