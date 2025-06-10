#!/usr/bin/env python3
"""
XGBoost Model Training for Train Comfort Predictor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from feature_selection import feature_selection_pipeline
import warnings
warnings.filterwarnings('ignore')

# Note: This will be updated to use target_definition_simple_5tier after feature_selection.py is updated


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    print("=== STARTING DATA SPLITTING ===")
    
    # Stratified split to ensure balanced classes in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Check class distribution
    print(f"\nTraining set class distribution:")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    for idx, count in train_dist.items():
        tier_name = ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'][idx]
        print(f"  {tier_name} ({idx}): {count} ({count/len(y_train)*100:.1f}%)")
    
    print(f"\nTest set class distribution:")
    test_dist = pd.Series(y_test).value_counts().sort_index()
    for idx, count in test_dist.items():
        tier_name = ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'][idx]
        print(f"  {tier_name} ({idx}): {count} ({count/len(y_test)*100:.1f}%)")
    
    print("=== COMPLETE ===")
    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline(X_train, X_test):
    """Create preprocessing pipeline (if needed)."""
    print("\n=== PREPROCESSING PIPELINE ===")
    
    # For XGBoost, we generally don't need scaling, but let's check for consistency
    # Most features are already properly encoded or scaled
    print("Checking for preprocessing needs...")
    
    # Check feature ranges
    feature_ranges = pd.DataFrame({
        'feature': X_train.columns,
        'min_train': X_train.min().values,
        'max_train': X_train.max().values,
        'mean_train': X_train.mean().values,
        'std_train': X_train.std().values
    })
    
    # Identify features with very different scales
    large_scale_features = feature_ranges[
        (feature_ranges['max_train'] > 1000) | (feature_ranges['std_train'] > 100)
    ]
    
    if len(large_scale_features) > 0:
        print(f"Features with large scale detected: {large_scale_features['feature'].tolist()}")
        print("Applying standard scaling to these features...")
        
        # Apply scaling only to large-scale features
        scaler = StandardScaler()
        large_scale_cols = large_scale_features['feature'].tolist()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[large_scale_cols] = scaler.fit_transform(X_train[large_scale_cols])
        X_test_scaled[large_scale_cols] = scaler.transform(X_test[large_scale_cols])
        
        print("=== PREPROCESSING APPLIED ===")
        return X_train_scaled, X_test_scaled, scaler
    else:
        print("No preprocessing needed - feature scales are appropriate for XGBoost")
        print("=== COMPLETE ===")
        return X_train, X_test, None


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier for 5-tier system."""
    print("\n=== TRAINING XGBOOST MODEL (5-TIER) ===")
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[i] for i in y_train])
    
    print(f"Class weights: {dict(zip(classes, class_weights))}")
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',  # Multi-class classification
        num_class=5,  # Changed from 3 to 5
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=10,
        verbosity=1
    )
    
    # Set up evaluation set for early stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    eval_names = ['train', 'test']
    
    print("Training XGBoost model with class weights...")
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=False
    )
    
    print(f"Training completed!")
    print(f"Best iteration: {xgb_model.best_iteration}")
    print(f"Best score: {xgb_model.best_score:.4f}")
    
    print("=== COMPLETE ===")
    return xgb_model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Model evaluation with comprehensive metrics."""
    print("\n=== MODEL EVALUATION ===")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    class_names = ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet']
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Feature importance
    plt.subplot(1, 2, 2)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=feature_importance, y='feature', x='importance')
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('models/model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Prediction probabilities analysis
    prob_df = pd.DataFrame(y_test_proba, columns=class_names)
    prob_df['actual'] = [class_names[i] for i in y_test]
    prob_df['predicted'] = [class_names[i] for i in y_test_pred]
    
    print(f"\nPrediction Confidence Analysis:")
    for class_name in class_names:
        class_mask = prob_df['actual'] == class_name
        avg_confidence = prob_df.loc[class_mask, class_name].mean()
        print(f"  {class_name}: Average confidence = {avg_confidence:.3f}")
    
    print("=== COMPLETE ===")
    
    evaluation_results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'classification_report': classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)
    }
    
    return evaluation_results


def hyperparameter_tuning(X_train, y_train):
    """Hyperparameter tuning using GridSearchCV."""
    print("\n=== HYPERPARAMETER TUNING ===")
    
    # Define parameter grid for tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Initialize base model
    xgb_base = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    print("Starting GridSearchCV (this may take a while)...")
    print(f"Parameter combinations to test: {len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])}")
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,  # 3-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    print("=== COMPLETE ===")
    return grid_search.best_estimator_, grid_search.best_params_


def save_model_and_artifacts(model, scaler, feature_info, evaluation_results, best_params=None):
    """Save trained model and supporting files."""
    print("\n=== SAVING MODEL AND ARTIFACTS ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model
    model_path = 'models/xgboost_comfort_classifier.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save scaler if it exists
    if scaler is not None:
        scaler_path = 'models/feature_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
    
    # Save target encoder
    target_encoder_path = 'models/target_encoder.joblib'
    joblib.dump(feature_info['target_encoder'], target_encoder_path)
    print(f"Target encoder saved to: {target_encoder_path}")
    
    # Save feature encoders
    encoders_path = 'models/feature_encoders.joblib'
    joblib.dump(feature_info['encoders'], encoders_path)
    print(f"Feature encoders saved to: {encoders_path}")
    
    # Save feature list
    feature_list_path = 'models/feature_list.joblib'
    joblib.dump(feature_info['feature_list'], feature_list_path)
    print(f"Feature list saved to: {feature_list_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'XGBoost Classifier (5-Tier)',
        'feature_count': feature_info['feature_count'],
        'target_classes': ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'],
        'train_accuracy': evaluation_results['train_accuracy'],
        'test_accuracy': evaluation_results['test_accuracy'],
        'feature_list': feature_info['feature_list'],
        'best_params': best_params,
        'model_path': model_path,
        'target_encoder_path': target_encoder_path,
        'encoders_path': encoders_path,
        'feature_list_path': feature_list_path,
        'scaler_path': scaler_path if scaler is not None else None
    }
    
    metadata_path = 'models/model_metadata.joblib'
    joblib.dump(metadata, metadata_path)
    print(f"Model metadata saved to: {metadata_path}")
    
    print("=== COMPLETE ===")
    return metadata


def xgboost_training_pipeline():
    """Complete XGBoost training pipeline."""
    print("=== STARTING XGBOOST TRAINING PIPELINE ===")
    
    # Get processed data from feature selection
    X, y, feature_info, df = feature_selection_pipeline()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Preprocessing
    X_train_processed, X_test_processed, scaler = create_preprocessing_pipeline(X_train, X_test)
    
    # Train initial model
    initial_model = train_xgboost_model(X_train_processed, y_train, X_test_processed, y_test)
    
    # Evaluate initial model
    initial_results = evaluate_model(initial_model, X_train_processed, X_test_processed, 
                                   y_train, y_test, feature_info['feature_list'])
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    tuned_model, best_params = hyperparameter_tuning(X_train_processed, y_train)
    
    # Evaluate tuned model
    print("\nEvaluating tuned model...")
    tuned_results = evaluate_model(tuned_model, X_train_processed, X_test_processed,
                                 y_train, y_test, feature_info['feature_list'])
    
    # Choose best model
    if tuned_results['test_accuracy'] > initial_results['test_accuracy']:
        print(f"\nTuned model performs better: {tuned_results['test_accuracy']:.4f} vs {initial_results['test_accuracy']:.4f}")
        final_model = tuned_model
        final_results = tuned_results
    else:
        print(f"\nInitial model performs better: {initial_results['test_accuracy']:.4f} vs {tuned_results['test_accuracy']:.4f}")
        final_model = initial_model
        final_results = initial_results
        best_params = None
    
    # Save model and artifacts
    metadata = save_model_and_artifacts(final_model, scaler, feature_info, final_results, best_params)
    
    print("\n=== XGBOOST TRAINING PIPELINE COMPLETE ===")
    print(f"Final model accuracy: {final_results['test_accuracy']:.4f}")
    print("Model and artifacts saved to models/ directory")
    print("Ready for API development!")
    
    return final_model, final_results, metadata


if __name__ == "__main__":
    model, results, metadata = xgboost_training_pipeline() 