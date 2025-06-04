#!/usr/bin/env python3
"""
Target Variable Definition for Train Comfort Predictor (Simplified)
Define Target Variable (comfort_tier) based on relevant_passengers_on_leg_departure
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
from feature_engineering import feature_engineering_pipeline


def define_comfort_tiers_simple(df):
    """Define comfort tiers based on occupancy percentage."""
    print("=== DEFINING COMFORT TIERS ===")
    
    # Calculate occupancy percentage
    df['occupancy_percentage'] = (df['relevant_passengers_on_leg_departure'] / 
                                 df['vehicle_capacity'] * 100)
    
    print("Occupancy percentage statistics:")
    print(df['occupancy_percentage'].describe())
    
    # Use quantile-based approach for balanced classes
    quantiles = df['occupancy_percentage'].quantile([0.33, 0.67]).values
    low_threshold = quantiles[0]
    high_threshold = quantiles[1]
    
    print(f"\nQuantile-based thresholds:")
    print(f"33rd percentile: {low_threshold:.1f}%")
    print(f"67th percentile: {high_threshold:.1f}%")
    
    # Create comfort tiers
    def assign_comfort_tier(occupancy_pct):
        if occupancy_pct <= low_threshold:
            return 'Quiet'
        elif occupancy_pct <= high_threshold:
            return 'Moderate'
        else:
            return 'Busy'
    
    df['comfort_tier'] = df['occupancy_percentage'].apply(assign_comfort_tier)
    
    print(f"\nThresholds used:")
    print(f"Quiet: ≤ {low_threshold:.1f}%")
    print(f"Moderate: {low_threshold:.1f}% - {high_threshold:.1f}%")
    print(f"Busy: > {high_threshold:.1f}%")
    
    # Show distribution
    tier_counts = df['comfort_tier'].value_counts()
    tier_percentages = df['comfort_tier'].value_counts(normalize=True) * 100
    
    print(f"\nComfort tier distribution:")
    for tier in ['Quiet', 'Moderate', 'Busy']:
        count = tier_counts.get(tier, 0)
        pct = tier_percentages.get(tier, 0)
        print(f"{tier}: {count} ({pct:.1f}%)")
    
    return df, low_threshold, high_threshold


def encode_target_variable(df):
    """Encode comfort_tier for machine learning."""
    print(f"\n=== ENCODING TARGET VARIABLE ===")
    
    # Label encode the target variable
    target_encoder = LabelEncoder()
    df['comfort_tier_encoded'] = target_encoder.fit_transform(df['comfort_tier'])
    
    # Show mapping
    print(f"Target variable encoding:")
    for i, class_name in enumerate(target_encoder.classes_):
        count = (df['comfort_tier_encoded'] == i).sum()
        print(f"{class_name}: {i} (n={count})")
    
    return df, target_encoder


def analyze_target_by_features(df):
    """Analyze target variable distribution by key features."""
    print(f"\n=== TARGET ANALYSIS BY FEATURES ===")
    
    # By coach type
    print("Comfort tier by coach type:")
    coach_analysis = pd.crosstab(df['coach_type'], df['comfort_tier'], normalize='index') * 100
    print(coach_analysis.round(1))
    
    # By time period
    print("\nComfort tier by time period:")
    time_analysis = pd.crosstab(df['time_period'], df['comfort_tier'], normalize='index') * 100
    print(time_analysis.round(1))
    
    # By weekend
    print("\nComfort tier by weekend:")
    weekend_analysis = pd.crosstab(df['is_weekend'], df['comfort_tier'], normalize='index') * 100
    weekend_analysis.index = ['Weekday', 'Weekend']
    print(weekend_analysis.round(1))


def target_variable_pipeline():
    """Complete target variable definition pipeline."""
    print("=== STARTING TARGET VARIABLE DEFINITION ===")
    
    # Get engineered features
    df, encoders = feature_engineering_pipeline()
    
    # Define comfort tiers
    df, low_threshold, high_threshold = define_comfort_tiers_simple(df)
    
    # Encode target variable
    df, target_encoder = encode_target_variable(df)
    
    # Analyze target by features
    analyze_target_by_features(df)
    
    # Summary
    print(f"\n=== COMPLETE ===")
    print(f"Target variable: comfort_tier")
    print(f"Classes: {list(target_encoder.classes_)}")
    print(f"Encoding: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
    print(f"Thresholds: Quiet ≤ {low_threshold:.1f}%, Moderate {low_threshold:.1f}%-{high_threshold:.1f}%, Busy > {high_threshold:.1f}%")
    print(f"Dataset shape: {df.shape}")
    
    print(f"\n=== COMPLETE ===")
    return df, target_encoder, encoders, (low_threshold, high_threshold)


if __name__ == "__main__":
    df_with_target, target_encoder, feature_encoders, thresholds = target_variable_pipeline() 