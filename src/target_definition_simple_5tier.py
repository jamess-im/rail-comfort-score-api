#!/usr/bin/env python3
"""
Target Variable Definition for Train Comfort Predictor (5-Tier System)
Minimal changes approach - reuses existing data structure
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import feature_engineering_pipeline


def define_comfort_tiers_five_level(df):
    """Define 5 comfort tiers based on occupancy percentage."""
    print("=== DEFINING 5-TIER COMFORT SYSTEM ===")
    
    # Calculate occupancy percentage (same as before)
    df['occupancy_percentage'] = (df['relevant_passengers_on_leg_departure'] / 
                                 df['vehicle_capacity'] * 100)
    
    print("Occupancy percentage statistics:")
    print(df['occupancy_percentage'].describe())
    
    # Use fixed thresholds (based on Phase 1 analysis)
    thresholds = {
        'very_quiet': 25,    # 0-25%
        'quiet': 50,         # 25-50%
        'moderate': 70,      # 50-70%
        'busy': 85,          # 70-85%
        'very_busy': 100     # 85%+
    }
    
    # Create comfort tiers
    def assign_comfort_tier(occupancy_pct):
        if occupancy_pct <= thresholds['very_quiet']:
            return 'Very Quiet'
        elif occupancy_pct <= thresholds['quiet']:
            return 'Quiet'
        elif occupancy_pct <= thresholds['moderate']:
            return 'Moderate'
        elif occupancy_pct <= thresholds['busy']:
            return 'Busy'
        else:
            return 'Very Busy'
    
    df['comfort_tier'] = df['occupancy_percentage'].apply(assign_comfort_tier)
    
    # Show distribution
    tier_counts = df['comfort_tier'].value_counts()
    tier_order = ['Very Quiet', 'Quiet', 'Moderate', 'Busy', 'Very Busy']
    
    print(f"\nComfort tier distribution:")
    for tier in tier_order:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            pct = count / len(df) * 100
            print(f"{tier}: {count} ({pct:.1f}%)")
    
    return df, thresholds


def encode_target_variable_five_tier(df):
    """Encode 5-tier comfort variable for machine learning."""
    print(f"\n=== ENCODING 5-TIER TARGET VARIABLE ===")
    
    # Create custom encoding (0 = Very Busy, 4 = Very Quiet)
    comfort_mapping = {
        'Very Busy': 0,
        'Busy': 1,
        'Moderate': 2,
        'Quiet': 3,
        'Very Quiet': 4
    }
    
    df['comfort_tier_encoded'] = df['comfort_tier'].map(comfort_mapping)
    
    # Create label encoder for consistency
    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'])
    
    return df, target_encoder


def target_variable_pipeline_five_tier():
    """Complete target variable definition pipeline for 5-tier system."""
    print("=== STARTING 5-TIER TARGET VARIABLE DEFINITION ===")
    
    # Get engineered features (using existing pipeline)
    df, encoders = feature_engineering_pipeline()
    
    # Define 5-tier comfort levels
    df, thresholds = define_comfort_tiers_five_level(df)
    
    # Encode target variable
    df, target_encoder = encode_target_variable_five_tier(df)
    
    return df, target_encoder, encoders, thresholds


if __name__ == "__main__":
    # Test the pipeline
    df, target_encoder, encoders, thresholds = target_variable_pipeline_five_tier()
    
    print("\n=== PIPELINE COMPLETE ===")
    print(f"Total records: {len(df)}")
    print(f"Features: {df.shape[1]}")
    print(f"Target classes: {list(target_encoder.classes_)}")
    print(f"Thresholds: {thresholds}")
    
    # Show sample of encoded values
    print("\nSample target encoding:")
    sample_df = df[['occupancy_percentage', 'comfort_tier', 'comfort_tier_encoded']].head(10)
    print(sample_df)