#!/usr/bin/env python3
"""
Target Variable Definition for Train Comfort Predictor
Define Target Variable (comfort_tier) based on relevant_passengers_on_leg_departure
"""

import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# First run feature engineering to get the enhanced dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import feature_engineering_pipeline


def analyze_occupancy_distribution(df):
    """Analyze the distribution of occupancy to determine comfort tier thresholds."""
    print("=== ANALYZING OCCUPANCY DISTRIBUTION ===")
    
    # Calculate occupancy percentage
    df['occupancy_percentage'] = (df['relevant_passengers_on_leg_departure'] / 
                                 df['vehicle_capacity'] * 100)
    
    print(f"Occupancy percentage statistics:")
    print(df['occupancy_percentage'].describe())
    
    # Plot distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['occupancy_percentage'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Occupancy Percentage')
    plt.xlabel('Occupancy Percentage (%)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['occupancy_percentage'])
    plt.title('Occupancy Percentage Box Plot')
    plt.ylabel('Occupancy Percentage (%)')
    
    plt.tight_layout()
    plt.savefig('models/occupancy_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate quantiles for tier definition
    quantiles = df['occupancy_percentage'].quantile([0.33, 0.67]).values
    print(f"\nQuantile-based thresholds:")
    print(f"33rd percentile: {quantiles[0]:.1f}%")
    print(f"67th percentile: {quantiles[1]:.1f}%")
    
    return df, quantiles


def define_comfort_tiers(df, method='quantile'):
    """Define comfort tiers based on occupancy percentage."""
    print(f"\n=== DEFINING COMFORT TIERS (Method: {method}) ===")
    
    df['occupancy_percentage'] = (df['relevant_passengers_on_leg_departure'] / 
                                 df['vehicle_capacity'] * 100)
    
    if method == 'quantile':
        # Use quantile-based approach (balanced classes)
        quantiles = df['occupancy_percentage'].quantile([0.33, 0.67]).values
        low_threshold = quantiles[0]
        high_threshold = quantiles[1]
        
    elif method == 'fixed':
        # Use fixed thresholds based on common sense
        low_threshold = 40  # Under 40% = Quiet
        high_threshold = 70  # Over 70% = Busy
        
    elif method == 'adaptive':
        # Use mean and standard deviation
        mean_occ = df['occupancy_percentage'].mean()
        std_occ = df['occupancy_percentage'].std()
        low_threshold = mean_occ - 0.5 * std_occ
        high_threshold = mean_occ + 0.5 * std_occ
    
    # Create comfort tiers
    def assign_comfort_tier(occupancy_pct):
        if occupancy_pct <= low_threshold:
            return 'Quiet'
        elif occupancy_pct <= high_threshold:
            return 'Moderate'
        else:
            return 'Busy'
    
    df['comfort_tier'] = df['occupancy_percentage'].apply(assign_comfort_tier)
    
    print(f"Thresholds used:")
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
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Coach type
    coach_counts = pd.crosstab(df['coach_type'], df['comfort_tier'])
    coach_counts.plot(kind='bar', ax=axes[0,0], rot=45)
    axes[0,0].set_title('Comfort Tier by Coach Type')
    axes[0,0].legend(title='Comfort Tier')
    
    # Time period
    time_counts = pd.crosstab(df['time_period'], df['comfort_tier'])
    time_counts.plot(kind='bar', ax=axes[0,1], rot=45)
    axes[0,1].set_title('Comfort Tier by Time Period')
    axes[0,1].legend(title='Comfort Tier')
    
    # Hour of day
    hour_comfort = df.groupby(['hour_of_day', 'comfort_tier']).size().unstack(fill_value=0)
    hour_comfort.plot(kind='bar', ax=axes[1,0], width=0.8)
    axes[1,0].set_title('Comfort Tier by Hour of Day')
    axes[1,0].legend(title='Comfort Tier')
    axes[1,0].tick_params(axis='x', rotation=0)
    
    # Vehicle capacity
    df.boxplot(column='vehicle_capacity', by='comfort_tier', ax=axes[1,1])
    axes[1,1].set_title('Vehicle Capacity by Comfort Tier')
    
    plt.suptitle('')  # Remove auto-generated title
    plt.tight_layout()
    plt.savefig('models/comfort_tier_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def target_variable_pipeline():
    """Complete target variable definition pipeline."""
    print("=== STARTING TARGET VARIABLE DEFINITION ===")
    
    # Get engineered features
    df, encoders = feature_engineering_pipeline()
    
    # Analyze occupancy distribution
    df, quantiles = analyze_occupancy_distribution(df)
    
    # Define comfort tiers using quantile method (balanced classes)
    df, low_threshold, high_threshold = define_comfort_tiers(df, method='quantile')
    
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