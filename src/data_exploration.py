#!/usr/bin/env python3
"""
Data Exploration and EDA for Train Comfort Predictor
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def explore_data():
    """Complete data exploration."""
    
    print("=== STARTING DATA EXPLORATION ===")
    
    # Load train_journey_legs into Pandas DataFrame
    print("\n--- Loading data ---")
    conn = duckdb.connect('duck')
    
    query = "SELECT * FROM train_journey_legs"
    df = conn.execute(query).fetchdf()
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rows: {len(df)}")
    
    # Exploratory Data Analysis
    print("\n--- Exploratory Data Analysis ---")
    
    # Basic info
    print("\n=== DATASET INFO ===")
    print(df.info())
    
    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())
    
    # Missing values
    print("\n=== MISSING VALUES ===")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percentage.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count', ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found!")
    
    # Statistical summary
    print("\n=== NUMERICAL COLUMNS SUMMARY ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numerical columns: {list(numerical_cols)}")
    print(df[numerical_cols].describe())
    
    # Categorical analysis
    print("\n=== CATEGORICAL COLUMNS ANALYSIS ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        if col not in ['leg_departure_dt', 'next_station_departure_dt']:
            print(f"\n{col}:")
            value_counts = df[col].value_counts()
            print(value_counts.head(10))  # Show top 10 values
    
    # DateTime analysis
    print("\n=== DATETIME ANALYSIS ===")
    df['leg_departure_dt'] = pd.to_datetime(df['leg_departure_dt'])
    df['next_station_departure_dt'] = pd.to_datetime(df['next_station_departure_dt'])
    
    print(f"Date range: {df['leg_departure_dt'].min()} to {df['leg_departure_dt'].max()}")
    print(f"Time span: {(df['leg_departure_dt'].max() - df['leg_departure_dt'].min()).days} days")
    
    # Target variable analysis
    print("\n=== TARGET VARIABLE ANALYSIS ===")
    target_col = 'relevant_passengers_on_leg_departure'
    print(f"Target variable: {target_col}")
    print(f"Min: {df[target_col].min()}")
    print(f"Max: {df[target_col].max()}")
    print(f"Mean: {df[target_col].mean():.2f}")
    print(f"Median: {df[target_col].median():.2f}")
    print(f"Std: {df[target_col].std():.2f}")
    
    # Outlier analysis
    print("\n=== OUTLIER ANALYSIS ===")
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    print(f"Outliers in target variable: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    
    # Station analysis
    print("\n=== STATION ANALYSIS ===")
    print("Top 10 departure stations:")
    print(df['stationName_from'].value_counts().head(10))
    print("\nTop 10 arrival stations:")
    print(df['stationName_to'].value_counts().head(10))
    
    # Coach type analysis
    print("\n=== COACH TYPE ANALYSIS ===")
    coach_counts = df['coach_type'].value_counts()
    print(coach_counts)
    
    # Time patterns
    print("\n=== TIME PATTERNS ===")
    df['hour'] = df['leg_departure_dt'].dt.hour
    df['day_of_week'] = df['leg_departure_dt'].dt.day_name()
    print("Departures by hour of day (top 10):")
    print(df['hour'].value_counts().sort_index().head(10))
    print("\nDepartures by day of week:")
    print(df['day_of_week'].value_counts())
    
    # Basic correlation analysis
    print("\n=== CORRELATION WITH TARGET ===")
    correlations = df[numerical_cols].corr()[target_col].sort_values(ascending=False)
    print("Top correlations with target variable:")
    print(correlations[correlations.index != target_col])
    
    # Data quality summary
    print("\n=== DATA QUALITY SUMMARY ===")
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Target variable range: {df[target_col].min():.1f} - {df[target_col].max():.1f}")
    print(f"Outliers in target: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    conn.close()
    print("\n=== COMPLETE ===")
    print("Database connection closed.")
    print("Ready to proceed with feature engineering")
    
    return df


if __name__ == "__main__":
    df = explore_data() 