#!/usr/bin/env python3
"""
Feature Engineering for Train Comfort Predictor
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def extract_time_features(df):
    """Extract time-based features from leg_departure_dt."""
    print("Extracting time features...")
    
    # Ensure datetime column is properly parsed
    df['leg_departure_dt'] = pd.to_datetime(df['leg_departure_dt'])
    
    # Extract time features
    df['hour_of_day'] = df['leg_departure_dt'].dt.hour
    df['day_of_week'] = df['leg_departure_dt'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['leg_departure_dt'].dt.month
    df['day_of_month'] = df['leg_departure_dt'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Sat=5, Sun=6
    
    # Create time periods
    df['time_period'] = pd.cut(df['hour_of_day'], 
                              bins=[0, 6, 9, 12, 17, 20, 24],
                              labels=['Night', 'Early', 'Morning', 'Afternoon', 'Evening', 'Late'],
                              include_lowest=True)
    
    # Peak hours (typical commuting times)
    df['is_peak_hour'] = ((df['hour_of_day'].between(7, 9)) | 
                          (df['hour_of_day'].between(17, 19))).astype(int)
    
    print(f"Added time features: hour_of_day, day_of_week, month, day_of_month, is_weekend, time_period, is_peak_hour")
    return df


def extract_location_features(df):
    """Parse string coordinates into numerical latitude and longitude."""
    print("Extracting location features...")
    
    # Handle empty coordinates by replacing with NaN, then filling with default values
    def safe_coordinate_parse(coord_series, index):
        """Safely parse coordinates, handling empty strings."""
        return coord_series.str.split(',').str[index].replace('', np.nan).astype(float)
    
    # Parse from station coordinates
    df['from_lat'] = safe_coordinate_parse(df['stationLocation_from'], 0)
    df['from_lon'] = safe_coordinate_parse(df['stationLocation_from'], 1)
    
    # Parse to station coordinates  
    df['to_lat'] = safe_coordinate_parse(df['stationLocation_to'], 0)
    df['to_lon'] = safe_coordinate_parse(df['stationLocation_to'], 1)
    
    # Fill missing coordinates with UK center coordinates (approximate)
    df['from_lat'].fillna(54.0, inplace=True)  # UK center latitude
    df['from_lon'].fillna(-2.0, inplace=True)  # UK center longitude
    df['to_lat'].fillna(54.0, inplace=True)
    df['to_lon'].fillna(-2.0, inplace=True)
    
    # Calculate distance between stations (Haversine formula approximation)
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth using Haversine formula."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    df['route_distance_km'] = haversine_distance(
        df['from_lat'], df['from_lon'], df['to_lat'], df['to_lon']
    )
    
    # Create location-based features
    # London stations (approximate area)
    london_stations = ['London Paddington', 'London Victoria', 'London Waterloo', 'London Kings Cross']
    df['from_london'] = df['stationName_from'].isin(london_stations).astype(int)
    df['to_london'] = df['stationName_to'].isin(london_stations).astype(int)
    
    # Major city indicators
    major_cities = ['Birmingham New Street', 'Manchester Piccadilly', 'Leeds', 
                   'Glasgow Central', 'Edinburgh Waverley']
    df['from_major_city'] = df['stationName_from'].isin(major_cities).astype(int)
    df['to_major_city'] = df['stationName_to'].isin(major_cities).astype(int)
    
    print(f"Added location features: from_lat, from_lon, to_lat, to_lon, route_distance_km, from_london, to_london, from_major_city, to_major_city")
    return df


def prepare_categorical_encoding(df):
    """Prepare categorical variables for encoding."""
    print("Preparing categorical encoding strategy...")
    
    # Define categorical columns that need encoding
    categorical_cols = ['coach_type', 'stationName_from', 'stationName_to', 
                       'headcode', 'rsid', 'time_period']
    
    # Initialize encoders dictionary
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[f'{col}_encoded'] = encoder.fit_transform(df[col])
            encoders[col] = encoder
            print(f"Encoded {col}: {len(encoder.classes_)} unique values")
    
    return df, encoders


def create_occupancy_features(df):
    """Create occupancy-based features."""
    print("Creating occupancy features...")
    
    # Calculate occupancy percentages
    df['occupancy_percentage_std'] = (df['vehicle_pax_on_arrival_std_at_from'] / 
                                     df['vehicle_capacity'] * 100)
    df['occupancy_percentage_first'] = (df['vehicle_pax_on_arrival_first_at_from'] / 
                                       df['vehicle_capacity'] * 100)
    
    # Total occupancy
    df['total_occupancy'] = (df['vehicle_pax_on_arrival_std_at_from'] + 
                            df['vehicle_pax_on_arrival_first_at_from'])
    df['total_occupancy_percentage'] = (df['total_occupancy'] / df['vehicle_capacity'] * 100)
    
    # Passenger flow ratios
    df['boarding_ratio'] = ((df['vehicle_pax_boarded_std_at_from'] + 
                            df['vehicle_pax_boarded_first_at_from']) / 
                           df['vehicle_capacity'] * 100)
    
    df['alighting_ratio'] = ((df['vehicle_pax_alighted_std_at_from'] + 
                             df['vehicle_pax_alighted_first_at_from']) / 
                            df['vehicle_capacity'] * 100)
    
    # Capacity utilization
    # df['capacity_utilization'] = (df['relevant_passengers_on_leg_departure'] / 
    #                              df['vehicle_capacity'] * 100)
    
    print(f"Added occupancy features: occupancy_percentage_std, occupancy_percentage_first, total_occupancy, total_occupancy_percentage, boarding_ratio, alighting_ratio, capacity_utilization")
    return df


def feature_engineering_pipeline():
    """Complete feature engineering pipeline"""
    print("=== STARTING FEATURE ENGINEERING ===")
    
    # Load data
    conn = duckdb.connect('duck')
    query = "SELECT * FROM train_journey_legs"
    df = conn.execute(query).fetchdf()
    
    print(f"Loaded dataset: {df.shape}")
    initial_columns = len(df.columns)
    
    # Apply feature engineering steps
    df = extract_time_features(df)
    df = extract_location_features(df)
    df = create_occupancy_features(df)
    df, encoders = prepare_categorical_encoding(df)
    
    # Summary of new features
    final_columns = len(df.columns)
    new_features = final_columns - initial_columns
    
    print(f"\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Original columns: {initial_columns}")
    print(f"Final columns: {final_columns}")
    print(f"New features created: {new_features}")
    
    # Display some sample engineered features
    feature_sample = df[['leg_departure_dt', 'hour_of_day', 'is_weekend', 'time_period',
                        'route_distance_km', 'from_london', 'occupancy_percentage_std',
                        ]].head()
    print(f"\nSample of engineered features:")
    print(feature_sample)
    
    conn.close()
    print(f"\n=== TASK COMPLETE ===")
    return df, encoders


if __name__ == "__main__":
    df_engineered, encoders = feature_engineering_pipeline() 