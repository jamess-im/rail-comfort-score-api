#!/usr/bin/env python3
"""
Feature Selection for Train Comfort Predictor
Select Features for Model Training (X variable)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from target_definition_simple_5tier import target_variable_pipeline_five_tier
import warnings
warnings.filterwarnings('ignore')


def select_model_features(df):
    """Select optimal features for XGBoost model training."""
    print("=== STARTING FEATURE SELECTION ===")
    
    # Define feature categories
    
    # Time features
    time_features = [
        'hour_of_day', 'day_of_week', 'month', 'is_weekend', 
        'time_period_encoded', 'is_peak_hour'
    ]
    
    # Location features (encoded station names and coordinates)
    location_features = [
        'stationName_from_encoded', 'stationName_to_encoded',
        'from_lat', 'from_lon', 'to_lat', 'to_lon',
        'route_distance_km', 'from_london', 'to_london',
        'from_major_city', 'to_major_city'
    ]
    
    # Coach/vehicle features
    vehicle_features = [
        'coach_type_encoded', 'vehicle_capacity'
    ]
    
    # Service identifier features (encoded)
    service_features = [
        'headcode_encoded', 'rsid_encoded'
    ]
    
    # Key contextual features - historical passenger counts and flows
    contextual_features = [
        'vehicle_pax_on_arrival_std_at_from',
        'vehicle_pax_on_arrival_first_at_from', 
        'totalUnitPassenger_at_leg_departure',
        'onUnitPassenger_at_from_station',
        'offUnitPassenger_at_from_station',
        'vehicle_pax_boarded_std_at_from',
        'vehicle_pax_boarded_first_at_from',
        'vehicle_pax_alighted_std_at_from',
        'vehicle_pax_alighted_first_at_from'
    ]
    
    # Engineered occupancy features
    occupancy_features = [
        'occupancy_percentage_std', 'occupancy_percentage_first',
        'total_occupancy', 'total_occupancy_percentage',
        'boarding_ratio', 'alighting_ratio', 'capacity_utilization'
    ]
    
    # Enhanced features
    enhanced_features = [
        'is_origin_major', 'is_destination_major', 'is_popular_route',
        'is_monday', 'is_friday', 'is_sunday'
    ]
    
    # Combine all feature categories
    selected_features = (time_features + location_features + vehicle_features +
                         service_features + contextual_features +
                         occupancy_features + enhanced_features)
    
    # Verify all features exist in the dataframe
    available_features = []
    missing_features = []
    
    for feature in selected_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"\n=== FEATURE SELECTION SUMMARY ===")
    print(f"Total features selected: {len(selected_features)}")
    print(f"Available features: {len(available_features)}")
    print(f"Missing features: {len(missing_features)}")
    
    if missing_features:
        print("Missing features:", missing_features)
    
    # Create feature matrix X and target vector y
    X = df[available_features].copy()
    y = df['comfort_tier_encoded'].copy()
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Display feature categories
    print("\n=== SELECTED FEATURE CATEGORIES ===")
    time_avail = [f for f in time_features if f in available_features]
    loc_avail = [f for f in location_features if f in available_features]
    veh_avail = [f for f in vehicle_features if f in available_features]
    svc_avail = [f for f in service_features if f in available_features]
    ctx_avail = [f for f in contextual_features if f in available_features]
    occ_avail = [f for f in occupancy_features if f in available_features]
    
    print(f"Time features ({len(time_avail)}): {time_avail}")
    print(f"Location features ({len(loc_avail)}): {loc_avail}")
    print(f"Vehicle features ({len(veh_avail)}): {veh_avail}")
    print(f"Service features ({len(svc_avail)}): {svc_avail}")
    print(f"Contextual features ({len(ctx_avail)}): {ctx_avail}")
    print(f"Occupancy features ({len(occ_avail)}): {occ_avail}")
    
    # Check for any missing values in selected features
    missing_values = X.isnull().sum()
    if missing_values.sum() > 0:
        print("\n=== MISSING VALUES WARNING ===")
        print(missing_values[missing_values > 0])
    else:
        print("\n✅ No missing values in selected features")
    
    # Check target distribution
    target_dist = y.value_counts().sort_index()
    print("\n=== TARGET DISTRIBUTION ===")
    for idx, count in target_dist.items():
        tier_name = ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'][idx]
        print(f"{tier_name} ({idx}): {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, available_features


def feature_selection_pipeline():
    """Complete feature selection pipeline."""
    print("=== FEATURE SELECTION PIPELINE ===")
    
    # Get processed data from previous tasks
    df, target_encoder, encoders, thresholds = target_variable_pipeline_five_tier()
    
    # Select features
    X, y, feature_list = select_model_features(df)
    
    # Save feature information for later use
    feature_info = {
        'feature_list': feature_list,
        'feature_count': len(feature_list),
        'target_encoder': target_encoder,
        'encoders': encoders,
        'X_shape': X.shape,
        'y_shape': y.shape
    }
    
    print(f"\n=== COMPLETE ===")
    print(f"Feature matrix ready: {X.shape}")
    print(f"Target vector ready: {y.shape}")
    print(f"Ready for XGBoost model training!")
    
    return X, y, feature_info, df


if __name__ == "__main__":
    X, y, feature_info, df = feature_selection_pipeline() 