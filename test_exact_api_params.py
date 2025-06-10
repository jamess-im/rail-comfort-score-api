#!/usr/bin/env python3
"""
Test the model with exact same parameters as API call
"""

import sys
sys.path.append('api')
import joblib
import numpy as np
import sqlite3
from datetime import datetime

def get_station_coordinates(tiploc):
    """Get station coordinates from database."""
    conn = sqlite3.connect('api/train_comfort_api_lookups.sqlite')
    try:
        query = "SELECT latitude, longitude FROM stations WHERE tiploc = ?"
        result = conn.execute(query, (tiploc,)).fetchone()
        if result:
            return result
        else:
            # Default coordinates if not found
            return (51.5, -0.1)  # London default
    finally:
        conn.close()

def get_station_name_from_tiploc(tiploc):
    """Get station name from TIPLOC."""
    conn = sqlite3.connect('api/train_comfort_api_lookups.sqlite')
    try:
        query = "SELECT station_name FROM stations WHERE tiploc = ?"
        result = conn.execute(query, (tiploc,)).fetchone()
        return result[0] if result else f"Unknown_{tiploc}"
    finally:
        conn.close()

def test_api_exact_features():
    print("=== TESTING MODEL WITH EXACT API PARAMETERS ===")
    
    # Load model artifacts
    model = joblib.load('models/xgboost_comfort_classifier.joblib')
    feature_list = joblib.load('models/feature_list.joblib')
    feature_encoders = joblib.load('models/feature_encoders.joblib')
    
    # Exact same inputs as API call
    from_tiploc = "KNGX"
    to_tiploc = "YORK"
    departure_dt = datetime(2025, 6, 14, 23, 30, 0)  # 11:30 PM
    coach_type = "Standard"
    
    print(f"Testing: {from_tiploc} -> {to_tiploc} at {departure_dt} ({coach_type})")
    
    # Get station info same as API
    from_station = get_station_name_from_tiploc(from_tiploc)
    to_station = get_station_name_from_tiploc(to_tiploc)
    from_coords = get_station_coordinates(from_tiploc)
    to_coords = get_station_coordinates(to_tiploc)
    
    print(f"Stations: {from_station} -> {to_station}")
    print(f"Coordinates: {from_coords} -> {to_coords}")
    
    # Mock service and historical data (use time-aware defaults like API now does)
    service_info = {
        "headcode": "1N33",
        "rsid": "GR470000"
    }
    
    # Calculate time-aware defaults like the API does
    hour = departure_dt.hour  # 23
    is_weekend = departure_dt.weekday() >= 5  # Saturday
    
    if 0 <= hour <= 5:  # Night/early morning (very quiet)
        base_std, base_first = (5, 2) if not is_weekend else (3, 1)
    elif 6 <= hour <= 8:  # Early morning (quiet)
        base_std, base_first = (15, 5) if not is_weekend else (10, 3)
    elif 9 <= hour <= 16:  # Daytime (moderate)
        base_std, base_first = (35, 8) if not is_weekend else (25, 6)
    elif 17 <= hour <= 19:  # Evening peak (busy)
        base_std, base_first = (60, 12) if not is_weekend else (45, 10)
    elif 20 <= hour <= 22:  # Evening (moderate)
        base_std, base_first = (25, 6) if not is_weekend else (20, 5)
    else:  # Late night (quiet) - this should apply to 23:30
        base_std, base_first = (8, 3) if not is_weekend else (5, 2)
    
    historical_data = {
        "avg_vehicle_pax_on_arrival_std": base_std,
        "avg_vehicle_pax_on_arrival_first": base_first,
        "avg_total_unit_pax_on_arrival": base_std + base_first,
        "avg_unit_boarders_at_station": max(2, base_std // 4),
        "avg_unit_alighters_at_station": max(1, base_std // 6),
    }
    
    print(f"Time-aware defaults for hour {hour}, weekend={is_weekend}:")
    print(f"  std passengers: {base_std}, first passengers: {base_first}")
    
    # Construct features exactly like API does
    from_lat, from_lon = from_coords
    to_lat, to_lon = to_coords
    route_distance = ((to_lat - from_lat) ** 2 + (to_lon - from_lon) ** 2) ** 0.5 * 111
    
    # Time features
    hour_of_day = departure_dt.hour  # Should be 1
    day_of_week = departure_dt.weekday()  # Saturday = 5
    month = departure_dt.month  # June = 6
    is_weekend = 1 if day_of_week >= 5 else 0  # Should be 1 (Saturday)
    is_peak_hour = 1 if (7 <= hour_of_day <= 9) or (17 <= hour_of_day <= 19) else 0  # Should be 0
    
    # Time period
    if 0 <= hour_of_day < 6:
        time_period = "Night"
    elif 6 <= hour_of_day < 9:
        time_period = "Early"
    elif 9 <= hour_of_day < 12:
        time_period = "Morning"
    elif 12 <= hour_of_day < 17:
        time_period = "Afternoon"
    elif 17 <= hour_of_day < 20:
        time_period = "Evening"
    else:
        time_period = "Late"
    
    print(f"\nTime features:")
    print(f"  hour_of_day: {hour_of_day}")
    print(f"  day_of_week: {day_of_week}")
    print(f"  is_weekend: {is_weekend}")
    print(f"  is_peak_hour: {is_peak_hour}")
    print(f"  time_period: {time_period}")
    
    # Build features dictionary exactly like API
    features = {
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "from_lat": from_lat,
        "from_lon": from_lon,
        "to_lat": to_lat,
        "to_lon": to_lon,
        "route_distance_km": route_distance,
        "from_london": 1 if "London" in from_station else 0,
        "to_london": 1 if "London" in to_station else 0,
        "from_major_city": 1 if any(city in from_station for city in ["Birmingham", "Manchester", "Leeds", "Glasgow", "Edinburgh"]) else 0,
        "to_major_city": 1 if any(city in to_station for city in ["Birmingham", "Manchester", "Leeds", "Glasgow", "Edinburgh"]) else 0,
        "vehicle_capacity": 89 if coach_type == "Standard" else 60,
        "vehicle_pax_on_arrival_std_at_from": historical_data["avg_vehicle_pax_on_arrival_std"],
        "vehicle_pax_on_arrival_first_at_from": historical_data["avg_vehicle_pax_on_arrival_first"],
        "totalUnitPassenger_at_leg_departure": historical_data["avg_total_unit_pax_on_arrival"],
        "onUnitPassenger_at_from_station": historical_data["avg_unit_boarders_at_station"],
        "offUnitPassenger_at_from_station": historical_data["avg_unit_alighters_at_station"],
        "vehicle_pax_boarded_std_at_from": historical_data["avg_unit_boarders_at_station"] * 0.7,
        "vehicle_pax_boarded_first_at_from": historical_data["avg_unit_boarders_at_station"] * 0.3,
        "vehicle_pax_alighted_std_at_from": historical_data["avg_unit_alighters_at_station"] * 0.7,
        "vehicle_pax_alighted_first_at_from": historical_data["avg_unit_alighters_at_station"] * 0.3,
        "occupancy_percentage_std": (historical_data["avg_vehicle_pax_on_arrival_std"] / 89) * 100,
        "occupancy_percentage_first": (historical_data["avg_vehicle_pax_on_arrival_first"] / 60) * 100,
        "total_occupancy": historical_data["avg_vehicle_pax_on_arrival_std"] + historical_data["avg_vehicle_pax_on_arrival_first"],
        "total_occupancy_percentage": ((historical_data["avg_vehicle_pax_on_arrival_std"] + historical_data["avg_vehicle_pax_on_arrival_first"]) / 89) * 100,
        "boarding_ratio": (historical_data["avg_unit_boarders_at_station"] / 89) * 100,
        "alighting_ratio": (historical_data["avg_unit_alighters_at_station"] / 89) * 100,
        "is_origin_major": 1 if from_station in ['London Kings Cross', 'Edinburgh Waverley', 'Leeds', 'Newcastle', 'Aberdeen'] else 0,
        "is_destination_major": 1 if to_station in ['London Kings Cross', 'Edinburgh Waverley', 'Leeds', 'Newcastle', 'Aberdeen'] else 0,
        "is_popular_route": 1 if (from_station, to_station) in [('London Kings Cross', 'Leeds'), ('London Kings Cross', 'Edinburgh Waverley'), ('London Kings Cross', 'York'), ('Leeds', 'London Kings Cross'), ('Edinburgh Waverley', 'London Kings Cross')] else 0,
        "is_monday": 1 if day_of_week == 0 else 0,
        "is_friday": 1 if day_of_week == 4 else 0,
        "is_sunday": 1 if day_of_week == 6 else 0,
    }
    
    # Add encoded features
    features["stationName_from_encoded"] = feature_encoders["stationName_from"].transform([from_station])[0] if "stationName_from" in feature_encoders else 0
    features["stationName_to_encoded"] = feature_encoders["stationName_to"].transform([to_station])[0] if "stationName_to" in feature_encoders else 0
    features["headcode_encoded"] = feature_encoders["headcode"].transform([service_info["headcode"]])[0] if "headcode" in feature_encoders else 0
    features["rsid_encoded"] = feature_encoders["rsid"].transform([service_info["rsid"]])[0] if "rsid" in feature_encoders else 0
    features["coach_type_encoded"] = feature_encoders["coach_type"].transform([coach_type])[0] if "coach_type" in feature_encoders else 0
    features["time_period_encoded"] = feature_encoders["time_period"].transform([time_period])[0] if "time_period" in feature_encoders else 0
    
    # Create feature vector in model order
    feature_vector = np.zeros(len(feature_list))
    for i, feature_name in enumerate(feature_list):
        if feature_name in features:
            feature_vector[i] = features[feature_name]
        else:
            if "encoded" in feature_name:
                feature_vector[i] = 0
            elif "percentage" in feature_name or "ratio" in feature_name:
                feature_vector[i] = 10.0
            else:
                feature_vector[i] = 0
    
    print(f"\nKey feature values:")
    for feat in ["hour_of_day", "is_weekend", "is_peak_hour", "occupancy_percentage_std", "total_occupancy_percentage"]:
        if feat in feature_list:
            idx = feature_list.index(feat)
            print(f"  {feat}: {feature_vector[idx]}")
    
    # Make prediction
    proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
    predicted_class = np.argmax(proba)
    class_names = ["Very Busy", "Busy", "Moderate", "Quiet", "Very Quiet"]
    
    print(f"\nPrediction probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, proba)):
        print(f"  {name}: {prob:.4f}")
    
    print(f"\nPredicted: {class_names[predicted_class]} (confidence: {np.max(proba):.4f})")
    
    if predicted_class in [0, 1]:  # Very Busy or Busy
        print("❌ PROBLEM: Model predicting Busy/Very Busy for 1:30 AM!")
        print("This suggests the feature values are wrong or the model has an issue.")
    else:
        print("✅ Good: Model correctly predicts quiet times for 1:30 AM")

if __name__ == "__main__":
    test_api_exact_features()