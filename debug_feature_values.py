#!/usr/bin/env python3
"""
Debug what exact feature values are being constructed for problematic times
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
        return result if result else (51.5, -0.1)
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

def analyze_feature_construction():
    print("=== ANALYZING FEATURE CONSTRUCTION FOR PROBLEMATIC TIMES ===")
    
    # Load model artifacts
    feature_list = joblib.load('models/feature_list.joblib')
    feature_encoders = joblib.load('models/feature_encoders.joblib')
    
    # Test times
    test_times = [
        ("01:36", datetime(2025, 6, 14, 1, 36, 0)),   # Should be Very Quiet but predicts Quiet
        ("06:15", datetime(2025, 6, 14, 6, 15, 0)),   # Predicts Very Quiet (unexpected)
    ]
    
    for time_str, departure_dt in test_times:
        print(f"\n=== ANALYZING {time_str} ===")
        
        # Basic info
        from_tiploc = "KNGX"
        to_tiploc = "PBRO"
        from_station = get_station_name_from_tiploc(from_tiploc)
        to_station = get_station_name_from_tiploc(to_tiploc)
        coach_type = "Standard"
        
        # Time-aware defaults (use the UPDATED values from API)
        hour = departure_dt.hour
        is_weekend = departure_dt.weekday() >= 5
        
        if 0 <= hour <= 5:  # Night/early morning (very quiet) - should be lowest
            base_std, base_first = (2, 1) if not is_weekend else (1, 0)
        elif 6 <= hour <= 8:  # Early morning (quiet) - slightly busier than night
            base_std, base_first = (8, 2) if not is_weekend else (5, 1)
        else:
            base_std, base_first = (25, 6) if not is_weekend else (20, 5)
        
        historical_data = {
            "avg_vehicle_pax_on_arrival_std": base_std,
            "avg_vehicle_pax_on_arrival_first": base_first,
            "avg_total_unit_pax_on_arrival": base_std + base_first,
            "avg_unit_boarders_at_station": max(2, base_std // 4),
            "avg_unit_alighters_at_station": max(1, base_std // 6),
        }
        
        print(f"Time-aware defaults: std={base_std}, first={base_first}")
        print(f"Expected occupancy_percentage_std: {(base_std/89)*100:.1f}%")
        
        # Build key features
        from_coords = get_station_coordinates(from_tiploc)
        to_coords = get_station_coordinates(to_tiploc)
        from_lat, from_lon = from_coords
        to_lat, to_lon = to_coords
        
        hour_of_day = departure_dt.hour
        day_of_week = departure_dt.weekday()
        is_weekend_val = 1 if day_of_week >= 5 else 0
        is_peak_hour = 1 if (7 <= hour_of_day <= 9) or (17 <= hour_of_day <= 19) else 0
        
        # Time period encoding
        if 0 <= hour_of_day < 6:
            time_period = "Night"
        elif 6 <= hour_of_day < 9:
            time_period = "Early"
        else:
            time_period = "Morning"
        
        print(f"Key time features:")
        print(f"  hour_of_day: {hour_of_day}")
        print(f"  is_weekend: {is_weekend_val}")
        print(f"  is_peak_hour: {is_peak_hour}")
        print(f"  time_period: {time_period}")
        
        # Calculate occupancy features
        occupancy_percentage_std = (historical_data["avg_vehicle_pax_on_arrival_std"] / 89) * 100
        total_occupancy_percentage = ((historical_data["avg_vehicle_pax_on_arrival_std"] + historical_data["avg_vehicle_pax_on_arrival_first"]) / 89) * 100
        
        print(f"Key occupancy features:")
        print(f"  occupancy_percentage_std: {occupancy_percentage_std:.1f}%")
        print(f"  total_occupancy_percentage: {total_occupancy_percentage:.1f}%")
        
        # Encode time period
        time_period_encoded = 0
        if "time_period" in feature_encoders:
            try:
                time_period_encoded = feature_encoders["time_period"].transform([time_period])[0]
                print(f"  time_period_encoded: {time_period_encoded}")
            except ValueError:
                print(f"  time_period_encoded: 0 (unknown)")

if __name__ == "__main__":
    analyze_feature_construction()