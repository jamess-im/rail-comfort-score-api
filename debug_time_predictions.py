#!/usr/bin/env python3
"""
Debug script to understand why early morning hours show unexpected predictions
"""

import requests
import json

def test_times():
    """Test various times to understand the pattern"""
    api_url = "http://localhost:8080/predict_comfort_first_leg"
    
    test_times = [
        "2025-06-14T01:36:00",  # Very early - should be Very Quiet
        "2025-06-14T05:47:00",  # Early morning - should be Very Quiet  
        "2025-06-14T06:15:00",  # Start of commute - should be busier than 1:36
        "2025-06-14T07:00:00",  # Peak commute - should be Very Busy
    ]
    
    for time_str in test_times:
        payload = {
            "from_tiploc": "KNGX", 
            "to_tiploc": "PBRO", 
            "departure_datetimes": [time_str]
        }
        
        response = requests.post(api_url, json=payload)
        data = response.json()
        
        prediction = data["predictions"][0]
        comfort = prediction["standard_class"]["comfort_tier"]
        confidence = prediction["standard_class"]["confidence"]
        service_info = prediction["service_info"]
        
        print(f"Time: {time_str}")
        print(f"  Prediction: {comfort} ({confidence:.3f})")
        print(f"  Time fallback: {service_info.get('time_fallback', 'N/A')}")
        print(f"  Route fallback: {service_info.get('route_fallback', 'N/A')}")
        print()

if __name__ == "__main__":
    test_times()