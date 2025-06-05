#!/usr/bin/env python3
"""
Example script demonstrating the batch comfort prediction API
"""

import requests
import json

def example_batch_request():
    """Example of making a batch request with multiple departure times."""
    
    # API endpoint
    api_url = "http://localhost:8001/predict_comfort_first_leg"
    
    # Example request with multiple departure times
    request_data = {
        "from_tiploc": "EUSTON",     # London Euston
        "to_tiploc": "BRMNGM",       # Birmingham New Street
        "departure_datetimes": [
            "2024-01-15T07:30:00",   # Early morning
            "2024-01-15T08:30:00",   # Morning rush hour
            "2024-01-15T12:00:00",   # Midday
            "2024-01-15T17:30:00",   # Evening rush hour
            "2024-01-15T20:00:00"    # Evening
        ]
    }
    
    print("🚂 Train Comfort Prediction - Batch Request Example")
    print("=" * 55)
    print(f"Route: London Euston → Birmingham New Street")
    print(f"Number of departure times: {len(request_data['departure_datetimes'])}")
    print()
    
    try:
        response = requests.post(api_url, json=request_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            predictions = result["predictions"]
            
            print(f"✅ Successfully received {len(predictions)} predictions!")
            print()
            
            for i, prediction in enumerate(predictions, 1):
                print(f"--- Prediction {i} ---")
                print(f"Departure: {prediction['departure_datetime']}")
                print(f"From: {prediction['from_station']}")
                print(f"To: {prediction['to_station']}")
                print(f"Standard Class: {prediction['standard_class']['comfort_tier']} "
                      f"(confidence: {prediction['standard_class']['confidence']:.2f})")
                print(f"First Class: {prediction['first_class']['comfort_tier']} "
                      f"(confidence: {prediction['first_class']['confidence']:.2f})")
                print(f"Service: {prediction['service_info']['headcode']} - {prediction['service_info']['next_stop']}")
                print()
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        print("Make sure the API server is running on http://localhost:8001")

def example_single_request():
    """Example of making a single request (still using array format)."""
    
    api_url = "http://localhost:8001/predict_comfort_first_leg"
    
    # Single departure time in array format
    request_data = {
        "from_tiploc": "PADTON",     # London Paddington
        "to_tiploc": "READING",      # Reading
        "departure_datetimes": ["2024-01-15T08:30:00"]  # Single time in array
    }
    
    print("\n🚂 Single Request Example")
    print("=" * 30)
    print(f"Route: London Paddington → Reading")
    print(f"Departure: {request_data['departure_datetimes'][0]}")
    print()
    
    try:
        response = requests.post(api_url, json=request_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result["predictions"][0]  # Get the single prediction
            
            print("✅ Prediction successful!")
            print(f"Standard Class: {prediction['standard_class']['comfort_tier']} "
                  f"(confidence: {prediction['standard_class']['confidence']:.2f})")
            print(f"First Class: {prediction['first_class']['comfort_tier']} "
                  f"(confidence: {prediction['first_class']['confidence']:.2f})")
            print()
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    # Run batch example
    example_batch_request()
    
    # Run single example  
    example_single_request()
    
    print("💡 Tip: You can now send multiple departure times in a single request!")
    print("   This is useful for comparing comfort levels across different times of day.") 