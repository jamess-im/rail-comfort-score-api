#!/usr/bin/env python3
"""Test API with 5-tier predictions."""

import requests
import json
from datetime import datetime
import sys

def test_api_5tier(base_url="http://localhost:8080"):
    """Test the API with various scenarios to validate 5-tier predictions."""
    print("=== TESTING 5-TIER COMFORT PREDICTION API ===")
    print(f"Base URL: {base_url}")
    
    # Test health check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"✅ Health Check: {response.json()}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test prediction with various occupancy scenarios
    print("\n2. Testing 5-Tier Predictions...")
    test_cases = [
        {
            "name": "London to Leeds - Peak and Off-Peak",
            "request": {
                "from_tiploc": "KNGX",
                "to_tiploc": "LEEDS",
                "departure_datetimes": [
                    "2024-01-15T05:30:00",  # Early morning - likely Very Quiet
                    "2024-01-15T08:30:00",  # Morning peak - likely Busy/Very Busy
                    "2024-01-15T11:00:00",  # Mid-day - likely Moderate
                    "2024-01-15T17:30:00",  # Evening peak - likely Busy/Very Busy
                    "2024-01-15T22:00:00"   # Late evening - likely Quiet
                ]
            }
        },
        {
            "name": "Edinburgh to London - Weekend vs Weekday",
            "request": {
                "from_tiploc": "EDINBURGH",
                "to_tiploc": "KNGX",
                "departure_datetimes": [
                    "2024-01-15T09:00:00",  # Monday morning
                    "2024-01-19T09:00:00",  # Friday morning
                    "2024-01-20T09:00:00"   # Saturday morning
                ]
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 Test Case: {test_case['name']}")
        
        try:
            response = requests.post(
                f"{base_url}/predict_comfort_first_leg",
                json=test_case["request"],
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data["predictions"]
                
                print(f"✅ Route: {test_case['request']['from_tiploc']} → {test_case['request']['to_tiploc']}")
                print(f"📊 Predictions ({len(predictions)} time slots):")
                
                # Track tier distribution
                tier_counts = {"Very Quiet": 0, "Quiet": 0, "Moderate": 0, "Busy": 0, "Very Busy": 0}
                
                for pred in predictions:
                    dt = pred["departure_datetime"]
                    std = pred["standard_class"]
                    first = pred["first_class"]
                    
                    # Count tiers
                    tier_counts[std["comfort_tier"]] += 1
                    
                    print(f"  🕐 {dt}")
                    print(f"    Standard: {std['comfort_tier']} (Score: {std['numeric_score']}, Confidence: {std['confidence']:.2f})")
                    print(f"    First:    {first['comfort_tier']} (Score: {first['numeric_score']}, Confidence: {first['confidence']:.2f})")
                
                # Show tier distribution
                print(f"📈 Tier Distribution for this route:")
                for tier, count in tier_counts.items():
                    if count > 0:
                        pct = count / len(predictions) * 100
                        print(f"  {tier}: {count}/{len(predictions)} ({pct:.0f}%)")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    # Test stations endpoint
    print("\n3. Testing Stations Endpoint...")
    try:
        response = requests.get(f"{base_url}/stations")
        if response.status_code == 200:
            stations = response.json()
            print(f"✅ Stations endpoint: {len(stations)} stations available")
            # Show a few examples
            for station in list(stations.items())[:3]:
                print(f"  {station[0]}: {station[1]}")
        else:
            print(f"❌ Stations endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stations request failed: {e}")
    
    # Test TIPLOC endpoint
    print("\n4. Testing TIPLOCs Endpoint...")
    try:
        response = requests.get(f"{base_url}/tiplocs")
        if response.status_code == 200:
            tiplocs = response.json()
            print(f"✅ TIPLOCs endpoint: {len(tiplocs)} TIPLOCs available")
            # Show a few examples
            for tiploc in list(tiplocs.items())[:3]:
                print(f"  {tiploc[0]}: {tiploc[1]}")
        else:
            print(f"❌ TIPLOCs endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ TIPLOCs request failed: {e}")
    
    print("\n=== 5-TIER API TESTING COMPLETE ===")
    
    # Validation summary
    print("\n📝 VALIDATION SUMMARY:")
    print("✅ Expected: 5-tier classification (Very Quiet, Quiet, Moderate, Busy, Very Busy)")
    print("✅ Expected: Numeric scores 1-5 (1=Very Busy, 5=Very Quiet)")
    print("✅ Expected: Realistic confidence scores (not always 99%)")
    print("✅ Expected: Different predictions for peak vs off-peak times")
    print("✅ Expected: Standard and First class predictions")


if __name__ == "__main__":
    # Allow custom API URL from command line
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    test_api_5tier(base_url)