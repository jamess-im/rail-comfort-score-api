#!/usr/bin/env python3
"""
Test script for Train Comfort Predictor API
Test the deployed API endpoint
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Configuration
API_BASE_URL = "http://localhost:8001"  # Change for deployed version
TEST_REQUESTS = [
    {
        "from_tiploc": "PADTON",  # London Paddington
        "to_tiploc": "READING",  # Reading
        "departure_datetimes": ["2024-01-15T08:30:00"]
    },
    {
        "from_tiploc": "MNCRPIC",  # Manchester Piccadilly
        "to_tiploc": "LIVRLST",   # Liverpool Lime Street
        "departure_datetimes": ["2024-01-15T17:45:00"]
    },
    {
        "from_tiploc": "BHMNEWST", # Birmingham New Street
        "to_tiploc": "EUSTON",    # London Euston
        "departure_datetimes": ["2024-01-15T12:15:00"]
    },
    {
        "from_tiploc": "EDINBUR",  # Edinburgh Waverley
        "to_tiploc": "GLAS",      # Glasgow Central
        "departure_datetimes": ["2024-01-15T09:00:00"]
    }
]


def test_health_endpoint():
    """Test the health check endpoint."""
    print("=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False


def test_stations_endpoint():
    """Test the stations endpoint."""
    print("\n=== Testing Stations Endpoint ===")
    try:
        response = requests.get(f"{API_BASE_URL}/stations", timeout=10)
        if response.status_code == 200:
            stations = response.json()
            print(f"✅ Stations endpoint working: {len(stations)} stations available")
            print(f"Sample stations: {list(stations.keys())[:5]}")
            return True
        else:
            print(f"❌ Stations endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Stations endpoint error: {e}")
        return False


def test_prediction_endpoint():
    """Test the main prediction endpoint."""
    print("\n=== Testing Prediction Endpoint ===")
    
    success_count = 0
    total_tests = len(TEST_REQUESTS)
    
    for i, test_request in enumerate(TEST_REQUESTS, 1):
        print(f"\nTest {i}/{total_tests}: {test_request['from_tiploc']} → {test_request['to_tiploc']}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict_comfort_first_leg",
                json=test_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Prediction successful")
                # Get the first (and only) prediction from the batch response
                prediction = result['predictions'][0]
                print(f"   From: {prediction['from_station']} → To: {prediction['to_station']}")
                print(f"   Standard Class: {prediction['standard_class']['comfort_tier']} "
                      f"(confidence: {prediction['standard_class']['confidence']:.2f})")
                print(f"   First Class: {prediction['first_class']['comfort_tier']} "
                      f"(confidence: {prediction['first_class']['confidence']:.2f})")
                print(f"   Service: {prediction['service_info']}")
                success_count += 1
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Prediction error: {e}")
        
        # Small delay between requests
        time.sleep(1)
    
    print(f"\n=== Prediction Tests Summary ===")
    print(f"Successful: {success_count}/{total_tests}")
    return success_count == total_tests


def test_invalid_requests():
    """Test API error handling with invalid requests."""
    print("\n=== Testing Error Handling ===")
    
    invalid_requests = [
        # Missing required fields
        {
            "from_tiploc": "PADTON",
            "departure_datetimes": ["2024-01-15T08:30:00"]
        },
        # Invalid datetime format
        {
            "from_tiploc": "PADTON",
            "to_tiploc": "READING",
            "departure_datetimes": ["invalid-date"]
        },
        # Non-existent station
        {
            "from_tiploc": "INVALID",
            "to_tiploc": "READING",
            "departure_datetimes": ["2024-01-15T08:30:00"]
        }
    ]
    
    error_handling_works = True
    
    for i, invalid_request in enumerate(invalid_requests, 1):
        print(f"\nInvalid Test {i}: {invalid_request}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict_comfort_first_leg",
                json=invalid_request,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code >= 400:
                print(f"✅ Correctly rejected with status {response.status_code}")
            else:
                print(f"❌ Should have been rejected but got status {response.status_code}")
                error_handling_works = False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            error_handling_works = False
    
    return error_handling_works


def test_api_documentation():
    """Test that API documentation is accessible."""
    print("\n=== Testing API Documentation ===")
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API documentation failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API documentation error: {e}")
        return False


def test_batch_prediction():
    """Test batch comfort prediction endpoint with multiple datetimes."""
    
    print("\n=== Testing Batch Prediction ===")
    
    # Test data with multiple departure times
    test_request = {
        "from_tiploc": "EUSTON",  # London Euston
        "to_tiploc": "BRMNGM",   # Birmingham
        "departure_datetimes": [
            "2024-01-15T08:30:00",  # Morning rush
            "2024-01-15T12:00:00",  # Midday
            "2024-01-15T17:30:00"   # Evening rush
        ]
    }
    
    print("Testing batch comfort prediction...")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_comfort_first_leg",
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"Number of predictions returned: {len(result['predictions'])}")
            
            for i, prediction in enumerate(result['predictions']):
                print(f"\n--- Prediction {i+1} ---")
                print(f"From: {prediction['from_station']} ({prediction['from_tiploc']})")
                print(f"To: {prediction['to_station']} ({prediction['to_tiploc']})")
                print(f"Departure: {prediction['departure_datetime']}")
                print(f"Standard Class: {prediction['standard_class']['comfort_tier']} "
                      f"(confidence: {prediction['standard_class']['confidence']:.2f}, "
                      f"score: {prediction['standard_class']['numeric_score']})")
                print(f"First Class: {prediction['first_class']['comfort_tier']} "
                      f"(confidence: {prediction['first_class']['confidence']:.2f}, "
                      f"score: {prediction['first_class']['numeric_score']})")
                print(f"Service: {prediction['service_info']['headcode']} / {prediction['service_info']['rsid']}")
                print(f"Next Stop: {prediction['service_info']['next_stop']}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def test_single_datetime_batch():
    """Test with a single datetime (should still work)."""
    
    print("\n=== Testing Single Datetime Batch ===")
    
    test_request = {
        "from_tiploc": "EUSTON",
        "to_tiploc": "BRMNGM", 
        "departure_datetimes": ["2024-01-15T08:30:00"]  # Single datetime in array
    }
    
    print("Testing single datetime in batch format...")
    print(f"Request: {json.dumps(test_request, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_comfort_first_leg",
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single datetime batch test successful!")
            print(f"Returned {len(result['predictions'])} prediction(s)")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def test_health_check():
    """Test health check endpoint."""
    print("\n=== Testing Health Check ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"Status: {result['status']}")
            print(f"Model loaded: {result['model_loaded']}")
            print(f"Database connected: {result['database_connected']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check request failed: {e}")
        return False


def main():
    """Run all API tests."""
    print("🚂 Train Comfort Predictor API Test Suite")
    print(f"Testing API at: {API_BASE_URL}")
    print(f"Test time: {datetime.now().isoformat()}")
    
    # Wait for API to be ready
    print("\nWaiting for API to be ready...")
    max_retries = 30
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready")
                break
        except:
            pass
        
        if attempt < max_retries - 1:
            print(f"Attempt {attempt + 1}/{max_retries} - waiting...")
            time.sleep(2)
    else:
        print("❌ API not ready after maximum retries")
        return False
    
    # Run all tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Stations Endpoint", test_stations_endpoint),
        ("Prediction Endpoint", test_prediction_endpoint),
        ("Error Handling", test_invalid_requests),
        ("API Documentation", test_api_documentation),
        ("Batch Prediction", test_batch_prediction),
        ("Single Datetime Batch", test_single_datetime_batch),
        ("Health Check", test_health_check)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        results[test_name] = test_func()
    
    # Summary
    print(f"\n{'='*50}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the logs above.")
        return False


if __name__ == "__main__":
    import sys
    
    # Allow custom API URL via command line
    if len(sys.argv) > 1:
        API_BASE_URL = sys.argv[1]
    
    success = main()
    sys.exit(0 if success else 1) 