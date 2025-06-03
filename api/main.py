#!/usr/bin/env python3
"""
FastAPI Application for Train Comfort Predictor
Tasks 7.1-7.4: API Development
"""

import os
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Train Comfort Predictor API",
    description="Predict train comfort levels (Quiet, Moderate, Busy) for UK rail journeys",
    version="1.0.0",
)

# Global variables for model and database
model = None
scaler = None
target_encoder = None
feature_encoders = None
feature_list = None
db_path = "train_comfort_api_lookups.sqlite"


class PredictionRequest(BaseModel):
    """Request model for comfort prediction."""

    from_station: str
    to_station: str
    departure_datetime: str  # ISO format: "2024-01-15T08:30:00"


class ComfortPrediction(BaseModel):
    """Response model for comfort prediction."""

    from_station: str
    to_station: str
    departure_datetime: str
    standard_class: Dict[str, Any]  # {"comfort_tier": "Moderate", "confidence": 0.85}
    first_class: Dict[str, Any]
    service_info: Dict[str, str]  # headcode, rsid, etc.


@app.on_event("startup")
async def load_model_and_database():
    """Task 7.2: Load model and supporting files at API startup."""
    global model, scaler, target_encoder, feature_encoders, feature_list

    print("=== LOADING MODEL AND ARTIFACTS ===")

    try:
        # Load trained model
        model_path = "../models/xgboost_comfort_classifier.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load target encoder
        target_encoder_path = "../models/target_encoder.joblib"
        if os.path.exists(target_encoder_path):
            target_encoder = joblib.load(target_encoder_path)
            print("✅ Target encoder loaded")

        # Load feature encoders
        encoders_path = "../models/feature_encoders.joblib"
        if os.path.exists(encoders_path):
            feature_encoders = joblib.load(encoders_path)
            print("✅ Feature encoders loaded")

        # Load feature list
        feature_list_path = "../models/feature_list.joblib"
        if os.path.exists(feature_list_path):
            feature_list = joblib.load(feature_list_path)
            print(f"✅ Feature list loaded: {len(feature_list)} features")

        # Load scaler if it exists
        scaler_path = "../models/feature_scaler.joblib"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded")
        else:
            print("ℹ️ No scaler found (not required)")

        # Check database connection
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            conn.close()
            print(f"✅ Database connected: {len(tables)} tables available")
        else:
            print(f"❌ Database not found: {db_path}")
            raise FileNotFoundError(f"Database not found: {db_path}")

        print("=== STARTUP COMPLETE ===")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


def parse_datetime(datetime_str: str) -> datetime:
    """Parse ISO datetime string."""
    try:
        return datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid datetime format: {datetime_str}"
        )


def get_station_coordinates(station_name: str) -> Optional[tuple]:
    """Get station coordinates from database."""
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT latitude, longitude FROM stations WHERE station_name = ?"
        result = conn.execute(query, (station_name,)).fetchone()
        return result if result else None
    finally:
        conn.close()


def identify_service_and_next_stop(from_station: str, to_station: str, departure_dt: datetime) -> Dict:
    """Identify relevant service and actual next stop from bundled SQLite."""
    conn = sqlite3.connect(db_path)
    try:
        # Extract time components
        hour = departure_dt.hour
        day_of_week = departure_dt.weekday()  # 0=Monday, 6=Sunday

        # First priority: Find direct service from origin to destination at exact time
        direct_query = """
        SELECT headcode, rsid, stationName_to, frequency
        FROM service_routes_summary_mvp 
        WHERE stationName_from = ? 
        AND stationName_to = ?
        AND hour = ?
        AND day_of_week = ?
        ORDER BY frequency DESC 
        LIMIT 1
        """

        result = conn.execute(direct_query, (from_station, to_station, hour, day_of_week)).fetchone()

        if result:
            return {
                "headcode": result[0],
                "rsid": result[1],
                "next_stop": result[2],
                "frequency": result[3],
            }
        
        # Second priority: Find direct service from origin to destination at similar time
        direct_similar_time_query = """
        SELECT headcode, rsid, stationName_to, frequency
        FROM service_routes_summary_mvp 
        WHERE stationName_from = ? 
        AND stationName_to = ?
        ORDER BY ABS(hour - ?) ASC, frequency DESC
        LIMIT 1
        """
        result = conn.execute(direct_similar_time_query, (from_station, to_station, hour)).fetchone()

        if result:
            return {
                "headcode": result[0],
                "rsid": result[1],
                "next_stop": result[2],
                "frequency": result[3],
                "time_fallback": True,
            }

        # Third priority: Find any service from origin at exact time (original logic)
        origin_query = """
        SELECT headcode, rsid, stationName_to, frequency
        FROM service_routes_summary_mvp 
        WHERE stationName_from = ? 
        AND hour = ?
        AND day_of_week = ?
        ORDER BY frequency DESC 
        LIMIT 1
        """

        result = conn.execute(origin_query, (from_station, hour, day_of_week)).fetchone()

        if result:
            return {
                "headcode": result[0],
                "rsid": result[1],
                "next_stop": result[2],
                "frequency": result[3],
                "route_fallback": True,
            }
        
        # Final fallback: find any service from this station at similar time
        fallback_query = """
        SELECT headcode, rsid, stationName_to, frequency
        FROM service_routes_summary_mvp 
        WHERE stationName_from = ? 
        ORDER BY ABS(hour - ?) ASC, frequency DESC
        LIMIT 1
        """
        result = conn.execute(fallback_query, (from_station, hour)).fetchone()

        if result:
            return {
                "headcode": result[0],
                "rsid": result[1],
                "next_stop": result[2],
                "frequency": result[3],
                "fallback": True,
            }
        else:
            raise HTTPException(
                status_code=404, detail=f"No services found from {from_station}"
            )

    finally:
        conn.close()


def get_historical_averages(
    headcode: str, rsid: str, station: str, departure_dt: datetime
) -> Dict:
    """Fetch historical averages for arrival state from bundled SQLite."""
    conn = sqlite3.connect(db_path)
    try:
        # Determine time bucket
        hour = departure_dt.hour
        if 6 <= hour <= 9:
            time_bucket = "morning_peak"
        elif 10 <= hour <= 15:
            time_bucket = "midday"
        elif 16 <= hour <= 19:
            time_bucket = "evening_peak"
        elif 20 <= hour <= 23:
            time_bucket = "evening"
        else:
            time_bucket = "night_early"

        day_of_week = departure_dt.weekday()

        query = """
        SELECT avg_vehicle_pax_on_arrival_std, avg_vehicle_pax_on_arrival_first,
               avg_total_unit_pax_on_arrival, avg_unit_boarders_at_station,
               avg_unit_alighters_at_station
        FROM historical_averages 
        WHERE headcode = ? AND rsid = ? AND station_of_arrival = ?
        AND day_of_week_bucket = ? AND time_of_day_bucket = ?
        """

        result = conn.execute(
            query, (headcode, rsid, station, day_of_week, time_bucket)
        ).fetchone()

        if result:
            return {
                "avg_vehicle_pax_on_arrival_std": result[0] or 0,
                "avg_vehicle_pax_on_arrival_first": result[1] or 0,
                "avg_total_unit_pax_on_arrival": result[2] or 0,
                "avg_unit_boarders_at_station": result[3] or 0,
                "avg_unit_alighters_at_station": result[4] or 0,
            }
        else:
            # Return default values if no historical data
            return {
                "avg_vehicle_pax_on_arrival_std": 50,
                "avg_vehicle_pax_on_arrival_first": 10,
                "avg_total_unit_pax_on_arrival": 60,
                "avg_unit_boarders_at_station": 15,
                "avg_unit_alighters_at_station": 10,
            }

    finally:
        conn.close()


def construct_feature_vector(
    from_station: str,
    to_station: str,
    departure_dt: datetime,
    service_info: Dict,
    historical_data: Dict,
    coach_type: str,
) -> pd.DataFrame:
    """Construct feature vector for XGBoost model."""

    # Get station coordinates
    from_coords = get_station_coordinates(from_station)
    to_coords = get_station_coordinates(to_station)

    if not from_coords or not to_coords:
        raise HTTPException(status_code=404, detail="Station coordinates not found")

    # Calculate route distance (simple Euclidean for now)
    from_lat, from_lon = from_coords
    to_lat, to_lon = to_coords
    route_distance = (
        (to_lat - from_lat) ** 2 + (to_lon - from_lon) ** 2
    ) ** 0.5 * 111  # Rough km conversion

    # Extract time features
    hour_of_day = departure_dt.hour
    day_of_week = departure_dt.weekday()
    month = departure_dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if (7 <= hour_of_day <= 9) or (17 <= hour_of_day <= 19) else 0

    # Time period encoding
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

    # Create feature dictionary
    features = {
        # Time features
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        # Location features
        "from_lat": from_lat,
        "from_lon": from_lon,
        "to_lat": to_lat,
        "to_lon": to_lon,
        "route_distance_km": route_distance,
        "from_london": 1 if "London" in from_station else 0,
        "to_london": 1 if "London" in to_station else 0,
        "from_major_city": (
            1
            if any(
                city in from_station
                for city in [
                    "Birmingham",
                    "Manchester",
                    "Leeds",
                    "Glasgow",
                    "Edinburgh",
                ]
            )
            else 0
        ),
        "to_major_city": (
            1
            if any(
                city in to_station
                for city in [
                    "Birmingham",
                    "Manchester",
                    "Leeds",
                    "Glasgow",
                    "Edinburgh",
                ]
            )
            else 0
        ),
        # Vehicle features
        "vehicle_capacity": (
            89
            if coach_type == "Standard"
            else (60 if coach_type == "First Class" else 115)
        ),
        # Contextual features from historical data
        "vehicle_pax_on_arrival_std_at_from": historical_data[
            "avg_vehicle_pax_on_arrival_std"
        ],
        "vehicle_pax_on_arrival_first_at_from": historical_data[
            "avg_vehicle_pax_on_arrival_first"
        ],
        "totalUnitPassenger_at_leg_departure": historical_data[
            "avg_total_unit_pax_on_arrival"
        ],
        "onUnitPassenger_at_from_station": historical_data[
            "avg_unit_boarders_at_station"
        ],
        "offUnitPassenger_at_from_station": historical_data[
            "avg_unit_alighters_at_station"
        ],
        "vehicle_pax_boarded_std_at_from": historical_data[
            "avg_unit_boarders_at_station"
        ]
        * 0.7,
        "vehicle_pax_boarded_first_at_from": historical_data[
            "avg_unit_boarders_at_station"
        ]
        * 0.3,
        "vehicle_pax_alighted_std_at_from": historical_data[
            "avg_unit_alighters_at_station"
        ]
        * 0.7,
        "vehicle_pax_alighted_first_at_from": historical_data[
            "avg_unit_alighters_at_station"
        ]
        * 0.3,
        # Engineered occupancy features
        "occupancy_percentage_std": (
            historical_data["avg_vehicle_pax_on_arrival_std"]
            / (89 if coach_type == "Standard" else 60)
        )
        * 100,
        "occupancy_percentage_first": (
            historical_data["avg_vehicle_pax_on_arrival_first"]
            / (60 if coach_type == "First Class" else 89)
        )
        * 100,
        "total_occupancy": historical_data["avg_vehicle_pax_on_arrival_std"]
        + historical_data["avg_vehicle_pax_on_arrival_first"],
        "total_occupancy_percentage": (
            (
                historical_data["avg_vehicle_pax_on_arrival_std"]
                + historical_data["avg_vehicle_pax_on_arrival_first"]
            )
            / (89 if coach_type == "Standard" else 115)
        )
        * 100,
        "boarding_ratio": (
            historical_data["avg_unit_boarders_at_station"]
            / (89 if coach_type == "Standard" else 115)
        )
        * 100,
        "alighting_ratio": (
            historical_data["avg_unit_alighters_at_station"]
            / (89 if coach_type == "Standard" else 115)
        )
        * 100,
        "capacity_utilization": (
            (
                historical_data["avg_vehicle_pax_on_arrival_std"]
                + historical_data["avg_vehicle_pax_on_arrival_first"]
            )
            / (89 if coach_type == "Standard" else 115)
        )
        * 100,
    }

    # Encode categorical features
    if feature_encoders:
        # Encode station names
        if "stationName_from" in feature_encoders:
            try:
                features["stationName_from_encoded"] = feature_encoders[
                    "stationName_from"
                ].transform([from_station])[0]
            except ValueError:
                features["stationName_from_encoded"] = 0  # Unknown station

        if "stationName_to" in feature_encoders:
            try:
                features["stationName_to_encoded"] = feature_encoders[
                    "stationName_to"
                ].transform([to_station])[0]
            except ValueError:
                features["stationName_to_encoded"] = 0  # Unknown station

        # Encode service identifiers
        if "headcode" in feature_encoders:
            try:
                features["headcode_encoded"] = feature_encoders["headcode"].transform(
                    [service_info["headcode"]]
                )[0]
            except ValueError:
                features["headcode_encoded"] = 0

        if "rsid" in feature_encoders:
            try:
                features["rsid_encoded"] = feature_encoders["rsid"].transform(
                    [service_info["rsid"]]
                )[0]
            except ValueError:
                features["rsid_encoded"] = 0

        # Encode coach type
        if "coach_type" in feature_encoders:
            try:
                features["coach_type_encoded"] = feature_encoders[
                    "coach_type"
                ].transform([coach_type])[0]
            except ValueError:
                features["coach_type_encoded"] = 0

        # Encode time period
        if "time_period" in feature_encoders:
            try:
                features["time_period_encoded"] = feature_encoders[
                    "time_period"
                ].transform([time_period])[0]
            except ValueError:
                features["time_period_encoded"] = 0

    # Create DataFrame with only the features needed by the model
    feature_df = pd.DataFrame([features])

    # Select only the features that were used in training
    if feature_list:
        available_features = [f for f in feature_list if f in feature_df.columns]
        feature_df = feature_df[available_features]

        # Add missing features with default values
        for feature in feature_list:
            if feature not in feature_df.columns:
                feature_df[feature] = 0

        # Reorder to match training order
        feature_df = feature_df[feature_list]

    return feature_df


@app.post("/predict_comfort_first_leg", response_model=ComfortPrediction)
async def predict_comfort(request: PredictionRequest):
    """Task 7.3: Implement API endpoint for comfort prediction."""

    try:
        # Parse and validate inputs
        departure_dt = parse_datetime(request.departure_datetime)

        # Identify relevant service and actual next stop
        service_info = identify_service_and_next_stop(
            request.from_station, request.to_station, departure_dt
        )

        # Fetch historical averages for arrival state
        historical_data = get_historical_averages(
            service_info["headcode"],
            service_info["rsid"],
            request.from_station,
            departure_dt,
        )

        # Construct feature vectors for both Standard and First Class
        std_features = construct_feature_vector(
            request.from_station,
            request.to_station,
            departure_dt,
            service_info,
            historical_data,
            "Standard",
        )

        first_features = construct_feature_vector(
            request.from_station,
            request.to_station,
            departure_dt,
            service_info,
            historical_data,
            "First Class",
        )

        # Apply scaling if needed
        if scaler:
            std_features_scaled = std_features.copy()
            first_features_scaled = first_features.copy()
            # Apply scaling to specific columns if scaler exists
        else:
            std_features_scaled = std_features
            first_features_scaled = first_features

        # Make predictions
        std_proba = model.predict_proba(std_features_scaled)[0]
        first_proba = model.predict_proba(first_features_scaled)[0]

        # Get class names
        class_names = ["Busy", "Moderate", "Quiet"]

        # Format predictions
        std_prediction = {
            "comfort_tier": class_names[np.argmax(std_proba)],
            "confidence": float(np.max(std_proba)),
        }

        first_prediction = {
            "comfort_tier": class_names[np.argmax(first_proba)],
            "confidence": float(np.max(first_proba)),
        }

        return ComfortPrediction(
            from_station=request.from_station,
            to_station=request.to_station,
            departure_datetime=request.departure_datetime,
            standard_class=std_prediction,
            first_class=first_prediction,
            service_info={
                "headcode": service_info["headcode"],
                "rsid": service_info["rsid"],
                "next_stop": service_info["next_stop"],
                "fallback_used": str(service_info.get("fallback", False)),
                "time_fallback": str(service_info.get("time_fallback", False)),
                "route_fallback": str(service_info.get("route_fallback", False)),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database_connected": os.path.exists(db_path),
    }


@app.get("/stations")
async def get_stations():
    """Get list of available stations."""
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT station_name FROM stations ORDER BY station_name"
        stations = [row[0] for row in conn.execute(query).fetchall()]
        return {"stations": stations}
    finally:
        conn.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
