# Train Comfort Predictor: 3-Tier to 5-Tier Enhancement Specification

## Executive Summary

This specification outlines the enhancement of the Train Comfort Predictor system from its current 3-tier comfort classification system to a more granular 5-tier system, while maintaining the existing data infrastructure for rapid MVP deployment.

## Current State

### What the System Does Now

The Train Comfort Predictor is an ML-powered API that predicts how busy train carriages will be on UK rail services. Currently, the system:

1. **Processes passenger count data** from train sensors showing real-time occupancy by carriage
2. **Transforms this data** through three stages:
   - Raw sensor data (`my_data`) → 
   - Enriched with carriage capacity (`train_journeys_enriched`) → 
   - Final training data with journey legs (`train_journey_legs`)
3. **Trains an XGBoost model** to classify comfort into 3 tiers:
   - **Quiet** (≤33rd percentile occupancy)
   - **Moderate** (33rd-67th percentile)
   - **Busy** (>67th percentile)
4. **Serves predictions via API** where users input:
   - Origin and destination stations (via TIPLOC codes)
   - Departure time
   - Receives comfort predictions for both Standard and First Class

### Current Limitations

- **Too coarse**: Only 3 tiers don't capture the nuance between "slightly busy" and "packed"
- **Forced distribution**: Quantile-based thresholds mean exactly 33% of trains are classified as "Quiet" even if they're not
- **Limited context**: The model doesn't understand where in a journey a particular leg occurs
- **Accuracy ceiling**: Current approach achieves ~70% accuracy

## Proposed Enhancement

### What We're Changing

We will enhance the system to provide 5-tier comfort predictions while keeping the existing data pipeline intact:

**New Comfort Tiers:**
- **Very Quiet** (0-25% occupancy) - Plenty of seats, can choose where to sit
- **Quiet** (25-50% occupancy) - Easy to find seats together
- **Moderate** (50-70% occupancy) - Seats available but choice limited
- **Busy** (70-85% occupancy) - Few seats, may need to walk through carriages
- **Very Busy** (85%+ occupancy) - Standing room likely

### What We're NOT Changing

To deliver quickly for the MVP:
- **No data structure changes** - The existing tables and schema remain as-is
- **No pipeline rewrites** - Current data flow from raw → enriched → final stays the same
- **No complex journey reconstruction** - We'll add simple journey context without restructuring
- **API interface stays the same** - Users still just provide from/to stations and time

### Key Improvements

1. **Better Granularity**: 5 tiers provide clearer expectations for passengers
2. **Domain-Based Thresholds**: Using sensible occupancy percentages rather than forced quantiles
3. **Simple Feature Enhancements**: Adding high-value features that don't require data restructuring
4. **Journey Context**: Basic journey progress indicators using existing data
5. **Maintained Simplicity**: All changes work within current architecture

### Expected Outcomes

- **Accuracy**: Improvement from ~70% to ~75-80% (acceptable for MVP)
- **User Experience**: More actionable predictions ("Very Quiet" vs just "Quiet")
- **Implementation Time**: 3 days instead of 8-10 days for full restructure
- **Risk**: Minimal - no breaking changes to data pipeline
- **Scalability**: Foundation for future enhancements without technical debt

## Implementation Philosophy

This enhancement follows the principle of "maximum impact, minimum disruption." Rather than perfecting the data pipeline (which would delay the MVP), we're making targeted improvements that:

1. Deliver immediate user value (5-tier predictions)
2. Improve accuracy through smart feature engineering
3. Maintain system stability
4. Allow for future enhancements

The result will be a system that provides more useful predictions to passengers while remaining maintainable and ready for rapid deployment in the POC environment.

### What We'll Keep As-Is
- Current data structure (no SQL changes needed)
- Existing `train_journey_legs` table remains unchanged
- Current data pipeline stays the same

### High-Impact Changes We'll Make

## 1. Fix Coach Type Logic (Quick Fix)

Update `feature_engineering.py`:

```python
def prepare_categorical_encoding(df):
    """Prepare categorical variables for encoding."""
    print("Preparing categorical encoding strategy...")
    
    # Fix: Remove 'Mixed' from coach_type options
    if 'coach_type' in df.columns:
        # Ensure only Standard or First Class
        df['coach_type'] = df['coach_type'].replace('Mixed', 'Standard')
    
    # Rest of existing code...
```

## 2. Update Target Variable to 5-Tier (Essential Change)

Create `target_definition_simple_5tier.py`:

```python
#!/usr/bin/env python3
"""
Target Variable Definition for Train Comfort Predictor (5-Tier System)
Minimal changes approach - reuses existing data structure
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import feature_engineering_pipeline


def define_comfort_tiers_five_level(df):
    """Define 5 comfort tiers based on occupancy percentage."""
    print("=== DEFINING 5-TIER COMFORT SYSTEM ===")
    
    # Calculate occupancy percentage (same as before)
    df['occupancy_percentage'] = (df['relevant_passengers_on_leg_departure'] / 
                                 df['vehicle_capacity'] * 100)
    
    print("Occupancy percentage statistics:")
    print(df['occupancy_percentage'].describe())
    
    # Use fixed thresholds for MVP
    # These can be tuned based on initial results
    thresholds = {
        'very_quiet': 25,    # 0-25%
        'quiet': 50,         # 25-50%
        'moderate': 70,      # 50-70%
        'busy': 85,          # 70-85%
        'very_busy': 100     # 85%+
    }
    
    # Create comfort tiers
    def assign_comfort_tier(occupancy_pct):
        if occupancy_pct <= thresholds['very_quiet']:
            return 'Very Quiet'
        elif occupancy_pct <= thresholds['quiet']:
            return 'Quiet'
        elif occupancy_pct <= thresholds['moderate']:
            return 'Moderate'
        elif occupancy_pct <= thresholds['busy']:
            return 'Busy'
        else:
            return 'Very Busy'
    
    df['comfort_tier'] = df['occupancy_percentage'].apply(assign_comfort_tier)
    
    print(f"\nThresholds used:")
    for tier, threshold in thresholds.items():
        print(f"{tier}: {threshold}%")
    
    # Show distribution
    tier_counts = df['comfort_tier'].value_counts()
    tier_order = ['Very Quiet', 'Quiet', 'Moderate', 'Busy', 'Very Busy']
    
    print(f"\nComfort tier distribution:")
    for tier in tier_order:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            pct = count / len(df) * 100
            print(f"{tier}: {count} ({pct:.1f}%)")
    
    return df, thresholds


def encode_target_variable_five_tier(df):
    """Encode 5-tier comfort variable for machine learning."""
    print(f"\n=== ENCODING 5-TIER TARGET VARIABLE ===")
    
    # Create custom encoding (0 = Very Busy, 4 = Very Quiet)
    comfort_mapping = {
        'Very Busy': 0,
        'Busy': 1,
        'Moderate': 2,
        'Quiet': 3,
        'Very Quiet': 4
    }
    
    df['comfort_tier_encoded'] = df['comfort_tier'].map(comfort_mapping)
    
    # Create label encoder for consistency
    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'])
    
    # Show mapping
    print(f"Target variable encoding:")
    for tier, encoding in comfort_mapping.items():
        count = (df['comfort_tier_encoded'] == encoding).sum()
        print(f"{tier}: {encoding} (n={count})")
    
    return df, target_encoder


def target_variable_pipeline_five_tier():
    """Complete target variable definition pipeline for 5-tier system."""
    print("=== STARTING 5-TIER TARGET VARIABLE DEFINITION ===")
    
    # Get engineered features (using existing pipeline)
    df, encoders = feature_engineering_pipeline()
    
    # Define 5-tier comfort levels
    df, thresholds = define_comfort_tiers_five_level(df)
    
    # Encode target variable
    df, target_encoder = encode_target_variable_five_tier(df)
    
    # Summary
    print(f"\n=== COMPLETE ===")
    print(f"Target variable: comfort_tier (5 levels)")
    print(f"Classes: {list(target_encoder.classes_)}")
    print(f"Dataset shape: {df.shape}")
    
    return df, target_encoder, encoders, thresholds


if __name__ == "__main__":
    df_with_target, target_encoder, feature_encoders, thresholds = target_variable_pipeline_five_tier()
```

## 3. Update Model Training (Simple Changes)

Update `xgboost_model_training.py` - just change the number of classes:

```python
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier for 5-tier system."""
    print("\n=== TRAINING XGBOOST MODEL (5-TIER) ===")
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=5,  # Changed from 3 to 5
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=10,
        verbosity=1,
        # Add class weights for imbalanced classes
        scale_pos_weight=1
    )
    
    # Rest of the code remains the same...
```

## 4. Update API for 5-Tier (Key Changes Only)

Update `api/main.py` - minimal changes:

```python
# Update the class names and mappings
def predict_comfort(request: PredictionRequest):
    """Updated for 5-tier predictions."""
    
    # ... existing code for service identification ...
    
    # Get class names and create numeric mapping
    class_names = ["Very Busy", "Busy", "Moderate", "Quiet", "Very Quiet"]
    # Numeric score: 1=Very Busy (least comfortable) to 5=Very Quiet (most comfortable)
    class_to_numeric = {
        "Very Busy": 1,
        "Busy": 2,
        "Moderate": 3,
        "Quiet": 4,
        "Very Quiet": 5
    }
    
    # ... rest of prediction logic remains the same ...
```

## 5. Add Journey Context Without Data Changes

Update `api_data_preparation.py` to create a service patterns lookup:

```python
def create_api_lookup_tables():
    """Create additional lookup tables for journey context."""
    
    # Add to existing API data preparation
    query = """
    -- Service stop patterns
    CREATE TABLE IF NOT EXISTS service_patterns AS
    SELECT 
        headcode,
        rsid,
        GROUP_CONCAT(DISTINCT stationName_from || '->' || stationName_to 
                     ORDER BY leg_departure_dt) as typical_route,
        COUNT(DISTINCT DATE(leg_departure_dt)) as days_observed,
        AVG(relevant_passengers_on_leg_departure) as avg_passengers
    FROM train_journey_legs
    GROUP BY headcode, rsid
    HAVING COUNT(*) >= 10;
    
    -- Station sequence for journey progress
    CREATE TABLE IF NOT EXISTS station_sequences AS
    SELECT DISTINCT
        headcode,
        rsid,
        stationName_from,
        stationName_to,
        AVG(CASE 
            WHEN totalUnitPassenger_at_leg_departure > 0 
            THEN relevant_passengers_on_leg_departure::float / totalUnitPassenger_at_leg_departure
            ELSE 0.5 
        END) as typical_retention_rate
    FROM train_journey_legs
    GROUP BY headcode, rsid, stationName_from, stationName_to;
    """
```

## 6. Add Simple Feature Enhancements

Add these to existing `feature_engineering.py` without changing structure:

```python
def create_simple_enhanced_features(df):
    """Add high-value features without changing data structure."""
    
    # Journey position estimate (simple heuristic)
    london_station = 'London Kings Cross'
    major_terminals = ['Edinburgh Waverley', 'Leeds', 'Newcastle', 'Aberdeen']
    
    # Simple journey progress indicator
    df['is_origin_major'] = df['stationName_from'].isin([london_station] + major_terminals)
    df['is_destination_major'] = df['stationName_to'].isin([london_station] + major_terminals)
    
    # Time until destination (simple estimate based on typical journey times)
    typical_journey_times = {
        ('London Kings Cross', 'Leeds'): 140,
        ('London Kings Cross', 'Edinburgh Waverley'): 270,
        ('London Kings Cross', 'Newcastle'): 180,
        ('Leeds', 'London Kings Cross'): 140,
        # Add more as needed
    }
    
    # Route popularity (simple encoding)
    popular_routes = [
        ('London Kings Cross', 'Leeds'),
        ('London Kings Cross', 'Edinburgh Waverley'),
        ('London Kings Cross', 'York'),
        ('Leeds', 'London Kings Cross'),
        ('Edinburgh Waverley', 'London Kings Cross')
    ]
    
    df['is_popular_route'] = df.apply(
        lambda x: (x['stationName_from'], x['stationName_to']) in popular_routes, 
        axis=1
    ).astype(int)
    
    # Add day of week patterns
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
    
    return df
```

## Implementation Order (Time Efficient)

1. **Day 1**: Update target variable to 5-tier (2-3 hours)
2. **Day 1**: Fix coach type issue (30 minutes)
3. **Day 1**: Update XGBoost training for 5 classes (1 hour)
4. **Day 2**: Add simple feature enhancements (2-3 hours)
5. **Day 2**: Update API for 5-tier predictions (2 hours)
6. **Day 3**: Add journey context lookups to API (3-4 hours)
7. **Day 3**: Test and validate (2-3 hours)

**Total time: 3 days instead of 8-10 days**

## Expected Accuracy Improvements

With these minimal changes:
- Base 3-tier accuracy: ~70%
- With 5-tier + simple features: ~75-80%
- Good enough for MVP/POC

## Station List Integration

Save the station list for validation in the API:

```python
# In api/main.py
VALID_STATIONS = [
    'Aberdeen', 'Grantham', 'Middlesbrough', 'Thornaby', 'Northallerton',
    'Stonehaven', 'Arbroath', 'Horsforth', 'Bowes Park', 'North Queensferry',
    'Keighley', 'Skipton', 'Welwyn Garden City', 'Cupar', 'Falkirk Grahamston',
    'Gleneagles', 'Kingussie', 'Morpeth', 'Lincoln', 'Hitchin', 'St Neots',
    'Newark North Gate', 'Berwick-upon-Tweed', 'Leeds', 'York', 'Perth',
    'Pitlochery', 'Aviemore', 'Sandy', 'Peterborough', 'Dunbar', 'Huntingdon',
    'Newcastle', 'Darlington', 'Inverkeithing', 'Alnmouth', 'Dunblane',
    'Stirling', 'Kirkcaldy', 'Inverness', 'Larbert', 'Haymarket', 'Selby',
    'Blair Atholl', 'Laurencekirk', 'Retford', 'Brough', 'Kinghorn',
    'London Kings Cross', 'Stevenage', 'Chester-le-Street', 'Thirsk',
    'Edinburgh Waverley', 'Montrose', 'Reston', 'Bradford Forster Square',
    'Leuchars', 'Harrogate', 'Carrbridge', 'Newtonmore', 'Musselburgh',
    'Sandal & Agbrigg', 'Knebworth', 'Ferriby', 'Finsbury Park', 'Shipley',
    'Hykeham', 'Outwood', 'Wakefield Westgate', 'Durham', 'Dundee',
    'Doncaster', 'Hull', 'Biggleswade', 'Manors'
]

def validate_station(station_name: str) -> bool:
    """Validate if station is in our network."""
    return station_name in VALID_STATIONS
```

This approach gives you:
- 5-tier comfort predictions ✅
- Better accuracy through simple improvements ✅
- No complex data restructuring needed ✅
- Quick implementation (3 days) ✅
- Working MVP/POC ready for testing ✅

Would you like me to provide the complete updated files for any of these components?