# 5-Tier Comfort Predictor Implementation Plan

## Executive Summary

This document provides a complete implementation plan for upgrading the UK Rail Comfort Score system from the current 3-tier classification (Quiet/Moderate/Busy) to a more granular 5-tier system (Very Quiet/Quiet/Moderate/Busy/Very Busy). The upgrade will use domain-based occupancy thresholds instead of the current quantile-based approach, providing more actionable comfort predictions for passengers.

**Key Benefits:**
- More granular comfort predictions (5 tiers vs 3)
- Domain-based thresholds that reflect real-world occupancy patterns
- Improved user experience with actionable descriptions
- Minimal disruption to existing infrastructure
- Quick implementation timeline (3 days)

## Phase 1: Data Analysis & Threshold Validation (Day 1 Morning)

### Task 1.1: Analyze Current Occupancy Distribution
**File to Create**: `src/analysis_occupancy_distribution.py`
**Objective**: Understand actual occupancy patterns in the current dataset

**Implementation Steps:**
```python
# Script should:
1. Connect to 'duck' database
2. Load train_journey_legs table
3. Calculate occupancy_percentage = (relevant_passengers_on_leg_departure / vehicle_capacity) * 100
4. Generate statistics:
   - Min, max, mean, median occupancy
   - Percentile values (10th, 25th, 33rd, 50th, 67th, 75th, 85th, 90th, 95th)
   - Distribution histogram
5. Compare current quantile thresholds (33rd, 67th) vs proposed fixed thresholds (25%, 50%, 70%, 85%)
6. Save visualizations to models/occupancy_distribution.png
7. Output summary statistics to console
```

### Task 1.2: Validate 5-Tier Thresholds
**Continue in**: `src/analysis_occupancy_distribution.py`
**Objective**: Check if proposed thresholds create reasonable class distribution

**Implementation Steps:**
```python
# Add to analysis script:
1. Apply proposed 5-tier classification:
   - Very Quiet: 0-25%
   - Quiet: 25-50%
   - Moderate: 50-70%
   - Busy: 70-85%
   - Very Busy: 85%+
2. Calculate resulting class distribution percentages
3. Identify class imbalance (likely fewer Very Quiet and Very Busy)
4. Test alternative thresholds if severe imbalance
5. Output recommended thresholds based on data
```

### Task 1.3: Document Threshold Analysis
**File to Create**: `THRESHOLD_ANALYSIS_REPORT.md`
**Objective**: Document findings and final threshold recommendations

**Content Structure:**
```markdown
# Occupancy Threshold Analysis Report

## Current 3-Tier Distribution
- Quiet: X% of journeys (≤33rd percentile = Y% occupancy)
- Moderate: X% of journeys (33rd-67th percentile = Y-Z% occupancy)
- Busy: X% of journeys (>67th percentile = >Z% occupancy)

## Proposed 5-Tier Distribution
- Very Quiet: X% of journeys (0-25% occupancy)
- Quiet: X% of journeys (25-50% occupancy)
- Moderate: X% of journeys (50-70% occupancy)
- Busy: X% of journeys (70-85% occupancy)
- Very Busy: X% of journeys (85%+ occupancy)

## Recommendations
- Final thresholds: [25, 50, 70, 85] or adjusted based on analysis
- Class weights needed for XGBoost: {...}
```

## Phase 2: Core 5-Tier Implementation (Day 1 Afternoon)

### Task 2.1: Create 5-Tier Target Definition
**File to Create**: `src/target_definition_simple_5tier.py`
**Objective**: Implement new target variable definition with domain-based thresholds

**Implementation Blueprint:**
```python
#!/usr/bin/env python3
"""
Target Variable Definition for Train Comfort Predictor (5-Tier System)
Minimal changes approach - reuses existing data structure
"""

import pandas as pd
import numpy as np
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
    
    # Use fixed thresholds (adjust based on Phase 1 analysis)
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
    
    return df, target_encoder, encoders, thresholds
```

### Task 2.2: Fix Coach Type Issue
**File to Update**: `src/feature_engineering.py`
**Function**: `prepare_categorical_encoding()`
**Line**: Around line 102

**Change Required:**
```python
def prepare_categorical_encoding(df):
    """Prepare categorical variables for encoding."""
    print("Preparing categorical encoding strategy...")
    
    # Fix: Remove 'Mixed' from coach_type options
    if 'coach_type' in df.columns:
        # Ensure only Standard or First Class
        df['coach_type'] = df['coach_type'].replace('Mixed', 'Standard')
    
    # Define categorical columns that need encoding
    categorical_cols = ['coach_type', 'stationName_from', 'stationName_to', 
                       'headcode', 'rsid', 'time_period']
    
    # Rest of existing code...
```

### Task 2.3: Update XGBoost Training for 5 Classes
**File to Update**: `src/xgboost_model_training.py`
**Multiple Changes Required**:

1. **Update imports to use 5-tier**:
```python
from target_definition_simple_5tier import target_variable_pipeline_five_tier
```

2. **Update split_data() function** (line ~42):
```python
# Update class names in print statements
tier_name = ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'][idx]
```

3. **Update train_xgboost_model() function** (line ~106):
```python
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier for 5-tier system."""
    print("\n=== TRAINING XGBOOST MODEL (5-TIER) ===")
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[i] for i in y_train])
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=5,  # Changed from 3 to 5
        random_state=42,
        eval_metric='mlogloss',
        early_stopping_rounds=10,
        verbosity=1
    )
    
    # Train with sample weights
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=False
    )
```

4. **Update evaluate_model() function** (line ~150):
```python
# Update class names
class_names = ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet']
```

5. **Update metadata in save_model_and_artifacts()** (line ~288):
```python
'target_classes': ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'],
```

### Task 2.4: Update Feature Selection Pipeline
**File to Update**: `src/feature_selection.py`
**Line**: Around line 10

**Changes Required:**
```python
# Update import
from target_definition_simple_5tier import target_variable_pipeline_five_tier

# Update function call in feature_selection_pipeline() (line ~133)
df, target_encoder, encoders, thresholds = target_variable_pipeline_five_tier()

# Update tier names in select_model_features() (line ~123)
tier_name = ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet'][idx]
```

## Phase 3: Feature Enhancements & API Updates (Day 2)

### Task 3.1: Add Simple Journey Context Features
**File to Update**: `src/feature_engineering.py`
**Add New Function** after `create_occupancy_features()`:

```python
def create_simple_enhanced_features(df):
    """Add high-value features without changing data structure."""
    print("Creating enhanced features...")
    
    # Journey position estimate (simple heuristic)
    london_station = 'London Kings Cross'
    major_terminals = ['Edinburgh Waverley', 'Leeds', 'Newcastle', 'Aberdeen']
    
    # Simple journey progress indicator
    df['is_origin_major'] = df['stationName_from'].isin([london_station] + major_terminals).astype(int)
    df['is_destination_major'] = df['stationName_to'].isin([london_station] + major_terminals).astype(int)
    
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
    
    print(f"Added enhanced features: is_origin_major, is_destination_major, is_popular_route, is_monday, is_friday, is_sunday")
    
    return df
```

**Update feature_engineering_pipeline()** (line ~173):
```python
# Add after create_occupancy_features
df = create_simple_enhanced_features(df)
```

### Task 3.2: Update API for 5-Tier Predictions
**File to Update**: `api/main.py`
**Multiple Changes Required**:

1. **Update predict_comfort() function** (line ~645):
```python
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
```

2. **Update construct_feature_vector()** to include enhanced features:
```python
# Add to features dictionary (after line ~499)
# Enhanced features
"is_origin_major": 1 if from_station in ['London Kings Cross', 'Edinburgh Waverley', 'Leeds', 'Newcastle', 'Aberdeen'] else 0,
"is_destination_major": 1 if to_station in ['London Kings Cross', 'Edinburgh Waverley', 'Leeds', 'Newcastle', 'Aberdeen'] else 0,
"is_popular_route": 1 if (from_station, to_station) in [
    ('London Kings Cross', 'Leeds'),
    ('London Kings Cross', 'Edinburgh Waverley'),
    ('London Kings Cross', 'York'),
    ('Leeds', 'London Kings Cross'),
    ('Edinburgh Waverley', 'London Kings Cross')
] else 0,
"is_monday": 1 if day_of_week == 0 else 0,
"is_friday": 1 if day_of_week == 4 else 0,
"is_sunday": 1 if day_of_week == 6 else 0,
```

### Task 3.3: Update Feature Selection for Enhanced Features
**File to Update**: `src/feature_selection.py`
**Add to feature lists** (after line ~63):

```python
# Enhanced features
enhanced_features = [
    'is_origin_major', 'is_destination_major', 'is_popular_route',
    'is_monday', 'is_friday', 'is_sunday'
]

# Add to selected_features (line ~66)
selected_features = (time_features + location_features + vehicle_features +
                     service_features + contextual_features +
                     occupancy_features + enhanced_features)
```

## Phase 4: Integration Testing & Validation (Day 3)

### Task 4.1: Train Complete 5-Tier Model
**Execute Commands**:
```bash
# 1. Test target definition
python src/target_definition_simple_5tier.py

# 2. Train full model
python src/xgboost_model_training.py

# 3. Verify model artifacts
ls -la models/
# Should see: xgboost_comfort_classifier.joblib, target_encoder.joblib, etc.
```

### Task 4.2: Create Performance Comparison
**File to Create**: `MODEL_COMPARISON_REPORT.md`
**Content Structure**:
```markdown
# 5-Tier vs 3-Tier Model Comparison

## Model Performance Metrics

### 3-Tier Model (Baseline)
- Overall Accuracy: X%
- Class Distribution: Balanced (33% each)
- Confusion Matrix: [...]

### 5-Tier Model (New)
- Overall Accuracy: X%
- Class Distribution: [Very Quiet: X%, Quiet: X%, Moderate: X%, Busy: X%, Very Busy: X%]
- Per-Class Metrics:
  - Very Quiet: Precision X%, Recall X%
  - Quiet: Precision X%, Recall X%
  - Moderate: Precision X%, Recall X%
  - Busy: Precision X%, Recall X%
  - Very Busy: Precision X%, Recall X%

## Key Findings
- [Analysis of where model performs well/poorly]
- [Recommendations for production deployment]
```

### Task 4.3: API Integration Testing
**File to Update/Create**: `test_api_5tier.py`
```python
#!/usr/bin/env python3
"""Test API with 5-tier predictions."""

import requests
import json
from datetime import datetime

# Test health check
response = requests.get("http://localhost:8080/health")
print(f"Health Check: {response.json()}")

# Test prediction with various occupancy scenarios
test_cases = [
    {
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
]

for test in test_cases:
    response = requests.post(
        "http://localhost:8080/predict_comfort_first_leg",
        json=test
    )
    
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        print(f"\nRoute: {test['from_tiploc']} to {test['to_tiploc']}")
        for pred in predictions:
            dt = pred["departure_datetime"]
            std = pred["standard_class"]
            first = pred["first_class"]
            print(f"  {dt}: Standard={std['comfort_tier']} ({std['numeric_score']}), First={first['comfort_tier']} ({first['numeric_score']})")
    else:
        print(f"Error: {response.status_code} - {response.text}")
```

### Task 4.4: Update API Data Preparation
**File to Update**: `src/api_data_preparation.py`
**Ensure enhanced features are included in lookups**:

No changes needed - the existing pipeline will work with the new model.

## Success Criteria Checklist

### Functional Requirements
- [ ] 5-tier classification implemented (Very Quiet, Quiet, Moderate, Busy, Very Busy)
- [ ] Domain-based thresholds working (not quantile-based)
- [ ] API returns 5-tier predictions with numeric scores 1-5
- [ ] Model trains successfully with class balancing
- [ ] All existing functionality preserved

### Performance Requirements
- [ ] Model accuracy ≥ 70% overall
- [ ] Per-class precision/recall documented
- [ ] API response time < 2 seconds
- [ ] No system stability issues
- [ ] Realistic confidence scores (not always 99%+)

### Technical Requirements
- [ ] No breaking changes to data pipeline
- [ ] Enhanced features integrated
- [ ] Model artifacts loadable by API
- [ ] Coach type "Mixed" issue fixed
- [ ] Documentation updated

## Risk Mitigation Strategies

### 1. Class Imbalance
**Risk**: Very Quiet and Very Busy classes may have few samples
**Mitigation**: 
- Use compute_class_weight('balanced') in XGBoost
- Monitor per-class metrics
- Adjust thresholds if necessary

### 2. Accuracy Degradation
**Risk**: 5-tier model may have lower accuracy than 3-tier
**Mitigation**:
- Set realistic expectations (70-75% is good for 5 classes)
- Focus on per-class performance for critical tiers
- Consider ensemble approaches if needed

### 3. API Breaking Changes
**Risk**: External systems may expect 3-tier responses
**Mitigation**:
- Maintain same API endpoint structure
- Only change comfort_tier values and numeric_score range
- Document changes clearly

### 4. Training Time
**Risk**: 5-class model may take longer to train
**Mitigation**:
- Use early_stopping_rounds in XGBoost
- Reduce hyperparameter search space if needed
- Train on subset first for quick validation

## Timeline

### Day 1 (8 hours)
- Morning (4h): Phase 1 - Data Analysis & Threshold Validation
- Afternoon (4h): Phase 2 - Core 5-Tier Implementation

### Day 2 (8 hours)
- Morning (4h): Phase 3 - Feature Enhancements
- Afternoon (4h): Phase 3 - API Updates & Initial Testing

### Day 3 (8 hours)
- Morning (4h): Phase 4 - Model Training & Validation
- Afternoon (4h): Phase 4 - Integration Testing & Documentation

## Deliverables

1. **Analysis Reports**
   - `THRESHOLD_ANALYSIS_REPORT.md`
   - `MODEL_COMPARISON_REPORT.md`

2. **Code Updates**
   - `src/target_definition_simple_5tier.py` (new)
   - `src/feature_engineering.py` (updated)
   - `src/xgboost_model_training.py` (updated)
   - `src/feature_selection.py` (updated)
   - `api/main.py` (updated)

3. **Model Artifacts**
   - `models/xgboost_comfort_classifier.joblib` (5-tier)
   - `models/target_encoder.joblib` (5 classes)
   - Supporting metadata files

4. **Test Results**
   - API test outputs showing 5-tier predictions
   - Performance metrics comparison
   - Validation of all success criteria

## Post-Implementation Considerations

1. **Monitor Production Performance**
   - Track prediction distribution across 5 tiers
   - Collect user feedback on granularity usefulness
   - Monitor API response times

2. **Future Enhancements**
   - Consider coach-specific thresholds
   - Add seasonal adjustments
   - Implement feedback loop for threshold tuning

3. **Documentation Updates**
   - Update README.md with 5-tier information
   - Update API documentation
   - Create user guide explaining comfort tiers

This implementation plan provides a complete roadmap for upgrading the comfort prediction system from 3-tier to 5-tier classification while maintaining system stability and improving user value.