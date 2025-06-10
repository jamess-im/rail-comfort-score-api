# 3-Tier to 5-Tier Comfort Prediction Upgrade Implementation Plan

## Executive Summary

This document outlines the implementation plan for upgrading the UK Rail Comfort Score system from the current 3-tier classification (Quiet, Moderate, Busy) to a 5-tier system (Very Quiet, Quiet, Moderate, Busy, Very Busy). This upgrade will provide more granular comfort predictions while maintaining backward compatibility.

## Current System Analysis

### Current 3-Tier System
- **Quiet**: ≤ 33rd percentile occupancy (~31.4%)
- **Moderate**: 33rd-67th percentile occupancy (31.4%-56.1%)
- **Busy**: > 67th percentile occupancy (>56.1%)

### Key Statistics
- Occupancy ranges from 0% to 500%+ (outliers)
- Median occupancy: ~29.6%
- Balanced distribution: ~33% each tier
- High prediction confidence: 99%+

## Proposed 5-Tier System

### New Tier Definitions
1. **Very Quiet** (Tier 1): 0-20th percentile
2. **Quiet** (Tier 2): 20th-40th percentile  
3. **Moderate** (Tier 3): 40th-60th percentile
4. **Busy** (Tier 4): 60th-80th percentile
5. **Very Busy** (Tier 5): 80th-100th percentile

### Expected Occupancy Thresholds
Based on current data distribution:
- Very Quiet: ≤ 19%
- Quiet: 19% - 35%
- Moderate: 35% - 50%
- Busy: 50% - 72%
- Very Busy: > 72%

## Implementation Steps

### Phase 1: Data Layer Updates

#### 1.1 Update Target Variable Definition
**File**: `src/target_definition_simple.py`

**Changes**:
- Modify `define_comfort_tiers_simple()` to use 5 quantiles (0.2, 0.4, 0.6, 0.8)
- Update tier assignment logic for 5 categories
- Adjust class names array

```python
# New quantiles
quantiles = df['occupancy_percentage'].quantile([0.2, 0.4, 0.6, 0.8]).values

# New tier assignment
def assign_comfort_tier(occupancy_pct):
    if occupancy_pct <= quantiles[0]:
        return 'Very Quiet'
    elif occupancy_pct <= quantiles[1]:
        return 'Quiet'
    elif occupancy_pct <= quantiles[2]:
        return 'Moderate'
    elif occupancy_pct <= quantiles[3]:
        return 'Busy'
    else:
        return 'Very Busy'
```

#### 1.2 Update Target Encoder
**Changes**:
- Encoder will automatically adapt to 5 classes
- Update encoding display logic

### Phase 2: Model Training Updates

#### 2.1 Update XGBoost Configuration
**File**: `src/xgboost_model_training.py`

**Changes**:
- Update `num_class` parameter from 3 to 5
- Modify class name arrays throughout
- Update evaluation metrics display

```python
# Update XGBoost initialization
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=5,  # Changed from 3
    random_state=42,
    eval_metric='mlogloss',
    early_stopping_rounds=10,
    verbosity=1
)

# Update class names
class_names = ['Very Busy', 'Busy', 'Moderate', 'Quiet', 'Very Quiet']
```

#### 2.2 Update Model Evaluation
**Changes**:
- Adjust confusion matrix size (5x5)
- Update classification report for 5 classes
- Modify visualization layouts

### Phase 3: API Updates

#### 3.1 Update API Response Model
**File**: `api/main.py`

**Changes**:
- Update class names in prediction logic
- Modify numeric score mapping (1-5 instead of 1-3)
- Ensure backward compatibility with response structure

```python
# Update class mapping
class_names = ["Very Busy", "Busy", "Moderate", "Quiet", "Very Quiet"]
class_to_numeric = {
    "Very Busy": 1,
    "Busy": 2, 
    "Moderate": 3,
    "Quiet": 4,
    "Very Quiet": 5
}
```

#### 3.2 Database Updates
**File**: `src/api_data_preparation.py`

**Changes**:
- No structural changes needed
- Historical averages remain the same
- Service lookup logic unchanged

### Phase 4: Testing & Validation

#### 4.1 Create Test Suite
**New File**: `tests/test_5_tier_upgrade.py`

**Tests**:
- Verify 5 balanced classes in training data
- Confirm model outputs 5 probability values
- Test API returns correct tier names
- Validate numeric scores (1-5)
- Check backward compatibility

#### 4.2 Model Performance Validation
- Compare accuracy metrics
- Analyze per-class precision/recall
- Validate threshold boundaries
- Test edge cases

### Phase 5: Migration Strategy

#### 5.1 Parallel Deployment
1. Train new 5-tier model alongside existing 3-tier
2. Deploy both models initially
3. Add API version flag for tier selection
4. Monitor performance differences

#### 5.2 Gradual Rollout
1. Deploy to staging environment
2. A/B test with subset of users
3. Monitor feedback and accuracy
4. Full production deployment

## Implementation Timeline

### Week 1: Development
- Day 1-2: Update data pipeline for 5 tiers
- Day 3-4: Retrain model with new tiers
- Day 5: Update API endpoints

### Week 2: Testing
- Day 1-2: Unit and integration tests
- Day 3-4: Performance validation
- Day 5: Documentation updates

### Week 3: Deployment
- Day 1-2: Staging deployment
- Day 3-4: A/B testing
- Day 5: Production rollout

## Risk Mitigation

### Technical Risks
1. **Model Performance Degradation**
   - Mitigation: Extensive testing, maintain 3-tier fallback
   
2. **API Breaking Changes**
   - Mitigation: Version endpoints, maintain backward compatibility

3. **Data Imbalance**
   - Mitigation: Monitor class distributions, adjust thresholds if needed

### Business Risks
1. **User Confusion**
   - Mitigation: Clear documentation, gradual rollout
   
2. **Integration Issues**
   - Mitigation: Comprehensive API testing, client communication

## Success Metrics

1. **Model Performance**
   - Maintain >95% accuracy
   - Balanced precision/recall across all 5 tiers
   
2. **API Performance**
   - Response time <100ms
   - 99.9% uptime
   
3. **User Adoption**
   - Positive feedback on granularity
   - Reduced complaints about edge cases

## Rollback Plan

If issues arise:
1. Revert API to use 3-tier model
2. Maintain 5-tier model for analysis
3. Address issues and redeploy
4. Keep both models available via versioning

## File Change Summary

### Files to Modify:
1. `src/target_definition_simple.py` - Add 5-tier logic
2. `src/xgboost_model_training.py` - Update model config
3. `api/main.py` - Update API responses
4. `src/feature_selection.py` - Verify compatibility
5. `test_api.py` - Add 5-tier tests

### New Files:
1. `tests/test_5_tier_upgrade.py` - Comprehensive test suite
2. `src/tier_migration.py` - Migration utilities
3. `docs/5_tier_api_guide.md` - API documentation

### Generated Artifacts:
1. `models/xgboost_comfort_classifier_5tier.joblib`
2. `models/target_encoder_5tier.joblib`
3. `models/model_metadata_5tier.joblib`

## Conclusion

This upgrade will provide users with more nuanced comfort predictions while maintaining system stability. The implementation focuses on backward compatibility, thorough testing, and gradual rollout to minimize risk. The 5-tier system will better serve users who need finer granularity in comfort predictions, particularly for journey planning during peak and off-peak hours.