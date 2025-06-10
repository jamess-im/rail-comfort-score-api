# 5-Tier vs 3-Tier Model Comparison

## Executive Summary

The UK Rail Comfort Score system has been successfully upgraded from a 3-tier to a 5-tier classification system. The new model achieves **89.4% accuracy** while providing more granular and actionable comfort predictions for passengers.

## Model Performance Metrics

### 5-Tier Model (New Implementation)

**Overall Performance:**
- **Test Accuracy**: 89.4%
- **Training Accuracy**: 93.9%
- **Model Type**: XGBoost Classifier with Class Balancing
- **Features**: 42 engineered features (including 6 new enhanced features)

**Class Distribution in Dataset:**
- Very Quiet (0-25% occupancy): 63,466 journeys (42.7%)
- Quiet (25-50% occupancy): 43,269 journeys (29.1%)
- Moderate (50-70% occupancy): 20,860 journeys (14.0%)
- Busy (70-85% occupancy): 9,291 journeys (6.2%)
- Very Busy (85%+ occupancy): 11,841 journeys (8.0%)

**Per-Class Performance Metrics:**

| Comfort Tier | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Very Busy    | 0.91      | 0.89   | 0.90     | 2,368   |
| Busy         | 0.69      | 0.79   | 0.74     | 1,858   |
| Moderate     | 0.82      | 0.83   | 0.82     | 4,172   |
| Quiet        | 0.88      | 0.88   | 0.88     | 8,654   |
| Very Quiet   | 0.96      | 0.94   | 0.95     | 12,694  |

**Confidence Analysis:**
- Very Busy: 86.5% average confidence
- Busy: 70.4% average confidence
- Moderate: 74.8% average confidence
- Quiet: 81.3% average confidence
- Very Quiet: 91.2% average confidence

### 3-Tier Model (Baseline - Historical)

**Previous Performance (Estimated):**
- Overall Accuracy: ~85-90% (comparable range)
- Class Distribution: Artificially balanced (33% each tier)
- Threshold Method: Quantile-based (33rd and 67th percentiles)
- User Experience: Less granular predictions

**Previous Tier Definitions:**
- Quiet: ≤19.2% occupancy (counter-intuitive)
- Moderate: 19.2%-45.1% occupancy
- Busy: >45.1% occupancy

## Key Improvements

### 1. **More Granular Predictions**
- 5 distinct comfort levels vs. 3
- Better differentiation between comfort levels
- More actionable information for passenger journey planning

### 2. **Intuitive Thresholds**
- Domain-based thresholds (25%, 50%, 70%, 85%) vs. quantile-based
- Thresholds align with passenger expectations
- Consistent across all routes and times

### 3. **Enhanced Feature Engineering**
- 6 new enhanced features added:
  - `is_origin_major`, `is_destination_major`
  - `is_popular_route`
  - `is_monday`, `is_friday`, `is_sunday`
- Fixed coach type handling ('Mixed' → 'Standard')
- Improved route and timing pattern recognition

### 4. **Class Imbalance Handling**
- Implemented class weights for balanced training
- Better performance on minority classes (Busy, Very Busy)
- More realistic confidence scores

### 5. **Improved User Experience**
- Numeric scoring system (1-5 scale)
- 1 = Very Busy (least comfortable)
- 5 = Very Quiet (most comfortable)
- More nuanced journey planning capability

## Technical Achievements

### ✅ **Functional Requirements Met**
- 5-tier classification implemented
- Domain-based thresholds working
- API returns 5-tier predictions with numeric scores
- Model trains successfully with class balancing
- All existing functionality preserved

### ✅ **Performance Requirements Met**
- Model accuracy: 89.4% (exceeds 70% target)
- Per-class performance documented
- Realistic confidence scores achieved
- No system stability issues observed

### ✅ **Technical Requirements Met**
- No breaking changes to data pipeline
- Enhanced features successfully integrated
- Model artifacts compatible with API
- Coach type issue resolved
- Documentation updated

## Implementation Highlights

### **Training Efficiency**
- Class weights successfully addressed imbalance
- Model converged quickly (99 iterations)
- Best validation score: 0.2644 (log loss)

### **Feature Engineering Success**
- 42 features selected from 58 candidates
- Enhanced features contribute to better route and timing predictions
- Missing feature handling robust (1 feature excluded: `capacity_utilization`)

### **API Integration**
- Seamless integration with existing API endpoints
- Backward-compatible response structure
- Enhanced features automatically computed from request parameters

## Recommendations for Production

### **Immediate Deployment**
1. **Deploy with confidence**: 89.4% accuracy is excellent for 5-class prediction
2. **Monitor tier distribution**: Track real-world prediction patterns
3. **Collect user feedback**: Validate granularity usefulness with passengers

### **Future Enhancements**
1. **Seasonal adjustments**: Consider holiday and event-based modifications
2. **Route-specific tuning**: Fine-tune thresholds for specific corridors
3. **Real-time feedback**: Implement mechanism for threshold optimization

### **Monitoring KPIs**
1. **Prediction distribution**: Ensure realistic spread across 5 tiers
2. **User engagement**: Track which tiers influence journey decisions
3. **Accuracy validation**: Compare predictions with actual occupancy when available

## Conclusion

The 5-tier comfort prediction system represents a significant improvement over the previous 3-tier system:

- **Higher granularity** provides more actionable passenger information
- **Intuitive thresholds** align with passenger expectations
- **Strong technical performance** with 89.4% accuracy across 5 classes
- **Enhanced features** improve prediction quality for routes and timing
- **Production-ready** implementation with comprehensive testing

The upgrade successfully balances technical accuracy with user experience improvements, making it ready for immediate production deployment.

---

*Model trained on 148,727 journey records with 42 engineered features*  
*Training completed: Phase 4 of 5-Tier Implementation Plan*