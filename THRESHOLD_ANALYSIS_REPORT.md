# Occupancy Threshold Analysis Report

## Executive Summary

This report presents the analysis of occupancy patterns in the UK rail comfort dataset and validates the proposed 5-tier classification system. Based on 148,727 journey records, we recommend proceeding with the proposed domain-based thresholds while implementing class balancing techniques.

## Current 3-Tier Distribution

The existing system uses quantile-based thresholds:
- **Quiet**: ≤19.2% occupancy (33rd percentile) - 50,917 journeys (34.2%)
- **Moderate**: 19.2%-45.1% occupancy - 48,774 journeys (32.8%)
- **Busy**: >45.1% occupancy (67th percentile) - 49,036 journeys (33.0%)

This approach ensures equal class distribution but results in counter-intuitive thresholds (e.g., 19% occupancy classified as "Moderate").

## Occupancy Statistics

- **Min occupancy**: 0.0%
- **Max occupancy**: 500.0% (data quality issue - outliers)
- **Mean occupancy**: 37.2%
- **Median occupancy**: 30.1%
- **Standard deviation**: 31.3%

### Key Percentiles
- 10th percentile: 3.9%
- 25th percentile: 13.7%
- 50th percentile: 30.1%
- 75th percentile: 53.4%
- 85th percentile: 68.6%
- 90th percentile: 79.5%
- 95th percentile: 96.1%

## Proposed 5-Tier Distribution

Using domain-based thresholds [25%, 50%, 70%, 85%]:
- **Very Quiet**: 0-25% occupancy - 63,466 journeys (42.7%)
- **Quiet**: 25-50% occupancy - 43,269 journeys (29.1%)
- **Moderate**: 50-70% occupancy - 20,860 journeys (14.0%)
- **Busy**: 70-85% occupancy - 9,291 journeys (6.2%)
- **Very Busy**: 85%+ occupancy - 5,613 journeys (3.8%)

## Alternative Threshold Analysis

We tested alternative thresholds [30%, 50%, 70%, 85%]:
- **Very Quiet**: 0-30% - 73,330 journeys (49.3%)
- **Quiet**: 30-50% - 33,405 journeys (22.5%)
- **Moderate**: 50-70% - 20,860 journeys (14.0%)
- **Busy**: 70-85% - 9,291 journeys (6.2%)
- **Very Busy**: 85%+ - 5,613 journeys (3.8%)

This creates even more imbalance in the "Very Quiet" class.

## Recommendations

### 1. Proceed with Original Thresholds
The proposed thresholds [25%, 50%, 70%, 85%] offer the best balance between:
- Intuitive understanding (25% = quarter full, 50% = half full, etc.)
- Reasonable class distribution
- Actionable information for passengers

### 2. Implement Class Balancing
To handle class imbalance, use the following weights in XGBoost:
- **Very Quiet**: 0.47
- **Quiet**: 0.69
- **Moderate**: 1.43
- **Busy**: 3.20
- **Very Busy**: 5.30

### 3. Data Quality Considerations
- Filter outliers (occupancy > 100%) during training
- These likely represent data collection errors or special circumstances

### 4. Benefits of Domain-Based Approach
- More interpretable for end users
- Aligns with passenger expectations
- Provides consistent thresholds across routes
- Enables better decision-making for journey planning

## Visualization

See `models/occupancy_distribution.png` for detailed visualizations including:
- Occupancy distribution histogram with threshold lines
- Box plots by coach type
- 3-tier vs 5-tier comparison
- Cumulative distribution function

## Conclusion

The proposed 5-tier system with domain-based thresholds provides a significant improvement over the current quantile-based approach. While class imbalance exists, it reflects real-world patterns where most journeys have lower occupancy. The recommended class weights will ensure the model learns to distinguish between all comfort levels effectively.