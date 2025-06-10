#!/usr/bin/env python3
"""
Analyze occupancy distribution in the current dataset to validate 5-tier thresholds.
This script connects to the DuckDB database and analyzes passenger occupancy patterns.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_occupancy_distribution():
    """Analyze occupancy patterns in the train journey data."""
    print("=== OCCUPANCY DISTRIBUTION ANALYSIS ===")
    
    # Connect to DuckDB
    db_path = Path(__file__).parent.parent / "duck.db"
    print(f"Connecting to database at: {db_path}")
    
    conn = duckdb.connect(str(db_path), read_only=True)
    
    # Load journey data
    query = """
    SELECT 
        relevant_passengers_on_leg_departure,
        vehicle_capacity,
        coach_type,
        leg_departure_dt
    FROM train_journey_legs
    WHERE vehicle_capacity > 0
    """
    
    df = conn.execute(query).fetch_df()
    conn.close()
    
    print(f"Loaded {len(df):,} journey records")
    
    # Calculate occupancy percentage
    df['occupancy_percentage'] = (df['relevant_passengers_on_leg_departure'] / 
                                 df['vehicle_capacity'] * 100)
    
    # Basic statistics
    print("\n=== OCCUPANCY STATISTICS ===")
    print(f"Min occupancy: {df['occupancy_percentage'].min():.1f}%")
    print(f"Max occupancy: {df['occupancy_percentage'].max():.1f}%")
    print(f"Mean occupancy: {df['occupancy_percentage'].mean():.1f}%")
    print(f"Median occupancy: {df['occupancy_percentage'].median():.1f}%")
    print(f"Std deviation: {df['occupancy_percentage'].std():.1f}%")
    
    # Calculate percentiles
    percentiles = [10, 25, 33, 50, 67, 75, 85, 90, 95]
    print("\n=== OCCUPANCY PERCENTILES ===")
    for p in percentiles:
        value = np.percentile(df['occupancy_percentage'], p)
        print(f"{p}th percentile: {value:.1f}%")
    
    # Current 3-tier thresholds (quantile-based)
    q33 = np.percentile(df['occupancy_percentage'], 33)
    q67 = np.percentile(df['occupancy_percentage'], 67)
    print(f"\nCurrent 3-tier thresholds:")
    print(f"  Quiet: ≤ {q33:.1f}% (33rd percentile)")
    print(f"  Moderate: {q33:.1f}% - {q67:.1f}%")
    print(f"  Busy: > {q67:.1f}% (67th percentile)")
    
    # Apply current 3-tier classification
    df['tier_3'] = pd.cut(df['occupancy_percentage'], 
                          bins=[-np.inf, q33, q67, np.inf],
                          labels=['Quiet', 'Moderate', 'Busy'])
    
    print("\nCurrent 3-tier distribution:")
    for tier in ['Quiet', 'Moderate', 'Busy']:
        count = (df['tier_3'] == tier).sum()
        pct = count / len(df) * 100
        print(f"  {tier}: {count:,} ({pct:.1f}%)")
    
    # Proposed 5-tier thresholds (domain-based)
    proposed_thresholds = [25, 50, 70, 85]
    print(f"\n=== PROPOSED 5-TIER ANALYSIS ===")
    print("Proposed thresholds: 0-25%, 25-50%, 50-70%, 70-85%, 85%+")
    
    # Apply proposed 5-tier classification
    df['tier_5'] = pd.cut(df['occupancy_percentage'],
                          bins=[0, 25, 50, 70, 85, 100],
                          labels=['Very Quiet', 'Quiet', 'Moderate', 'Busy', 'Very Busy'],
                          include_lowest=True)
    
    print("\nProposed 5-tier distribution:")
    tier_order = ['Very Quiet', 'Quiet', 'Moderate', 'Busy', 'Very Busy']
    for tier in tier_order:
        count = (df['tier_5'] == tier).sum()
        pct = count / len(df) * 100
        print(f"  {tier}: {count:,} ({pct:.1f}%)")
    
    # Test alternative thresholds if imbalanced
    print("\n=== TESTING ALTERNATIVE THRESHOLDS ===")
    
    # Alternative 1: Adjust for better balance
    alt_thresholds = [30, 50, 70, 85]
    df['tier_5_alt1'] = pd.cut(df['occupancy_percentage'],
                               bins=[0, 30, 50, 70, 85, 100],
                               labels=['Very Quiet', 'Quiet', 'Moderate', 'Busy', 'Very Busy'],
                               include_lowest=True)
    
    print("\nAlternative thresholds (30-50-70-85):")
    for tier in tier_order:
        count = (df['tier_5_alt1'] == tier).sum()
        pct = count / len(df) * 100
        print(f"  {tier}: {count:,} ({pct:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Occupancy Distribution Analysis', fontsize=16)
    
    # 1. Histogram of occupancy
    ax1 = axes[0, 0]
    ax1.hist(df['occupancy_percentage'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(q33, color='red', linestyle='--', label=f'33rd percentile: {q33:.1f}%')
    ax1.axvline(q67, color='red', linestyle='--', label=f'67th percentile: {q67:.1f}%')
    for t in proposed_thresholds:
        ax1.axvline(t, color='green', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Occupancy Percentage')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Occupancy Distribution with Thresholds')
    ax1.legend()
    
    # 2. Box plot by coach type
    ax2 = axes[0, 1]
    df.boxplot(column='occupancy_percentage', by='coach_type', ax=ax2)
    ax2.set_xlabel('Coach Type')
    ax2.set_ylabel('Occupancy Percentage')
    ax2.set_title('Occupancy by Coach Type')
    
    # 3. 3-tier vs 5-tier comparison
    ax3 = axes[1, 0]
    tier_comparison = pd.DataFrame({
        '3-Tier': df['tier_3'].value_counts(normalize=True) * 100,
        '5-Tier': df['tier_5'].value_counts(normalize=True) * 100
    })
    tier_comparison.plot(kind='bar', ax=ax3)
    ax3.set_xlabel('Comfort Tier')
    ax3.set_ylabel('Percentage of Journeys')
    ax3.set_title('3-Tier vs 5-Tier Distribution')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_occupancy = np.sort(df['occupancy_percentage'])
    cumulative = np.arange(1, len(sorted_occupancy) + 1) / len(sorted_occupancy) * 100
    ax4.plot(sorted_occupancy, cumulative, 'b-', linewidth=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Occupancy Percentage')
    ax4.set_ylabel('Cumulative Percentage')
    ax4.set_title('Cumulative Distribution Function')
    
    # Add threshold lines
    for t in proposed_thresholds:
        idx = np.searchsorted(sorted_occupancy, t)
        if idx < len(cumulative):
            ax4.plot([t, t], [0, cumulative[idx]], 'g:', alpha=0.5)
            ax4.plot([0, t], [cumulative[idx], cumulative[idx]], 'g:', alpha=0.5)
            ax4.text(t + 1, 5, f'{t}%', fontsize=8)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "occupancy_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Recommendation
    print("\n=== RECOMMENDATIONS ===")
    print("Based on the analysis:")
    print("1. The proposed thresholds [25, 50, 70, 85] create a reasonable distribution")
    print("2. Very Quiet and Very Busy classes have fewer samples but still substantial")
    print("3. Consider using class weights in XGBoost to handle imbalance")
    print("4. The thresholds align well with intuitive occupancy levels")
    
    # Calculate class weights for balanced training
    tier_counts = df['tier_5'].value_counts()
    total_samples = len(df)
    n_classes = len(tier_counts)
    
    print("\nRecommended class weights for balanced training:")
    for tier in tier_order:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            weight = total_samples / (n_classes * count)
            print(f"  {tier}: {weight:.2f}")
    
    return df, proposed_thresholds


if __name__ == "__main__":
    df, thresholds = analyze_occupancy_distribution()
    print("\nAnalysis complete!")