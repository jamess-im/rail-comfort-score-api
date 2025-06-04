#!/usr/bin/env python3
"""
API Data Preparation for Train Comfort Predictor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import duckdb
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta # timedelta is not used, but datetime is
import re # For string normalization
import warnings
warnings.filterwarnings('ignore')

# --- Constants for the external TIPLOC database ---
# **** USER ACTION: Update these paths and names as per your TIPLOC database setup ****
TIPLOC_DB_PATH = 'ukra.db'
TIPLOC_TABLE_NAME = 'tiplocs'                       # Table name for the tiploc codes
TIPLOC_CODE_COLUMN = 'tiploc_code'                  # Column name for TIPLOC codes
TIPLOC_DESCRIPTION_COLUMN = 'description'           # Primary station name column in tiploc DB
TIPLOC_TPS_DESCRIPTION_COLUMN = 'tps_description'   # Secondary station name column
TIPLOC_CRS_COLUMN = 'crs_code'                      # CRS code column

# --- Normalization function for station names ---
def normalize_station_name(name):
    if pd.isna(name) or name is None: # Check for None as well
        return None
    name = str(name).strip().lower()
    # Remove common suffixes that might cause mismatches, be careful with these
    # name = re.sub(r'\s+(station|plat\w*|halt|jn|jcn|jct|sdgs|depot|platform\s*\d*)\b', '', name, flags=re.IGNORECASE)
    # Remove trailing numbers that might be platform numbers if not part of the core name
    name = re.sub(r'\s+\d+$', '', name)
    name = name.replace('&', 'and') # Standardize ampersand
    name = re.sub(r'[^\w\s]', '', name) # Remove most punctuation, keeps alphanumeric and space
    name = ' '.join(name.split()) # Normalize multiple spaces to single space
    return name if name else None # Return None if normalization results in empty string


def connect_to_duckdb():
    """Connect to the main DuckDB database containing the MVP dataset."""
    print("=== CONNECTING TO DUCKDB DATABASE ===")
    try:
        # Assuming your DuckDB database file is named 'duck' in the project root
        # Adjust if it's in a subdirectory like 'data/duck.db'
        conn = duckdb.connect('duck')
        print("Successfully connected to DuckDB database ('duck')")
        return conn
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        return None


def identify_api_required_data(duck_conn):
    """Identify data required by the API at prediction time."""
    print("\n=== IDENTIFYING API REQUIRED DATA ===")
    
    # Get basic dataset info
    query = "SELECT COUNT(*) as total_records FROM train_journey_legs"
    total_records_legs = duck_conn.execute(query).fetchone()[0]
    print(f"Total records in train_journey_legs (MVP dataset): {total_records_legs}")
    
    # --- Fetch TIPLOC data ---
    print(f"\nFetching TIPLOC data from '{TIPLOC_DB_PATH}'...")
    tiploc_df = pd.DataFrame(columns=['tiploc', 'primary_normalized_name_for_match', 'tiploc_orig_desc', 'tiploc_orig_tps_desc', 'crs']) # Default empty
    if os.path.exists(TIPLOC_DB_PATH):
        try:
            conn_tiploc = sqlite3.connect(TIPLOC_DB_PATH)
            tiploc_query = f"""
                SELECT
                    TRIM({TIPLOC_CODE_COLUMN}) AS tiploc,
                    {TIPLOC_DESCRIPTION_COLUMN} AS desc_name,
                    {TIPLOC_TPS_DESCRIPTION_COLUMN} AS tps_desc_name,
                    {TIPLOC_CRS_COLUMN} AS crs
                FROM {TIPLOC_TABLE_NAME}
                WHERE {TIPLOC_CODE_COLUMN} IS NOT NULL AND TRIM({TIPLOC_CODE_COLUMN}) <> ''
            """
            raw_tiploc_df = pd.read_sql_query(tiploc_query, conn_tiploc)
            conn_tiploc.close()

            raw_tiploc_df['normalized_desc_name'] = raw_tiploc_df['desc_name'].apply(normalize_station_name)
            raw_tiploc_df['normalized_tps_desc_name'] = raw_tiploc_df['tps_desc_name'].apply(normalize_station_name)
            
            # MODIFIED LOGIC (Option 1): Choose the best normalized name for matching
            # Prioritize tps_description if it's good, then description, then skip if neither is good.
            def choose_best_tiploc_name_for_matching(row):
                norm_tps = row['normalized_tps_desc_name']
                norm_desc = row['normalized_desc_name']

                # Prefer tps_description if it's valid and longer or if description is too short/invalid
                if pd.notna(norm_tps) and len(norm_tps) > 3:
                    if pd.notna(norm_desc) and len(norm_desc) > 3:
                        # If both are good, prefer the longer one, or tps if lengths are similar
                        # This heuristic might need tuning based on your specific name variations
                        if len(norm_tps) >= len(norm_desc) - 2: # Allow desc to be slightly shorter
                            return norm_tps
                        else:
                            return norm_desc # desc is significantly longer and valid
                    return norm_tps # tps is good, desc is not
                elif pd.notna(norm_desc) and len(norm_desc) > 3:
                    return norm_desc # desc is good, tps is not
                return None # Neither is suitable

            raw_tiploc_df['primary_normalized_name_for_match'] = raw_tiploc_df.apply(choose_best_tiploc_name_for_matching, axis=1)

            # Keep original names for reference
            raw_tiploc_df.rename(columns={'desc_name': 'tiploc_orig_desc', 'tps_desc_name': 'tiploc_orig_tps_desc'}, inplace=True)
            
            # Select necessary columns, keeping both normalized names for fallback matching
            tiploc_df = raw_tiploc_df[['tiploc', 'primary_normalized_name_for_match', 'normalized_desc_name', 'normalized_tps_desc_name', 'tiploc_orig_desc', 'tiploc_orig_tps_desc', 'crs']].copy()
            tiploc_df.dropna(subset=['primary_normalized_name_for_match'], inplace=True) # Critical
            
            # Deduplicate on tiploc first (tiploc should be unique key from source)
            tiploc_df.drop_duplicates(subset=['tiploc'], keep='first', inplace=True)
            # Then, deduplicate on the name used for matching. If multiple TIPLOCs somehow normalize to the exact same name,
            # this picks one. This helps ensure the right side of the merge is unique on the join key.
            tiploc_df.drop_duplicates(subset=['primary_normalized_name_for_match'], keep='first', inplace=True)

            print(f"  Fetched and processed {len(tiploc_df)} unique TIPLOC records for matching (after deduplication on name and tiploc).")

        except Exception as e:
            print(f"  ERROR fetching/processing TIPLOC data: {e}. Proceeding without TIPLOCs for station table.")
            # Ensure tiploc_df has the correct columns even if empty, for schema consistency in merge
            tiploc_df = pd.DataFrame(columns=['tiploc', 'primary_normalized_name_for_match', 'normalized_desc_name', 'normalized_tps_desc_name', 'tiploc_orig_desc', 'tiploc_orig_tps_desc', 'crs'])
    else:
        print(f"  WARNING: TIPLOC database not found at '{TIPLOC_DB_PATH}'. Proceeding without TIPLOCs.")
        # Ensure tiploc_df has the correct columns even if empty
        tiploc_df = pd.DataFrame(columns=['tiploc', 'primary_normalized_name_for_match', 'normalized_desc_name', 'normalized_tps_desc_name', 'tiploc_orig_desc', 'tiploc_orig_tps_desc', 'crs'])


    # --- Station Information (from train_journey_legs) ---
    print("\n1. Station Information (from train_journey_legs):")
    station_locations_query = """
    SELECT stationName, stationLocation
    FROM (
        SELECT stationName_from AS stationName, stationLocation_from AS stationLocation FROM train_journey_legs
        UNION
        SELECT stationName_to AS stationName, stationLocation_to AS stationLocation FROM train_journey_legs WHERE stationName_to IS NOT NULL
    ) all_station_events
    WHERE stationName IS NOT NULL AND TRIM(stationName) <> '' AND stationLocation IS NOT NULL AND TRIM(stationLocation) <> '';
    """
    all_distinct_station_loc_df = duck_conn.execute(station_locations_query).fetchdf()
    print(f"  Found {len(all_distinct_station_loc_df)} distinct (station name, station location) pairs in journey data before name grouping.")

    all_distinct_station_loc_df['normalized_journey_station_name'] = all_distinct_station_loc_df['stationName'].apply(normalize_station_name)
    all_distinct_station_loc_df.dropna(subset=['normalized_journey_station_name'], inplace=True)

    stations_df_from_journeys = all_distinct_station_loc_df.groupby('normalized_journey_station_name', as_index=False).agg(
        station_name_journey_orig=('stationName', 'first'),
        station_location=('stationLocation', 'first')
    )
    print(f"  Consolidated to {len(stations_df_from_journeys)} unique normalized station names from journey data.")

    # --- Join stations_df_from_journeys with tiploc_df ---
    # Two-step matching process: first try primary match, then fallback to alternative
    print("  Performing two-step TIPLOC matching...")
    
    # Step 1: Try matching with primary normalized name
    stations_with_tiploc_df = pd.merge(
        stations_df_from_journeys,
        tiploc_df,
        left_on='normalized_journey_station_name',
        right_on='primary_normalized_name_for_match',
        how='left'
    )
    
    initial_matches = stations_with_tiploc_df['tiploc'].notna().sum()
    print(f"    Step 1 - Primary match: {initial_matches} stations matched")
    
    # Step 2: For unmatched stations, try fallback matching with the alternative normalized name
    if not tiploc_df.empty:
        unmatched_mask = stations_with_tiploc_df['tiploc'].isna()
        unmatched_stations = stations_with_tiploc_df[unmatched_mask].copy()
        
        if len(unmatched_stations) > 0:
            print(f"    Step 2 - Attempting fallback matching for {len(unmatched_stations)} unmatched stations...")
            
            # Create alternative matching dataframes
            # First try with desc_name if it wasn't the primary choice
            tiploc_desc_alt = tiploc_df[tiploc_df['normalized_desc_name'].notna() & (tiploc_df['normalized_desc_name'] != tiploc_df['primary_normalized_name_for_match'])][
                ['tiploc', 'normalized_desc_name', 'tiploc_orig_desc', 'tiploc_orig_tps_desc', 'crs']
            ].copy()
            tiploc_desc_alt.drop_duplicates(subset=['normalized_desc_name'], keep='first', inplace=True)
            
            # Try with tps_desc_name if it wasn't the primary choice  
            tiploc_tps_alt = tiploc_df[tiploc_df['normalized_tps_desc_name'].notna() & (tiploc_df['normalized_tps_desc_name'] != tiploc_df['primary_normalized_name_for_match'])][
                ['tiploc', 'normalized_tps_desc_name', 'tiploc_orig_desc', 'tiploc_orig_tps_desc', 'crs']
            ].copy()
            tiploc_tps_alt.drop_duplicates(subset=['normalized_tps_desc_name'], keep='first', inplace=True)
            
            # Try fallback match with desc_name
            fallback_matches_desc = pd.merge(
                unmatched_stations[['normalized_journey_station_name', 'station_name_journey_orig', 'station_location']],
                tiploc_desc_alt,
                left_on='normalized_journey_station_name',
                right_on='normalized_desc_name',
                how='inner'
            )
            
            # Try fallback match with tps_desc_name (only for stations not matched by desc)
            remaining_unmatched = unmatched_stations[~unmatched_stations['normalized_journey_station_name'].isin(fallback_matches_desc['normalized_journey_station_name'])]
            fallback_matches_tps = pd.merge(
                remaining_unmatched[['normalized_journey_station_name', 'station_name_journey_orig', 'station_location']],
                tiploc_tps_alt,
                left_on='normalized_journey_station_name',
                right_on='normalized_tps_desc_name',
                how='inner'
            )
            
            # Combine fallback matches
            all_fallback_matches = pd.concat([fallback_matches_desc, fallback_matches_tps], ignore_index=True)
            
            if len(all_fallback_matches) > 0:
                print(f"    Step 2 - Fallback matching found {len(all_fallback_matches)} additional matches")
                
                # Update the unmatched rows with fallback matches
                for _, fallback_row in all_fallback_matches.iterrows():
                    mask = (stations_with_tiploc_df['normalized_journey_station_name'] == fallback_row['normalized_journey_station_name']) & stations_with_tiploc_df['tiploc'].isna()
                    stations_with_tiploc_df.loc[mask, 'tiploc'] = fallback_row['tiploc']
                    stations_with_tiploc_df.loc[mask, 'tiploc_orig_desc'] = fallback_row['tiploc_orig_desc']
                    stations_with_tiploc_df.loc[mask, 'tiploc_orig_tps_desc'] = fallback_row['tiploc_orig_tps_desc']
                    stations_with_tiploc_df.loc[mask, 'crs'] = fallback_row['crs']
            else:
                print(f"    Step 2 - No additional matches found through fallback")
    
    final_matches = stations_with_tiploc_df['tiploc'].notna().sum()
    print(f"  Total TIPLOC matches: {final_matches} out of {len(stations_with_tiploc_df)} stations")
    
    # Clean up columns we don't need in the final output
    columns_to_drop = ['primary_normalized_name_for_match', 'normalized_desc_name', 'normalized_tps_desc_name']
    stations_with_tiploc_df = stations_with_tiploc_df.drop(columns=[col for col in columns_to_drop if col in stations_with_tiploc_df.columns], errors='ignore')

    # Select final columns. 'station_name_journey_orig' is our canonical name from journey data.
    # Ensure all desired columns are present, even if they come from the right side of the merge (tiploc_df)
    # and might be all NaNs if tiploc_df was empty or merge failed.
    final_cols = ['station_name_journey_orig', 'station_location', 'tiploc', 'crs', 'tiploc_orig_desc', 'tiploc_orig_tps_desc']
    for col in final_cols:
        if col not in stations_with_tiploc_df.columns:
            stations_with_tiploc_df[col] = None # Add missing columns as None

    final_stations_df = stations_with_tiploc_df[final_cols].copy()
    final_stations_df.rename(columns={'station_name_journey_orig': 'station_name'}, inplace=True)
    
    # 'station_name' should be unique here because stations_df_from_journeys had unique 'normalized_journey_station_name'
    # and we merged onto that.
    if final_stations_df['station_name'].duplicated().any():
        print("  CRITICAL WARNING: Duplicate station_names found in final_stations_df after merge. This indicates an issue in the station deduplication logic.")
        final_stations_df.drop_duplicates(subset=['station_name'], keep='first', inplace=True)

    # Parse coordinates
    final_stations_df['latitude'] = pd.NA
    final_stations_df['longitude'] = pd.NA
    if 'station_location' in final_stations_df.columns and not final_stations_df['station_location'].isnull().all():
        valid_locations_mask = final_stations_df['station_location'].notna()
        if valid_locations_mask.any():
            # Ensure splitting only happens on rows with valid string data
            # And handle cases where split might not produce 2 columns
            coords = final_stations_df.loc[valid_locations_mask, 'station_location'].astype(str).str.split(',', expand=True, n=1)
            final_stations_df.loc[valid_locations_mask, 'latitude'] = pd.to_numeric(coords[0], errors='coerce')
            if coords.shape[1] > 1:
                 final_stations_df.loc[valid_locations_mask, 'longitude'] = pd.to_numeric(coords[1], errors='coerce')
            else:
                 final_stations_df.loc[valid_locations_mask, 'longitude'] = pd.NA


    final_stations_df['latitude'] = final_stations_df['latitude'].fillna(54.0)
    final_stations_df['longitude'] = final_stations_df['longitude'].fillna(-2.0)
    final_stations_df = final_stations_df.drop(columns=['station_location'], errors='ignore')
    print(f"  Station coordinates parsed/imputed for {len(final_stations_df)} stations.")
    
    missing_tiplocs = final_stations_df['tiploc'].isnull().sum()
    if not tiploc_df.empty and missing_tiplocs > 0 : # Only warn if tiploc_df was supposed to have data
        print(f"  WARNING: {missing_tiplocs} out of {len(final_stations_df)} stations from journey data could not be matched with a TIPLOC.")
        # For debugging, print some unmatched stations:
        # unmatched_df = final_stations_df[final_stations_df['tiploc'].isnull()]
        # print("  DEBUG: Sample unmatched stations (journey name):")
        # for index, row in unmatched_df.head().iterrows():
        #     print(f"    '{row['station_name']}' (Normalized: '{normalize_station_name(row['station_name'])}')")
    
    # --- Service Identification & Routing Summary (from train_journey_legs) ---
    # (This part of your script remains the same as it uses station_name from train_journey_legs)
    print("\n2. Service Identification & Routing Summary:")
    service_query = """
    SELECT 
        headcode,
        rsid,
        stationName_from,
        stationName_to,
        EXTRACT(hour FROM leg_departure_dt) as hour,
        CAST(strftime(leg_departure_dt, '%w') AS INTEGER) as day_of_week, -- Sunday=0 to Saturday=6
        COUNT(*) as frequency
    FROM train_journey_legs
    WHERE stationName_from IS NOT NULL AND stationName_to IS NOT NULL AND headcode IS NOT NULL AND rsid IS NOT NULL
    GROUP BY headcode, rsid, stationName_from, stationName_to, 
             EXTRACT(hour FROM leg_departure_dt), CAST(strftime(leg_departure_dt, '%w') AS INTEGER)
    ORDER BY frequency DESC
    """
    service_routes_df = duck_conn.execute(service_query).fetchdf()
    print(f"  Service route patterns: {len(service_routes_df)}")
    
    # --- Historical Average Stats (from train_journey_legs) ---
    # (This part of your script remains the same)
    print("\n3. Historical Average Stats for Arrival State Estimation:")
    stats_query = """
    SELECT 
        headcode,
        rsid,
        stationName_from as station_of_arrival,
        CAST(strftime(leg_departure_dt, '%w') AS INTEGER) as day_of_week_bucket, -- Sunday=0
        CASE 
            WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 6 AND 9 THEN 'morning_peak'
            WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 10 AND 15 THEN 'midday'
            WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 16 AND 19 THEN 'evening_peak'
            WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 20 AND 23 THEN 'evening'
            ELSE 'night_early'
        END as time_of_day_bucket,
        COALESCE(AVG(vehicle_pax_on_arrival_std_at_from), 0) as avg_vehicle_pax_on_arrival_std,
        COALESCE(AVG(vehicle_pax_on_arrival_first_at_from), 0) as avg_vehicle_pax_on_arrival_first,
        COALESCE(AVG(totalUnitPassenger_at_leg_departure), 0) as avg_total_unit_pax_on_arrival,
        COALESCE(AVG(onUnitPassenger_at_from_station), 0) as avg_unit_boarders_at_station,
        COALESCE(AVG(offUnitPassenger_at_from_station), 0) as avg_unit_alighters_at_station,
        COUNT(*) as observations
    FROM train_journey_legs
    WHERE headcode IS NOT NULL AND rsid IS NOT NULL AND stationName_from IS NOT NULL
    GROUP BY headcode, rsid, stationName_from, 
             CAST(strftime(leg_departure_dt, '%w') AS INTEGER),
             CASE 
                WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 6 AND 9 THEN 'morning_peak'
                WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 10 AND 15 THEN 'midday'
                WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 16 AND 19 THEN 'evening_peak'
                WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 20 AND 23 THEN 'evening'
                ELSE 'night_early'
             END
    HAVING COUNT(*) >= 3
    ORDER BY observations DESC
    """
    historical_stats_df = duck_conn.execute(stats_query).fetchdf()
    print(f"  Historical stat patterns: {len(historical_stats_df)}")
    
    # --- Coach type information (from train_journey_legs) ---
    # (This part of your script remains the same)
    print("\n4. Coach Type Information:")
    coach_query = """
    SELECT 
        coach_type,
        COALESCE(AVG(vehicle_capacity), 0) as avg_capacity,
        COUNT(*) as frequency
    FROM train_journey_legs
    WHERE coach_type IS NOT NULL
    GROUP BY coach_type
    ORDER BY frequency DESC
    """
    coach_info_df = duck_conn.execute(coach_query).fetchdf()
    print(f"  Coach types: {len(coach_info_df)}")
    for _, row in coach_info_df.iterrows():
        print(f"    {row['coach_type']}: avg capacity {row['avg_capacity']:.1f}, frequency {row['frequency']}")
    
    print("\n=== IDENTIFY API REQUIRED DATA COMPLETE ===")
    
    return {
        'stations': final_stations_df,
        'service_routes': service_routes_df,
        'historical_stats': historical_stats_df,
        'coach_info': coach_info_df
    }

def create_sqlite_database_and_populate(api_data):
    """Create SQLite database and populate tables."""
    print("\n=== CREATING SQLITE DATABASE ===")
    
    db_path = 'api/train_comfort_api_lookups.sqlite'
    os.makedirs('api', exist_ok=True)
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    print(f"Created new SQLite database: {db_path}")
    
    try:
        # 1. Create and populate stations table
        print("\n1. Creating stations table...")
        # Ensure all expected columns exist, even if mostly NULL (like tiploc if merge failed)
        cols_to_write = ['station_name', 'latitude', 'longitude', 'tiploc', 'crs', 'tiploc_orig_desc', 'tiploc_orig_tps_desc']
        df_to_write_stations = pd.DataFrame(columns=cols_to_write)
        for col in cols_to_write:
            if col in api_data['stations'].columns:
                df_to_write_stations[col] = api_data['stations'][col]
            else:
                df_to_write_stations[col] = None # Add column with NULLs if missing
        
        df_to_write_stations.to_sql('stations', conn, index=False, if_exists='replace')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stations_name ON stations(station_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stations_tiploc ON stations(tiploc)") # Add index for tiploc
        station_count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
        print(f"   Inserted {station_count} stations")
        
        # 2. Create and populate service_routes_summary_mvp table
        print("\n2. Creating service_routes_summary_mvp table...")
        api_data['service_routes'].to_sql('service_routes_summary_mvp', conn, index=False, if_exists='replace')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sr_headcode ON service_routes_summary_mvp(headcode)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sr_rsid ON service_routes_summary_mvp(rsid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sr_from ON service_routes_summary_mvp(stationName_from)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sr_hour ON service_routes_summary_mvp(hour)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sr_dow ON service_routes_summary_mvp(day_of_week)")
        routes_count = conn.execute("SELECT COUNT(*) FROM service_routes_summary_mvp").fetchone()[0]
        print(f"   Inserted {routes_count} service route patterns")
        
        # 3. Create and populate historical_averages table
        print("\n3. Creating historical_averages table...")
        api_data['historical_stats'].to_sql('historical_averages', conn, index=False, if_exists='replace')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ha_headcode ON historical_averages(headcode)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ha_rsid ON historical_averages(rsid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ha_station ON historical_averages(station_of_arrival)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ha_dow ON historical_averages(day_of_week_bucket)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ha_time ON historical_averages(time_of_day_bucket)")
        hist_count = conn.execute("SELECT COUNT(*) FROM historical_averages").fetchone()[0]
        print(f"   Inserted {hist_count} historical average patterns")
        
        # 4. Create and populate coach_info table
        print("\n4. Creating coach_info table...")
        api_data['coach_info'].to_sql('coach_info', conn, index=False, if_exists='replace')
        coach_count = conn.execute("SELECT COUNT(*) FROM coach_info").fetchone()[0]
        print(f"   Inserted {coach_count} coach type records")
        
        # 5. Create metadata table
        print("\n5. Creating metadata table...")
        total_source_recs_val = total_records_legs if 'total_records_legs' in locals() else 0
        metadata_df = pd.DataFrame([{ # Corrected to metadata_df
            'created_at': datetime.now().isoformat(),
            'source_dataset': 'train_journey_legs (MVP fixed dataset)',
            'total_source_records_in_legs': total_source_recs_val,
            'stations_count': station_count,
            'service_patterns_count': routes_count,
            'historical_patterns_count': hist_count,
            'coach_types_count': coach_count,
            'purpose': 'API lookup database for Train Comfort Predictor MVP'
        }])
        metadata_df.to_sql('metadata', conn, index=False, if_exists='replace') # Corrected variable name
        print(f"   Created metadata table")
        
        conn.commit()
        print(f"\n✅ SQLite database created successfully: {db_path}")
        
    except Exception as e:
        print(f"Error creating SQLite database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
    
    print("=== CREATE SQLITE DB COMPLETE ===")
    return db_path


def validate_api_database(db_path):
    """Validate the created API database."""
    # This function remains largely the same, but ensure your test queries
    # still make sense with the potentially modified data.
    # For instance, the station lookup test might need a station name
    # that you know exists after normalization and TIPLOC merge.
    # And you might add a test for TIPLOC lookup.
    print("\n=== VALIDATING API DATABASE ===")
    if not os.path.exists(db_path):
        print(f"❌ Database file not found for validation: {db_path}")
        return False
        
    conn = sqlite3.connect(db_path)
    try:
        required_tables = ['stations', 'service_routes_summary_mvp', 'historical_averages', 'coach_info', 'metadata']
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Required tables: {required_tables}")
        print(f"Existing tables: {existing_tables}")
        
        missing_tables = set(required_tables) - set(existing_tables)
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        else:
            print("✅ All required tables present")
        
        for table in required_tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} records (ensure these counts are reasonable)")
            if table == 'stations' and count == 0:
                print(f"  WARNING: 'stations' table is empty. Check TIPLOC merge and station processing.")

        print("\n=== TESTING API-LIKE QUERIES (post-TIPLOC integration) ===")
        sample_station_name_for_test = "London Kings Cross" # Choose a name you expect to be present
        
        station_coord_query = "SELECT latitude, longitude FROM stations WHERE station_name = ?"
        station_coord_result = conn.execute(station_coord_query, (sample_station_name_for_test,)).fetchone()
        if station_coord_result:
            print(f"✅ Station coordinate lookup test for '{sample_station_name_for_test}': {station_coord_result}")
        else:
            print(f"⚠️ Station coordinate lookup test failed for '{sample_station_name_for_test}' or station not found.")

        # Find a TIPLOC that should exist for a known station
        sample_tiploc_query = "SELECT tiploc FROM stations WHERE station_name = ? AND tiploc IS NOT NULL LIMIT 1"
        tiploc_result_for_known_station = conn.execute(sample_tiploc_query, (sample_station_name_for_test,)).fetchone()
        
        if tiploc_result_for_known_station and tiploc_result_for_known_station[0]:
            known_tiploc = tiploc_result_for_known_station[0]
            station_name_from_tiploc_query = "SELECT station_name FROM stations WHERE tiploc = ?"
            name_from_tiploc_result = conn.execute(station_name_from_tiploc_query, (known_tiploc,)).fetchone()
            if name_from_tiploc_result:
                print(f"✅ TIPLOC lookup test: '{known_tiploc}' -> '{name_from_tiploc_result[0]}'")
            else:
                print(f"⚠️ TIPLOC lookup test failed for known TIPLOC '{known_tiploc}'.")
        else:
            print(f"⚠️ Could not find a TIPLOC for '{sample_station_name_for_test}' to test TIPLOC lookup, or TIPLOC data might be missing.")

        # ... (other validation queries for service_routes and historical_averages as before) ...
        print("✅ Database validation checks complete (review warnings if any).")
        return True
        
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        return False
    finally:
        conn.close()


def api_data_preparation_pipeline():
    """Complete API data preparation pipeline."""
    global total_records_legs # Make it global to be accessible in metadata creation
    print("=== STARTING API DATA PREPARATION PIPELINE ===")
    
    duck_conn = connect_to_duckdb()
    if not duck_conn:
        print("❌ Failed to connect to DuckDB database")
        return None
    
    try:
        # This needs to be fetched here for the metadata table later
        total_records_legs_query = "SELECT COUNT(*) as total_records FROM train_journey_legs"
        total_records_legs = duck_conn.execute(total_records_legs_query).fetchone()[0]

        api_data = identify_api_required_data(duck_conn)
        db_path = create_sqlite_database_and_populate(api_data) # total_records_legs is now in scope
        
        if db_path: # Only validate if db_path was successfully created
            validation_success = validate_api_database(db_path)
            if validation_success:
                print("\n=== API DATA PREPARATION PIPELINE COMPLETE ===")
                # ... (rest of the success print messages) ...
                return db_path
            else:
                print("❌ Database validation failed during API data prep pipeline.")
                return None
        else:
            print("❌ SQLite database creation failed.")
            return None
        
    except Exception as e:
        print(f"❌ An error occurred in the API data preparation pipeline: {e}")
        return None
    finally:
        if duck_conn:
            duck_conn.close()
            print("DuckDB connection closed.")


if __name__ == "__main__":
    # Ensure this script is run from the project root or adjust paths accordingly
    # Example: python src/api_data_preparation.py
    
    # For TIPLOC_DB_PATH, you might set it via an environment variable for flexibility
    # or ensure it's correctly pathed if relative.
    # For testing, you might place a sample tiploc.sqlite in 'data/' directory
    # and set TIPLOC_DB_PATH = 'data/tiplocs.sqlite'
    
    # Check if TIPLOC_DB_PATH is default and warn if so, as it needs to be user-set
    if TIPLOC_DB_PATH == 'path/to/your/tiploc_database.sqlite':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("WARNING: TIPLOC_DB_PATH is set to its default placeholder.")
        print("Please update TIPLOC_DB_PATH at the top of this script with the actual path")
        print("to your TIPLOC SQLite database file before running.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # sys.exit(1) # Optionally exit if not configured

    db_path_result = api_data_preparation_pipeline()
    if db_path_result:
        print(f"\nProcess finished. API Lookup DB is at: {db_path_result}")
    else:
        print("\nProcess finished with errors.")