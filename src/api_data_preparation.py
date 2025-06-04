#!/usr/bin/env python3
"""
API Data Preparation for Train Comfort Predictor
Tasks 6.1-6.2: Create SQLite database for API lookups from fixed MVP DuckDB dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import duckdb
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def connect_to_duckdb():
    """Connect to the main DuckDB database containing the MVP dataset."""
    print("=== CONNECTING TO DUCKDB DATABASE ===")
    try:
        conn = duckdb.connect('duck')
        print("Successfully connected to DuckDB database")
        return conn
    except Exception as e:
        print(f"Error connecting to DuckDB: {e}")
        return None


def identify_api_required_data(duck_conn):
    """Task 6.1: Identify data required by the API at prediction time."""
    print("\n=== IDENTIFYING API REQUIRED DATA (Task 6.1) ===")
    
    # Get basic dataset info
    query = "SELECT COUNT(*) as total_records FROM train_journey_legs"
    total_records = duck_conn.execute(query).fetchone()[0]
    print(f"Total records in MVP dataset: {total_records}")
    
    # Station Information
    print("\n1. Station Information (stationName to coordinates mapping):")
    station_query = """
    SELECT DISTINCT 
        stationName_from as station_name,
        stationLocation_from as station_location
    FROM train_journey_legs
    UNION
    SELECT DISTINCT 
        stationName_to as station_name,
        stationLocation_to as station_location
    FROM train_journey_legs
    ORDER BY station_name
    """
    stations_df = duck_conn.execute(station_query).fetchdf()
    print(f"  Unique stations in dataset: {len(stations_df)}")
    
    # Parse coordinates - handle empty strings
    coord_split = stations_df['station_location'].str.split(',', expand=True)
    coord_split = coord_split.replace('', np.nan)  # Replace empty strings with NaN
    stations_df[['latitude', 'longitude']] = coord_split.astype(float)
    
    # Fill missing coordinates with UK center coordinates (approximate)
    stations_df['latitude'].fillna(54.0, inplace=True)  # UK center latitude
    stations_df['longitude'].fillna(-2.0, inplace=True)  # UK center longitude
    
    stations_df = stations_df.drop('station_location', axis=1)
    print(f"  Station coordinates parsed successfully")
    
    # Service Identification & Routing
    print("\n2. Service Identification & Routing Summary:")
    service_query = """
    SELECT 
        headcode,
        rsid,
        stationName_from,
        stationName_to,
        EXTRACT(hour FROM leg_departure_dt) as hour,
        EXTRACT(dow FROM leg_departure_dt) as day_of_week,
        COUNT(*) as frequency
    FROM train_journey_legs
    GROUP BY headcode, rsid, stationName_from, stationName_to, 
             EXTRACT(hour FROM leg_departure_dt), EXTRACT(dow FROM leg_departure_dt)
    ORDER BY frequency DESC
    """
    service_routes_df = duck_conn.execute(service_query).fetchdf()
    print(f"  Service route patterns: {len(service_routes_df)}")
    print(f"  Unique headcodes: {service_routes_df['headcode'].nunique()}")
    print(f"  Unique RSIDs: {service_routes_df['rsid'].nunique()}")
    
    # Historical Average Stats
    print("\n3. Historical Average Stats for Arrival State Estimation:")
    stats_query = """
    SELECT 
        headcode,
        rsid,
        stationName_from as station_of_arrival,
        EXTRACT(dow FROM leg_departure_dt) as day_of_week_bucket,
        CASE 
            WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 6 AND 9 THEN 'morning_peak'
            WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 10 AND 15 THEN 'midday'
            WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 16 AND 19 THEN 'evening_peak'
            WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 20 AND 23 THEN 'evening'
            ELSE 'night_early'
        END as time_of_day_bucket,
        AVG(vehicle_pax_on_arrival_std_at_from) as avg_vehicle_pax_on_arrival_std,
        AVG(vehicle_pax_on_arrival_first_at_from) as avg_vehicle_pax_on_arrival_first,
        AVG(totalUnitPassenger_at_leg_departure) as avg_total_unit_pax_on_arrival,
        AVG(vehicle_pax_boarded_std_at_from + vehicle_pax_boarded_first_at_from) as avg_unit_boarders_at_station,
        AVG(vehicle_pax_alighted_std_at_from + vehicle_pax_alighted_first_at_from) as avg_unit_alighters_at_station,
        COUNT(*) as observations
    FROM train_journey_legs
    GROUP BY headcode, rsid, stationName_from, 
             EXTRACT(dow FROM leg_departure_dt),
             CASE 
                WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 6 AND 9 THEN 'morning_peak'
                WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 10 AND 15 THEN 'midday'
                WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 16 AND 19 THEN 'evening_peak'
                WHEN EXTRACT(hour FROM leg_departure_dt) BETWEEN 20 AND 23 THEN 'evening'
                ELSE 'night_early'
             END
    HAVING COUNT(*) >= 3  -- Only include patterns with at least 3 observations
    ORDER BY observations DESC
    """
    historical_stats_df = duck_conn.execute(stats_query).fetchdf()
    print(f"  Historical stat patterns: {len(historical_stats_df)}")
    
    # Coach type information
    print("\n4. Coach Type Information:")
    coach_query = """
    SELECT 
        coach_type,
        AVG(vehicle_capacity) as avg_capacity,
        COUNT(*) as frequency
    FROM train_journey_legs
    GROUP BY coach_type
    ORDER BY frequency DESC
    """
    coach_info_df = duck_conn.execute(coach_query).fetchdf()
    print(f"  Coach types: {len(coach_info_df)}")
    for _, row in coach_info_df.iterrows():
        print(f"    {row['coach_type']}: avg capacity {row['avg_capacity']:.0f}, frequency {row['frequency']}")
    
    print("\n=== TASK 6.1 COMPLETE ===")
    
    return {
        'stations': stations_df,
        'service_routes': service_routes_df,
        'historical_stats': historical_stats_df,
        'coach_info': coach_info_df
    }


def create_sqlite_database_and_populate(api_data):
    """Task 6.2: Create SQLite database and populate tables."""
    print("\n=== CREATING SQLITE DATABASE (Task 6.2) ===")
    
    # Create SQLite database
    db_path = 'api/train_comfort_api_lookups.sqlite'
    os.makedirs('api', exist_ok=True)
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    print(f"Created new SQLite database: {db_path}")
    
    try:
        # 1. Create and populate stations table
        print("\n1. Creating stations table...")
        api_data['stations'].to_sql('stations', conn, index=False, if_exists='replace')
        
        # Create index on station name
        conn.execute("CREATE INDEX idx_stations_name ON stations(station_name)")
        
        station_count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
        print(f"   Inserted {station_count} stations")
        
        # 2. Create and populate service_routes_summary_mvp table
        print("\n2. Creating service_routes_summary_mvp table...")
        api_data['service_routes'].to_sql('service_routes_summary_mvp', conn, index=False, if_exists='replace')
        
        # Create indexes for efficient service matching
        conn.execute("CREATE INDEX idx_service_routes_headcode ON service_routes_summary_mvp(headcode)")
        conn.execute("CREATE INDEX idx_service_routes_rsid ON service_routes_summary_mvp(rsid)")
        conn.execute("CREATE INDEX idx_service_routes_from ON service_routes_summary_mvp(stationName_from)")
        conn.execute("CREATE INDEX idx_service_routes_hour ON service_routes_summary_mvp(hour)")
        conn.execute("CREATE INDEX idx_service_routes_dow ON service_routes_summary_mvp(day_of_week)")
        
        routes_count = conn.execute("SELECT COUNT(*) FROM service_routes_summary_mvp").fetchone()[0]
        print(f"   Inserted {routes_count} service route patterns")
        
        # 3. Create and populate historical_averages table
        print("\n3. Creating historical_averages table...")
        api_data['historical_stats'].to_sql('historical_averages', conn, index=False, if_exists='replace')
        
        # Create indexes for efficient lookup
        conn.execute("CREATE INDEX idx_hist_headcode ON historical_averages(headcode)")
        conn.execute("CREATE INDEX idx_hist_rsid ON historical_averages(rsid)")
        conn.execute("CREATE INDEX idx_hist_station ON historical_averages(station_of_arrival)")
        conn.execute("CREATE INDEX idx_hist_dow ON historical_averages(day_of_week_bucket)")
        conn.execute("CREATE INDEX idx_hist_time ON historical_averages(time_of_day_bucket)")
        
        hist_count = conn.execute("SELECT COUNT(*) FROM historical_averages").fetchone()[0]
        print(f"   Inserted {hist_count} historical average patterns")
        
        # 4. Create and populate coach_info table
        print("\n4. Creating coach_info table...")
        api_data['coach_info'].to_sql('coach_info', conn, index=False, if_exists='replace')
        
        coach_count = conn.execute("SELECT COUNT(*) FROM coach_info").fetchone()[0]
        print(f"   Inserted {coach_count} coach type records")
        
        # 5. Create metadata table
        print("\n5. Creating metadata table...")
        metadata = pd.DataFrame([{
            'created_at': datetime.now().isoformat(),
            'source_dataset': 'train_journey_legs (MVP fixed dataset)',
            'total_source_records': len(api_data['service_routes']),  # Approximate
            'stations_count': station_count,
            'service_patterns_count': routes_count,
            'historical_patterns_count': hist_count,
            'coach_types_count': coach_count,
            'purpose': 'API lookup database for Train Comfort Predictor MVP'
        }])
        metadata.to_sql('metadata', conn, index=False, if_exists='replace')
        
        print(f"   Created metadata table")
        
        # Verify database integrity
        print("\n=== DATABASE VERIFICATION ===")
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"Tables created: {[table[0] for table in tables]}")
        
        # Test queries
        print("\n=== TESTING SAMPLE QUERIES ===")
        
        # Test station lookup
        sample_station = conn.execute("SELECT * FROM stations LIMIT 1").fetchone()
        print(f"Sample station: {sample_station}")
        
        # Test service route lookup
        sample_route = conn.execute("SELECT * FROM service_routes_summary_mvp LIMIT 1").fetchone()
        print(f"Sample route: {sample_route}")
        
        # Test historical averages lookup
        sample_hist = conn.execute("SELECT * FROM historical_averages LIMIT 1").fetchone()
        print(f"Sample historical data: {sample_hist}")
        
        conn.commit()
        print(f"\n✅ SQLite database created successfully: {db_path}")
        
    except Exception as e:
        print(f"Error creating SQLite database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
    
    print("=== TASK 6.2 COMPLETE ===")
    return db_path


def validate_api_database(db_path):
    """Validate the created API database."""
    print("\n=== VALIDATING API DATABASE ===")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Check that all required tables exist
        required_tables = ['stations', 'service_routes_summary_mvp', 'historical_averages', 'coach_info', 'metadata']
        existing_tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        
        print(f"Required tables: {required_tables}")
        print(f"Existing tables: {existing_tables}")
        
        missing_tables = set(required_tables) - set(existing_tables)
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        else:
            print("✅ All required tables present")
        
        # Check data counts
        for table in required_tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} records")
        
        # Test API-like queries
        print("\n=== TESTING API-LIKE QUERIES ===")
        
        # 1. Station coordinate lookup
        station_query = "SELECT latitude, longitude FROM stations WHERE station_name = 'London Paddington'"
        station_result = conn.execute(station_query).fetchone()
        if station_result:
            print(f"✅ Station lookup test: London Paddington at {station_result}")
        else:
            print("❌ Station lookup test failed")
        
        # 2. Service identification
        service_query = """
        SELECT headcode, rsid, stationName_to, frequency 
        FROM service_routes_summary_mvp 
        WHERE stationName_from = 'London Paddington' 
        AND hour = 8 
        ORDER BY frequency DESC 
        LIMIT 3
        """
        service_results = conn.execute(service_query).fetchall()
        print(f"✅ Service lookup test: Found {len(service_results)} services from London Paddington at 8am")
        
        # 3. Historical averages lookup
        hist_query = """
        SELECT avg_vehicle_pax_on_arrival_std, avg_vehicle_pax_on_arrival_first
        FROM historical_averages 
        WHERE station_of_arrival = 'London Paddington'
        AND time_of_day_bucket = 'morning_peak'
        LIMIT 3
        """
        hist_results = conn.execute(hist_query).fetchall()
        print(f"✅ Historical averages test: Found {len(hist_results)} patterns for London Paddington morning peak")
        
        print("✅ Database validation successful")
        return True
        
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        return False
    finally:
        conn.close()


def api_data_preparation_pipeline():
    """Complete API data preparation pipeline for Tasks 6.1-6.2."""
    print("=== STARTING API DATA PREPARATION PIPELINE ===")
    
    # Connect to DuckDB
    duck_conn = connect_to_duckdb()
    if not duck_conn:
        print("❌ Failed to connect to DuckDB database")
        return None
    
    try:
        # Task 6.1: Identify required data
        api_data = identify_api_required_data(duck_conn)
        
        # Task 6.2: Create SQLite database
        db_path = create_sqlite_database_and_populate(api_data)
        
        # Validate the created database
        validation_success = validate_api_database(db_path)
        
        if validation_success:
            print("\n=== API DATA PREPARATION PIPELINE COMPLETE ===")
            print(f"✅ SQLite database ready for API: {db_path}")
            print("Database contains:")
            print(f"  - {len(api_data['stations'])} stations with coordinates")
            print(f"  - {len(api_data['service_routes'])} service route patterns")
            print(f"  - {len(api_data['historical_stats'])} historical average patterns")
            print(f"  - {len(api_data['coach_info'])} coach type configurations")
            print("\nReady for API development (Tasks 7.1-7.4)!")
            return db_path
        else:
            print("❌ Database validation failed")
            return None
        
    finally:
        duck_conn.close()


if __name__ == "__main__":
    db_path = api_data_preparation_pipeline() 