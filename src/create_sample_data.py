#!/usr/bin/env python3
"""
Create sample train journey data for the MVP.
Based on the schema shown in the project documentation.
"""

import duckdb
import pandas as pd
from datetime import datetime, timedelta
import random


def create_sample_train_data():
    """Create sample train journey legs data for MVP development."""
    
    # Create connection to DuckDB
    conn = duckdb.connect('duck')
    
    # Sample data parameters
    num_records = 10000
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Sample stations (major UK stations)
    stations = [
        ('London Paddington', '51.5154,-0.1755'),
        ('London Victoria', '51.4952,-0.1441'),
        ('London Waterloo', '51.5036,-0.1143'),
        ('Birmingham New Street', '52.4778,-1.8987'),
        ('Manchester Piccadilly', '53.4775,-2.2308'),
        ('Leeds', '53.7955,-1.5491'),
        ('Glasgow Central', '55.8588,-4.2577'),
        ('Edinburgh Waverley', '55.9521,-3.1900'),
        ('Bristol Temple Meads', '51.4489,-2.5813'),
        ('Cardiff Central', '51.4755,-3.1786'),
        ('Newcastle', '54.9662,-1.6117'),
        ('Liverpool Lime Street', '53.4074,-2.9772'),
        ('Reading', '51.4587,-0.9700'),
        ('Oxford', '51.7537,-1.2684'),
        ('Cambridge', '52.1943,0.1371')
    ]
    
    # Sample coach types
    coach_types = ['Standard', 'First Class', 'Mixed']
    
    # Sample service codes
    headcodes = ['1A01', '1B02', '1C03', '2A01', '2B02', '2C03', '5A01', '5B02']
    rsids = ['GW100001', 'GW100002', 'SW200001', 'SW200002',
             'VT300001', 'VT300002']
    
    # Generate sample data
    records = []
    
    for i in range(num_records):
        # Random timestamp within the year
        random_days = random.randint(0, (end_date - start_date).days)
        random_hours = random.randint(5, 23)  # Typical train operating hours
        random_minutes = random.choice([0, 15, 30, 45])  # Typical departure times
        
        leg_departure = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
        
        # Next station departure (typically 20-60 minutes later)
        next_departure = leg_departure + timedelta(minutes=random.randint(20, 60))
        
        # Random stations
        from_station, from_location = random.choice(stations)
        to_station, to_location = random.choice([s for s in stations if s[0] != from_station])
        
        # Vehicle capacity based on coach type
        coach_type = random.choice(coach_types)
        if coach_type == 'First Class':
            capacity = random.randint(40, 80)
        elif coach_type == 'Standard':
            capacity = random.randint(60, 120)
        else:  # Mixed
            capacity = random.randint(80, 150)
        
        # Passenger counts (realistic occupancy patterns)
        base_occupancy = random.uniform(0.2, 0.9)  # 20-90% occupancy
        
        vehicle_pax_on_arrival_std = int(capacity * base_occupancy * random.uniform(0.7, 1.3))
        vehicle_pax_on_arrival_first = int(capacity * 0.2 * base_occupancy * random.uniform(0.5, 1.5)) if coach_type != 'Standard' else 0
        
        # Boarding and alighting (realistic flow)
        alighting_rate = random.uniform(0.1, 0.4)  # 10-40% of passengers alight
        boarding_rate = random.uniform(0.05, 0.3)   # 5-30% new passengers board
        
        vehicle_pax_alighted_std = int(vehicle_pax_on_arrival_std * alighting_rate)
        vehicle_pax_boarded_std = int(capacity * boarding_rate)
        
        vehicle_pax_alighted_first = int(vehicle_pax_on_arrival_first * alighting_rate) if vehicle_pax_on_arrival_first > 0 else 0
        vehicle_pax_boarded_first = int(capacity * 0.2 * boarding_rate) if coach_type != 'Standard' else 0
        
        # Unit-level aggregates
        on_unit = random.randint(5, 25)  # Passengers boarding at station
        off_unit = random.randint(3, 20)  # Passengers alighting at station
        total_unit = vehicle_pax_on_arrival_std + vehicle_pax_on_arrival_first + on_unit - off_unit
        
        # Relevant passengers (those continuing on this leg)
        relevant_passengers = vehicle_pax_on_arrival_std + vehicle_pax_on_arrival_first - vehicle_pax_alighted_std - vehicle_pax_alighted_first + vehicle_pax_boarded_std + vehicle_pax_boarded_first
        
        record = {
            'coach_type': coach_type,
            'driverId': f'DR{random.randint(1000, 9999)}',
            'headcode': random.choice(headcodes),
            'leg_departure_dt': leg_departure,
            'messageId_leg_start': f'MSG{i:06d}',
            'next_station_departure_dt': next_departure,
            'offUnitPassenger_at_from_station': off_unit,
            'onUnitPassenger_at_from_station': on_unit,
            'relevant_passengers_on_leg_departure': float(relevant_passengers),
            'rsid': random.choice(rsids),
            'stationLocation_from': from_location,
            'stationLocation_to': to_location,
            'stationName_from': from_station,
            'stationName_to': to_station,
            'totalUnitPassenger_at_leg_departure': total_unit,
            'unitId': f'UNIT{random.randint(100, 999)}',
            'vehicle_capacity': float(capacity),
            'vehicle_pax_alighted_first_at_from': float(vehicle_pax_alighted_first),
            'vehicle_pax_alighted_std_at_from': float(vehicle_pax_alighted_std),
            'vehicle_pax_boarded_first_at_from': float(vehicle_pax_boarded_first),
            'vehicle_pax_boarded_std_at_from': float(vehicle_pax_boarded_std),
            'vehicle_pax_on_arrival_first_at_from': float(vehicle_pax_on_arrival_first),
            'vehicle_pax_on_arrival_std_at_from': float(vehicle_pax_on_arrival_std),
            'vehicleNo': f'VEH{random.randint(10000, 99999)}'
        }
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Drop table if exists and create new one with data
    conn.execute("DROP TABLE IF EXISTS train_journey_legs")
    conn.execute("CREATE TABLE train_journey_legs AS SELECT * FROM df")
    
    # Verify data
    result = conn.execute("SELECT COUNT(*) FROM train_journey_legs").fetchone()
    print(f"Created sample database with {result[0]} records")
    
    # Show sample data
    sample = conn.execute("SELECT * FROM train_journey_legs LIMIT 5").fetchdf()
    print("\nSample data:")
    print(sample)
    
    conn.close()

if __name__ == "__main__":
    create_sample_train_data() 