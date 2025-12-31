import sqlite3
import config

def init_database():
    """Create the database and tables"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    # Fire detections - date-aware
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fires (
            id INTEGER PRIMARY KEY,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            date TEXT NOT NULL,
            time TEXT,
            frp REAL,
            brightness REAL,
            confidence TEXT,
            daynight TEXT,
            satellite TEXT,
            UNIQUE(lat, lon, date, time)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_fires_location 
        ON fires(lat, lon, date)
    """)
    
    # Elevation data cache - ONE per area (no date)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS elevation_cache (
            id INTEGER PRIMARY KEY,
            lat_min REAL,
            lat_max REAL,
            lon_min REAL,
            lon_max REAL,
            file_path TEXT,
            downloaded_at TEXT,
            metadata TEXT,
            UNIQUE(lat_min, lat_max, lon_min, lon_max)
        )
    """)
    
    # Wind data cache - date-aware (one per day)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wind_cache (
            id INTEGER PRIMARY KEY,
            date TEXT,
            lat_min REAL,
            lat_max REAL,
            lon_min REAL,
            lon_max REAL,
            file_path TEXT,
            downloaded_at TEXT,
            UNIQUE(date, lat_min, lat_max, lon_min, lon_max)
        )
    """)
    
    # Predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            lat REAL,
            lon REAL,
            date TEXT,
            risk_level TEXT,
            probability REAL
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✓ Database created at: {config.DB_PATH}")

if __name__ == "__main__":
    init_database()