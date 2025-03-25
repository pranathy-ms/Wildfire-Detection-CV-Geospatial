import os
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import folium
from io import StringIO
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler

# Configuration
MAP_KEY = "1be8ec47202191da44c455d68bad5edc"
SENSOR = "VIIRS_NOAA20_NRT"
COUNTRY_CODE = "USA"
DB_PATH = os.path.join(os.path.dirname(__file__), "wildfire_db.sqlite")
UPDATE_INTERVAL = 300  # 5 minutes in seconds

def init_database():
    """Initialize SQLite database with spatial support"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Enable spatialite extension
    c.execute("SELECT load_extension('mod_spatialite')")
    
    # Create fires table
    c.execute('''
        CREATE TABLE IF NOT EXISTS fires (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            bright_ti4 REAL,
            frp REAL,
            confidence TEXT,
            acq_date DATETIME,
            geometry GEOMETRY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create spatial index
    c.execute("SELECT CreateSpatialIndex('fires', 'geometry')")
    conn.commit()
    conn.close()

def fetch_new_fires(last_fetch_time):
    """Fetch new fire data from FIRMS API"""
    try:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/{SENSOR}/{COUNTRY_CODE}/1"
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        
        # Filter new fires
        new_fires = df[df['acq_date'] > last_fetch_time]
        return new_fires
    
    except Exception as e:
        print(f"API Error: {str(e)}")
        return pd.DataFrame()

def ingest_data():
    """Main data ingestion workflow"""
    conn = sqlite3.connect(DB_PATH)
    last_fetch = pd.read_sql("SELECT MAX(acq_date) as last FROM fires", conn).iloc[0]['last']
    last_fetch_time = pd.to_datetime(last_fetch) if last_fetch else datetime.now() - timedelta(hours=1)
    
    new_data = fetch_new_fires(last_fetch_time)
    
    if not new_data.empty:
        gdf = gpd.GeoDataFrame(
            new_data,
            geometry=gpd.points_from_xy(new_data.longitude, new_data.latitude),
            crs="EPSG:4326"
        )
        
        # Store in SQLite with spatial indexing
        gdf.to_postgis("fires", f"sqlite:///{DB_PATH}", if_exists='append')
        print(f"Ingested {len(gdf)} new fire detections")
    
    conn.close()

def streamlit_app():
    """Streamlit web application component"""
    import streamlit as st
    from streamlit_folium import folium_static
    
    st.title("Real-time Wildfire Monitoring")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Get latest data
    gdf = gpd.read_postgis('''
        SELECT *, GeomFromWKB(geometry) as geometry 
        FROM fires 
        ORDER BY acq_date DESC 
        LIMIT 1000
    ''', conn)
    
    if not gdf.empty:
        # Create map
        m = folium.Map(
            location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()],
            zoom_start=4,
            tiles='CartoDB dark_matter'
        )
        
        # Add time-slider visualization
        from folium.plugins import TimeSliderChoropleth
        
        styledict = {
            str(idx): {
                'fillColor': _get_fire_color(row),
                'radius': row['frp']/10
            } for idx, row in gdf.iterrows()
        }
        
        TimeSliderChoropleth(
            gdf.to_json(),
            styledict=styledict,
            init_timestamp=-1
        ).add_to(m)
        
        folium_static(m)
        
        # Display statistics
        st.metric("Total Active Fires", len(gdf))
        st.line_chart(gdf.set_index('acq_date')['frp'])
        
    conn.close()

def _get_fire_color(row):
    """Get color based on fire intensity and confidence"""
    if row['confidence'] == 'h' and row['frp'] > 50:
        return '#ff0000'  # High confidence, high FRP
    elif row['confidence'] == 'h':
        return '#ff4500'  # High confidence
    else:
        return '#ffd700'  # Nominal confidence

if __name__ == "__main__":
    # Initialize database
    init_database()
    
    # Set up scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(ingest_data, 'interval', seconds=UPDATE_INTERVAL)
    scheduler.start()
    
    # Run Streamlit app
    import subprocess
    subprocess.run(["streamlit", "run", __file__])
