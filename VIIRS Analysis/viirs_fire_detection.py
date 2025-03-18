import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from io import StringIO

# Configuration
MAP_KEY = "1be8ec47202191da44c455d68bad5edc"  # Replace with your FIRMS MAP_KEY
SENSOR = "VIIRS_NOAA20_NRT"    # Choose from supported sensors
COUNTRY_CODE = "USA"           # Replace with the 3-letter country code (e.g., USA, IND, BRA)

def get_country_fire_data(country_code, days_back=7):
    """
    Fetch VIIRS fire data for a specific country using FIRMS API.
    
    Args:
        country_code (str): 3-letter country code (e.g., USA, IND, BRA).
        days_back (int): Number of days to query (1-10).
        
    Returns:
        pd.DataFrame: Fire detection data as a DataFrame.
    """
    # Construct API URL
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/{SENSOR}/{country_code}/{days_back}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Check for API errors before processing
        if "Invalid" in response.text:
            raise ValueError(f"API Error: {response.text.strip()}")
        
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        print(f"Data download failed: {str(e)}")
        return None

def process_fire_data(df):
    """
    Process and validate VIIRS fire data structure.
    
    Args:
        df (pd.DataFrame): Raw fire detection data.
        
    Returns:
        gpd.GeoDataFrame: Processed fire detections as a GeoDataFrame.
    """
    required_columns = [
        'latitude', 'longitude', 'bright_ti4',
        'frp', 'confidence', 'acq_date', 'acq_time'
    ]
    
    if not set(required_columns).issubset(df.columns):
        missing = set(required_columns) - set(df.columns)
        raise ValueError(f"Missing columns: {missing}\nActual: {df.columns.tolist()}")
    
    # Filter valid detections
    valid_fires = df[
        (df['confidence'].isin(['h', 'n'])) &  # 'h'=high, 'n'=nominal
        (df['bright_ti4'] > 330) &
        (df['frp'] > 5)
    ].copy()
    
    # Convert to GeoDataFrame
    return gpd.GeoDataFrame(
        valid_fires,
        geometry=gpd.points_from_xy(valid_fires.longitude, valid_fires.latitude),
        crs="EPSG:4326"
    )

def main():
    # Query fire data for the last 7 days in the USA
    fire_data = get_country_fire_data(COUNTRY_CODE, days_back=7)
    
    if fire_data is not None:
        try:
            gdf = process_fire_data(fire_data)
            print(f"Processed {len(gdf)} valid fire detections")
            
            # Save processed data to GeoJSON file
            gdf.to_file('viirs_fires_country.geojson', driver='GeoJSON')
            print("Processed fire data saved as 'viirs_fires_country.geojson'")
            
            # Display summary statistics
            print(gdf[['latitude', 'longitude', 'bright_ti4', 'frp']].describe())
        except ValueError as e:
            print(f"Processing error: {str(e)}")
    else:
        print("Failed to retrieve VIIRS fire data.")

if __name__ == "__main__":
    main()
