import os
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from io import StringIO

# Configuration
MAP_KEY = "1be8ec47202191da44c455d68bad5edc"
SENSOR = "VIIRS_NOAA20_NRT"
COUNTRY_CODE = "USA"

def get_country_fire_data(country_code, start_date, end_date):
    """Fetch VIIRS fire data for a specific country and date range"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days_back = (end - start).days + 1
        
        url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/{SENSOR}/{country_code}/{days_back}/{end_date}"
        
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        mask = (df['acq_date'] >= start) & (df['acq_date'] <= end)
        return df[mask]
        
    except Exception as e:
        print(f"Data download failed: {str(e)}")
        return None

def save_to_current_dir(gdf, filename):
    """Save GeoDataFrame to specified filename in current directory"""
    current_dir = os.getcwd()
    print(current_dir)
    output_path = os.path.join(current_dir, filename)
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"File saved to: {output_path}")

if __name__ == "__main__":
    start_date = "2025-01-07"
    end_date = "2025-01-12"
    
    fire_data = get_country_fire_data(COUNTRY_CODE, start_date, end_date)
    
    if fire_data is not None:
        try:
            gdf = gpd.GeoDataFrame(
                fire_data,
                geometry=gpd.points_from_xy(fire_data.longitude, fire_data.latitude),
                crs="EPSG:4326"
            )
            print(f"Found {len(gdf)} fires between {start_date} and {end_date}")
            
            # Save to current directory with explicit path handling
            save_to_current_dir(gdf, 'custom_date_fires.geojson')
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
    else:
        print("Failed to retrieve fire data")
