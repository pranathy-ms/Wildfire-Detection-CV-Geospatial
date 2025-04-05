import requests
import os
import VIIRS_API_keys as keys

# Configuration - Replace with your valid API key
API_KEY = keys.OPEN_TOPOGRAPHY_MAP_KEY
BASE_URL = "https://portal.opentopography.org/API/usgsdem?"

def fetch_usgs_3dep(lat_min, lat_max, lon_min, lon_max, resolution="10m", output_dir="."):
    """
    Fetch USGS 3DEP DEM data from OpenTopography API.
    
    Args:
        lat_min (float): Minimum latitude (decimal degrees)
        lat_max (float): Maximum latitude (decimal degrees)
        lon_min (float): Minimum longitude (decimal degrees)
        lon_max (float): Maximum longitude (decimal degrees)
        resolution (str): "1m", "10m", or "30m"
        output_dir (str): Directory to save the GeoTIFF
        
    Returns:
        str: Path to downloaded DEM file
    """
    try:
        params = {
            "demtype": "USGS{}".format(resolution),
            "south": lat_min,
            "north": lat_max,
            "west": lon_min,
            "east": lon_max,
            "outputFormat": "GTiff",
            "API_Key": API_KEY,
            "csr": "4326"  # Coordinate Reference System (WGS84)
        }
        
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"USGS3DEP_{resolution}_{lat_min}_{lat_max}_{lon_min}_{lon_max}.tif"
            output_path = os.path.join(output_dir, filename)
            
            with open(output_path, "wb") as f:
                f.write(response.content)
                
            print(f"Successfully downloaded DEM to: {output_path}")
            return output_path
            
        elif response.status_code == 400:
            error_msg = response.json().get('error', {}).get('message', 'Bad request')
            print(f"API Error (400): {error_msg}")
            
        elif response.status_code == 401:
            print("Authentication failed. Verify your API key at: https://opentopography.org/developers")
            
        else:
            print(f"Unexpected error ({response.status_code}): {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        
    return None

# Example usage for Los Angeles area
if __name__ == "__main__":
    # LA bounding box coordinates
    la_bbox = {
        "lat_min": 33.5,
        "lat_max": 34.5,
        "lon_min": -119.0,
        "lon_max": -118.0
    }
    
    # Fetch 10m resolution DEM
    dem_path = fetch_usgs_3dep(
        resolution="10m",
        output_dir="./elevation_data",
        **la_bbox
    )
    
    if dem_path:
        print(f"DEM file ready for analysis at: {dem_path}")
