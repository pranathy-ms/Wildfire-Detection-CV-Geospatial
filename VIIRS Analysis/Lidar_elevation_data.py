import requests
import os
import VIIRS_API_keys as keys

# Configuration
API_KEY = keys.OPEN_TOPOGRAPHY_MAP_KEY
BASE_URL = "https://portal.opentopography.org/API/usgsdem"

def fetch_usgs_3dep(lat_min, lat_max, lon_min, lon_max, resolution="30m", output_dir=os.getcwd()):
    """
    Fetch USGS 3DEP DEM data from OpenTopography API.
    
    Args:
        lat_min (float): Minimum latitude (decimal degrees)
        lat_max (float): Maximum latitude (decimal degrees)
        lon_min (float): Minimum longitude (decimal degrees)
        lon_max (float): Maximum longitude (decimal degrees)
        resolution (str): "1m", "3m", "10m", or "30m"
        output_dir (str): Directory to save the GeoTIFF (default: current directory)
        
    Returns:
        str: Path to downloaded DEM file
    """
    try:
        url = f"{BASE_URL}?datasetName=USGS{resolution}&south={lat_min}&north={lat_max}&west={lon_min}&east={lon_max}&outputFormat=GTiff&API_Key={API_KEY}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            filename = f"USGS3DEP_{resolution}_{lat_min}_{lat_max}_{lon_min}_{lon_max}.tif"
            output_path = os.path.join(output_dir, filename)
            
            with open(output_path, "wb") as f:
                f.write(response.content)
                
            print(f"Successfully downloaded DEM to: {output_path}")
            return output_path
            
        elif response.status_code == 400:
            print(f"API Error (400): Bad request - check parameters")
            
        elif response.status_code == 401:
            print("Authentication failed. Verify your API key at: https://opentopography.org/developers")
            
        else:
            print(f"Unexpected error ({response.status_code})")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        
    return None

# Example usage (saves to current directory)
if __name__ == "__main__":
    la_bbox = {
        "lat_min": 33.5,
        "lat_max": 34.5,
        "lon_min": -119.0,
        "lon_max": -118.0
    }
    
    dem_path = fetch_usgs_3dep(**la_bbox)  # No output_dir specified
