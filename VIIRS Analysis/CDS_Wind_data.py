import cdsapi
import os

# Configuration
start_date = "2025-01-07"
lat_min = 33.5
lat_max = 34.5
lon_min = -119.0
lon_max = -118.0

# Set output path to current working directory
output_dir = os.getcwd()
output_file = os.path.join(output_dir, "era5_wind_la.nc")

# Initialize CDS API client
c = cdsapi.Client()

# Download ERA5 wind data
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind'
        ],
        'year': start_date[:4],
        'month': start_date[5:7],
        'day': start_date[8:10],
        'time': [
            '00:00', '06:00', '12:00', '18:00'
        ],
        'area': [
            lat_max, lon_min,  # North-West corner
            lat_min, lon_max   # South-East corner
        ],
        'format': 'netcdf'
    },
    output_file
)

print(f"Data saved to: {output_file}")
