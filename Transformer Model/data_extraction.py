"""
Unified Data Extraction Module for Wildfire Analysis
Combines VIIRS fire data, ERA5 wind data, and USGS 3DEP elevation data
"""

import os
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
import numpy as np
import richdem as rd
from rasterio.transform import rowcol
from datetime import datetime
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WildfireDataExtractor:
    """Unified data extraction for wildfire analysis"""
    
    def __init__(self, viirs_path: str, era5_path: str, dem_path: str):
        """
        Initialize data extractor
        
        Args:
            viirs_path: Path to VIIRS fire GeoJSON file
            era5_path: Path to ERA5 NetCDF file
            dem_path: Path to DEM GeoTIFF file
        """
        self.viirs_path = viirs_path
        self.era5_path = era5_path
        self.dem_path = dem_path
        
        # Placeholders for loaded data
        self.gdf = None
        self.era5_wind = None
        self.dem = None
        self.slope = None
        self.transform = None
        self.dem_bounds = None
        self.dem_crs = None
        self.dem_height = None
        self.dem_width = None
        
    def load_all_data(self):
        """Load all data sources"""
        logger.info("Loading all data sources...")
        self._load_viirs()
        self._load_era5()
        self._load_dem()
        logger.info("All data loaded successfully")
        
    def _load_viirs(self):
        """Load and prepare VIIRS fire data"""
        logger.info(f"Loading VIIRS data from {self.viirs_path}")
        
        # Load with forced lowercase columns
        self.gdf = gpd.read_file(self.viirs_path).rename(columns=lambda x: x.lower())
        
        # Ensure confidence is lowercase
        if 'confidence' in self.gdf.columns:
            self.gdf['confidence'] = self.gdf['confidence'].str.lower()
            
        logger.info(f"Loaded {len(self.gdf)} VIIRS fire detections")
        logger.info(f"Unique confidence values: {self.gdf['confidence'].unique()}")
        
    def _load_era5(self):
        """Load ERA5 wind data"""
        logger.info(f"Loading ERA5 data from {self.era5_path}")
        
        self.era5_wind = xr.open_dataset(self.era5_path, engine='netcdf4')
        
        # Standardize time coordinate name
        if 'valid_time' in self.era5_wind.coords:
            self.era5_wind = self.era5_wind.rename({'valid_time': 'time'})
            
        logger.info(f"ERA5 variables: {list(self.era5_wind.data_vars)}")
        
    def _load_dem(self):
        """Load DEM and calculate slope"""
        logger.info(f"Loading DEM from {self.dem_path}")
        
        with rasterio.open(self.dem_path) as src:
            self.dem = src.read(1)
            self.transform = src.transform
            self.dem_bounds = src.bounds
            self.dem_crs = src.crs
            self.dem_height, self.dem_width = self.dem.shape
            
        # Convert VIIRS to DEM CRS
        self.gdf = self.gdf.to_crs(self.dem_crs)
        logger.info(f"Converted VIIRS to DEM CRS: {self.dem_crs}")
        
        # Calculate slope
        logger.info("Calculating slope...")
        dem_rd = rd.rdarray(self.dem, no_data=-9999)
        self.slope = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
        logger.info("Slope calculation complete")
        
    def clip_to_dem_bounds(self):
        """Clip VIIRS points to DEM spatial extent"""
        if self.gdf is None or self.dem_bounds is None:
            raise ValueError("Must load data before clipping")
            
        original_count = len(self.gdf)
        
        self.gdf = self.gdf.cx[
            self.dem_bounds.left:self.dem_bounds.right,
            self.dem_bounds.bottom:self.dem_bounds.top
        ]
        
        logger.info(f"Clipped VIIRS: {original_count} → {len(self.gdf)} points")
        
    def extract_features(self) -> pd.DataFrame:
        """
        Extract all features for each fire detection point
        
        Returns:
            DataFrame with columns: latitude, longitude, frp, u10, v10, 
                                   elevation, slope, confidence, confidence_num
        """
        if self.gdf is None:
            raise ValueError("Must load data before extracting features")
            
        logger.info("Extracting features from all data sources...")
        
        features = []
        skipped = 0
        
        for idx, row in self.gdf.iterrows():
            try:
                # Extract wind data
                wind = self._get_wind_at_point(row)
                
                # Extract elevation and slope
                elevation, slope_val = self._get_terrain_at_point(row)
                
                # Skip if terrain data is invalid
                if np.isnan(elevation) or np.isnan(slope_val):
                    skipped += 1
                    continue
                
                features.append({
                    'latitude': row.geometry.y,
                    'longitude': row.geometry.x,
                    'frp': row['frp'],
                    'u10': wind.u10.item(),
                    'v10': wind.v10.item(),
                    'elevation': elevation,
                    'slope': slope_val,
                    'confidence': row['confidence'],
                    'confidence_num': self._map_confidence(row['confidence'])
                })
                
            except Exception as e:
                logger.warning(f"Skipping row {idx}: {str(e)}")
                skipped += 1
                continue
        
        features_df = pd.DataFrame(features)
        
        logger.info(f"Extracted {len(features_df)} valid features ({skipped} skipped)")
        logger.info(f"Elevation range: {features_df['elevation'].min():.1f} - {features_df['elevation'].max():.1f} m")
        logger.info(f"Slope range: {features_df['slope'].min():.1f} - {features_df['slope'].max():.1f}°")
        
        return features_df
    
    def _get_wind_at_point(self, row) -> xr.Dataset:
        """Extract wind data at fire detection point"""
        era5_time = np.datetime64(row['acq_date'])
        
        return self.era5_wind.sel(
            time=era5_time,
            latitude=row.geometry.y,
            longitude=row.geometry.x,
            method='nearest'
        )
    
    def _get_terrain_at_point(self, row) -> Tuple[float, float]:
        """Extract elevation and slope at fire detection point"""
        x, y = row.geometry.x, row.geometry.y
        row_idx, col_idx = rowcol(self.transform, x, y)
        
        # Validate indices
        if (0 <= row_idx < self.dem_height) and (0 <= col_idx < self.dem_width):
            elevation = self.dem[row_idx, col_idx]
            slope_val = self.slope[row_idx, col_idx]
            return elevation, slope_val
        else:
            return np.nan, np.nan
    
    @staticmethod
    def _map_confidence(conf_str: str) -> int:
        """Map confidence string to numerical value"""
        mapping = {'l': 0, 'n': 1, 'h': 2}
        return mapping.get(conf_str, -1)
    
    def get_summary_stats(self, features_df: pd.DataFrame) -> dict:
        """Generate summary statistics"""
        return {
            'total_features': len(features_df),
            'confidence_distribution': features_df['confidence'].value_counts().to_dict(),
            'frp_stats': {
                'mean': features_df['frp'].mean(),
                'max': features_df['frp'].max(),
                'min': features_df['frp'].min()
            },
            'elevation_stats': {
                'mean': features_df['elevation'].mean(),
                'max': features_df['elevation'].max(),
                'min': features_df['elevation'].min()
            },
            'slope_stats': {
                'mean': features_df['slope'].mean(),
                'max': features_df['slope'].max(),
                'min': features_df['slope'].min()
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = WildfireDataExtractor(
        viirs_path='custom_date_fires.geojson',
        era5_path='era5_wind_la.nc',
        dem_path='USGS3DEP_30m_33.5_34.5_-119.0_-118.0.tif'
    )
    
    # Load all data
    extractor.load_all_data()
    
    # Clip to DEM bounds
    extractor.clip_to_dem_bounds()
    
    # Extract features
    features_df = extractor.extract_features()
    
    # Save to CSV
    features_df.to_csv('wildfire_features.csv', index=False)
    logger.info(f"Saved features to wildfire_features.csv")
    
    # Print summary
    stats = extractor.get_summary_stats(features_df)
    print("\n=== Summary Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
