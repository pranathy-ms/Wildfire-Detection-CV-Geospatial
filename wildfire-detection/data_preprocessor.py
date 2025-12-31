"""
Data Processor - Creates 5x5 spatial patches for training
Converts raw fire/elevation/wind data into model-ready features
"""

import sqlite3
import config
import numpy as np
import pandas as pd
import rasterio
import richdem as rd
import xarray as xr
from pathlib import Path
from rasterio.transform import rowcol
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.lat_min = config.LAT_MIN
        self.lat_max = config.LAT_MAX
        self.lon_min = config.LON_MIN
        self.lon_max = config.LON_MAX
        self.grid_size_km = config.GRID_SIZE_KM
        self.patch_size = config.PATCH_SIZE
        
        # Grid parameters
        self.deg_per_km = 0.01
        self.grid_step = self.grid_size_km * self.deg_per_km
        
        # Create grid
        self.lats = np.arange(self.lat_min, self.lat_max, self.grid_step)
        self.lons = np.arange(self.lon_min, self.lon_max, self.grid_step)
        
        print(f"Grid: {len(self.lats)} x {len(self.lons)} = {len(self.lats) * len(self.lons)} cells")
    
    def load_fires(self):
        """Load fire data from database"""
        print("\n📥 Loading fire data...")
        
        conn = sqlite3.connect(config.DB_PATH)
        query = """
            SELECT lat, lon, date, frp, brightness, confidence
            FROM fires
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"  ✓ Loaded {len(df)} fire detections")
        print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def load_elevation(self):
        """Load elevation and calculate slope"""
        print("\n📥 Loading elevation data...")
        
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM elevation_cache LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError("No elevation data found")
        
        dem_file = result[0]
        
        with rasterio.open(dem_file) as src:
            dem = src.read(1)
            transform = src.transform
        
        # Calculate slope using richdem
        dem_rd = rd.rdarray(dem, no_data=-999999)
        slope = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
        
        print(f"  ✓ DEM shape: {dem.shape}")
        print(f"  ✓ Slope calculated")
        
        return dem, slope, transform
    
    def load_wind(self, date):
        """Load wind data for specific date"""
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT file_path FROM wind_cache
            WHERE date = ?
        """, (date,))
        result = cursor.fetchone()
        conn.close()
        
        if not result or not Path(result[0]).exists():
            return None
        
        try:
            wind_data = xr.open_dataset(result[0])
            if 'valid_time' in wind_data.coords:
                wind_data = wind_data.rename({'valid_time': 'time'})
            return wind_data
        except Exception as e:
            print(f"  ⚠️  Wind error for {date}: {e}")
            return None
    
    def precompute_features(self, dem, slope, transform, wind_data_dict):
        """Pre-compute static features for all grid cells"""
        print("\n🔧 Pre-computing features for all grid cells...")
        
        feature_grid = {}
        dem_height, dem_width = dem.shape
        
        for i, lat in enumerate(self.lats):
            for j, lon in enumerate(self.lons):
                # Get terrain features
                row_idx, col_idx = rowcol(transform, lon, lat)
                
                if not (0 <= row_idx < dem_height and 0 <= col_idx < dem_width):
                    continue
                
                elevation = float(dem[row_idx, col_idx])
                slope_val = float(slope[row_idx, col_idx])
                
                # Skip invalid terrain
                if elevation < 0:# or slope_val == 0:
                    continue
                
                # Get wind features (average across all available dates)
                u10_vals, v10_vals = [], []
                for wind_data in wind_data_dict.values():
                    if wind_data is not None:
                        try:
                            wind = wind_data.sel(latitude=lat, longitude=lon, method='nearest')
                            u10_vals.append(float(wind.u10.values.mean()))
                            v10_vals.append(float(wind.v10.values.mean()))
                        except:
                            pass
                
                u10 = np.mean(u10_vals) if u10_vals else 0
                v10 = np.mean(v10_vals) if v10_vals else 0
                wind_speed = np.sqrt(u10**2 + v10**2)
                
                feature_grid[(i, j)] = {
                    'lat': lat,
                    'lon': lon,
                    'elevation': elevation,
                    'slope': slope_val,
                    'u10': u10,
                    'v10': v10,
                    'wind_speed': wind_speed
                }
        
        print(f"  ✓ Computed features for {len(feature_grid)} valid cells")
        return feature_grid
    
    def create_patches(self, fires_df, feature_grid):
        """Create 5x5 spatial patches with labels"""
        print("\n🎯 Creating 5x5 spatial patches...")
        
        dates = sorted(fires_df['date'].unique())
        print(f"  Processing {len(dates)} dates...")
        
        examples = []
        half_patch = self.patch_size // 2
        
        # For each day except last 2 (need 48hr prediction window)
        for day_idx in range(len(dates) - 2):
            current_date = dates[day_idx]
            next_date_1 = dates[day_idx + 1]
            next_date_2 = dates[day_idx + 2]
            
            current_fires = fires_df[fires_df['date'] == current_date]
            next_fires_1 = fires_df[fires_df['date'] == next_date_1]
            next_fires_2 = fires_df[fires_df['date'] == next_date_2]
            
            if len(current_fires) == 0:
                continue
            
            # Combine next 2 days for spread labels
            next_fires = pd.concat([next_fires_1, next_fires_2], ignore_index=True)
            
            # Create fire cell maps
            current_fire_cells = set()
            next_fire_cells = set()
            current_fire_locations = []
            
            for _, fire in current_fires.iterrows():
                lat_idx = int((fire['lat'] - self.lat_min) / self.grid_step)
                lon_idx = int((fire['lon'] - self.lon_min) / self.grid_step)
                if 0 <= lat_idx < len(self.lats) and 0 <= lon_idx < len(self.lons):
                    current_fire_cells.add((lat_idx, lon_idx))
                    current_fire_locations.append((fire['lat'], fire['lon'], fire['frp']))
            
            for _, fire in next_fires.iterrows():
                lat_idx = int((fire['lat'] - self.lat_min) / self.grid_step)
                lon_idx = int((fire['lon'] - self.lon_min) / self.grid_step)
                if 0 <= lat_idx < len(self.lats) and 0 <= lon_idx < len(self.lons):
                    next_fire_cells.add((lat_idx, lon_idx))
            
            # For each burning cell, check neighbors
            day_patches = 0
            for lat_idx, lon_idx in current_fire_cells:
                # Check immediate neighbors (8 directions)
                for dlat in [-1, 0, 1]:
                    for dlon in [-1, 0, 1]:
                        if dlat == 0 and dlon == 0:
                            continue
                        
                        target_lat_idx = lat_idx + dlat
                        target_lon_idx = lon_idx + dlon
                        
                        # Check if target is valid
                        if not (half_patch <= target_lat_idx < len(self.lats) - half_patch and
                                half_patch <= target_lon_idx < len(self.lons) - half_patch):
                            continue
                        
                        # Skip if already burning
                        if (target_lat_idx, target_lon_idx) in current_fire_cells:
                            continue
                        
                        # Label: did fire spread here?
                        spread = 1 if (target_lat_idx, target_lon_idx) in next_fire_cells else 0
                        
                        # Extract 5x5 patch centered on target
                        patch_features = []
                        patch_valid = True
                        
                        for pi in range(-half_patch, half_patch + 1):
                            for pj in range(-half_patch, half_patch + 1):
                                patch_lat_idx = target_lat_idx + pi
                                patch_lon_idx = target_lon_idx + pj
                                
                                if (patch_lat_idx, patch_lon_idx) not in feature_grid:
                                    patch_valid = False
                                    break
                                
                                cell_features = feature_grid[(patch_lat_idx, patch_lon_idx)]
                                
                                # Is this cell burning?
                                is_burning = 1.0 if (patch_lat_idx, patch_lon_idx) in current_fire_cells else 0.0
                                
                                # Distance and direction to nearest fire
                                min_dist = float('inf')
                                fire_direction = 0
                                nearest_frp = 0
                                
                                for fire_lat, fire_lon, frp in current_fire_locations:
                                    dist = np.sqrt((cell_features['lat'] - fire_lat)**2 +
                                                 (cell_features['lon'] - fire_lon)**2) * 111
                                    if dist < min_dist:
                                        min_dist = dist
                                        fire_direction = np.arctan2(
                                            cell_features['lat'] - fire_lat,
                                            cell_features['lon'] - fire_lon
                                        )
                                        nearest_frp = frp
                                
                                # Wind alignment with fire direction
                                wind_direction = np.arctan2(cell_features['v10'], cell_features['u10'])
                                wind_alignment = np.cos(wind_direction - fire_direction)
                                
                                # 7 features per cell
                                patch_features.extend([
                                    cell_features['elevation'],
                                    cell_features['slope'],
                                    cell_features['wind_speed'],
                                    wind_alignment,
                                    min_dist,
                                    nearest_frp,
                                    is_burning
                                ])
                            
                            if not patch_valid:
                                break
                        
                        if not patch_valid or len(patch_features) != self.patch_size * self.patch_size * 7:
                            continue
                        
                        examples.append({
                            'target_lat': self.lats[target_lat_idx],
                            'target_lon': self.lons[target_lon_idx],
                            'date': current_date,
                            'patch_features': patch_features,
                            'spread': spread
                        })
                        day_patches += 1
            
            if day_patches > 0:
                print(f"  {current_date}: {len(current_fires)} fires → {day_patches} patches")
        
        print(f"\n  ✓ Generated {len(examples)} total patches")
        return examples
    
    def process(self):
        """Main processing pipeline"""
        print("=" * 60)
        print("DATA PROCESSING PIPELINE")
        print("=" * 60)
        
        # Load all data
        fires_df = self.load_fires()
        dem, slope, transform = self.load_elevation()
        
        # Load wind for all dates
        print("\n📥 Loading wind data for all dates...")
        dates = sorted(fires_df['date'].unique())
        wind_data_dict = {}
        for date in dates:
            wind_data_dict[date] = self.load_wind(date)
            status = "✓" if wind_data_dict[date] is not None else "✗"
            print(f"  {date}: {status}")
        
        # Pre-compute features
        feature_grid = self.precompute_features(dem, slope, transform, wind_data_dict)
        
        # Create patches
        examples = self.create_patches(fires_df, feature_grid)
        
        if len(examples) == 0:
            print("\n❌ No training examples generated!")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(examples)
        
        # Statistics
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total examples: {len(df)}")
        print(f"Spread = 1: {(df['spread'] == 1).sum()} ({(df['spread'] == 1).mean() * 100:.1f}%)")
        print(f"Spread = 0: {(df['spread'] == 0).sum()} ({(df['spread'] == 0).mean() * 100:.1f}%)")
        print(f"Features per example: {self.patch_size * self.patch_size * 7}")
        
        # Save dataset
        output_file = config.DATA_DIR / "processed" / "training_dataset.pkl"
        output_file.parent.mkdir(exist_ok=True)
        df.to_pickle(output_file)
        
        print(f"\n✅ Dataset saved: {output_file}")
        print("=" * 60)
        
        return df

if __name__ == "__main__":
    processor = DataProcessor()
    dataset = processor.process()