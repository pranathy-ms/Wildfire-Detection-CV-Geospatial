"""
WILDFIRE SPREAD PREDICTION - TRANSFORMER MODEL
Clean implementation: Multi-day data → Spatial grid → Predict spread

Goal: Predict which cells near active fires will burn in next 48 hours
"""

import os
import requests
import cdsapi
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import richdem as rd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from rasterio.transform import rowcol
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

try:
    import VIIRS_API_keys as keys
    FIRMS_KEY = keys.MAP_KEY
    OPENTOPO_KEY = keys.OPEN_TOPOGRAPHY_MAP_KEY
except:
    FIRMS_KEY = "YOUR_FIRMS_KEY_HERE"
    OPENTOPO_KEY = "YOUR_OPENTOPO_KEY_HERE"

# Study area and time period
LAT_MIN = 33.5
LAT_MAX = 34.5
LON_MIN = -119.0
LON_MAX = -118.0
START_DATE = "2025-01-07"
NUM_DAYS = 14  # Increased from 8 to 14

# Grid parameters
GRID_SIZE_KM = 1.0  # 1km x 1km cells
PATCH_SIZE = 5  # Extract 5x5 patches (25 cells) around each target
# Center cell of patch is the target we're predicting

# Transformer parameters
D_MODEL = 128
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 32


# ============================================================================
# STEP 1: DOWNLOAD MULTI-DAY DATA
# ============================================================================

def download_multiday_data():
    """Download fire, wind, and elevation data for 14 days"""
    logger.info("="*60)
    logger.info("STEP 1: DOWNLOADING DATA")
    logger.info("="*60)
    
    os.makedirs('data/multiday', exist_ok=True)
    
    # Generate date list
    start = datetime.strptime(START_DATE, '%Y-%m-%d')
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(NUM_DAYS)]
    
    logger.info(f"\nDate range: {dates[0]} to {dates[-1]} ({NUM_DAYS} days)")
    
    # Download VIIRS for each day
    logger.info("\n1.1 Downloading fire detections...")
    fire_files = []
    
    for date in dates:
        file_path = f'data/multiday/fires_{date}.csv'
        
        if Path(file_path).exists():
            logger.info(f"  ✓ Using existing {date}")
            fire_files.append(file_path)
            continue
        
        try:
            query_date = datetime.strptime(date, '%Y-%m-%d')
            days_ago = (datetime.now() - query_date).days
            sensor = "VIIRS_NOAA20_SP" if days_ago > 105 else "VIIRS_NOAA20_NRT"
            
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/{sensor}/{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}/1/{date}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'w') as f:
                f.write(response.text)
            
            df = pd.read_csv(file_path)
            logger.info(f"  ✓ {date}: {len(df)} detections")
            fire_files.append(file_path)
            
        except Exception as e:
            logger.warning(f"  ✗ {date}: {e}")
            fire_files.append(None)
    
    # Download elevation (once)
    logger.info("\n1.2 Downloading elevation data...")
    dem_file = 'data/multiday/elevation.tif'
    
    if Path(dem_file).exists():
        logger.info("  ✓ Using existing elevation.tif")
    else:
        try:
            url = f"https://portal.opentopography.org/API/usgsdem?datasetName=USGS30m&south={LAT_MIN}&north={LAT_MAX}&west={LON_MIN}&east={LON_MAX}&outputFormat=GTiff&API_Key={OPENTOPO_KEY}"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(dem_file, 'wb') as f:
                f.write(response.content)
            logger.info("  ✓ Downloaded elevation.tif")
        except Exception as e:
            logger.error(f"  ✗ Elevation download failed: {e}")
            return None
    
    # Download wind
    logger.info("\n1.3 Downloading wind data...")
    wind_file = 'data/multiday/wind.nc'
    
    if Path(wind_file).exists():
        logger.info("  ✓ Using existing wind.nc")
    else:
        try:
            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                    'year': START_DATE[:4],
                    'month': START_DATE[5:7],
                    'day': START_DATE[8:10],
                    'time': ['00:00', '06:00', '12:00', '18:00'],
                    'area': [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
                    'format': 'netcdf'
                },
                wind_file
            )
            logger.info("  ✓ Downloaded wind.nc")
        except Exception as e:
            logger.warning(f"  ✗ Wind download failed: {e}")
    
    logger.info("\n✓ Data download complete!")
    return {'fires': fire_files, 'dem': dem_file, 'wind': wind_file, 'dates': dates}


# ============================================================================
# STEP 2: CREATE SPATIAL GRID & LABELS
# ============================================================================

def create_grid_and_labels(data_files):
    """Create spatial grid and extract 5x5 patches for spatial context"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: CREATING SPATIAL GRID & PATCHES")
    logger.info("="*60)
    
    dates = data_files['dates']
    fire_files = data_files['fires']
    
    # Load all fire data
    logger.info("\n2.1 Loading fire detections...")
    all_fires = []
    for date, file_path in zip(dates, fire_files):
        if file_path and Path(file_path).exists():
            df = pd.read_csv(file_path)
            if len(df) > 0:
                df['date'] = date
                all_fires.append(df)
    
    if not all_fires:
        logger.error("No fire data available!")
        return None
    
    fires_df = pd.concat(all_fires, ignore_index=True)
    logger.info(f"  Total detections: {len(fires_df)} across {len(all_fires)} days")
    
    # Create spatial grid
    logger.info("\n2.2 Creating spatial grid...")
    deg_per_km = 0.01
    grid_step = GRID_SIZE_KM * deg_per_km
    
    lats = np.arange(LAT_MIN, LAT_MAX, grid_step)
    lons = np.arange(LON_MIN, LON_MAX, grid_step)
    
    logger.info(f"  Grid dimensions: {len(lats)} x {len(lons)} = {len(lats)*len(lons)} cells")
    
    # Load terrain data
    logger.info("\n2.3 Loading terrain data...")
    with rasterio.open(data_files['dem']) as src:
        dem = src.read(1)
        transform = src.transform
        dem_height, dem_width = dem.shape
    
    # Calculate slope
    dem_rd = rd.rdarray(dem, no_data=-9999)
    slope = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
    logger.info("  ✓ Calculated slope")
    
    # Load wind
    if Path(data_files['wind']).exists():
        wind_data = xr.open_dataset(data_files['wind'], engine='netcdf4')
        if 'valid_time' in wind_data.coords:
            wind_data = wind_data.rename({'valid_time': 'time'})
        logger.info("  ✓ Loaded wind data")
    else:
        wind_data = None
        logger.warning("  ⚠ No wind data")
    
    # Build feature grid (pre-compute features for all cells)
    logger.info("\n2.4 Pre-computing features for all grid cells...")
    feature_grid = {}
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Get terrain features
            row_idx, col_idx = rowcol(transform, lon, lat)
            if not (0 <= row_idx < dem_height and 0 <= col_idx < dem_width):
                continue
            
            elevation = float(dem[row_idx, col_idx])
            slope_val = float(slope[row_idx, col_idx])
            
            # Skip invalid terrain
            if elevation < 0 or slope_val == 0:
                continue
            
            # Get wind features
            if wind_data:
                try:
                    wind = wind_data.sel(latitude=lat, longitude=lon, method='nearest')
                    u10 = float(wind.u10.values.mean())
                    v10 = float(wind.v10.values.mean())
                    wind_speed = np.sqrt(u10**2 + v10**2)
                except:
                    u10, v10, wind_speed = 0, 0, 0
            else:
                u10, v10, wind_speed = 0, 0, 0
            
            feature_grid[(i, j)] = {
                'lat': lat,
                'lon': lon,
                'elevation': elevation,
                'slope': slope_val,
                'u10': u10,
                'v10': v10,
                'wind_speed': wind_speed
            }
    
    logger.info(f"  ✓ Computed features for {len(feature_grid)} valid cells")
    
    # Generate training examples with 5x5 spatial patches
    logger.info("\n2.5 Extracting 5x5 spatial patches (48hr window)...")
    
    examples = []
    half_patch = PATCH_SIZE // 2  # 2 cells on each side of center
    
    # For each day except last 2
    for day_idx in range(len(dates) - 2):
        current_date = dates[day_idx]
        next_date_1 = dates[day_idx + 1]
        next_date_2 = dates[day_idx + 2]
        
        current_fires = fires_df[fires_df['date'] == current_date]
        next_fires_1 = fires_df[fires_df['date'] == next_date_1]
        next_fires_2 = fires_df[fires_df['date'] == next_date_2]
        
        if len(current_fires) == 0:
            continue
        
        logger.info(f"  {current_date}: {len(current_fires)} → {len(next_fires_1)}(+1d), {len(next_fires_2)}(+2d)")
        
        # Combine next 2 days for spread labels
        next_fires = pd.concat([next_fires_1, next_fires_2], ignore_index=True)
        
        # Create fire maps
        current_fire_cells = set()
        next_fire_cells = set()
        current_fire_locations = []
        
        for _, fire in current_fires.iterrows():
            lat_idx = int((fire['latitude'] - LAT_MIN) / grid_step)
            lon_idx = int((fire['longitude'] - LON_MIN) / grid_step)
            if 0 <= lat_idx < len(lats) and 0 <= lon_idx < len(lons):
                current_fire_cells.add((lat_idx, lon_idx))
                current_fire_locations.append((fire['latitude'], fire['longitude'], fire.get('frp', 0)))
        
        for _, fire in next_fires.iterrows():
            lat_idx = int((fire['latitude'] - LAT_MIN) / grid_step)
            lon_idx = int((fire['longitude'] - LON_MIN) / grid_step)
            if 0 <= lat_idx < len(lats) and 0 <= lon_idx < len(lons):
                next_fire_cells.add((lat_idx, lon_idx))
        
        # For each burning cell, check neighbors and extract 5x5 patches
        for lat_idx, lon_idx in current_fire_cells:
            # Check immediate neighbors (1 cell away)
            for dlat in [-1, 0, 1]:
                for dlon in [-1, 0, 1]:
                    if dlat == 0 and dlon == 0:
                        continue
                    
                    target_lat_idx = lat_idx + dlat
                    target_lon_idx = lon_idx + dlon
                    
                    # Check if target is valid and not already burning
                    if not (half_patch <= target_lat_idx < len(lats) - half_patch and 
                            half_patch <= target_lon_idx < len(lons) - half_patch):
                        continue
                    
                    if (target_lat_idx, target_lon_idx) in current_fire_cells:
                        continue
                    
                    # Label: did fire spread to this target cell?
                    spread = 1 if (target_lat_idx, target_lon_idx) in next_fire_cells else 0
                    
                    # Extract 5x5 patch centered on target cell
                    patch_features = []
                    patch_valid = True
                    
                    for pi in range(-half_patch, half_patch + 1):
                        for pj in range(-half_patch, half_patch + 1):
                            patch_lat_idx = target_lat_idx + pi
                            patch_lon_idx = target_lon_idx + pj
                            
                            # Check if this patch cell has features
                            if (patch_lat_idx, patch_lon_idx) not in feature_grid:
                                patch_valid = False
                                break
                            
                            cell_features = feature_grid[(patch_lat_idx, patch_lon_idx)]
                            
                            # Is this patch cell currently on fire? (binary feature)
                            is_burning = 1.0 if (patch_lat_idx, patch_lon_idx) in current_fire_cells else 0.0
                            
                            # Calculate distance and direction from this patch cell to nearest fire
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
                            
                            # Wind alignment: is wind blowing toward this cell from fire?
                            wind_direction = np.arctan2(cell_features['v10'], cell_features['u10'])
                            wind_alignment = np.cos(wind_direction - fire_direction)
                            
                            # 7 features per cell in patch
                            patch_features.extend([
                                cell_features['elevation'],
                                cell_features['slope'],
                                cell_features['wind_speed'],
                                wind_alignment,
                                min_dist,
                                nearest_frp,
                                is_burning  # NEW: Binary flag if cell is burning
                            ])
                        
                        if not patch_valid:
                            break
                    
                    if not patch_valid or len(patch_features) != PATCH_SIZE * PATCH_SIZE * 7:
                        continue
                    
                    # Store example with 5x5 spatial context
                    examples.append({
                        'target_lat': lats[target_lat_idx],
                        'target_lon': lons[target_lon_idx],
                        'patch_features': patch_features,  # List of 175 features (25 cells × 7 features)
                        'spread': spread
                    })
    
    logger.info(f"\n✓ Generated {len(examples)} spatial patches")
    
    # Convert to DataFrame
    df = pd.DataFrame(examples)
    
    if len(df) == 0:
        logger.error("No valid patches generated!")
        return None
    
    logger.info(f"  Spread=1: {(df['spread']==1).sum()} ({(df['spread']==1).mean()*100:.1f}%)")
    logger.info(f"  Spread=0: {(df['spread']==0).sum()} ({(df['spread']==0).mean()*100:.1f}%)")
    
    df.to_pickle('data/multiday/spatial_patches.pkl')  # Use pickle for list columns
    return df
    
    dates = data_files['dates']
    fire_files = data_files['fires']
    
    # Load all fire data
    logger.info("\n2.1 Loading fire detections...")
    all_fires = []
    for date, file_path in zip(dates, fire_files):
        if file_path and Path(file_path).exists():
            df = pd.read_csv(file_path)
            if len(df) > 0:
                df['date'] = date
                all_fires.append(df)
    
    if not all_fires:
        logger.error("No fire data available!")
        return None
    
    fires_df = pd.concat(all_fires, ignore_index=True)
    logger.info(f"  Total detections: {len(fires_df)} across {len(all_fires)} days")
    
    # Create spatial grid
    logger.info("\n2.2 Creating spatial grid...")
    deg_per_km = 0.01
    grid_step = GRID_SIZE_KM * deg_per_km
    
    lats = np.arange(LAT_MIN, LAT_MAX, grid_step)
    lons = np.arange(LON_MIN, LON_MAX, grid_step)
    
    logger.info(f"  Grid dimensions: {len(lats)} x {len(lons)} = {len(lats)*len(lons)} cells")
    
    # Load terrain data
    logger.info("\n2.3 Loading terrain data...")
    with rasterio.open(data_files['dem']) as src:
        dem = src.read(1)
        transform = src.transform
        dem_height, dem_width = dem.shape
    
    # Calculate slope
    dem_rd = rd.rdarray(dem, no_data=-9999)
    slope = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
    logger.info("  ✓ Calculated slope")
    
    # Load wind
    if Path(data_files['wind']).exists():
        wind_data = xr.open_dataset(data_files['wind'], engine='netcdf4')
        if 'valid_time' in wind_data.coords:
            wind_data = wind_data.rename({'valid_time': 'time'})
        logger.info("  ✓ Loaded wind data")
    else:
        wind_data = None
        logger.warning("  ⚠ No wind data")
    
    # Generate training examples
    logger.info("\n2.4 Generating spread labels (48hr window, 5x5 neighborhood)...")
    
    examples = []
    
    # For each day except last 2
    for day_idx in range(len(dates) - 2):
        current_date = dates[day_idx]
        next_date_1 = dates[day_idx + 1]
        next_date_2 = dates[day_idx + 2]
        
        current_fires = fires_df[fires_df['date'] == current_date]
        next_fires_1 = fires_df[fires_df['date'] == next_date_1]
        next_fires_2 = fires_df[fires_df['date'] == next_date_2]
        
        if len(current_fires) == 0:
            continue
        
        logger.info(f"  {current_date}: {len(current_fires)} → {len(next_fires_1)}(+1d), {len(next_fires_2)}(+2d)")
        
        # Combine next 2 days for spread labels
        next_fires = pd.concat([next_fires_1, next_fires_2], ignore_index=True)
        
        # Create fire maps
        current_fire_cells = set()
        next_fire_cells = set()
        current_fire_locations = []
        
        for _, fire in current_fires.iterrows():
            lat_idx = int((fire['latitude'] - LAT_MIN) / grid_step)
            lon_idx = int((fire['longitude'] - LON_MIN) / grid_step)
            if 0 <= lat_idx < len(lats) and 0 <= lon_idx < len(lons):
                current_fire_cells.add((lat_idx, lon_idx))
                current_fire_locations.append((fire['latitude'], fire['longitude'], fire.get('frp', 0)))
        
        for _, fire in next_fires.iterrows():
            lat_idx = int((fire['latitude'] - LAT_MIN) / grid_step)
            lon_idx = int((fire['longitude'] - LON_MIN) / grid_step)
            if 0 <= lat_idx < len(lats) and 0 <= lon_idx < len(lons):
                next_fire_cells.add((lat_idx, lon_idx))
        
        # Check 5x5 neighborhood (24 neighbors) around each burning cell
        for lat_idx, lon_idx in current_fire_cells:
            for dlat in [-2, -1, 0, 1, 2]:
                for dlon in [-2, -1, 0, 1, 2]:
                    if dlat == 0 and dlon == 0:
                        continue
                    
                    neighbor_lat_idx = lat_idx + dlat
                    neighbor_lon_idx = lon_idx + dlon
                    
                    if not (0 <= neighbor_lat_idx < len(lats) and 0 <= neighbor_lon_idx < len(lons)):
                        continue
                    
                    if (neighbor_lat_idx, neighbor_lon_idx) in current_fire_cells:
                        continue
                    
                    # Label: did fire spread here in 48hrs?
                    spread = 1 if (neighbor_lat_idx, neighbor_lon_idx) in next_fire_cells else 0
                    
                    # Extract features
                    neighbor_lat = LAT_MIN + neighbor_lat_idx * grid_step
                    neighbor_lon = LON_MIN + neighbor_lon_idx * grid_step
                    
                    # Terrain
                    row_idx, col_idx = rowcol(transform, neighbor_lon, neighbor_lat)
                    if not (0 <= row_idx < dem_height and 0 <= col_idx < dem_width):
                        continue
                    
                    elevation = float(dem[row_idx, col_idx])
                    slope_val = float(slope[row_idx, col_idx])
                    
                    # Skip invalid terrain
                    if elevation < 0 or slope_val == 0:
                        continue
                    
                    # Wind
                    if wind_data:
                        try:
                            wind = wind_data.sel(latitude=neighbor_lat, longitude=neighbor_lon, method='nearest')
                            u10 = float(wind.u10.values.mean())
                            v10 = float(wind.v10.values.mean())
                        except:
                            u10, v10 = 0, 0
                    else:
                        u10, v10 = 0, 0
                    
                    # Calculate distance and direction to nearest fire
                    min_distance = float('inf')
                    nearest_frp = 0
                    fire_direction = 0
                    
                    for fire_lat, fire_lon, frp in current_fire_locations:
                        dist = np.sqrt((neighbor_lat - fire_lat)**2 + (neighbor_lon - fire_lon)**2) * 111
                        if dist < min_distance:
                            min_distance = dist
                            nearest_frp = frp
                            fire_direction = np.arctan2(neighbor_lat - fire_lat, neighbor_lon - fire_lon)
                    
                    # Wind features
                    wind_direction = np.arctan2(v10, u10)
                    wind_speed = np.sqrt(u10**2 + v10**2)
                    wind_alignment = np.cos(wind_direction - fire_direction)
                    
                    examples.append({
                        'lat': neighbor_lat,
                        'lon': neighbor_lon,
                        'elevation': elevation,
                        'slope': slope_val,
                        'wind_speed': wind_speed,
                        'wind_alignment': wind_alignment,
                        'distance_to_fire': min_distance,
                        'fire_intensity': nearest_frp,
                        'spread': spread
                    })
    
    df = pd.DataFrame(examples)
    logger.info(f"\n✓ Generated {len(df)} training examples")
    logger.info(f"  Spread=1: {(df['spread']==1).sum()} ({(df['spread']==1).mean()*100:.1f}%)")
    logger.info(f"  Spread=0: {(df['spread']==0).sum()} ({(df['spread']==0).mean()*100:.1f}%)")
    
    df.to_csv('data/multiday/grid_features.csv', index=False)
    return df


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class SpreadDataset(Dataset):
    """Dataset for spatial patches"""
    def __init__(self, patches, labels):
        # patches: (N, 175) → reshape to (N, 25, 7) for transformer
        self.patches = torch.FloatTensor(np.array(patches.tolist())).reshape(-1, PATCH_SIZE*PATCH_SIZE, 7)
        self.labels = torch.LongTensor(labels.astype(np.int64))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]


class SpatialTransformer(nn.Module):
    """Spatial Transformer - processes 5x5 patches to predict center cell spread"""
    def __init__(self, feature_dim=7, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Project each cell's features to d_model dimensions
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Positional encoding for spatial positions
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder - learns spatial relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head - uses center cell's encoding
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch, 25 cells, 7 features per cell)
        
        # Project each cell to d_model
        x = self.input_proj(x)  # (batch, 25, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer learns which cells matter for prediction
        x = self.transformer(x)  # (batch, 25, d_model)
        
        # Use CENTER cell (index 12 in 5x5 flattened grid) for prediction
        center_idx = (PATCH_SIZE * PATCH_SIZE) // 2  # Cell 12 is center of 5x5
        x = x[:, center_idx, :]  # (batch, d_model)
        
        # Classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# STEP 3: TRAIN TRANSFORMER
# ============================================================================

def train_transformer(df):
    """Train spatial transformer model"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: TRAINING SPATIAL TRANSFORMER")
    logger.info("="*60)
    
    # Prepare data - each example is a 5x5 patch (25 cells × 7 features)
    patches = df['patch_features'].values  # Array of lists
    labels = df['spread'].values.astype(np.int64)
    
    # Class balance
    pos_weight = (labels == 0).sum() / (labels == 1).sum() if (labels == 1).sum() > 0 else 1.0
    logger.info(f"\nClass balance: {pos_weight:.1f}:1 (negative:positive)")
    logger.info(f"Each example: {PATCH_SIZE}x{PATCH_SIZE} spatial patch = {PATCH_SIZE*PATCH_SIZE} cells × 7 features")
    
    # Standardize features (across all cells in all patches)
    logger.info("\n3.1 Standardizing features across all patches...")
    all_features = np.array(patches.tolist()).reshape(-1, 7)  # Flatten all cells
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # Reshape back to patches
    patches_scaled = all_features_scaled.reshape(len(patches), -1)  # (N, 175)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        patches_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = SpreadDataset(X_train, y_train)
    val_dataset = SpreadDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"\nTraining: {len(train_dataset)} | Validation: {len(val_dataset)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    model = SpatialTransformer(
        feature_dim=7,
        d_model=D_MODEL,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    logger.info(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss with moderate class weighting
    weights = torch.FloatTensor([1.0, pos_weight * 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop
    logger.info(f"\n3.2 Training for up to {NUM_EPOCHS} epochs...")
    best_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                outputs = model(patches)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        history['train_loss'].append(train_loss)
        history['val_f1'].append(f1)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'history': history
            }, 'data/multiday/spatial_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                f"P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}"
            )
    
    # Final evaluation
    logger.info(f"\n✓ Training complete! Best F1: {best_f1:.3f}")
    
    logger.info("\nFinal Performance:")
    logger.info(f"  Precision: {history['val_precision'][-1]:.3f}")
    logger.info(f"  Recall:    {history['val_recall'][-1]:.3f}")
    logger.info(f"  F1 Score:  {history['val_f1'][-1]:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
    logger.info(f"  FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")
    logger.info(f"\n  False Negatives (missed fires): {cm[1,0]} ⚠️")
    logger.info(f"  False Positives (false alarms): {cm[0,1]}")
    
    # Improvement metrics
    logger.info("\n📊 Improvement vs Single-Cell Model:")
    logger.info(f"  Previous F1: 0.368")
    logger.info(f"  Current F1:  {history['val_f1'][-1]:.3f}")
    logger.info(f"  Improvement: {(history['val_f1'][-1] - 0.368) / 0.368 * 100:+.1f}%")
    
    return model, scaler, history
    
    # Class balance
    pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
    logger.info(f"\nClass balance: {pos_weight:.1f}:1 (negative:positive)")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = SpreadDataset(X_train, y_train)
    val_dataset = SpreadDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    model = SpreadTransformer(
        input_dim=len(feature_cols), d_model=D_MODEL,
        nhead=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    
    # Loss with class weighting (moderate weight to balance precision/recall)
    weights = torch.FloatTensor([1.0, pos_weight * 0.4]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop
    logger.info(f"\nTraining for up to {NUM_EPOCHS} epochs...")
    best_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        history['train_loss'].append(train_loss)
        history['val_f1'].append(f1)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'history': history,
                'feature_cols': feature_cols
            }, 'data/multiday/spread_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                f"P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}"
            )
    
    # Final evaluation
    logger.info(f"\n✓ Training complete! Best F1: {best_f1:.3f}")
    
    logger.info("\nFinal Performance:")
    logger.info(f"  Precision: {history['val_precision'][-1]:.3f}")
    logger.info(f"  Recall:    {history['val_recall'][-1]:.3f}")
    logger.info(f"  F1 Score:  {history['val_f1'][-1]:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
    logger.info(f"  FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")
    logger.info(f"\n  False Negatives (missed fires): {cm[1,0]} ⚠️")
    logger.info(f"  False Positives (false alarms): {cm[0,1]}")
    
    return model, scaler, history


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("WILDFIRE SPREAD PREDICTION - TRANSFORMER")
    print("="*60)
    print(f"\nStudy area: LA fires ({LAT_MIN},{LON_MIN}) to ({LAT_MAX},{LON_MAX})")
    print(f"\nTime period: {START_DATE} + {NUM_DAYS} days")
    print(f"Grid: {GRID_SIZE_KM}km cells")
    print(f"Spatial context: {PATCH_SIZE}x{PATCH_SIZE} patches ({PATCH_SIZE*PATCH_SIZE} cells per example)")
    print(f"Model: {NUM_LAYERS} layers, {NUM_HEADS} heads, d_model={D_MODEL}")
    print("="*60)
    
    # Step 1: Download data
    data_files = download_multiday_data()
    if not data_files:
        logger.error("Data download failed")
        return
    
    # Step 2: Create grid and labels
    df = create_grid_and_labels(data_files)
    if df is None or len(df) == 0:
        logger.error("Grid creation failed")
        return
    
    # Step 3: Train transformer
    model, scaler, history = train_transformer(df)
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput files:")
    print("  data/multiday/fires_*.csv        - Daily fire detections")
    print("  data/multiday/spatial_patches.pkl - Training dataset (5x5 patches)")
    print("  data/multiday/spatial_model.pth   - Trained spatial transformer")
    print("="*60)


if __name__ == "__main__":
    main()