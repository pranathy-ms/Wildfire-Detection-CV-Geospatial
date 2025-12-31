"""
Predictor - Makes fire spread predictions for new dates
"""

import torch
import numpy as np
import pandas as pd
import sqlite3
import pickle
import config
import rasterio
import richdem as rd
import xarray as xr
from pathlib import Path
from rasterio.transform import rowcol
from datetime import datetime

# Import model architecture
import sys
sys.path.append(str(Path(__file__).parent))
from model_trainer import SpatialTransformer

class FirePredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        
        # Grid setup
        self.lat_min = config.LAT_MIN
        self.lat_max = config.LAT_MAX
        self.lon_min = config.LON_MIN
        self.lon_max = config.LON_MAX
        self.grid_step = config.GRID_SIZE_KM * 0.01
        
        self.lats = np.arange(self.lat_min, self.lat_max, self.grid_step)
        self.lons = np.arange(self.lon_min, self.lon_max, self.grid_step)
        
        print(f"🔮 Predictor initialized")
        print(f"   Grid: {len(self.lats)} x {len(self.lons)} cells")
    
    def load_model(self):
        """Load trained model"""
        print("\n Loading trained model...")
        
        model_path = config.DATA_DIR / "models" / "trained_model.pth"
        
        if not model_path.exists():
            raise ValueError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Initialize model
        self.model = SpatialTransformer(
            feature_dim=7,
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.1,
            patch_size=config.PATCH_SIZE
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        self.scaler = checkpoint['scaler']
        
        #print(f"  ✓ Model loaded from {model_path}")
        print(f"  ✓ Model loaded from data/models/trained_model.pth")
        print(f"  ✓ Training F1: {checkpoint['history']['val_f1'][-1]:.3f}")
    
    def load_fires(self, date):
        """Load fire data for specific date"""
        conn = sqlite3.connect(config.DB_PATH)
        query = """
            SELECT lat, lon, frp
            FROM fires
            WHERE date = ?
        """
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()
        
        return df
    
    def load_elevation(self):
        """Load elevation and slope"""
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM elevation_cache LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        with rasterio.open(result[0]) as src:
            dem = src.read(1)
            transform = src.transform
        
        dem_rd = rd.rdarray(dem, no_data=-999999)
        slope = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
        
        return dem, slope, transform
    
    def load_wind(self, date):
        """Load wind data"""
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM wind_cache WHERE date = ?", (date,))
        result = cursor.fetchone()
        conn.close()
        
        if not result or not Path(result[0]).exists():
            return None
        
        wind_data = xr.open_dataset(result[0])
        if 'valid_time' in wind_data.coords:
            wind_data = wind_data.rename({'valid_time': 'time'})
        
        return wind_data
    
    def precompute_features(self, dem, slope, transform, wind_data):
        """Pre-compute features for all grid cells"""
        feature_grid = {}
        dem_height, dem_width = dem.shape
        
        for i, lat in enumerate(self.lats):
            for j, lon in enumerate(self.lons):
                row_idx, col_idx = rowcol(transform, lon, lat)
                
                if not (0 <= row_idx < dem_height and 0 <= col_idx < dem_width):
                    continue
                
                elevation = float(dem[row_idx, col_idx])
                slope_val = float(slope[row_idx, col_idx])
                
                if elevation <= 0:
                    continue
                
                # Wind features
                if wind_data is not None:
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
        
        return feature_grid
    
    def predict(self, date):
        """Make predictions for a specific date"""
        print("=" * 60)
        print(f"PREDICTING FIRE SPREAD FOR {date}")
        print("=" * 60)
        
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        # Load data
        print(f"\n Loading data for {date}...")
        fires_df = self.load_fires(date)
        
        if len(fires_df) == 0:
            print(f"  ✗ No fires found on {date}")
            return None
        
        print(f"  ✓ {len(fires_df)} active fires")
        
        dem, slope, transform = self.load_elevation()
        wind_data = self.load_wind(date)
        
        print(f"  ✓ Elevation loaded")
        print(f"  ✓ Wind {'loaded' if wind_data else 'not available'}")
        
        # Pre-compute features
        print(f"\n🔧 Pre-computing features...")
        feature_grid = self.precompute_features(dem, slope, transform, wind_data)
        print(f"  ✓ {len(feature_grid)} valid cells")
        
        # Create fire cell map
        fire_cells = set()
        fire_locations = []
        
        for _, fire in fires_df.iterrows():
            lat_idx = int((fire['lat'] - self.lat_min) / self.grid_step)
            lon_idx = int((fire['lon'] - self.lon_min) / self.grid_step)
            if 0 <= lat_idx < len(self.lats) and 0 <= lon_idx < len(self.lons):
                fire_cells.add((lat_idx, lon_idx))
                fire_locations.append((fire['lat'], fire['lon'], fire['frp']))
        
        print(f"  ✓ {len(fire_cells)} burning cells")
        
        # Generate predictions
        print(f"\n Generating predictions...")
        predictions = []
        half_patch = config.PATCH_SIZE // 2
        
        for lat_idx, lon_idx in fire_cells:
            # Check neighbors
            for dlat in [-1, 0, 1]:
                for dlon in [-1, 0, 1]:
                    if dlat == 0 and dlon == 0:
                        continue
                    
                    target_lat_idx = lat_idx + dlat
                    target_lon_idx = lon_idx + dlon
                    
                    # Check validity
                    if not (half_patch <= target_lat_idx < len(self.lats) - half_patch and
                            half_patch <= target_lon_idx < len(self.lons) - half_patch):
                        continue
                    
                    if (target_lat_idx, target_lon_idx) in fire_cells:
                        continue
                    
                    # Extract 5x5 patch
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
                            is_burning = 1.0 if (patch_lat_idx, patch_lon_idx) in fire_cells else 0.0
                            
                            # Distance to nearest fire
                            min_dist = float('inf')
                            fire_direction = 0
                            nearest_frp = 0
                            
                            for fire_lat, fire_lon, frp in fire_locations:
                                dist = np.sqrt((cell_features['lat'] - fire_lat)**2 +
                                             (cell_features['lon'] - fire_lon)**2) * 111
                                if dist < min_dist:
                                    min_dist = dist
                                    fire_direction = np.arctan2(
                                        cell_features['lat'] - fire_lat,
                                        cell_features['lon'] - fire_lon
                                    )
                                    nearest_frp = frp
                            
                            wind_direction = np.arctan2(cell_features['v10'], cell_features['u10'])
                            wind_alignment = np.cos(wind_direction - fire_direction)
                            
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
                    
                    if not patch_valid or len(patch_features) != config.PATCH_SIZE ** 2 * 7:
                        continue
                    
                    predictions.append({
                        'lat': self.lats[target_lat_idx],
                        'lon': self.lons[target_lon_idx],
                        'features': patch_features
                    })
        
        if len(predictions) == 0:
            print("  ✗ No valid predictions generated")
            return None
        
        print(f"  ✓ Analyzing {len(predictions)} cells")
        
        # Run model inference
        print(f"\n Running model inference...")
        
        # Prepare features
        X = np.array([p['features'] for p in predictions])
        X_scaled = self.scaler.transform(X.reshape(-1, 7)).reshape(len(predictions), -1)
        X_tensor = torch.FloatTensor(X_scaled).reshape(-1, config.PATCH_SIZE**2, 7).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            spread_probs = probabilities[:, 1].cpu().numpy()
        
        # Add probabilities to predictions
        for pred, prob in zip(predictions, spread_probs):
            pred['spread_probability'] = float(prob)
            pred['risk_level'] = 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.5 else 'LOW'
        
        # Statistics
        high_risk = sum(1 for p in predictions if p['spread_probability'] > 0.7)
        medium_risk = sum(1 for p in predictions if 0.5 < p['spread_probability'] <= 0.7)
        low_risk = sum(1 for p in predictions if 0.3 < p['spread_probability'] <= 0.5)
        
        print(f"\n Risk Assessment:")
        print(f"HIGH risk (>70%):     {high_risk:3d} cells")
        print(f"MEDIUM risk (50-70%): {medium_risk:3d} cells")
        print(f"LOW risk (30-50%):    {low_risk:3d} cells")
        
        #overall_risk = np.mean([p['spread_probability'] for p in predictions if p['spread_probability'] > 0.3])
        high_risk_cells = [p['spread_probability'] for p in predictions if p['spread_probability'] > 0.3]
        overall_risk = np.mean(high_risk_cells) if high_risk_cells else np.mean([p['spread_probability'] for p in predictions])
        print(f"\n  Overall risk score: {overall_risk:.1%}")
        
        # Save results
        output_file = config.DATA_DIR / "predictions" / f"prediction_{date}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(output_file, 'w') as f:
            json.dump({
                'date': date,
                'num_fires': len(fires_df),
                'predictions': predictions,
                'summary': {
                    'high_risk': high_risk,
                    'medium_risk': medium_risk,
                    'low_risk': low_risk,
                    'overall_risk': float(overall_risk)
                }
            }, f, indent=2)
        
        #print(f"\n Predictions saved: {output_file}")
        print(f"\nPredictions saved: data/predictions/prediction_{date}.json")

        
        print("=" * 60)
        
        return predictions

if __name__ == "__main__":
    # Test on high-activity day
    predictor = FirePredictor()
    
    print("\n" + "=" * 60)
    print("TESTING ON MULTIPLE DATES")
    print("=" * 60)
    
    test_dates = ["2025-01-08", "2025-01-10", "2025-01-15"]
    
    for date in test_dates:
        predictions = predictor.predict(date)
        
        if predictions:
            print(f"\n🗺️  Top 5 highest risk cells for {date}:")
            sorted_preds = sorted(predictions, key=lambda x: x['spread_probability'], reverse=True)[:5]
            for pred in sorted_preds:
                print(f"  ({pred['lat']:.4f}, {pred['lon']:.4f}): {pred['spread_probability']:.1%} - {pred['risk_level']}")
        
        print("\n" + "-" * 60)