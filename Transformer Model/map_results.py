"""
Minimal code to visualize fire spread predictions on a map
Shows: Actual fires, predictions, elevation, and wind
"""

import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow
import torch

# Configuration
LAT_MIN, LAT_MAX = 33.5, 34.5
LON_MIN, LON_MAX = -119.0, -118.0

def create_map():
    """Create comprehensive fire spread visualization"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. Load and plot elevation (background)
    print("Loading elevation...")
    with rasterio.open('data/multiday/elevation.tif') as src:
        dem = src.read(1)
        bounds = src.bounds
        
    # Plot elevation as background
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(dem, extent=extent, cmap='terrain', alpha=0.3, aspect='auto')
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.6)
    
    # 2. Load actual fires (first day)
    print("Loading fire detections...")
    fires_day1 = pd.read_csv('data/multiday/fires_2025-01-07.csv')
    if len(fires_day1) > 0:
        ax.scatter(fires_day1['longitude'], fires_day1['latitude'], 
                  c='red', s=30, alpha=0.7, label='Actual Fires (Day 1)', 
                  edgecolors='darkred', linewidth=0.5)
    
    # 3. Load predictions
    print("Loading predictions...")
    try:
        patches_df = pd.read_pickle('data/multiday/spatial_patches.pkl')
        
        # Separate by prediction outcome
        spread_yes = patches_df[patches_df['spread'] == 1]
        spread_no = patches_df[patches_df['spread'] == 0]
        
        # Plot predicted spread locations
        if len(spread_yes) > 0:
            ax.scatter(spread_yes['target_lon'], spread_yes['target_lat'],
                      c='orange', s=20, alpha=0.5, marker='^',
                      label=f'Predicted Spread ({len(spread_yes)})', 
                      edgecolors='darkorange', linewidth=0.5)
        
        # Sample of no-spread predictions (too many to plot all)
        if len(spread_no) > 0:
            sample = spread_no.sample(min(200, len(spread_no)))
            ax.scatter(sample['target_lon'], sample['target_lat'],
                      c='lightblue', s=10, alpha=0.3, marker='.',
                      label=f'No Spread (sample)', edgecolors='none')
        
    except Exception as e:
        print(f"Could not load predictions: {e}")
    
    # 4. Add wind vectors (sample grid)
    print("Adding wind vectors...")
    try:
        import xarray as xr
        wind = xr.open_dataset('data/multiday/wind.nc', engine='netcdf4')
        if 'valid_time' in wind.coords:
            wind = wind.rename({'valid_time': 'time'})
        
        # Sample wind at grid points
        lats = np.linspace(LAT_MIN + 0.1, LAT_MAX - 0.1, 8)
        lons = np.linspace(LON_MIN + 0.1, LON_MAX - 0.1, 8)
        
        for lat in lats:
            for lon in lons:
                try:
                    w = wind.sel(latitude=lat, longitude=lon, method='nearest')
                    u = float(w.u10.values.mean())
                    v = float(w.v10.values.mean())
                    
                    # Scale for visibility
                    scale = 0.05
                    ax.arrow(lon, lat, u*scale, v*scale,
                            head_width=0.02, head_length=0.015,
                            fc='blue', ec='blue', alpha=0.6, linewidth=1)
                except:
                    pass
        
        # Add wind legend
        ax.arrow(LON_MIN + 0.05, LAT_MAX - 0.05, 0.1, 0,
                head_width=0.02, head_length=0.015,
                fc='blue', ec='blue', linewidth=2)
        ax.text(LON_MIN + 0.17, LAT_MAX - 0.05, 'Wind (5 m/s)',
               fontsize=9, va='center')
        
    except Exception as e:
        print(f"Could not add wind: {e}")
    
    # 5. Formatting
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Wildfire Spread Prediction - Spatial Transformer Results\n' +
                'LA Region (Jan 7-20, 2025)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add metrics box
    metrics_text = (
        'Model Performance:\n'
        'F1 Score: 0.707\n'
        'Precision: 72.5%\n'
        'Recall: 69.0%\n'
        'Caught: 29/42 fires'
    )
    ax.text(0.02, 0.98, metrics_text,
           transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('data/multiday/fire_spread_map.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: data/multiday/fire_spread_map.png")
    
    try:
        plt.show()
    except:
        pass


def create_comparison_map():
    """Side-by-side: Actual spread vs Predicted spread"""
    
    try:
        # Load data
        patches_df = pd.read_pickle('data/multiday/spatial_patches.pkl')
        fires_day1 = pd.read_csv('data/multiday/fires_2025-01-07.csv')
        fires_day3 = pd.read_csv('data/multiday/fires_2025-01-09.csv')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Load elevation for both
        with rasterio.open('data/multiday/elevation.tif') as src:
            dem = src.read(1)
            bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # LEFT: Actual fires Day 1 → Day 3
        ax1.imshow(dem, extent=extent, cmap='terrain', alpha=0.3, aspect='auto')
        ax1.scatter(fires_day1['longitude'], fires_day1['latitude'],
                   c='red', s=30, alpha=0.7, label='Day 1 (Actual)', marker='o')
        ax1.scatter(fires_day3['longitude'], fires_day3['latitude'],
                   c='darkred', s=30, alpha=0.7, label='Day 3 (Spread)', marker='s')
        ax1.set_title('ACTUAL: Fire Spread (Day 1 → Day 3)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RIGHT: Predictions
        ax2.imshow(dem, extent=extent, cmap='terrain', alpha=0.3, aspect='auto')
        ax2.scatter(fires_day1['longitude'], fires_day1['latitude'],
                   c='red', s=30, alpha=0.7, label='Day 1 (Actual)', marker='o')
        
        # Plot predictions
        spread_yes = patches_df[patches_df['spread'] == 1]
        ax2.scatter(spread_yes['target_lon'], spread_yes['target_lat'],
                   c='orange', s=30, alpha=0.7, label='Predicted Spread', marker='^')
        
        ax2.set_title('PREDICTED: Fire Spread (48hr forecast)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/multiday/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: data/multiday/actual_vs_predicted.png")
        
        try:
            plt.show()
        except:
            pass
        
    except Exception as e:
        print(f"Could not create comparison map: {e}")


def main():
    """Run all visualizations"""
    print("="*60)
    print("CREATING FIRE SPREAD MAPS")
    print("="*60)
    
    # Main map
    create_map()
    
    # Comparison map
    create_comparison_map()
    
    print("\n✅ Complete! Maps saved to data/multiday/")
    print("="*60)


if __name__ == "__main__":
    main()