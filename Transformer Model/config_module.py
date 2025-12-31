"""
Configuration module for wildfire analysis pipeline
Centralized configuration management
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataPaths:
    """Data file paths configuration"""
    viirs_geojson: str = 'custom_date_fires.geojson'
    era5_netcdf: str = 'era5_wind_la.nc'
    dem_tif: str = 'USGS3DEP_30m_33.5_34.5_-119.0_-118.0.tif'
    output_dir: str = 'outputs'
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class GeographicBounds:
    """Geographic bounding box configuration"""
    lat_min: float = 33.5
    lat_max: float = 34.5
    lon_min: float = -119.0
    lon_max: float = -118.0
    
    @property
    def center(self):
        """Get center coordinates"""
        return (self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2


@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 5
    
    # Transformer model parameters
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # Feature columns
    feature_columns: list = None
    target_column: str = 'confidence_num'
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = ['frp', 'u10', 'v10', 'elevation', 'slope']


@dataclass
class VisualizationConfig:
    """Visualization settings"""
    map_center: tuple = (34.05, -118.25)
    zoom_start: int = 9
    tiles: str = 'CartoDB dark_matter'
    slope_opacity: float = 0.4
    fire_point_base_radius: float = 2
    frp_scale_factor: float = 5
    wind_vector_scale: float = 100
    wind_vector_color: str = '#00b3ff'
    
    # Color schemes
    slope_colors: list = None
    confidence_colors: dict = None
    frp_colors: list = None
    
    def __post_init__(self):
        if self.slope_colors is None:
            self.slope_colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']
        
        if self.confidence_colors is None:
            self.confidence_colors = {
                0: 'gray',    # low
                1: 'orange',  # nominal
                2: 'red'      # high
            }
        
        if self.frp_colors is None:
            self.frp_colors = ['yellow', 'orange', 'red']


@dataclass
class APIConfig:
    """API keys and endpoints (load from environment or keys file)"""
    cds_api_url: str = 'https://cds.climate.copernicus.eu/api/v2'
    opentopo_api_url: str = 'https://portal.opentopography.org/API/usgsdem'
    firms_api_url: str = 'https://firms.modaps.eosdis.nasa.gov/api/country/csv'
    
    # Load from environment variables or keys file
    cds_api_key: Optional[str] = None
    opentopo_api_key: Optional[str] = None
    firms_map_key: Optional[str] = None
    
    def __post_init__(self):
        """Try to load API keys from environment or keys file"""
        self.cds_api_key = os.getenv('CDS_API_KEY')
        self.opentopo_api_key = os.getenv('OPENTOPO_API_KEY')
        self.firms_map_key = os.getenv('FIRMS_MAP_KEY')
        
        # Try importing from keys file if env vars not set
        if not all([self.cds_api_key, self.opentopo_api_key, self.firms_map_key]):
            try:
                import VIIRS_API_keys as keys
                self.opentopo_api_key = self.opentopo_api_key or getattr(keys, 'OPEN_TOPOGRAPHY_MAP_KEY', None)
                self.firms_map_key = self.firms_map_key or getattr(keys, 'MAP_KEY', None)
            except ImportError:
                pass


@dataclass
class Config:
    """Main configuration class combining all settings"""
    data: DataPaths = None
    bounds: GeographicBounds = None
    model: ModelConfig = None
    viz: VisualizationConfig = None
    api: APIConfig = None
    
    def __post_init__(self):
        """Initialize nested configurations"""
        self.data = self.data or DataPaths()
        self.bounds = self.bounds or GeographicBounds()
        self.model = self.model or ModelConfig()
        self.viz = self.viz or VisualizationConfig()
        self.api = self.api or APIConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(
            data=DataPaths(**config_dict.get('data', {})),
            bounds=GeographicBounds(**config_dict.get('bounds', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            viz=VisualizationConfig(**config_dict.get('viz', {})),
            api=APIConfig(**config_dict.get('api', {}))
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'data': self.data.__dict__,
            'bounds': self.bounds.__dict__,
            'model': self.model.__dict__,
            'viz': self.viz.__dict__,
            'api': {k: v for k, v in self.api.__dict__.items() if not k.endswith('_key')}
        }


# Global config instance
config = Config()


# Example usage
if __name__ == "__main__":
    # Access configuration
    print(f"VIIRS path: {config.data.viirs_geojson}")
    print(f"Geographic center: {config.bounds.center}")
    print(f"Model features: {config.model.feature_columns}")
    print(f"Map tiles: {config.viz.tiles}")
    
    # Modify configuration
    config.bounds.lat_min = 33.0
    config.model.n_estimators = 300
    
    # Export to dict
    import json
    print("\nConfiguration as JSON:")
    print(json.dumps(config.to_dict(), indent=2))
