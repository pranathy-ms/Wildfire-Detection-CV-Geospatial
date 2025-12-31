from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "wildfire.db"

# Make sure data directory exists
DATA_DIR.mkdir(exist_ok=True)

try:
    import VIIRS_API_keys as keys
    FIRMS_KEY = keys.MAP_KEY
    OPENTOPO_KEY = keys.OPEN_TOPOGRAPHY_MAP_KEY
except ImportError:
    print("⚠️  VIIRS_API_keys.py not found - create it with your keys")
    FIRMS_KEY = None
    OPENTOPO_KEY = None

# LA Wildfires - January 2025
LAT_MIN = 33.5
LAT_MAX = 34.5
LON_MIN = -119.0
LON_MAX = -118.0
START_DATE = "2025-01-07"
NUM_DAYS = 14

# Grid parameters
GRID_SIZE_KM = 1.0
PATCH_SIZE = 5