import requests
import sqlite3
import config
import json
from datetime import datetime, timedelta
from pathlib import Path

def check_database(lat_min, lat_max, lon_min, lon_max, start_date, num_days):
    """Check if we have fire data"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(DISTINCT date) FROM fires
        WHERE lat BETWEEN ? AND ?
        AND lon BETWEEN ? AND ?
        AND date >= ?
    """, (lat_min, lat_max, lon_min, lon_max, start_date))
    
    days_in_db = cursor.fetchone()[0]
    conn.close()
    
    return days_in_db >= num_days

def check_elevation(lat_min, lat_max, lon_min, lon_max):
    """Check if elevation exists for this area"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT file_path, downloaded_at FROM elevation_cache
        WHERE lat_min <= ? AND lat_max >= ?
        AND lon_min <= ? AND lon_max >= ?
    """, (lat_min, lat_max, lon_min, lon_max))
    
    result = cursor.fetchone()
    conn.close()
    
    if result and Path(result[0]).exists():
        return result[0], result[1]
    return None, None

def check_wind(start_date, lat_min, lat_max, lon_min, lon_max):
    """Check if wind data exists for specific date"""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT file_path FROM wind_cache
        WHERE date = ?
        AND lat_min <= ? AND lat_max >= ?
        AND lon_min <= ? AND lon_max >= ?
    """, (start_date, lat_min, lat_max, lon_min, lon_max))
    
    result = cursor.fetchone()
    conn.close()
    
    if result and Path(result[0]).exists():
        return result[0]
    return None

def fetch_fires(lat_min=None, lat_max=None, lon_min=None, lon_max=None, start_date=None, num_days=None):
    """Fetch fire data"""
    lat_min = lat_min or config.LAT_MIN
    lat_max = lat_max or config.LAT_MAX
    lon_min = lon_min or config.LON_MIN
    lon_max = lon_max or config.LON_MAX
    start_date = start_date or config.START_DATE
    num_days = num_days or config.NUM_DAYS
    
    print(f"\n🔥 FIRE DATA")
    print(f"📍 Area: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")
    print(f"📅 From: {start_date} for {num_days} days")
    
    print(f"🔍 Checking database...")
    if check_database(lat_min, lat_max, lon_min, lon_max, start_date, num_days):
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM fires
            WHERE lat BETWEEN ? AND ?
            AND lon BETWEEN ? AND ?
            AND date >= ?
        """, (lat_min, lat_max, lon_min, lon_max, start_date))
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"✅ Found {count} fires in database")
        return count
    
    print(f"📡 Downloading from VIIRS API...")
    
    if not config.FIRMS_KEY:
        print("❌ No API key!")
        return 0
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
    
    total_fires = 0
    
    for date in dates:
        query_date = datetime.strptime(date, '%Y-%m-%d')
        days_ago = (datetime.now() - query_date).days
        sensor = "VIIRS_NOAA20_SP" if days_ago > 105 else "VIIRS_NOAA20_NRT"
        
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{config.FIRMS_KEY}/{sensor}/{lon_min},{lat_min},{lon_max},{lat_max}/1/{date}"
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            if len(lines) < 2:
                print(f"  ○ {date}: No fires")
                continue
            
            header = lines[0].split(',')
            data = [line.split(',') for line in lines[1:]]
            
            lat_idx = header.index('latitude')
            lon_idx = header.index('longitude')
            date_idx = header.index('acq_date')
            time_idx = header.index('acq_time')
            frp_idx = header.index('frp')
            brightness_idx = header.index('bright_ti4')
            confidence_idx = header.index('confidence')
            daynight_idx = header.index('daynight')
            satellite_idx = header.index('satellite')
            
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            
            day_count = 0
            for row in data:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO fires 
                        (lat, lon, date, time, frp, brightness, confidence, daynight, satellite)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        float(row[lat_idx]),
                        float(row[lon_idx]),
                        row[date_idx],
                        row[time_idx],
                        float(row[frp_idx]),
                        float(row[brightness_idx]),
                        row[confidence_idx],
                        row[daynight_idx],
                        row[satellite_idx]
                    ))
                    day_count += 1
                except (ValueError, IndexError):
                    continue
            
            conn.commit()
            conn.close()
            
            total_fires += day_count
            print(f"  ✓ {date}: {day_count} fires")
            
        except Exception as e:
            print(f"  ✗ {date}: {e}")
    
    print(f"✅ Stored {total_fires} fires\n")
    return total_fires

def fetch_elevation(lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    """Fetch elevation data (downloads once per area, reuses forever)"""
    lat_min = lat_min or config.LAT_MIN
    lat_max = lat_max or config.LAT_MAX
    lon_min = lon_min or config.LON_MIN
    lon_max = lon_max or config.LON_MAX
    
    print(f"\n⛰️  ELEVATION DATA")
    print(f"📍 Area: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")
    
    print(f"🔍 Checking database...")
    cached, downloaded_at = check_elevation(lat_min, lat_max, lon_min, lon_max)
    if cached:
        print(f"✅ Found in database (downloaded: {downloaded_at})")
        print(f"   {cached}\n")
        return cached
    
    print(f"📡 Downloading from OpenTopography...")
    print(f"   (USGS 30m DEM - latest available)")
    
    if not config.OPENTOPO_KEY:
        print("❌ No API key!")
        return None
    
    elev_dir = config.DATA_DIR / "elevation"
    elev_dir.mkdir(exist_ok=True)
    
    file_path = elev_dir / "elevation_LA.tif"
    
    url = f"https://portal.opentopography.org/API/usgsdem?datasetName=USGS30m&south={lat_min}&north={lat_max}&west={lon_min}&east={lon_max}&outputFormat=GTiff&API_Key={config.OPENTOPO_KEY}"
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        # Extract metadata
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                metadata = {
                    'crs': str(src.crs),
                    'bounds': list(src.bounds),
                    'shape': src.shape,
                    'resolution': src.res
                }
        except:
            metadata = {}
        
        # Store in database
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO elevation_cache
            (lat_min, lat_max, lon_min, lon_max, file_path, downloaded_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (lat_min, lat_max, lon_min, lon_max, str(file_path), 
              datetime.now().isoformat(), json.dumps(metadata)))
        conn.commit()
        conn.close()
        
        print(f"✅ Stored: {file_path}\n")
        return str(file_path)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return None

def fetch_wind(start_date, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    """Fetch wind data for specific date"""
    lat_min = lat_min or config.LAT_MIN
    lat_max = lat_max or config.LAT_MAX
    lon_min = lon_min or config.LON_MIN
    lon_max = lon_max or config.LON_MAX
    
    print(f"  📅 {start_date}...", end=" ")
    
    # Check database first
    cached = check_wind(start_date, lat_min, lat_max, lon_min, lon_max)
    if cached:
        print("✓ (cached)")
        return cached
    
    try:
        import cdsapi
    except ImportError:
        print("✗ (cdsapi not installed)")
        return None
    
    wind_dir = config.DATA_DIR / "wind"
    wind_dir.mkdir(exist_ok=True)
    
    file_path = wind_dir / f"wind_{start_date}.nc"
    
    try:
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                'year': start_date[:4],
                'month': start_date[5:7],
                'day': start_date[8:10],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': [lat_max, lon_min, lat_min, lon_max],
                'format': 'netcdf'
            },
            str(file_path)
        )
        
        # Store in database
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO wind_cache
            (date, lat_min, lat_max, lon_min, lon_max, file_path, downloaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (start_date, lat_min, lat_max, lon_min, lon_max, str(file_path), 
              datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        print("✓")
        return str(file_path)
        
    except Exception as e:
        print(f"✗ ({e})")
        return None

def fetch_all(start_date=None, num_days=None):
    """Fetch all data for a date range"""
    start_date = start_date or config.START_DATE
    num_days = num_days or config.NUM_DAYS
    
    print("=" * 60)
    print(f"FETCHING ALL DATA")
    print(f"Date range: {start_date} for {num_days} days")
    print("=" * 60)
    
    # 1. Fires for entire range
    fires = fetch_fires(start_date=start_date, num_days=num_days)
    
    # 2. Elevation once (reuses forever)
    elevation = fetch_elevation()
    
    # 3. Wind for EACH day
    print(f"\n🌬️  WIND DATA ({num_days} days)")
    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
    
    wind_files = []
    for date in dates:
        wind_file = fetch_wind(start_date=date)
        wind_files.append(wind_file)
    
    wind_success = sum(1 for f in wind_files if f is not None)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"🔥 Fires: {fires} detections")
    print(f"⛰️  Elevation: {'✓' if elevation else '✗'}")
    print(f"🌬️  Wind: {wind_success}/{num_days} days")
    print("=" * 60)
    
    return {
        'fires': fires,
        'elevation': elevation,
        'wind_files': wind_files,
        'dates': dates
    }

if __name__ == "__main__":
    fetch_all()