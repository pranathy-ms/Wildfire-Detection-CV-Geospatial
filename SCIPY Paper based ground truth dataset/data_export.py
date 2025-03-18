import ee
import time
import math

# Initialize Earth Engine
ee.Initialize(project='ground-truth-dataset')

def calculate_pixels(geometry, scale):
    """Calculate approximate pixel count using California Albers projection (EPSG:3310)"""
    try:
        projected = geometry.transform('EPSG:3310', 1)
        area = projected.area(1).getInfo()  # Area in square meters
        return math.ceil(area / (scale ** 2))
    except Exception as e:
        print(f"Pixel calculation error: {str(e)}")
        return None

def monitor_task(task_id, timeout=7200):
    """Monitor task status with timeout (2 hours default)"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = ee.data.getTaskStatus(task_id)[0]
        state = status['state']
        
        if state in ['COMPLETED', 'FAILED', 'CANCELED']:
            print(f"\nTask {task_id} {state}")
            if 'error_message' in status:
                print(f"Error: {status['error_message']}")
            return state
        
        # Progress estimation
        if 'start_timestamp_ms' in status and status['start_timestamp_ms']:
            elapsed = (time.time() * 1000 - status['start_timestamp_ms']) / 1000
            print(f"\rProgress: {elapsed:.0f}s elapsed | Status: {state}", end='')
        
        time.sleep(30)  # Check every 30 seconds
    
    print(f"\nTimeout after {timeout//3600}h")
    return 'TIMEOUT'

# Main workflow
california = ee.Geometry.Rectangle([-124.48, 32.53, -114.13, 42.01])
collection = (ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(california)
    .filterDate('2023-06-01', '2023-09-30')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)))

image_list = collection.toList(3)

for i in range(image_list.size().getInfo()):
    try:
        image = ee.Image(image_list.get(i))
        scale = 100  # Start with 100m resolution
        
        # Pixel count validation
        pixel_count = calculate_pixels(california, scale)
        if not pixel_count:
            continue
            
        print(f"\nImage {i} - Estimated pixels: {pixel_count:,}")
        
        if pixel_count > 1e8:
            # Auto-adjust scale to stay under limit
            new_scale = math.sqrt(pixel_count / 1e8) * scale
            new_scale = 10 * math.ceil(new_scale / 10)  # Round up to nearest 10m
            print(f"Adjusting scale from {scale}m to {new_scale}m")
            scale = new_scale
        
        # Export configuration
        task = ee.batch.Export.image.toDrive(
            image=image.select(['B2','B3','B4','B8','B11','B12']),
            description=f'fire_image_{i}',
            folder='GEE_Exports',
            scale=scale,
            region=california,
            maxPixels=1e9,
            fileFormat='GeoTIFF',
            fileDimensions=[256, 256]  # Better for tile processing
        )
        task.start()
        print(f"Started Task {i}: {task.id}")
        
        # Monitoring with timeout
        final_state = monitor_task(task.id)
        
        if final_state != 'COMPLETED':
            print(f"Task {i} failed to complete successfully")
            time.sleep(300)  # Cool-down period after failure
            continue
            
    except Exception as e:
        print(f"\nCritical error on image {i}: {str(e)}")
        break

print("\nExport workflow completed. Check Google Drive for outputs.")
