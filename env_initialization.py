import ee

# Initialize with explicit project assets path
ee.Initialize(project='ground-truth-dataset')

# Now access the shared folder explicitly
asset_path = 'projects/ground-truth-dataset/assets/shared_assets'
assets = ee.data.listAssets({'parent': asset_path})['assets']

print(f"Found {len(assets)} assets in shared folder")
