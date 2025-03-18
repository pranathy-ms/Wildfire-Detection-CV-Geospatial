# Quick test in Python
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    'ground-truth-dataset-fc84ca909a29.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

print(f"Authenticated as: {credentials.service_account_email}")
# Should output: wildfire-minimal-access@PROJECT_ID.iam.gserviceaccount.com

###############################################################################
import os
from google.colab import drive

# Mount Google Drive (if using Colab)
drive.mount('/content/drive')

# Verify file paths (update with your actual path)
export_path = '/content/drive/MyDrive/GEE_Exports/'
files = [f for f in os.listdir(export_path) if f.endswith('.tif')]
print(f"Found {len(files)} GeoTIFF files:")
for f in files:
    print(f"- {f}")

