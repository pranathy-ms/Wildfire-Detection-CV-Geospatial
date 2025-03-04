# Quick test in Python
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    'ground-truth-dataset-fc84ca909a29.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

print(f"Authenticated as: {credentials.service_account_email}")
# Should output: wildfire-minimal-access@PROJECT_ID.iam.gserviceaccount.com
