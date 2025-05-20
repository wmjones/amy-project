import os

print("Checking Dropbox credentials...")
print(f"DROPBOX_APP_KEY: {'Found' if os.getenv('DROPBOX_APP_KEY') else 'Not Found'}")
print(
    f"DROPBOX_APP_SECRET: {'Found' if os.getenv('DROPBOX_APP_SECRET') else 'Not Found'}"
)
print(
    f"DROPBOX_REFRESH_TOKEN: {'Found' if os.getenv('DROPBOX_REFRESH_TOKEN') else 'Not Found'}"
)
