# Dropbox App Setup Guide

The current app key `9a6hiz26bhy58kv` is invalid. You need to create a new Dropbox app and update the credentials.

## Step 1: Create a New Dropbox App

1. Go to https://www.dropbox.com/developers/apps

2. Click "Create app"

3. Choose these settings:
   - **Choose an API**: Scoped access
   - **Choose the type of access**: Full Dropbox (to access shared folders)
   - **Name your app**: Give it a unique name (e.g., "FileOrganizer2024")

4. Click "Create app"

## Step 2: Configure App Permissions

1. In your app settings, go to the "Permissions" tab

2. Enable these permissions:
   - `files.content.read` - View content of your Dropbox files and folders
   - `files.content.write` - Edit content of your Dropbox files and folders
   - `files.metadata.read` - View information about your Dropbox files and folders
   - `files.metadata.write` - Edit information about your Dropbox files and folders
   - `sharing.read` - View your shared files and folders
   - `sharing.write` - Edit your shared files and folders (if needed)

3. Click "Submit" to save permissions

## Step 3: Get App Credentials

1. Go to the "Settings" tab

2. Copy your **App key** and **App secret**

3. Update your `.env` file:
   ```
   DROPBOX_APP_KEY=your_new_app_key
   DROPBOX_APP_SECRET=your_new_app_secret
   ```

## Step 4: Generate Access Token

1. Still in the Settings tab, under "OAuth 2", click "Generate" for the access token

2. Copy the generated token

3. Add it to your `.env` file temporarily:
   ```
   DROPBOX_ACCESS_TOKEN=your_generated_token
   ```

## Step 5: Get Refresh Token

1. Update your `.env` file with the new app credentials

2. Run the OAuth flow script:
   ```bash
   cd /workspaces/amy-project
   source venv/bin/activate
   python get_dropbox_token.py
   ```

3. Follow the prompts to get a refresh token

4. Update your `.env` file with the refresh token:
   ```
   DROPBOX_REFRESH_TOKEN=your_new_refresh_token
   ```

## Step 6: Test the Connection

Run the test script:
```bash
cd /workspaces/amy-project
source venv/bin/activate
python test_dropbox_connection.py
```

## Example .env Configuration

After setup, your `.env` should have:
```
DROPBOX_APP_KEY=your_new_app_key
DROPBOX_APP_SECRET=your_new_app_secret
DROPBOX_REFRESH_TOKEN=your_new_refresh_token
```

## Important Notes

- The refresh token provides long-term access without needing to re-authenticate
- Keep your app secret and refresh token secure
- The app needs permissions to access shared folders to find the file "100_4247"