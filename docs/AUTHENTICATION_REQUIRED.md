# Dropbox Authentication Required

The current refresh token in the `.env` file is invalid or malformed. To connect to Dropbox and search for the file, you need to:

## Step 1: Get a New Refresh Token

1. Open this URL in your browser:
   ```
   https://www.dropbox.com/oauth2/authorize?response_type=code&client_id=9a6hiz26bhy58kv&token_access_type=offline
   ```

2. Log in to your Dropbox account if needed

3. Click "Allow" to grant the app permission

4. Copy the authorization code that Dropbox provides

5. Run this command in your terminal:
   ```bash
   cd /workspaces/amy-project
   source venv/bin/activate
   python get_dropbox_token.py
   ```

6. Paste the authorization code when prompted

7. The script will display a new refresh token

## Step 2: Update the .env File

Replace the current DROPBOX_REFRESH_TOKEN in your `.env` file with the new token from step 1.

Current line:
```
DROPBOX_REFRESH_TOKEN=sl.u.AFuVj... (invalid)
```

Replace with:
```
DROPBOX_REFRESH_TOKEN=<your-new-refresh-token>
```

## Step 3: Run the Test

Once you have the new refresh token, run:
```bash
cd /workspaces/amy-project
source venv/bin/activate
python test_dropbox_connection.py
```

This will search for file `100_4247` in the shared folder `Hansman Syracuse photo docs July 2015`.

## Note

The current app credentials in `.env` are:
- DROPBOX_APP_KEY=9a6hiz26bhy58kv
- DROPBOX_APP_SECRET=bsx7phlyxotnln9

These are correct and don't need to be changed. Only the refresh token needs to be updated.
