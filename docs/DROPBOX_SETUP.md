# Dropbox Setup Instructions

To test the Dropbox connection and find file `100_4247` in the shared folder `Hansman Syracuse photo docs July 2015`, you need to:

## 1. Obtain a Dropbox Refresh Token

You need to authorize the Dropbox app and get a refresh token. Follow these steps:

1. Run the OAuth script:
   ```bash
   cd /workspaces/amy-project
   source venv/bin/activate
   python get_dropbox_token.py
   ```

2. The script will show you a URL like:
   ```
   https://www.dropbox.com/oauth2/authorize?response_type=code&client_id=9a6hiz26bhy58kv&token_access_type=offline
   ```

3. Open this URL in your browser where you're logged into Dropbox

4. Click "Allow" to grant permissions to the app

5. Copy the authorization code from Dropbox

6. Paste the code into the terminal when prompted

7. The script will display your refresh token

## 2. Add the Refresh Token to .env

Add the refresh token to your `.env` file:

```bash
DROPBOX_REFRESH_TOKEN=your_refresh_token_here
```

Replace the line:
```
# DROPBOX_REFRESH_TOKEN=to_be_obtained_during_first_auth
```

with:
```
DROPBOX_REFRESH_TOKEN=<your actual token>
```

## 3. Run the Test

Once you have the refresh token, run the test:

```bash
cd /workspaces/amy-project
source venv/bin/activate
python test_dropbox_connection.py
```

This will:
- Connect to your Dropbox account
- Search for file "100_4247" in the shared folder "Hansman Syracuse photo docs July 2015"
- Display information about the file if found

## Note on Shared Folders

Make sure your Dropbox app has permissions to access shared folders. If you get permission errors, you may need to:
1. Check the app permissions in your Dropbox App Console
2. Ensure the shared folder is accessible to your account
3. Try using the full path to the shared folder

## Current Dropbox Credentials

The `.env` file already contains:
- DROPBOX_APP_KEY=9a6hiz26bhy58kv
- DROPBOX_APP_SECRET=bsx7phlyxotnln9

You only need to add the refresh token.
