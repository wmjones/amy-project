# Add Required Dropbox Permissions

Your Dropbox app needs additional permissions to search for files. Follow these steps:

## Step 1: Add Permissions

1. Go to https://www.dropbox.com/developers/apps
2. Click on your app
3. Go to the **Permissions** tab
4. Enable these permissions:
   - `files.metadata.read` - View information about your Dropbox files and folders
   - `files.content.read` - View content of your Dropbox files and folders
   - `sharing.read` - View your shared files and folders (for accessing shared folders)

5. Click **Submit** to save the changes

## Step 2: Generate New Access Token

After adding permissions, you need a new access token:

1. Go to the **Settings** tab
2. Under "OAuth 2" section, find "Generated access token"
3. Click **Generate** (if there's already a token, you might need to regenerate it)
4. Copy the new token
5. Update your `.env` file:
   ```
   DROPBOX_ACCESS_TOKEN=your_new_access_token_with_permissions
   ```

## Step 3: Run the Test Again

```bash
cd /workspaces/amy-project
source venv/bin/activate
python test_with_access_token.py
```

## Important Notes

- Access tokens expire after a few hours
- When the token expires, you'll need to generate a new one
- Make sure to enable all three permissions mentioned above
- The token must be generated AFTER you add the permissions
