# Fix Dropbox App Permissions

Your access token is working (connected to wyattjo116@gmail.com) but your app needs permissions.

## Required Steps:

1. **Go to your Dropbox App Console**
   - https://www.dropbox.com/developers/apps
   - Click on your app

2. **Add Required Permissions** (Permissions tab)
   - ✅ `files.metadata.read` - To list and search files
   - ✅ `files.content.read` - To download files
   - ✅ `sharing.read` - To access shared folders
   
   Click **Submit** after selecting these.

3. **Generate NEW Access Token** (Settings tab)
   - IMPORTANT: You must generate a NEW token AFTER adding permissions
   - The current token doesn't have the permissions
   - Click "Generate" under "Generated access token"
   - Copy the new token

4. **Update .env File**
   ```
   DROPBOX_ACCESS_TOKEN=<paste_your_new_token_here>
   ```

5. **Test Again**
   ```bash
   cd /workspaces/amy-project
   source venv/bin/activate
   python test_basic_access.py
   ```

## Why This Is Happening

Access tokens capture the permissions at the time they're created. Your current token was created before the permissions were added, so it doesn't have them. A new token will include the permissions.

## Alternative: Use OAuth Flow

If you prefer a long-term solution, use a refresh token instead:
```bash
python get_dropbox_token.py
```

This will give you a refresh token that automatically gets new access tokens as needed.