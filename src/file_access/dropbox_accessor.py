"""
Dropbox API integration for file access and organization.
Provides methods to list, download, upload, and organize files in Dropbox.
"""

import os
import dropbox
from dropbox.files import FileMetadata, FolderMetadata
from dropbox.exceptions import ApiError, AuthError
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import time
import tempfile
import hashlib
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DropboxFile:
    """Represents a file in Dropbox."""
    id: str
    name: str
    path: str
    size: int
    modified: datetime
    content_hash: str
    is_downloadable: bool
    
    @property
    def extension(self) -> str:
        """Get file extension."""
        return Path(self.name).suffix.lower()


class DropboxAccessor:
    """Access and manage files in Dropbox."""
    
    def __init__(self, app_key: str, app_secret: str, 
                 refresh_token: Optional[str] = None,
                 access_token: Optional[str] = None):
        """
        Initialize Dropbox accessor.
        
        Args:
            app_key: Dropbox app key
            app_secret: Dropbox app secret
            refresh_token: OAuth refresh token (preferred)
            access_token: Short-lived access token
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.refresh_token = refresh_token
        self.access_token = access_token
        self.client: Optional[dropbox.Dropbox] = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Dropbox API."""
        try:
            if self.refresh_token:
                # Use refresh token for long-lived access
                self.client = dropbox.Dropbox(
                    app_key=self.app_key,
                    app_secret=self.app_secret,
                    oauth2_refresh_token=self.refresh_token
                )
                # Test the connection
                self.client.users_get_current_account()
                logger.info("Successfully authenticated with Dropbox using refresh token")
                
            elif self.access_token:
                # Use access token (short-lived)
                self.client = dropbox.Dropbox(self.access_token)
                # Test the connection
                self.client.users_get_current_account()
                logger.info("Successfully authenticated with Dropbox using access token")
                
            else:
                # Need to perform OAuth flow
                self._oauth_flow()
                
        except AuthError as e:
            logger.error(f"Authentication error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {e}")
            raise
    
    def _oauth_flow(self):
        """Perform OAuth authentication flow."""
        auth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
            self.app_key, 
            self.app_secret,
            token_access_type='offline'  # Request refresh token
        )
        
        auth_url = auth_flow.start()
        print(f"1. Go to: {auth_url}")
        print("2. Click 'Allow' (you might have to log in first)")
        print("3. Copy the authorization code.")
        auth_code = input("Enter the authorization code here: ").strip()
        
        try:
            oauth_result = auth_flow.finish(auth_code)
            self.refresh_token = oauth_result.refresh_token
            self.access_token = oauth_result.access_token
            
            # Create client with refresh token
            self.client = dropbox.Dropbox(
                app_key=self.app_key,
                app_secret=self.app_secret,
                oauth2_refresh_token=self.refresh_token
            )
            
            print(f"Successfully authenticated!")
            print(f"Refresh token: {self.refresh_token}")
            print("Save this refresh token for future use.")
            
            logger.info("OAuth flow completed successfully")
            
        except Exception as e:
            logger.error(f"Error during OAuth flow: {e}")
            raise
    
    def list_files(self, folder_path: str = "", 
                  recursive: bool = True,
                  file_types: Optional[List[str]] = None) -> List[DropboxFile]:
        """
        List files in Dropbox folder.
        
        Args:
            folder_path: Path to folder (empty string for root)
            recursive: Whether to list files recursively
            file_types: List of file extensions to filter (e.g., ['.pdf', '.jpg'])
            
        Returns:
            List of DropboxFile objects
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        files = []
        try:
            result = self.client.files_list_folder(folder_path, recursive=recursive)
            
            while True:
                for entry in result.entries:
                    if isinstance(entry, FileMetadata):
                        file = self._create_file_object(entry)
                        
                        # Filter by file type if specified
                        if file_types is None or file.extension in file_types:
                            files.append(file)
                
                if not result.has_more:
                    break
                
                result = self.client.files_list_folder_continue(result.cursor)
            
            logger.info(f"Listed {len(files)} files in {folder_path}")
            return files
            
        except ApiError as e:
            logger.error(f"Error listing files in {folder_path}: {e}")
            raise
    
    def _create_file_object(self, metadata: FileMetadata) -> DropboxFile:
        """Create DropboxFile object from metadata."""
        return DropboxFile(
            id=metadata.id,
            name=metadata.name,
            path=metadata.path_display,
            size=metadata.size,
            modified=metadata.server_modified,
            content_hash=metadata.content_hash,
            is_downloadable=metadata.is_downloadable
        )
    
    def download_file(self, dropbox_path: str, local_path: Path,
                     show_progress: bool = True) -> Path:
        """
        Download a file from Dropbox.
        
        Args:
            dropbox_path: Path to file in Dropbox
            local_path: Local path to save file
            show_progress: Whether to show download progress
            
        Returns:
            Path to downloaded file
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            # Create parent directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get file size for progress bar
            metadata = self.client.files_get_metadata(dropbox_path)
            file_size = metadata.size if isinstance(metadata, FileMetadata) else 0
            
            # Download with progress
            with open(local_path, 'wb') as f:
                if show_progress and file_size > 0:
                    with tqdm(total=file_size, desc=f"Downloading {Path(dropbox_path).name}",
                             unit='B', unit_scale=True) as pbar:
                        
                        metadata, response = self.client.files_download(dropbox_path)
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    self.client.files_download_to_file(str(local_path), dropbox_path)
            
            logger.info(f"Downloaded {dropbox_path} to {local_path}")
            return local_path
            
        except ApiError as e:
            logger.error(f"Error downloading {dropbox_path}: {e}")
            raise
    
    def download_batch(self, file_paths: List[str], local_dir: Path,
                      max_concurrent: int = 5) -> Dict[str, Path]:
        """
        Download multiple files in batch.
        
        Args:
            file_paths: List of Dropbox file paths
            local_dir: Local directory to save files
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            Dict mapping Dropbox paths to local paths
        """
        results = {}
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(file_paths), max_concurrent):
            batch = file_paths[i:i + max_concurrent]
            
            for dropbox_path in batch:
                try:
                    filename = Path(dropbox_path).name
                    local_path = local_dir / filename
                    
                    self.download_file(dropbox_path, local_path, show_progress=False)
                    results[dropbox_path] = local_path
                    
                except Exception as e:
                    logger.error(f"Error downloading {dropbox_path}: {e}")
                    results[dropbox_path] = None
            
            # Brief pause between batches
            if i + max_concurrent < len(file_paths):
                time.sleep(0.5)
        
        return results
    
    def create_folder(self, folder_path: str) -> bool:
        """
        Create a folder in Dropbox.
        
        Args:
            folder_path: Path for new folder
            
        Returns:
            True if successful (or folder already exists)
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            self.client.files_create_folder_v2(folder_path)
            logger.info(f"Created folder: {folder_path}")
            return True
            
        except ApiError as e:
            # Check if folder already exists
            if hasattr(e, 'error') and hasattr(e.error, 'is_path'):
                try:
                    if e.error.is_path() and e.error.get_path().is_conflict():
                        logger.info(f"Folder already exists: {folder_path}")
                        return True
                except:
                    pass
            
            logger.error(f"Error creating folder {folder_path}: {e}")
            raise
    
    def move_file(self, from_path: str, to_path: str,
                 autorename: bool = True) -> bool:
        """
        Move a file within Dropbox.
        
        Args:
            from_path: Current path of file
            to_path: Destination path
            autorename: Automatically rename if destination exists
            
        Returns:
            True if successful
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            self.client.files_move_v2(
                from_path, 
                to_path,
                autorename=autorename
            )
            logger.info(f"Moved {from_path} to {to_path}")
            return True
            
        except ApiError as e:
            logger.error(f"Error moving file from {from_path} to {to_path}: {e}")
            raise
    
    def copy_file(self, from_path: str, to_path: str,
                 autorename: bool = True) -> bool:
        """
        Copy a file within Dropbox.
        
        Args:
            from_path: Source path
            to_path: Destination path
            autorename: Automatically rename if destination exists
            
        Returns:
            True if successful
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            self.client.files_copy_v2(
                from_path,
                to_path,
                autorename=autorename
            )
            logger.info(f"Copied {from_path} to {to_path}")
            return True
            
        except ApiError as e:
            logger.error(f"Error copying file from {from_path} to {to_path}: {e}")
            raise
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from Dropbox.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if successful
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            self.client.files_delete_v2(file_path)
            logger.info(f"Deleted {file_path}")
            return True
            
        except ApiError as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise
    
    def get_temporary_link(self, file_path: str) -> str:
        """
        Get a temporary download link for a file.
        
        Args:
            file_path: Path to file in Dropbox
            
        Returns:
            Temporary download URL
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            link = self.client.files_get_temporary_link(file_path)
            return link.link
            
        except ApiError as e:
            logger.error(f"Error getting temporary link for {file_path}: {e}")
            raise
    
    def upload_file(self, local_path: Path, dropbox_path: str,
                   overwrite: bool = False) -> FileMetadata:
        """
        Upload a file to Dropbox.
        
        Args:
            local_path: Local file path
            dropbox_path: Destination path in Dropbox
            overwrite: Whether to overwrite existing file
            
        Returns:
            File metadata
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            mode = dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add
            
            with open(local_path, 'rb') as f:
                file_size = os.path.getsize(local_path)
                
                # Upload in chunks for large files
                if file_size > 150 * 1024 * 1024:  # 150MB
                    return self._chunked_upload(f, dropbox_path, file_size)
                else:
                    metadata = self.client.files_upload(
                        f.read(),
                        dropbox_path,
                        mode=mode,
                        mute=True
                    )
                    logger.info(f"Uploaded {local_path} to {dropbox_path}")
                    return metadata
                    
        except ApiError as e:
            logger.error(f"Error uploading {local_path} to {dropbox_path}: {e}")
            raise
    
    def _chunked_upload(self, file_obj, dropbox_path: str, file_size: int):
        """Upload large file in chunks."""
        CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
        
        with tqdm(total=file_size, desc=f"Uploading {Path(dropbox_path).name}",
                 unit='B', unit_scale=True) as pbar:
            
            # Start upload session
            upload_session_start_result = self.client.files_upload_session_start(
                file_obj.read(CHUNK_SIZE)
            )
            cursor = dropbox.files.UploadSessionCursor(
                session_id=upload_session_start_result.session_id,
                offset=file_obj.tell()
            )
            pbar.update(cursor.offset)
            
            # Upload remaining chunks
            while file_obj.tell() < file_size:
                chunk = file_obj.read(CHUNK_SIZE)
                if file_obj.tell() < file_size:
                    self.client.files_upload_session_append_v2(chunk, cursor)
                    cursor.offset = file_obj.tell()
                    pbar.update(len(chunk))
                else:
                    # Final chunk
                    commit = dropbox.files.CommitInfo(path=dropbox_path)
                    metadata = self.client.files_upload_session_finish(
                        chunk, cursor, commit
                    )
                    pbar.update(len(chunk))
                    return metadata
    
    def get_space_usage(self) -> Dict[str, int]:
        """
        Get Dropbox space usage information.
        
        Returns:
            Dict with 'used' and 'allocated' space in bytes
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            space_usage = self.client.users_get_space_usage()
            return {
                'used': space_usage.used,
                'allocated': space_usage.allocation.get_individual().allocated
            }
        except Exception as e:
            logger.error(f"Error getting space usage: {e}")
            raise
    
    def search_files(self, query: str, 
                    max_results: int = 100) -> List[DropboxFile]:
        """
        Search for files in Dropbox.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching files
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            result = self.client.files_search(query, max_results=max_results)
            files = []
            
            for match in result.matches:
                if isinstance(match.metadata, FileMetadata):
                    files.append(self._create_file_object(match.metadata))
            
            logger.info(f"Found {len(files)} files matching '{query}'")
            return files
            
        except ApiError as e:
            logger.error(f"Error searching for '{query}': {e}")
            raise
    
    def get_file_metadata(self, file_path: str) -> FileMetadata:
        """
        Get metadata for a specific file.
        
        Args:
            file_path: Path to file in Dropbox
            
        Returns:
            File metadata
        """
        if not self.client:
            raise RuntimeError("Not authenticated with Dropbox")
        
        try:
            metadata = self.client.files_get_metadata(file_path)
            if isinstance(metadata, FileMetadata):
                return metadata
            else:
                raise ValueError(f"{file_path} is not a file")
                
        except ApiError as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            raise