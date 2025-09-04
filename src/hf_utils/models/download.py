# src/models/download.py
import os
import shutil
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from huggingface_hub import snapshot_download, hf_hub_download, model_info
from huggingface_hub.utils import HfHubHTTPError

from ..common.config import HFConfig
from ..common.exceptions import DownloadError, ModelNotFoundError
from ..common.validators import validate_repo_name, sanitize_filename
from ..common.logger import setup_logger
from ..auth.manager import AuthManager

logger = setup_logger(__name__)

class ModelDownloader:
    """Handle model downloads from Hugging Face Hub."""
    
    def __init__(self, config: Optional[HFConfig] = None, auth_manager: Optional[AuthManager] = None):
        self.config = config or HFConfig()
        self.auth_manager = auth_manager or AuthManager(self.config)
    
    def download_model(
        self,
        repo_id: str,
        destination: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        force_download: bool = False,
        ignore_patterns: Optional[List[str]] = None,
        allow_patterns: Optional[List[str]] = None,
        max_workers: int = 8
    ) -> str:
        """
        Download a complete model from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID (username/model-name)
            destination: Local destination directory
            revision: Specific revision/branch to download
            cache_dir: Cache directory override
            local_files_only: Only use local cache
            resume_download: Resume interrupted downloads
            force_download: Force re-download even if cached
            ignore_patterns: Patterns to ignore during download
            allow_patterns: Patterns to allow during download
            max_workers: Number of concurrent download workers
            
        Returns:
            str: Path to downloaded model directory
        """
        # Validation
        if not validate_repo_name(repo_id):
            raise ValidationError(f"Invalid repository name: {repo_id}")
        
        try:
            # Prepare destination
            if destination is None:
                destination = self._get_default_destination(repo_id)
            
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            
            # Set cache directory
            cache_dir = cache_dir or self.config.cache_dir
            
            logger.info(f"Downloading model {repo_id} to {destination}")
            
            # Download model snapshot
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                cache_dir=cache_dir,
                local_dir=str(destination),
                local_files_only=local_files_only,
                resume_download=resume_download,
                force_download=force_download,
                ignore_patterns=ignore_patterns,
                allow_patterns=allow_patterns,
                max_workers=max_workers,
                token=self.config.token
            )
            
            logger.info(f"Model downloaded successfully to {downloaded_path}")
            return downloaded_path
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {repo_id}")
            elif e.response.status_code == 401:
                raise DownloadError(f"Access denied to model: {repo_id}")
            else:
                error_msg = f"Download failed: {e.response.status_code} - {e.response.reason}"
                raise DownloadError(error_msg)
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(error_msg)
            raise DownloadError(error_msg)
    
    def download_file(
        self,
        repo_id: str,
        filename: str,
        destination: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        force_download: bool = False
    ) -> str:
        """
        Download a specific file from a model repository.
        
        Args:
            repo_id: Repository ID
            filename: File to download
            destination: Local destination path
            revision: Specific revision/branch
            cache_dir: Cache directory override
            local_files_only: Only use local cache
            resume_download: Resume interrupted downloads
            force_download: Force re-download even if cached
            
        Returns:
            str: Path to downloaded file
        """
        try:
            # Prepare destination
            if destination is None:
                safe_filename = sanitize_filename(filename)
                destination = Path.cwd() / safe_filename
            
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Set cache directory
            cache_dir = cache_dir or self.config.cache_dir
            
            logger.info(f"Downloading file {filename} from {repo_id}")
            
            # Download specific file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                revision=revision,
                cache_dir=cache_dir,
                local_dir=str(destination.parent),
                local_dir_use_symlinks=False,
                local_files_only=local_files_only,
                resume_download=resume_download,
                force_download=force_download,
                token=self.config.token
            )
            
            # Move to final destination if needed
            if str(destination) != downloaded_path:
                shutil.move(downloaded_path, destination)
                downloaded_path = str(destination)
            
            logger.info(f"File downloaded successfully to {downloaded_path}")
            return downloaded_path
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"File not found: {filename} in {repo_id}")
            elif e.response.status_code == 401:
                raise DownloadError(f"Access denied to model: {repo_id}")
            else:
                error_msg = f"Download failed: {e.response.status_code} - {e.response.reason}"
                raise DownloadError(error_msg)
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(error_msg)
            raise DownloadError(error_msg)
    
    def get_model_info(self, repo_id: str, revision: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model repository.
        
        Args:
            repo_id: Repository ID
            revision: Specific revision/branch
            
        Returns:
            Dict containing model information
        """
        try:
            info = model_info(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                token=self.config.token
            )
            
            return {
                "id": info.id,
                "sha": info.sha,
                "created_at": info.created_at,
                "last_modified": info.last_modified,
                "private": info.private,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "siblings": [{"filename": f.rfilename, "size": f.size} for f in info.siblings],
                "card_data": info.card_data,
                "spaces": info.spaces
            }
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {repo_id}")
            raise DownloadError(f"Failed to get model info: {str(e)}")
    
    def list_model_files(self, repo_id: str, revision: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all files in a model repository.
        
        Args:
            repo_id: Repository ID
            revision: Specific revision/branch
            
        Returns:
            List of file information dictionaries
        """
        info = self.get_model_info(repo_id, revision)
        return info["siblings"]
    
    def _get_default_destination(self, repo_id: str) -> str:
        """Get default destination path for model."""
        # Create safe directory name from repo_id
        safe_name = repo_id.replace("/", "_")
        safe_name = sanitize_filename(safe_name)
        
        # Use configured destination or current directory
        base_dest = Path.cwd() / self.config.default_model_destination
        return str(base_dest / safe_name)