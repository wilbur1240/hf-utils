# src/models/upload.py
import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
from huggingface_hub.utils import HfHubHTTPError

from ..common.config import HFConfig
from ..common.exceptions import UploadError, ValidationError, RepoExistsError
from ..common.validators import validate_repo_name, validate_directory_path, validate_file_path
from ..common.logger import setup_logger
from ..auth.manager import AuthManager

logger = setup_logger(__name__)

class ModelUploader:
    """Handle model uploads to Hugging Face Hub."""
    
    def __init__(self, config: Optional[HFConfig] = None, auth_manager: Optional[AuthManager] = None):
        self.config = config or HFConfig()
        self.auth_manager = auth_manager or AuthManager(self.config)
        self.api = HfApi(endpoint=self.config.endpoint, token=self.config.token)
    
    def upload_model(
        self,
        model_path: str,
        repo_id: str,
        destination: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: bool = False,
        revision: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        model_card: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Upload a model to Hugging Face Hub.
        
        Args:
            model_path: Path to model directory or file
            repo_id: Repository ID (username/model-name)
            destination: Destination path in repo (None for root)
            private: Whether to make repository private
            commit_message: Commit message
            commit_description: Commit description
            create_pr: Whether to create a pull request
            revision: Branch/revision to upload to
            ignore_patterns: Patterns to ignore during upload
            model_card: Model card information
            tags: Model tags
            
        Returns:
            str: Repository URL
        """
        # Validation
        if not validate_repo_name(repo_id):
            raise ValidationError(f"Invalid repository name: {repo_id}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValidationError(f"Model path does not exist: {model_path}")
        
        # Authenticate and validate access
        if not self.auth_manager.is_authenticated():
            raise UploadError("Authentication required for upload")
        
        try:
            # Create repository if it doesn't exist
            repo_url = self._create_repository(repo_id, private, tags)
            
            # Prepare commit message
            if commit_message is None:
                commit_message = f"Upload {'folder' if model_path.is_dir() else 'file'} {model_path.name}"
            
            # Set default ignore patterns
            if ignore_patterns is None:
                ignore_patterns = ["*.git*", "*.DS_Store", "__pycache__", "*.pyc"]
            
            # Upload model files
            if model_path.is_dir():
                commit_info = self._upload_folder(
                    folder_path=str(model_path),
                    repo_id=repo_id,
                    destination=destination,
                    commit_message=commit_message,
                    commit_description=commit_description,
                    create_pr=create_pr,
                    revision=revision,
                    ignore_patterns=ignore_patterns
                )
            else:
                commit_info = self._upload_file(
                    file_path=str(model_path),
                    repo_id=repo_id,
                    destination=destination or model_path.name,
                    commit_message=commit_message,
                    commit_description=commit_description,
                    create_pr=create_pr,
                    revision=revision
                )
            
            # Update model card if provided
            if model_card:
                self._update_model_card(repo_id, model_card, revision)
            
            logger.info(f"Successfully uploaded model to {repo_url}")
            logger.info(f"Commit SHA: {commit_info.oid}")
            
            return repo_url
            
        except HfHubHTTPError as e:
            error_msg = f"Upload failed: {e.response.status_code} - {e.response.reason}"
            logger.error(error_msg)
            raise UploadError(error_msg)
        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            logger.error(error_msg)
            raise UploadError(error_msg)
    
    def _create_repository(self, repo_id: str, private: bool = False, 
                          tags: Optional[List[str]] = None) -> str:
        """Create repository if it doesn't exist."""
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                token=self.config.token,
                exist_ok=True
            )
            logger.info(f"Repository ready: {repo_url}")
            return repo_url
            
        except HfHubHTTPError as e:
            if e.response.status_code == 409:  # Repository already exists
                repo_url = f"{self.config.endpoint}/{repo_id}"
                logger.info(f"Repository already exists: {repo_url}")
                return repo_url
            raise
    
    def _upload_folder(self, folder_path: str, repo_id: str, **kwargs) -> Any:
        """Upload entire folder to repository."""
        logger.info(f"Uploading folder {folder_path} to {repo_id}")
        
        # Remove destination from kwargs for folder upload
        destination = kwargs.pop('destination', None)
        if destination:
            kwargs['path_in_repo'] = destination
        
        return upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            token=self.config.token,
            **kwargs
        )
    
    def _upload_file(self, file_path: str, repo_id: str, destination: str, **kwargs) -> Any:
        """Upload single file to repository."""
        logger.info(f"Uploading file {file_path} to {repo_id}/{destination}")
        
        return upload_file(
            path_or_fileobj=file_path,
            path_in_repo=destination,
            repo_id=repo_id,
            repo_type="model",
            token=self.config.token,
            **kwargs
        )
    
    def _update_model_card(self, repo_id: str, model_card: Dict[str, Any], 
                          revision: Optional[str] = None) -> None:
        """Update model card with metadata."""
        try:
            from huggingface_hub import ModelCard
            
            # Load existing model card or create new one
            try:
                card = ModelCard.load(repo_id, token=self.config.token)
            except:
                card = ModelCard("")
            
            # Update card data
            if not hasattr(card, 'data'):
                card.data = {}
            
            for key, value in model_card.items():
                card.data[key] = value
            
            # Save updated card
            card.push_to_hub(
                repo_id=repo_id,
                repo_type="model",
                token=self.config.token,
                revision=revision
            )
            
            logger.info("Model card updated successfully")
            
        except Exception as e:
            logger.warning(f"Failed to update model card: {str(e)}")