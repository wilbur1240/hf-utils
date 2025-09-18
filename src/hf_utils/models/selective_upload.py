"""
Selective file upload utilities for Hugging Face Hub
Part of the hf-utils toolkit

This module provides functions to upload specific files or file patterns
to the Hugging Face Hub without uploading entire repositories.
"""

import os
import fnmatch
from pathlib import Path
from typing import List, Optional, Union, Dict, Callable, Any, Tuple
from dataclasses import dataclass, field
import logging

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from tqdm import tqdm

from ..common.config import HFConfig
from ..common.logger import setup_logger
from ..common.exceptions import UploadError, ValidationError
from ..common.validators import validate_repo_name, validate_file_path
from ..auth.manager import AuthManager

logger = setup_logger(__name__)


@dataclass
class FileUploadConfig:
    """Configuration for selective file uploads."""
    
    # Target repository
    repo_id: str
    repo_type: str = "model"  # "model", "dataset", or "space"
    
    # File selection
    files: List[Union[str, Path]] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    
    # Upload options
    commit_message: Optional[str] = None
    commit_description: Optional[str] = None
    revision: str = "main"
    create_pr: bool = False
    
    # Path mapping
    local_base_path: Union[str, Path] = "."
    remote_base_path: str = ""
    
    # Behavior options
    overwrite_existing: bool = True
    dry_run: bool = False
    show_progress: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not validate_repo_name(self.repo_id):
            raise ValidationError(f"Invalid repository name: {self.repo_id}")
        
        if self.repo_type not in ["model", "dataset", "space"]:
            raise ValueError(f"repo_type must be 'model', 'dataset', or 'space', got: {self.repo_type}")
        
        # Convert paths to Path objects
        self.local_base_path = Path(self.local_base_path)
        
        # Ensure files are Path objects
        self.files = [Path(f) for f in self.files]


class SelectiveUploader:
    """Main class for handling selective file uploads to Hugging Face Hub."""
    
    def __init__(self, config: Optional[HFConfig] = None, auth_manager: Optional[AuthManager] = None):
        """Initialize the uploader."""
        self.config = config or HFConfig()
        self.auth_manager = auth_manager or AuthManager(self.config)
        self.api = HfApi(endpoint=self.config.endpoint, token=self.config.token)
    
    def upload_files(self, config: FileUploadConfig) -> Dict[str, Any]:
        """
        Upload specific files to Hugging Face Hub.
        
        Args:
            config: Upload configuration
            
        Returns:
            Dictionary with upload results and metadata
        """
        try:
            # Resolve files to upload
            files_to_upload = self._resolve_files(config)
            
            if not files_to_upload:
                logger.warning("No files found matching the specified criteria")
                return {"status": "warning", "message": "No files found", "files": []}
            
            # Log what will be uploaded
            logger.info(f"Found {len(files_to_upload)} files to upload to {config.repo_id}")
            for local_path, remote_path in files_to_upload:
                logger.debug(f"  {local_path} -> {remote_path}")
            
            if config.dry_run:
                return {
                    "status": "dry_run",
                    "message": f"Dry run: would upload {len(files_to_upload)} files",
                    "files": [(str(local), remote) for local, remote in files_to_upload]
                }
            
            # Create commit operations
            operations = []
            for local_path, remote_path in files_to_upload:
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=remote_path,
                        path_or_fileobj=str(local_path)
                    )
                )
            
            # Generate commit message if not provided
            commit_message = config.commit_message or self._generate_commit_message(files_to_upload)
            
            # Upload files
            commit_info = self.api.create_commit(
                repo_id=config.repo_id,
                repo_type=config.repo_type,
                operations=operations,
                commit_message=commit_message,
                commit_description=config.commit_description,
                revision=config.revision,
                create_pr=config.create_pr,
            )
            
            logger.info(f"Successfully uploaded {len(files_to_upload)} files")
            logger.info(f"Commit URL: {commit_info.commit_url}")
            
            return {
                "status": "success",
                "commit_info": commit_info,
                "files_uploaded": len(files_to_upload),
                "files": [(str(local), remote) for local, remote in files_to_upload]
            }
            
        except Exception as e:
            logger.error(f"Failed to upload files: {str(e)}")
            raise UploadError(f"Upload failed: {str(e)}") from e
    
    def _resolve_files(self, config: FileUploadConfig) -> List[Tuple[Path, str]]:
        """
        Resolve which files to upload based on configuration.
        
        Returns:
            List of tuples (local_path, remote_path)
        """
        files_to_upload = []
        
        # Add explicitly specified files
        for file_path in config.files:
            if not file_path.is_absolute():
                file_path = config.local_base_path / file_path
            
            if file_path.exists():
                remote_path = self._get_remote_path(file_path, config)
                files_to_upload.append((file_path, remote_path))
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Add files matching patterns
        for pattern in config.file_patterns:
            matched_files = self._find_files_by_pattern(pattern, config.local_base_path)
            for file_path in matched_files:
                remote_path = self._get_remote_path(file_path, config)
                files_to_upload.append((file_path, remote_path))
        
        # Remove files matching exclude patterns
        if config.exclude_patterns:
            files_to_upload = [
                (local, remote) for local, remote in files_to_upload
                if not self._matches_exclude_patterns(local, config.exclude_patterns, config.local_base_path)
            ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for local, remote in files_to_upload:
            key = (str(local), remote)
            if key not in seen:
                seen.add(key)
                unique_files.append((local, remote))
        
        return unique_files
    
    def _find_files_by_pattern(self, pattern: str, base_path: Path) -> List[Path]:
        """Find files matching a glob pattern."""
        if base_path.is_dir():
            return list(base_path.rglob(pattern))
        else:
            # If base_path is a file, check if it matches the pattern
            if fnmatch.fnmatch(base_path.name, pattern):
                return [base_path]
            return []
    
    def _matches_exclude_patterns(self, file_path: Path, exclude_patterns: List[str], base_path: Path) -> bool:
        """Check if a file matches any exclude pattern."""
        # Get relative path for pattern matching
        try:
            rel_path = file_path.relative_to(base_path)
        except ValueError:
            # File is not under base_path, use full path
            rel_path = file_path
        
        rel_path_str = str(rel_path)
        return any(fnmatch.fnmatch(rel_path_str, pattern) for pattern in exclude_patterns)
    
    def _get_remote_path(self, local_path: Path, config: FileUploadConfig) -> str:
        """Get the remote path for a local file."""
        try:
            # Try to get relative path from base
            rel_path = local_path.relative_to(config.local_base_path)
        except ValueError:
            # File is not under base_path, use filename only
            rel_path = local_path.name
        
        # Combine with remote base path
        if config.remote_base_path:
            return f"{config.remote_base_path.rstrip('/')}/{rel_path}"
        else:
            return str(rel_path)
    
    def _generate_commit_message(self, files_to_upload: List[Tuple[Path, str]]) -> str:
        """Generate a commit message based on files being uploaded."""
        if len(files_to_upload) == 1:
            _, remote_path = files_to_upload[0]
            return f"Update {remote_path}"
        else:
            return f"Update {len(files_to_upload)} files"


# Convenience functions for common use cases

def upload_single_file(
    repo_id: str,
    local_file: Union[str, Path],
    remote_path: Optional[str] = None,
    repo_type: str = "model",
    commit_message: Optional[str] = None,
    revision: str = "main",
    create_pr: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Upload a single file to Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (e.g., "username/repo-name")
        local_file: Path to local file
        remote_path: Path in the repository (if None, uses filename)
        repo_type: Type of repo ("model", "dataset", or "space")
        commit_message: Commit message
        revision: Branch or tag to upload to
        create_pr: Whether to create a pull request
        **kwargs: Additional arguments passed to FileUploadConfig
    
    Returns:
        Upload result dictionary
    """
    local_file = Path(local_file)
    
    config = FileUploadConfig(
        repo_id=repo_id,
        repo_type=repo_type,
        files=[local_file],
        commit_message=commit_message,
        revision=revision,
        create_pr=create_pr,
        **kwargs
    )
    
    # Override remote path if specified
    if remote_path:
        config.remote_base_path = str(Path(remote_path).parent)
        # We'll handle the filename part in the uploader
    
    uploader = SelectiveUploader()
    return uploader.upload_files(config)


def upload_files_by_pattern(
    repo_id: str,
    patterns: List[str],
    local_base_path: Union[str, Path] = ".",
    remote_base_path: str = "",
    exclude_patterns: Optional[List[str]] = None,
    repo_type: str = "model",
    commit_message: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Upload files matching glob patterns to Hugging Face Hub.
    
    Args:
        repo_id: Repository ID
        patterns: List of glob patterns to match files
        local_base_path: Base directory for local files
        remote_base_path: Base directory in the repository
        exclude_patterns: Patterns to exclude from upload
        repo_type: Type of repo ("model", "dataset", or "space")
        commit_message: Commit message
        **kwargs: Additional arguments passed to FileUploadConfig
    
    Returns:
        Upload result dictionary
    """
    config = FileUploadConfig(
        repo_id=repo_id,
        repo_type=repo_type,
        file_patterns=patterns,
        exclude_patterns=exclude_patterns or [],
        local_base_path=local_base_path,
        remote_base_path=remote_base_path,
        commit_message=commit_message,
        **kwargs
    )
    
    uploader = SelectiveUploader()
    return uploader.upload_files(config)


def update_model_files(
    repo_id: str,
    files: List[Union[str, Path]],
    commit_message: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Update specific model files.
    
    Args:
        repo_id: Model repository ID
        files: List of files to update
        commit_message: Commit message
        **kwargs: Additional arguments
    
    Returns:
        Upload result dictionary
    """
    config = FileUploadConfig(
        repo_id=repo_id,
        repo_type="model",
        files=files,
        commit_message=commit_message or "Update model files",
        **kwargs
    )
    
    uploader = SelectiveUploader()
    return uploader.upload_files(config)


def update_dataset_files(
    repo_id: str,
    files: List[Union[str, Path]],
    commit_message: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Update specific dataset files.
    
    Args:
        repo_id: Dataset repository ID
        files: List of files to update
        commit_message: Commit message
        **kwargs: Additional arguments
    
    Returns:
        Upload result dictionary
    """
    config = FileUploadConfig(
        repo_id=repo_id,
        repo_type="dataset",
        files=files,
        commit_message=commit_message or "Update dataset files",
        **kwargs
    )
    
    uploader = SelectiveUploader()
    return uploader.upload_files(config)


# Example usage and CLI integration
if __name__ == "__main__":
    # Example: Upload specific files
    config = FileUploadConfig(
        repo_id="myusername/my-model",
        files=["model.safetensors", "config.json"],
        commit_message="Update model weights and config",
        dry_run=True  # Set to False to actually upload
    )
    
    uploader = SelectiveUploader()
    result = uploader.upload_files(config)
    print(result)