# src/__init__.py
"""
HF Utils - Comprehensive utilities for Hugging Face Hub operations.

A modular toolkit for uploading, downloading, and managing models, datasets, 
and spaces on the Hugging Face Hub with clear destination specification 
and separated application logic.
"""

__version__ = "0.1.0"
__author__ = "Wilbur"
__email__ = "your.email@example.com"

from .auth.manager import AuthManager
from .models.upload import ModelUploader
from .models.download import ModelDownloader
from .models.manage import ModelManager
from .datasets.upload import DatasetUploader
from .datasets.download import DatasetDownloader
from .datasets.manage import DatasetManager
from .common.config import HFConfig
from .common.exceptions import (
    HFUtilsError,
    AuthenticationError,
    ValidationError,
    UploadError,
    DownloadError,
    ModelNotFoundError,
    DatasetNotFoundError,
    RepoExistsError,
)

__all__ = [
    # Core classes
    "AuthManager",
    "HFConfig",
    # Model operations
    "ModelUploader",
    "ModelDownloader", 
    "ModelManager",
    # Dataset operations
    "DatasetUploader",
    "DatasetDownloader",
    "DatasetManager",
    # Exceptions
    "HFUtilsError",
    "AuthenticationError",
    "ValidationError", 
    "UploadError",
    "DownloadError",
    "ModelNotFoundError",
    "DatasetNotFoundError",
    "RepoExistsError",
]