# src/datasets/download.py
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import snapshot_download, hf_hub_download, dataset_info
from huggingface_hub.utils import HfHubHTTPError

from ..common.config import HFConfig
from ..common.exceptions import DownloadError, DatasetNotFoundError
from ..common.validators import validate_repo_name, sanitize_filename
from ..common.logger import setup_logger
from ..auth.manager import AuthManager

logger = setup_logger(__name__)

class DatasetDownloader:
    """Handle dataset downloads from Hugging Face Hub."""
    
    def __init__(self, config: Optional[HFConfig] = None, auth_manager: Optional[AuthManager] = None):
        self.config = config or HFConfig()
        self.auth_manager = auth_manager or AuthManager(self.config)
    
    def download_dataset(
        self,
        repo_id: str,
        destination: Optional[str] = None,
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        num_proc: Optional[int] = None
    ) -> Union[Dataset, DatasetDict]:
        """
        Download a dataset from Hugging Face Hub using the datasets library.
        
        Args:
            repo_id: Repository ID (username/dataset-name)
            destination: Local destination directory
            config_name: Dataset configuration name
            split: Specific split to download
            revision: Specific revision/branch to download
            cache_dir: Cache directory override
            streaming: Enable streaming mode
            num_proc: Number of processes for multiprocessing
            
        Returns:
            Dataset or DatasetDict object
        """
        if not validate_repo_name(repo_id):
            raise ValidationError(f"Invalid repository name: {repo_id}")
        
        try:
            cache_dir = cache_dir or self.config.cache_dir
            
            logger.info(f"Downloading dataset {repo_id}")
            
            dataset = load_dataset(
                repo_id,
                name=config_name,
                split=split,
                revision=revision,
                cache_dir=cache_dir,
                streaming=streaming,
                num_proc=num_proc,
                token=self.config.token
            )
            
            # Save to destination if specified and not streaming
            if destination and not streaming:
                self._save_dataset_locally(dataset, destination, repo_id)
            
            logger.info(f"Dataset downloaded successfully: {repo_id}")
            return dataset
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise DatasetNotFoundError(f"Dataset not found: {repo_id}")
            error_msg = f"Dataset download failed: {str(e)}"
            logger.error(error_msg)
            raise DownloadError(error_msg)
    
    def download_dataset_files(
        self,
        repo_id: str,
        destination: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ) -> str:
        """
        Download raw dataset files from repository.
        
        Args:
            repo_id: Repository ID
            destination: Local destination directory
            revision: Specific revision/branch
            cache_dir: Cache directory override
            allow_patterns: Patterns to allow
            ignore_patterns: Patterns to ignore
            
        Returns:
            str: Path to downloaded files
        """
        try:
            if destination is None:
                destination = self._get_default_destination(repo_id)
            
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            
            cache_dir = cache_dir or self.config.cache_dir
            
            logger.info(f"Downloading dataset files {repo_id} to {destination}")
            
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
                local_dir=str(destination),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                token=self.config.token
            )
            
            logger.info(f"Dataset files downloaded to {downloaded_path}")
            return downloaded_path
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise DatasetNotFoundError(f"Dataset not found: {repo_id}")
            raise DownloadError(f"Failed to get dataset info: {str(e)}")
    
    def _save_dataset_locally(self, dataset: Union[Dataset, DatasetDict], 
                             destination: str, repo_id: str) -> None:
        """Save dataset to local directory."""
        try:
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(dataset, DatasetDict):
                # Save each split
                for split_name, split_dataset in dataset.items():
                    split_path = dest_path / f"{split_name}.parquet"
                    split_dataset.to_parquet(str(split_path))
                    logger.info(f"Saved {split_name} split to {split_path}")
            else:
                # Single dataset
                file_path = dest_path / "data.parquet"
                dataset.to_parquet(str(file_path))
                logger.info(f"Saved dataset to {file_path}")
                
        except Exception as e:
            logger.warning(f"Failed to save dataset locally: {str(e)}")
    
    def _get_default_destination(self, repo_id: str) -> str:
        """Get default destination path for dataset."""
        safe_name = repo_id.replace("/", "_")
        safe_name = sanitize_filename(safe_name)
        
        base_dest = Path.cwd() / self.config.default_dataset_destination
        return str(base_dest / safe_name)