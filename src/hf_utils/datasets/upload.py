# src/datasets/upload.py
import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from huggingface_hub.utils import HfHubHTTPError

from ..common.config import HFConfig
from ..common.exceptions import UploadError, ValidationError
from ..common.validators import validate_repo_name, validate_dataset_format, validate_file_path
from ..common.logger import setup_logger
from ..auth.manager import AuthManager

logger = setup_logger(__name__)

class DatasetUploader:
    """Handle dataset uploads to Hugging Face Hub."""
    
    def __init__(self, config: Optional[HFConfig] = None, auth_manager: Optional[AuthManager] = None):
        self.config = config or HFConfig()
        self.auth_manager = auth_manager or AuthManager(self.config)
        self.api = HfApi(endpoint=self.config.endpoint, token=self.config.token)
    
    def upload_dataset(
        self,
        data_path: str,
        repo_id: str,
        destination: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: bool = False,
        revision: Optional[str] = None,
        dataset_card: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        convert_to_parquet: bool = True
    ) -> str:
        """
        Upload a dataset to Hugging Face Hub.
        
        Args:
            data_path: Path to dataset file or directory
            repo_id: Repository ID (username/dataset-name)
            destination: Destination path in repo
            private: Whether to make repository private
            commit_message: Commit message
            commit_description: Commit description
            create_pr: Whether to create a pull request
            revision: Branch/revision to upload to
            dataset_card: Dataset card information
            tags: Dataset tags
            convert_to_parquet: Convert CSV/JSON to Parquet format
            
        Returns:
            str: Repository URL
        """
        # Validation
        if not validate_repo_name(repo_id):
            raise ValidationError(f"Invalid repository name: {repo_id}")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise ValidationError(f"Data path does not exist: {data_path}")
        
        # Authenticate
        if not self.auth_manager.is_authenticated():
            raise UploadError("Authentication required for upload")
        
        try:
            # Create repository
            repo_url = self._create_repository(repo_id, private, tags)
            
            # Prepare commit message
            if commit_message is None:
                commit_message = f"Upload dataset from {data_path.name}"
            
            # Process and upload data
            if data_path.is_file():
                # Single file upload
                processed_files = self._process_single_file(
                    data_path, convert_to_parquet
                )
                for file_path, repo_path in processed_files:
                    self._upload_file(
                        file_path=file_path,
                        repo_id=repo_id,
                        destination=destination or repo_path,
                        commit_message=commit_message,
                        commit_description=commit_description,
                        create_pr=create_pr,
                        revision=revision
                    )
            else:
                # Directory upload
                processed_dir = self._process_directory(
                    data_path, convert_to_parquet
                )
                upload_folder(
                    folder_path=str(processed_dir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo=destination,
                    commit_message=commit_message,
                    commit_description=commit_description,
                    create_pr=create_pr,
                    revision=revision,
                    token=self.config.token
                )
            
            # Update dataset card
            if dataset_card:
                self._update_dataset_card(repo_id, dataset_card, revision)
            
            logger.info(f"Successfully uploaded dataset to {repo_url}")
            return repo_url
            
        except Exception as e:
            error_msg = f"Dataset upload failed: {str(e)}"
            logger.error(error_msg)
            raise UploadError(error_msg)
    
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        repo_id: str,
        filename: str = "data.parquet",
        destination: Optional[str] = None,
        **kwargs
    ) -> str:
        """Upload pandas DataFrame directly to Hub."""
        
        # Create temporary file
        temp_path = Path.cwd() / f"temp_{filename}"
        
        try:
            # Save DataFrame to file
            if filename.endswith('.parquet'):
                df.to_parquet(temp_path, index=False)
            elif filename.endswith('.csv'):
                df.to_csv(temp_path, index=False)
            elif filename.endswith('.json'):
                df.to_json(temp_path, orient='records', lines=True)
            else:
                raise ValidationError(f"Unsupported file format: {filename}")
            
            # Upload the file
            return self.upload_dataset(
                data_path=str(temp_path),
                repo_id=repo_id,
                destination=destination or filename,
                **kwargs
            )
        
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    def _create_repository(self, repo_id: str, private: bool = False,
                          tags: Optional[List[str]] = None) -> str:
        """Create dataset repository if it doesn't exist."""
        try:
            repo_url = create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=self.config.token,
                exist_ok=True
            )
            logger.info(f"Dataset repository ready: {repo_url}")
            return repo_url
        except Exception as e:
            raise UploadError(f"Failed to create repository: {str(e)}")
    
    def _process_single_file(self, file_path: Path, convert_to_parquet: bool) -> List[tuple]:
        """Process a single data file."""
        if not validate_dataset_format(str(file_path)):
            raise ValidationError(f"Unsupported file format: {file_path.suffix}")
        
        processed_files = []
        
        if convert_to_parquet and file_path.suffix.lower() in ['.csv', '.json', '.jsonl']:
            # Convert to Parquet
            parquet_path = file_path.with_suffix('.parquet')
            self._convert_to_parquet(file_path, parquet_path)
            processed_files.append((str(parquet_path), parquet_path.name))
            
            # Keep original as well
            processed_files.append((str(file_path), file_path.name))
        else:
            processed_files.append((str(file_path), file_path.name))
        
        return processed_files
    
    def _process_directory(self, dir_path: Path, convert_to_parquet: bool) -> Path:
        """Process a directory of data files."""
        if not convert_to_parquet:
            return dir_path
        
        # Create temporary processed directory
        processed_dir = dir_path.parent / f"{dir_path.name}_processed"
        processed_dir.mkdir(exist_ok=True)
        
        for file_path in dir_path.iterdir():
            if file_path.is_file() and validate_dataset_format(str(file_path)):
                if file_path.suffix.lower() in ['.csv', '.json', '.jsonl']:
                    # Convert to Parquet
                    parquet_path = processed_dir / file_path.with_suffix('.parquet').name
                    self._convert_to_parquet(file_path, parquet_path)
        
        return processed_dir
    
    def _convert_to_parquet(self, input_path: Path, output_path: Path) -> None:
        """Convert CSV/JSON to Parquet format."""
        try:
            if input_path.suffix.lower() == '.csv':
                df = pd.read_csv(input_path)
            elif input_path.suffix.lower() in ['.json', '.jsonl']:
                df = pd.read_json(input_path, lines=input_path.suffix.lower() == '.jsonl')
            else:
                raise ValidationError(f"Cannot convert {input_path.suffix} to Parquet")
            
            df.to_parquet(output_path, index=False)
            logger.info(f"Converted {input_path.name} to {output_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to convert {input_path.name} to Parquet: {str(e)}")
            raise ValidationError(f"File conversion failed: {str(e)}")
    
    def _upload_file(self, file_path: str, repo_id: str, destination: str, **kwargs) -> Any:
        """Upload single file to dataset repository."""
        logger.info(f"Uploading file {file_path} to {repo_id}/{destination}")
        
        return upload_file(
            path_or_fileobj=file_path,
            path_in_repo=destination,
            repo_id=repo_id,
            repo_type="dataset",
            token=self.config.token,
            **kwargs
        )
    
    def _update_dataset_card(self, repo_id: str, dataset_card: Dict[str, Any],
                           revision: Optional[str] = None) -> None:
        """Update dataset card with metadata."""
        try:
            from huggingface_hub import DatasetCard
            
            # Load existing card or create new one
            try:
                card = DatasetCard.load(repo_id, token=self.config.token)
            except:
                card = DatasetCard("")
            
            # Update card data
            if not hasattr(card, 'data'):
                card.data = {}
            
            for key, value in dataset_card.items():
                card.data[key] = value
            
            # Save updated card
            card.push_to_hub(
                repo_id=repo_id,
                repo_type="dataset",
                token=self.config.token,
                revision=revision
            )
            
            logger.info("Dataset card updated successfully")
            
        except Exception as e:
            logger.warning(f"Failed to update dataset card: {str(e)}")
