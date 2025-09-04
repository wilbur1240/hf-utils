# src/datasets/manage.py
from typing import List, Dict, Any, Optional
from pathlib import Path
from huggingface_hub import HfApi, list_datasets, delete_repo
from huggingface_hub.utils import HfHubHTTPError

from ..common.config import HFConfig
from ..common.exceptions import DatasetNotFoundError, UploadError
from ..common.logger import setup_logger
from ..auth.manager import AuthManager

logger = setup_logger(__name__)

class DatasetManager:
    """Manage dataset repositories on Hugging Face Hub."""
    
    def __init__(self, config: Optional[HFConfig] = None, auth_manager: Optional[AuthManager] = None):
        self.config = config or HFConfig()
        self.auth_manager = auth_manager or AuthManager(self.config)
        self.api = HfApi(endpoint=self.config.endpoint, token=self.config.token)
    
    def list_user_datasets(self, author: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List datasets owned by user or organization."""
        try:
            if author is None:
                if not self.auth_manager.is_authenticated():
                    raise UploadError("Authentication required to list user datasets")
                author = self.auth_manager.get_username()
            
            datasets = list_datasets(
                author=author,
                token=self.config.token,
                limit=limit
            )
            
            dataset_list = []
            for dataset in datasets:
                dataset_list.append({
                    "id": dataset.id,
                    "created_at": dataset.created_at,
                    "last_modified": dataset.last_modified,
                    "private": dataset.private,
                    "downloads": dataset.downloads,
                    "likes": dataset.likes,
                    "tags": dataset.tags
                })
            
            logger.info(f"Found {len(dataset_list)} datasets for author: {author}")
            return dataset_list
            
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            raise
    
    def delete_dataset(self, repo_id: str, missing_ok: bool = False) -> bool:
        """Delete a dataset repository."""
        try:
            if not self.auth_manager.is_authenticated():
                raise UploadError("Authentication required to delete datasets")
            
            self.auth_manager.validate_repo_access(repo_id, "dataset", "delete")
            
            delete_repo(
                repo_id=repo_id,
                repo_type="dataset",
                token=self.config.token
            )
            
            logger.info(f"Dataset repository deleted: {repo_id}")
            return True
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                if missing_ok:
                    logger.info(f"Dataset repository not found (ignored): {repo_id}")
                    return True
                raise DatasetNotFoundError(f"Dataset not found: {repo_id}")
            raise UploadError(f"Failed to delete dataset: {str(e)}")
    
    def update_dataset_visibility(self, repo_id: str, private: bool) -> bool:
        """Update dataset repository visibility."""
        try:
            if not self.auth_manager.is_authenticated():
                raise UploadError("Authentication required to update dataset visibility")
            
            self.auth_manager.validate_repo_access(repo_id, "dataset", "write")
            
            self.api.update_repo_visibility(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=self.config.token
            )
            
            visibility = "private" if private else "public"
            logger.info(f"Dataset repository visibility updated to {visibility}: {repo_id}")
            return True
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise DatasetNotFoundError(f"Dataset not found: {repo_id}")
            raise UploadError(f"Failed to update visibility: {str(e)}")
    
    def search_datasets(self,
                       query: Optional[str] = None,
                       author: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       language: Optional[str] = None,
                       multilinguality: Optional[str] = None,
                       size: Optional[str] = None,
                       limit: int = 20) -> List[Dict[str, Any]]:
        """Search for datasets on Hugging Face Hub."""
        try:
            datasets = list_datasets(
                search=query,
                author=author,
                tags=tags,
                language=language,
                multilinguality=multilinguality,
                size=size,
                limit=limit,
                token=self.config.token
            )
            
            results = []
            for dataset in datasets:
                results.append({
                    "id": dataset.id,
                    "downloads": dataset.downloads,
                    "likes": dataset.likes,
                    "created_at": dataset.created_at,
                    "last_modified": dataset.last_modified,
                    "tags": dataset.tags
                })
            
            logger.info(f"Found {len(results)} datasets matching search criteria")
            return results
            
        except Exception as e:
            logger.error(f"Dataset search failed: {str(e)}")
            raise
    
    def get_dataset_stats(self, repo_id: str) -> Dict[str, Any]:
        """Get statistics for a dataset repository."""
        try:
            info = self.api.dataset_info(
                repo_id=repo_id,
                token=self.config.token
            )
            
            return {
                "downloads": info.downloads,
                "likes": info.likes,
                "created_at": info.created_at,
                "last_modified": info.last_modified,
                "total_size": sum(f.size or 0 for f in info.siblings),
                "num_files": len(info.siblings),
                "tags": info.tags
            }
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise DatasetNotFoundError(f"Dataset not found: {repo_id}")
            raise UploadError(f"Failed to get dataset stats: {str(e)}")
    
    def move_dataset(self, from_id: str, to_id: str) -> bool:
        """Move/rename a dataset repository."""
        try:
            if not self.auth_manager.is_authenticated():
                raise UploadError("Authentication required to move datasets")
            
            self.auth_manager.validate_repo_access(from_id, "dataset", "delete")
            self.auth_manager.validate_repo_access(to_id, "dataset", "write")
            
            self.api.move_repo(
                from_id=from_id,
                to_id=to_id,
                repo_type="dataset",
                token=self.config.token
            )
            
            logger.info(f"Dataset repository moved from {from_id} to {to_id}")
            return True
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise DatasetNotFoundError(f"Dataset not found: {from_id}")
            raise UploadError(f"Failed to move dataset: {str(e)}")
    
    def clone_dataset(self, source_repo_id: str, target_repo_id: str, 
                     private: bool = False) -> str:
        """Clone a dataset repository."""
        try:
            if not self.auth_manager.is_authenticated():
                raise UploadError("Authentication required to clone datasets")
            
            # Download source dataset
            from .download import DatasetDownloader
            downloader = DatasetDownloader(self.config, self.auth_manager)
            dataset = downloader.download_dataset(source_repo_id)
            
            # Upload to target repository
            from .upload import DatasetUploader
            uploader = DatasetUploader(self.config, self.auth_manager)
            
            # Create temp directory and save dataset
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                if hasattr(dataset, 'to_parquet'):
                    # Single dataset
                    dataset.to_parquet(str(temp_path / "data.parquet"))
                else:
                    # Handle DatasetDict with multiple splits
                    for split_name, split_data in dataset.items():
                        split_path = temp_path / f"{split_name}.parquet"
                        split_data.to_parquet(str(split_path))
                
                repo_url = uploader.upload_dataset(
                    data_path=str(temp_path),
                    repo_id=target_repo_id,
                    private=private,
                    commit_message=f"Clone from {source_repo_id}",
                    convert_to_parquet=False  # Already in Parquet format
                )
            
            logger.info(f"Dataset cloned from {source_repo_id} to {target_repo_id}")
            return repo_url
            
        except Exception as e:
            error_msg = f"Failed to clone dataset: {str(e)}"
            logger.error(error_msg)
            raise UploadError(error_msg)
    
    def add_dataset_tags(self, repo_id: str, tags: List[str]) -> bool:
        """Add tags to a dataset repository."""
        try:
            if not self.auth_manager.is_authenticated():
                raise UploadError("Authentication required to modify dataset tags")
            
            self.auth_manager.validate_repo_access(repo_id, "dataset", "write")
            
            # Get current dataset card
            from huggingface_hub import DatasetCard
            try:
                card = DatasetCard.load(repo_id, token=self.config.token)
            except:
                card = DatasetCard("")
            
            # Update tags
            if not hasattr(card, 'data'):
                card.data = {}
            
            existing_tags = card.data.get('tags', [])
            new_tags = list(set(existing_tags + tags))  # Remove duplicates
            card.data['tags'] = new_tags
            
            # Save updated card
            card.push_to_hub(
                repo_id=repo_id,
                repo_type="dataset",
                token=self.config.token
            )
            
            logger.info(f"Added tags {tags} to dataset {repo_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to add tags: {str(e)}"
            logger.error(error_msg)
            raise UploadError(error_msg)
    
    def validate_dataset_format(self, repo_id: str) -> Dict[str, Any]:
        """Validate dataset format and structure."""
        try:
            files = self.list_dataset_files(repo_id)
            
            validation_results = {
                "valid": True,
                "issues": [],
                "recommendations": [],
                "file_count": len(files),
                "total_size": sum(f.get("size", 0) or 0 for f in files),
                "formats": []
            }
            
            # Check file formats
            formats = set()
            for file_info in files:
                filename = file_info["filename"]
                if "." in filename:
                    ext = filename.split(".")[-1].lower()
                    formats.add(ext)
            
            validation_results["formats"] = list(formats)
            
            # Common validations
            has_readme = any(f["filename"].lower() == "readme.md" for f in files)
            if not has_readme:
                validation_results["issues"].append("Missing README.md file")
                validation_results["recommendations"].append("Add a README.md with dataset description")
            
            # Check for common data formats
            data_formats = {"csv", "json", "jsonl", "parquet", "arrow", "txt"}
            has_data_files = any(ext in data_formats for ext in formats)
            
            if not has_data_files:
                validation_results["issues"].append("No recognized data format files found")
                validation_results["valid"] = False
            
            logger.info(f"Dataset validation completed for {repo_id}")
            return validation_results
            
        except Exception as e:
            error_msg = f"Failed to validate dataset: {str(e)}"
            logger.error(error_msg)
            raise UploadError(error_msg)
    
    def list_dataset_files(self, repo_id: str, revision: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all files in a dataset repository."""
        try:
            info = self.api.dataset_info(
                repo_id=repo_id,
                revision=revision,
                token=self.config.token
            )
            
            files = []
            for sibling in info.siblings:
                files.append({
                    "filename": sibling.rfilename,
                    "size": sibling.size,
                    "blob_id": getattr(sibling, 'blob_id', None),
                    "lfs": getattr(sibling, 'lfs', None)
                })
            
            return files
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise DatasetNotFoundError(f"Dataset not found: {repo_id}")
            raise UploadError(f"Failed to list dataset files: {str(e)}")