# src/models/manage.py
from typing import List, Dict, Any, Optional
from huggingface_hub import HfApi, list_models, delete_repo
from huggingface_hub.utils import HfHubHTTPError

from ..common.config import HFConfig
from ..common.exceptions import ModelNotFoundError, UploadError
from ..common.logger import setup_logger
from ..auth.manager import AuthManager

logger = setup_logger(__name__)

class ModelManager:
    """Manage model repositories on Hugging Face Hub."""
    
    def __init__(self, config: Optional[HFConfig] = None, auth_manager: Optional[AuthManager] = None):
        self.config = config or HFConfig()
        self.auth_manager = auth_manager or AuthManager(self.config)
        self.api = HfApi(endpoint=self.config.endpoint, token=self.config.token)
    
    def list_user_models(self, author: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List models owned by user or organization.
        
        Args:
            author: Author/organization name (defaults to current user)
            limit: Maximum number of models to return
            
        Returns:
            List of model information
        """
        try:
            if author is None:
                if not self.auth_manager.is_authenticated():
                    raise UploadError("Authentication required to list user models")
                author = self.auth_manager.get_username()
            
            models = list_models(
                author=author,
                token=self.config.token,
                limit=limit
            )
            
            model_list = []
            for model in models:
                model_list.append({
                    "id": model.id,
                    "created_at": model.created_at,
                    "last_modified": model.last_modified,
                    "private": model.private,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags,
                    "pipeline_tag": model.pipeline_tag
                })
            
            logger.info(f"Found {len(model_list)} models for author: {author}")
            return model_list
            
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise
    
    def delete_model(self, repo_id: str, missing_ok: bool = False) -> bool:
        """
        Delete a model repository.
        
        Args:
            repo_id: Repository ID to delete
            missing_ok: Don't raise error if repository doesn't exist
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            # Validate user has access to delete
            if not self.auth_manager.is_authenticated():
                raise UploadError("Authentication required to delete models")
            
            self.auth_manager.validate_repo_access(repo_id, "model", "delete")
            
            delete_repo(
                repo_id=repo_id,
                repo_type="model",
                token=self.config.token
            )
            
            logger.info(f"Model repository deleted: {repo_id}")
            return True
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                if missing_ok:
                    logger.info(f"Model repository not found (ignored): {repo_id}")
                    return True
                raise ModelNotFoundError(f"Model not found: {repo_id}")
            raise UploadError(f"Failed to delete model: {str(e)}")
    
    def update_model_visibility(self, repo_id: str, private: bool) -> bool:
        """
        Update model repository visibility.
        
        Args:
            repo_id: Repository ID
            private: True for private, False for public
            
        Returns:
            bool: True if updated successfully
        """
        try:
            if not self.auth_manager.is_authenticated():
                raise UploadError("Authentication required to update model visibility")
            
            self.auth_manager.validate_repo_access(repo_id, "model", "write")
            
            self.api.update_repo_visibility(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                token=self.config.token
            )
            
            visibility = "private" if private else "public"
            logger.info(f"Model repository visibility updated to {visibility}: {repo_id}")
            return True
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {repo_id}")
            raise UploadError(f"Failed to update visibility: {str(e)}")
    
    def move_model(self, from_id: str, to_id: str) -> bool:
        """
        Move/rename a model repository.
        
        Args:
            from_id: Current repository ID
            to_id: New repository ID
            
        Returns:
            bool: True if moved successfully
        """
        try:
            if not self.auth_manager.is_authenticated():
                raise UploadError("Authentication required to move models")
            
            self.auth_manager.validate_repo_access(from_id, "model", "delete")
            self.auth_manager.validate_repo_access(to_id, "model", "write")
            
            self.api.move_repo(
                from_id=from_id,
                to_id=to_id,
                repo_type="model",
                token=self.config.token
            )
            
            logger.info(f"Model repository moved from {from_id} to {to_id}")
            return True
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {from_id}")
            raise UploadError(f"Failed to move model: {str(e)}")
    
    def get_model_stats(self, repo_id: str) -> Dict[str, Any]:
        """
        Get statistics for a model repository.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            Dict containing model statistics
        """
        try:
            info = self.api.model_info(
                repo_id=repo_id,
                repo_type="model",
                token=self.config.token
            )
            
            return {
                "downloads": info.downloads,
                "likes": info.likes,
                "created_at": info.created_at,
                "last_modified": info.last_modified,
                "total_size": sum(f.size or 0 for f in info.siblings),
                "num_files": len(info.siblings),
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag
            }
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {repo_id}")
            raise UploadError(f"Failed to get model stats: {str(e)}")
    
    def search_models(self, 
                     query: Optional[str] = None,
                     author: Optional[str] = None, 
                     tags: Optional[List[str]] = None,
                     pipeline_tag: Optional[str] = None,
                     library: Optional[str] = None,
                     language: Optional[str] = None,
                     limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for models on Hugging Face Hub.
        
        Args:
            query: Search query
            author: Filter by author
            tags: Filter by tags
            pipeline_tag: Filter by pipeline tag
            library: Filter by library (transformers, diffusers, etc.)
            language: Filter by language
            limit: Maximum results to return
            
        Returns:
            List of model information
        """
        try:
            models = list_models(
                search=query,
                author=author,
                tags=tags,
                pipeline_tag=pipeline_tag,
                library=library,
                language=language,
                limit=limit,
                token=self.config.token
            )
            
            results = []
            for model in models:
                results.append({
                    "id": model.id,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "created_at": model.created_at,
                    "last_modified": model.last_modified,
                    "tags": model.tags,
                    "pipeline_tag": model.pipeline_tag,
                    "library": getattr(model, 'library_name', None)
                })
            
            logger.info(f"Found {len(results)} models matching search criteria")
            return results
            
        except Exception as e:
            logger.error(f"Model search failed: {str(e)}")
            raise