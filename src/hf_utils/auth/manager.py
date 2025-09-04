# src/auth/manager.py
import os
import getpass
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import HfApi, login, logout, whoami
from huggingface_hub.utils import HfHubHTTPError

from ..common.config import HFConfig
from ..common.exceptions import AuthenticationError
from ..common.logger import setup_logger

logger = setup_logger(__name__)

class AuthManager:
    """Manage Hugging Face authentication."""
    
    def __init__(self, config: Optional[HFConfig] = None):
        self.config = config or HFConfig()
        self.api = HfApi(endpoint=self.config.endpoint)
        self._user_info: Optional[Dict[str, Any]] = None
    
    def authenticate(self, token: Optional[str] = None, add_to_git_credential: bool = True) -> bool:
        """
        Authenticate with Hugging Face Hub.
        
        Args:
            token: HF token. If None, will try to get from config or prompt user
            add_to_git_credential: Whether to add token to git credential store
            
        Returns:
            bool: True if authentication successful
        """
        auth_token = token or self.config.token
        
        if not auth_token:
            auth_token = self._prompt_for_token()
        
        try:
            login(token=auth_token, add_to_git_credential=add_to_git_credential)
            self.config.token = auth_token
            
            # Verify authentication by getting user info
            self._user_info = self.get_user_info()
            logger.info(f"Successfully authenticated as: {self._user_info.get('name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise AuthenticationError(f"Failed to authenticate: {str(e)}")
    
    def logout(self) -> None:
        """Logout from Hugging Face Hub."""
        try:
            logout()
            self.config.token = None
            self._user_info = None
            
            # Remove saved token file
            token_path = Path.home() / ".huggingface" / "token"
            if token_path.exists():
                try:
                    token_path.unlink()
                    logger.info("Removed saved token file")
                except Exception as e:
                    logger.warning(f"Could not remove token file: {e}")
            
            logger.info("Successfully logged out")
        except Exception as e:
            logger.error(f"Logout failed: {str(e)}")
            raise AuthenticationError(f"Failed to logout: {str(e)}")
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        try:
            if not self.config.token:
                return False
            
            # Try to get user info to verify token is valid
            self.get_user_info()
            return True
            
        except Exception:
            return False
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get current user information."""
        try:
            if self._user_info is None:
                self._user_info = whoami(token=self.config.token)
            return self._user_info
            
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid or expired token")
            raise AuthenticationError(f"Failed to get user info: {str(e)}")
        except Exception as e:
            raise AuthenticationError(f"Failed to get user info: {str(e)}")
    
    def get_username(self) -> str:
        """Get current username."""
        user_info = self.get_user_info()
        return user_info.get("name", "unknown")
    
    def get_organizations(self) -> list:
        """Get list of organizations user belongs to."""
        user_info = self.get_user_info()
        return user_info.get("orgs", [])
    
    def can_write_to_repo(self, repo_id: str, repo_type: str = "model") -> bool:
        """Check if user can write to a repository."""
        try:
            # Try to get repo info - will fail if no access
            repo_info = self.api.repo_info(
                repo_id=repo_id,
                repo_type=repo_type,
                token=self.config.token
            )
            return True
        except HfHubHTTPError as e:
            if e.response.status_code in [401, 403, 404]:
                return False
            raise AuthenticationError(f"Error checking repo access: {str(e)}")
    
    def validate_repo_access(self, repo_id: str, repo_type: str = "model", 
                           operation: str = "write") -> None:
        """Validate user has required access to repository."""
        if not self.is_authenticated():
            raise AuthenticationError("Not authenticated. Please login first.")
        
        username = self.get_username()
        repo_owner = repo_id.split("/")[0] if "/" in repo_id else username
        
        # Check if user owns the repo or belongs to the organization
        if repo_owner != username and repo_owner not in [org["name"] for org in self.get_organizations()]:
            if not self.can_write_to_repo(repo_id, repo_type):
                raise AuthenticationError(
                    f"You don't have {operation} access to {repo_type} repository: {repo_id}"
                )
    
    def _prompt_for_token(self) -> str:
        """Interactively prompt user for token."""
        print("Please enter your Hugging Face token.")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        
        token = getpass.getpass("Token: ").strip()
        if not token:
            raise AuthenticationError("Token is required for authentication")
        
        return token
    
    def save_token_to_file(self, token_path: Optional[str] = None) -> None:
        """Save token to file for future use."""
        if not self.config.token:
            raise AuthenticationError("No token to save")
        
        if token_path is None:
            token_path = Path.home() / ".huggingface" / "token"
        
        token_file = Path(token_path)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(token_file, "w") as f:
            f.write(self.config.token)
        
        # Set restrictive permissions
        token_file.chmod(0o600)
        logger.info(f"Token saved to {token_file}")
    
    def load_token_from_file(self, token_path: Optional[str] = None) -> Optional[str]:
        """Load token from file."""
        if token_path is None:
            token_path = Path.home() / ".huggingface" / "token"
        
        token_file = Path(token_path)
        if token_file.exists():
            try:
                with open(token_file, "r") as f:
                    token = f.read().strip()
                return token if token else None
            except Exception as e:
                logger.warning(f"Failed to load token from file: {str(e)}")
        
        return None