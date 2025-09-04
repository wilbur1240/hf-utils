# tests/test_auth/test_auth_manager.py
# Unit tests for authentication manager

import pytest
from unittest.mock import patch, Mock
from hf_utils.auth.manager import AuthManager
from hf_utils.common.exceptions import AuthenticationError

@pytest.mark.unit
class TestAuthManager:
    """Test cases for AuthManager class."""
    
    def test_init_with_config(self, mock_config):
        """Test AuthManager initialization."""
        auth_manager = AuthManager(mock_config)
        assert auth_manager.config == mock_config
        assert auth_manager._user_info is None
    
    def test_init_without_config(self):
        """Test AuthManager initialization without config."""
        auth_manager = AuthManager()
        assert auth_manager.config is not None
        assert hasattr(auth_manager.config, 'token')
    
    @patch('hf_utils.auth.manager.login')
    @patch('hf_utils.auth.manager.whoami')
    def test_authenticate_success(self, mock_whoami, mock_login, mock_config):
        """Test successful authentication."""
        # Setup mocks
        mock_whoami.return_value = {"name": "test_user"}
        
        auth_manager = AuthManager(mock_config)
        result = auth_manager.authenticate(token="test_token")
        
        # Assertions
        assert result is True
        mock_login.assert_called_once_with(token="test_token", add_to_git_credential=True)
        mock_whoami.assert_called_once()
        assert auth_manager.config.token == "test_token"
    
    @patch('hf_utils.auth.manager.login')
    def test_authenticate_failure(self, mock_login, mock_config):
        """Test authentication failure."""
        # Setup mock to raise exception
        mock_login.side_effect = Exception("Invalid token")
        
        auth_manager = AuthManager(mock_config)
        
        with pytest.raises(AuthenticationError):
            auth_manager.authenticate(token="invalid_token")
    
    @patch('hf_utils.auth.manager.whoami')
    def test_is_authenticated_true(self, mock_whoami, mock_config):
        """Test is_authenticated returns True for valid token."""
        mock_whoami.return_value = {"name": "test_user"}
        
        auth_manager = AuthManager(mock_config)
        assert auth_manager.is_authenticated() is True
    
    @patch('hf_utils.auth.manager.whoami')
    def test_is_authenticated_false(self, mock_whoami, mock_config):
        """Test is_authenticated returns False for invalid token."""
        from huggingface_hub.utils import HfHubHTTPError
        import requests
        
        # Create mock response for 401 error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_whoami.side_effect = HfHubHTTPError("Unauthorized", response=mock_response)
        
        auth_manager = AuthManager(mock_config)
        assert auth_manager.is_authenticated() is False
    
    def test_get_username(self, mock_auth_manager):
        """Test getting username."""
        username = mock_auth_manager.get_username()
        assert username == "test_user"
