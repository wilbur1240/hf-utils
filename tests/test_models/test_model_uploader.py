# tests/test_models/test_model_uploader.py
# Unit tests for model uploader

import pytest
from unittest.mock import patch, Mock
from hf_utils.models.upload import ModelUploader
from hf_utils.common.exceptions import ValidationError, UploadError

@pytest.mark.unit
class TestModelUploader:
    """Test cases for ModelUploader class."""
    
    def test_init(self, mock_config, mock_auth_manager):
        """Test ModelUploader initialization."""
        uploader = ModelUploader(mock_config, mock_auth_manager)
        assert uploader.config == mock_config
        assert uploader.auth_manager == mock_auth_manager
    
    def test_upload_model_invalid_repo_name(self, mock_config, mock_auth_manager, temp_dir):
        """Test upload with invalid repository name."""
        uploader = ModelUploader(mock_config, mock_auth_manager)
        
        with pytest.raises(ValidationError, match="Invalid repository name"):
            uploader.upload_model(
                model_path=str(temp_dir),
                repo_id="invalid-repo-name"  # Missing username/
            )
    
    def test_upload_model_nonexistent_path(self, mock_config, mock_auth_manager):
        """Test upload with non-existent model path."""
        uploader = ModelUploader(mock_config, mock_auth_manager)
        
        with pytest.raises(ValidationError, match="Model path does not exist"):
            uploader.upload_model(
                model_path="/nonexistent/path",
                repo_id="user/model"
            )
    
    @patch('hf_utils.models.upload.create_repo')
    @patch('hf_utils.models.upload.upload_folder')
    def test_upload_model_directory_success(self, mock_upload_folder, mock_create_repo,
                                          mock_config, mock_auth_manager, sample_model_files):
        """Test successful model directory upload."""
        # Setup mocks
        mock_create_repo.return_value = "https://huggingface.co/user/model"
        mock_upload_folder.return_value = Mock(oid="abc123")
        
        uploader = ModelUploader(mock_config, mock_auth_manager)
        result = uploader.upload_model(
            model_path=str(sample_model_files),
            repo_id="user/test-model"
        )
        
        # Assertions
        assert result == "https://huggingface.co/user/model"
        mock_create_repo.assert_called_once()
        mock_upload_folder.assert_called_once()