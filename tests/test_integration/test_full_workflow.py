# tests/test_integration/test_full_workflow.py
# Integration tests for complete workflows

import pytest
from unittest.mock import patch, Mock

@pytest.mark.integration
@pytest.mark.slow
class TestFullWorkflow:
    """Integration tests for complete workflows."""
    
    @patch('hf_utils.models.upload.HfApi')
    @patch('hf_utils.models.upload.create_repo')
    @patch('hf_utils.models.upload.upload_folder')
    def test_model_upload_workflow(self, mock_upload, mock_create, mock_api,
                                 mock_config, mock_auth_manager, sample_model_files):
        """Test complete model upload workflow."""
        from hf_utils.models.upload import ModelUploader
        
        # Setup mocks
        mock_create.return_value = "https://huggingface.co/user/model"
        mock_upload.return_value = Mock(oid="abc123")
        
        # Test workflow
        uploader = ModelUploader(mock_config, mock_auth_manager)
        result = uploader.upload_model(
            model_path=str(sample_model_files),
            repo_id="user/test-model",
            model_card={"license": "apache-2.0"}
        )
        
        # Verify result
        assert result == "https://huggingface.co/user/model"
        assert mock_create.called
        assert mock_upload.called