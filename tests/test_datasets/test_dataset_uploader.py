# tests/test_datasets/test_dataset_uploader.py
# Unit tests for dataset uploader

import pytest
from unittest.mock import patch, Mock
import pandas as pd
from hf_utils.datasets.upload import DatasetUploader

@pytest.mark.unit
class TestDatasetUploader:
    """Test cases for DatasetUploader class."""
    
    def test_upload_dataframe(self, mock_config, mock_auth_manager, temp_dir):
        """Test uploading pandas DataFrame."""
        uploader = DatasetUploader(mock_config, mock_auth_manager)
        
        # Create test DataFrame
        df = pd.DataFrame({
            'text': ['Hello', 'World'], 
            'label': [0, 1]
        })
        
        with patch.object(uploader, 'upload_dataset') as mock_upload:
            mock_upload.return_value = "https://huggingface.co/user/dataset"
            
            result = uploader.upload_dataframe(
                df=df,
                repo_id="user/test-dataset",
                filename="data.parquet"
            )
            
            assert result == "https://huggingface.co/user/dataset"
            mock_upload.assert_called_once()
    
    def test_convert_to_parquet(self, mock_config, mock_auth_manager, sample_dataset_csv, temp_dir):
        """Test CSV to Parquet conversion."""
        uploader = DatasetUploader(mock_config, mock_auth_manager)
        
        output_path = temp_dir / "output.parquet"
        uploader._convert_to_parquet(sample_dataset_csv, output_path)
        
        # Verify Parquet file was created
        assert output_path.exists()
        
        # Verify data integrity
        original_df = pd.read_csv(sample_dataset_csv)
        converted_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(original_df, converted_df)