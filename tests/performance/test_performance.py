# tests/performance/test_performance.py
# Performance tests

import pytest
import time
from unittest.mock import Mock, patch

@pytest.mark.slow
class TestPerformance:
    """Performance tests."""
    
    def test_large_dataset_processing_performance(self, mock_config, mock_auth_manager):
        """Test performance with large datasets."""
        from hf_utils.datasets.upload import DatasetUploader
        import pandas as pd
        
        # Create large DataFrame (10k rows)
        large_df = pd.DataFrame({
            'text': [f'Sample text {i}' for i in range(10000)],
            'label': [i % 5 for i in range(10000)],
            'score': [0.1 * (i % 10) for i in range(10000)]
        })
        
        uploader = DatasetUploader(mock_config, mock_auth_manager)
        
        # Measure time for DataFrame operations
        start_time = time.time()
        
        # Mock the actual upload to avoid network calls
        with patch.object(uploader, 'upload_dataset') as mock_upload:
            mock_upload.return_value = "https://huggingface.co/user/dataset"
            
            result = uploader.upload_dataframe(
                df=large_df,
                repo_id="user/large-dataset"
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 10k rows in under 5 seconds
        assert processing_time < 5.0
        assert result == "https://huggingface.co/user/dataset"