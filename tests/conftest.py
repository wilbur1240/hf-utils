# tests/conftest.py
# Central configuration file for pytest - fixtures and test setup

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from hf_utils.common.config import HFConfig
from hf_utils.auth.manager import AuthManager

# =============================================================================
# TEST FIXTURES (Reusable test components)
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def mock_config():
    """Mock HF configuration for testing."""
    config = HFConfig(
        cache_dir="/tmp/test_cache",
        token="test_token_123",
        endpoint="https://huggingface.co",
        max_retries=1,
        timeout=30
    )
    return config

@pytest.fixture
def mock_auth_manager(mock_config):
    """Mock authentication manager."""
    auth_manager = Mock(spec=AuthManager)
    auth_manager.config = mock_config
    auth_manager.is_authenticated.return_value = True
    auth_manager.get_username.return_value = "test_user"
    auth_manager.get_user_info.return_value = {
        "name": "test_user",
        "email": "test@example.com"
    }
    return auth_manager

@pytest.fixture
def sample_model_files(temp_dir):
    """Create sample model files for testing."""
    model_dir = temp_dir / "sample_model"
    model_dir.mkdir()
    
    # Create dummy model files
    (model_dir / "config.json").write_text('{"model_type": "bert"}')
    (model_dir / "pytorch_model.bin").write_bytes(b"fake_model_data")
    (model_dir / "tokenizer.json").write_text('{"tokenizer": "test"}')
    (model_dir / "README.md").write_text("# Test Model")
    
    return model_dir

@pytest.fixture
def sample_dataset_csv(temp_dir):
    """Create sample CSV dataset for testing."""
    data = {
        'text': ['Hello world', 'How are you?', 'Testing data'],
        'label': [0, 1, 2],
        'score': [0.95, 0.87, 0.92]
    }
    df = pd.DataFrame(data)
    csv_path = temp_dir / "sample_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def sample_dataset_json(temp_dir):
    """Create sample JSON dataset for testing."""
    import json
    data = [
        {"text": "Hello world", "label": 0, "score": 0.95},
        {"text": "How are you?", "label": 1, "score": 0.87},
        {"text": "Testing data", "label": 2, "score": 0.92}
    ]
    json_path = temp_dir / "sample_dataset.json"
    with open(json_path, 'w') as f:
        json.dump(data, f)
    return json_path

@pytest.fixture
def mock_hf_api():
    """Mock Hugging Face API responses."""
    with patch('hf_utils.models.upload.HfApi') as mock_api:
        # Configure mock responses
        mock_instance = mock_api.return_value
        mock_instance.create_repo.return_value = "https://huggingface.co/test_user/test_repo"
        mock_instance.upload_folder.return_value = Mock(oid="abc123")
        mock_instance.upload_file.return_value = Mock(oid="def456")
        
        yield mock_instance

# =============================================================================
# TEST MARKERS (Categories of tests)
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_auth: Tests that need authentication")
    config.addinivalue_line("markers", "requires_internet: Tests that need internet")