# HF Utils - Hugging Face Hub Utilities

A comprehensive, modular toolkit for working with Hugging Face Hub. Upload, download, and manage models, datasets, and spaces with clear destination specification and separated application logic.

## 🚀 Features

### 🤖 Model Operations
- **Upload**: Push models with metadata, model cards, and version control
- **Download**: Fetch complete models or specific files with caching
- **Manage**: List, delete, update visibility, search, and get statistics

### 📊 Dataset Operations  
- **Upload**: Push datasets with automatic format conversion (CSV/JSON → Parquet)
- **Download**: Fetch datasets with streaming support and DataFrame conversion
- **Manage**: Repository management, search, cloning, and statistics

### 🚀 Spaces Operations (Coming Soon)
- **Deploy**: Create Gradio and Streamlit applications
- **Manage**: Update configurations, secrets, and monitor status
- **Monitor**: Check deployment status, logs, and usage metrics

### 🔐 Authentication
- Multiple authentication methods (token, environment, interactive)
- Token validation and management
- Organization access control

### ⚙️ Configuration
- Environment-based configuration
- Flexible destination specification
- Caching and performance settings

## 📦 Installation

```bash
# Install from source
git clone https://github.com/wilbur1240/hf-utils.git
cd hf-utils
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## 🔧 Quick Start

### Authentication

```python
from hf_utils import HFConfig, AuthManager

# Setup configuration
config = HFConfig.from_env()
auth_manager = AuthManager(config)

# Authenticate (interactive prompt if no token)
auth_manager.authenticate()

# Or provide token directly
auth_manager.authenticate(token="your_hf_token_here")
```

### Model Operations

```python
from hf_utils.models import ModelUploader, ModelDownloader, ModelManager

# Upload a model
uploader = ModelUploader(config, auth_manager)
repo_url = uploader.upload_model(
    model_path="./my-model",
    repo_id="username/my-model",
    private=False,
    model_card={"license": "apache-2.0", "tags": ["nlp"]}
)

# Download a model
downloader = ModelDownloader(config, auth_manager)
model_path = downloader.download_model(
    repo_id="bert-base-uncased",
    destination="./models/bert"
)

# Manage models
manager = ModelManager(config, auth_manager)
models = manager.list_user_models(limit=10)
```

### Dataset Operations

```python
from hf_utils.datasets import DatasetUploader, DatasetDownloader
import pandas as pd

# Upload DataFrame directly
uploader = DatasetUploader(config, auth_manager)
df = pd.DataFrame({"text": ["Hello", "World"], "label": [0, 1]})

repo_url = uploader.upload_dataframe(
    df=df,
    repo_id="username/my-dataset",
    filename="data.parquet"
)

# Download as DataFrame
downloader = DatasetDownloader(config, auth_manager)
df = downloader.download_as_dataframe(
    repo_id="imdb",
    split="train[:1000]"
)
```

## 🖥️ CLI Usage

```bash
# Authentication
hf-utils auth login
hf-utils auth whoami

# Model operations
hf-utils model upload ./my-model username/my-model --private
hf-utils model download bert-base-uncased --destination ./models/bert
hf-utils model list --author username
hf-utils model search "text classification" --limit 10

# Dataset operations  
hf-utils dataset upload ./data.csv username/my-dataset
hf-utils dataset download squad --split train --destination ./datasets/
hf-utils dataset download-df imdb --split test --output ./imdb_test.parquet
hf-utils dataset list --author username

# Get help
hf-utils --help
hf-utils model --help
hf-utils dataset --help
```

## 📁 Project Structure

```
hf-utils/
├── src/hf_utils/
│   ├── auth/           # Authentication management
│   ├── models/         # Model operations (upload, download, manage)
│   ├── datasets/       # Dataset operations (upload, download, manage)  
│   ├── spaces/         # Space operations (coming soon)
│   ├── common/         # Shared utilities (config, logging, validation)
│   └── cli/            # Command-line interface
├── examples/           # Usage examples
├── tests/             # Test suite
├── docs/              # Documentation
└── requirements.txt   # Dependencies
```

## ⚙️ Configuration

Create a `.env` file or set environment variables:

```bash
# Hugging Face token
HF_TOKEN=your_token_here

# Cache directory
HF_CACHE_DIR=~/.cache/huggingface

# Default destinations
DEFAULT_MODEL_DESTINATION=models
DEFAULT_DATASET_DESTINATION=datasets

# Performance settings
MAX_RETRIES=3
TIMEOUT=300
CHUNK_SIZE=8192
```

## 🧪 Examples

Check the `examples/` directory for comprehensive usage examples:

- `model_examples.py` - Model upload/download/management
- `dataset_examples.py` - Dataset operations and processing  

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face team for the excellent Hub platform and libraries