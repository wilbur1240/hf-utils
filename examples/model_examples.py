# examples/model_examples.py
#!/usr/bin/env python3
"""
Example usage of HF Utils for model operations.
"""

import os
from pathlib import Path
import pandas as pd

# Import HF Utils components
from hf_utils import HFConfig, AuthManager
from hf_utils.models.upload import ModelUploader
from hf_utils.models.download import ModelDownloader
from hf_utils.models.manage import ModelManager

def setup_authentication():
    """Setup authentication with Hugging Face."""
    config = HFConfig()
    auth_manager = AuthManager(config)
    
    # Authenticate (will prompt for token if needed)
    if not auth_manager.is_authenticated():
        print("Please authenticate with Hugging Face...")
        auth_manager.authenticate()
    
    return config, auth_manager

def example_upload_model():
    """Example: Upload a model to Hugging Face Hub."""
    print("=== Model Upload Example ===")
    
    config, auth_manager = setup_authentication()
    uploader = ModelUploader(config, auth_manager)
    
    # Upload model directory
    repo_url = uploader.upload_model(
        model_path="./my-model",  # Local model directory
        repo_id="username/my-awesome-model",
        destination=None,  # Upload to root of repo
        private=False,
        commit_message="Upload my awesome model",
        model_card={
            "license": "apache-2.0",
            "tags": ["text-classification", "pytorch"],
            "datasets": ["imdb"],
            "language": ["en"]
        }
    )
    
    print(f"✅ Model uploaded to: {repo_url}")

def example_download_model():
    """Example: Download a model from Hugging Face Hub."""
    print("=== Model Download Example ===")
    
    config, auth_manager = setup_authentication()
    downloader = ModelDownloader(config, auth_manager)
    
    # Download complete model
    model_path = downloader.download_model(
        repo_id="bert-base-uncased",
        destination="./downloaded-models/bert-base-uncased",
        revision=None,  # Latest version
        force_download=False
    )
    
    print(f"✅ Model downloaded to: {model_path}")
    
    # Download specific file
    config_path = downloader.download_file(
        repo_id="bert-base-uncased",
        filename="config.json",
        destination="./configs/bert-config.json"
    )
    
    print(f"✅ Config file downloaded to: {config_path}")

def example_manage_models():
    """Example: Manage model repositories."""
    print("=== Model Management Example ===")
    
    config, auth_manager = setup_authentication()
    manager = ModelManager(config, auth_manager)
    
    # List user's models
    models = manager.list_user_models(limit=5)
    print(f"Found {len(models)} models:")
    for model in models:
        print(f"  - {model['id']} ({model['downloads']:,} downloads)")
    
    # Search for models
    results = manager.search_models(
        query="text classification",
        pipeline_tag="text-classification",
        limit=10
    )
    print(f"\nFound {len(results)} text classification models")
    
    # Get model info
    try:
        info = manager.get_model_stats("bert-base-uncased")
        print(f"\nBERT base stats:")
        print(f"  Downloads: {info['downloads']:,}")
        print(f"  Likes: {info['likes']}")
        print(f"  Files: {info['num_files']}")
    except Exception as e:
        print(f"Could not get model stats: {e}")

def example_batch_operations():
    """Example: Batch model operations."""
    print("=== Batch Model Operations Example ===")
    
    config, auth_manager = setup_authentication()
    uploader = ModelUploader(config, auth_manager)
    downloader = ModelDownloader(config, auth_manager)
    
    # List of models to download
    models_to_download = [
        "distilbert-base-uncased",
        "roberta-base",
        "albert-base-v2"
    ]
    
    download_dir = Path("./batch-downloads")
    download_dir.mkdir(exist_ok=True)
    
    for model_id in models_to_download:
        try:
            print(f"Downloading {model_id}...")
            destination = download_dir / model_id.replace("/", "_")
            
            downloader.download_model(
                repo_id=model_id,
                destination=str(destination),
                allow_patterns=["*.json", "*.txt"]  # Only config and text files
            )
            
            print(f"✅ Downloaded {model_id}")
        except Exception as e:
            print(f"❌ Failed to download {model_id}: {e}")

if __name__ == "__main__":
    try:
        example_upload_model()
        example_download_model()
        example_manage_models()
        example_batch_operations()
    except KeyboardInterrupt:
        print("\nOperations cancelled by user")
    except Exception as e:
        print(f"Error: {e}")