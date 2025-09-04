# examples/dataset_examples.py
#!/usr/bin/env python3
"""
Example usage of HF Utils for dataset operations.
"""

import pandas as pd
from pathlib import Path
import json

from hf_utils import HFConfig, AuthManager
from hf_utils.datasets.upload import DatasetUploader
from hf_utils.datasets.download import DatasetDownloader
from hf_utils.datasets.manage import DatasetManager

def setup_authentication():
    """Setup authentication with Hugging Face."""
    config = HFConfig()
    auth_manager = AuthManager(config)
    
    if not auth_manager.is_authenticated():
        print("Please authenticate with Hugging Face...")
        auth_manager.authenticate()
    
    return config, auth_manager

def example_upload_dataset():
    """Example: Upload a dataset to Hugging Face Hub."""
    print("=== Dataset Upload Example ===")
    
    config, auth_manager = setup_authentication()
    uploader = DatasetUploader(config, auth_manager)
    
    # Create sample dataset
    sample_data = {
        'text': ['Hello world', 'How are you?', 'Machine learning is fun'],
        'label': [0, 1, 2],
        'score': [0.95, 0.87, 0.92]
    }
    df = pd.DataFrame(sample_data)
    
    # Upload DataFrame directly
    repo_url = uploader.upload_dataframe(
        df=df,
        repo_id="username/my-sample-dataset",
        filename="data.parquet",
        private=False,
        commit_message="Upload sample dataset",
        dataset_card={
            "license": "mit",
            "tags": ["text-classification", "sample-data"],
            "task_categories": ["text-classification"],
            "language": ["en"],
            "size_categories": ["n<1K"]
        }
    )
    
    print(f"✅ Dataset uploaded to: {repo_url}")
    
    # Upload CSV file
    csv_path = "sample_data.csv"
    df.to_csv(csv_path, index=False)
    
    try:
        repo_url = uploader.upload_dataset(
            data_path=csv_path,
            repo_id="username/csv-dataset-example",
            convert_to_parquet=True,  # Also create Parquet version
            private=False
        )
        print(f"✅ CSV dataset uploaded to: {repo_url}")
    finally:
        # Clean up
        if Path(csv_path).exists():
            Path(csv_path).unlink()

def example_download_dataset():
    """Example: Download a dataset from Hugging Face Hub."""
    print("=== Dataset Download Example ===")
    
    config, auth_manager = setup_authentication()
    downloader = DatasetDownloader(config, auth_manager)
    
    # Download dataset using datasets library
    dataset = downloader.download_dataset(
        repo_id="squad",
        split="train[:1000]",  # First 1000 samples
        streaming=False
    )
    
    print(f"✅ Downloaded dataset with {len(dataset)} samples")
    print(f"Dataset features: {list(dataset.features.keys())}")
    
    # Download as pandas DataFrame
    df = downloader.download_as_dataframe(
        repo_id="imdb",
        split="train[:100]",  # First 100 samples
    )
    
    print(f"✅ Downloaded as DataFrame: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    
    # Save DataFrame to local file
    output_path = "./downloaded_data.parquet"
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved to: {output_path}")

def example_manage_datasets():
    """Example: Manage dataset repositories."""
    print("=== Dataset Management Example ===")
    
    config, auth_manager = setup_authentication()
    manager = DatasetManager(config, auth_manager)
    
    # List user's datasets
    datasets = manager.list_user_datasets(limit=5)
    print(f"Found {len(datasets)} datasets:")
    for dataset in datasets:
        print(f"  - {dataset['id']} ({dataset['downloads']:,} downloads)")
    
    # Search datasets
    results = manager.search_datasets(
        query="sentiment analysis",
        tags=["sentiment-analysis"],
        language="en",
        limit=5
    )
    print(f"\nFound {len(results)} sentiment analysis datasets")
    
    # Get dataset info
    try:
        info = downloader.get_dataset_info("imdb")
        print(f"\nIMDB dataset info:")
        print(f"  Downloads: {info['downloads']:,}")
        print(f"  Likes: {info['likes']}")
        print(f"  Files: {len(info['siblings'])}")
    except Exception as e:
        print(f"Could not get dataset info: {e}")

def example_dataset_processing():
    """Example: Dataset processing and conversion."""
    print("=== Dataset Processing Example ===")
    
    # Create sample datasets in different formats
    data = {
        'id': range(1, 101),
        'text': [f"Sample text {i}" for i in range(1, 101)],
        'category': ['A', 'B', 'C'] * 33 + ['A'],
        'score': [i * 0.01 for i in range(1, 101)]
    }
    df = pd.DataFrame(data)
    
    # Save in different formats
    formats = {
        'csv': 'sample.csv',
        'json': 'sample.json',
        'parquet': 'sample.parquet'
    }
    
    # Create files
    df.to_csv(formats['csv'], index=False)
    df.to_json(formats['json'], orient='records', lines=True)
    df.to_parquet(formats['parquet'], index=False)
    
    config, auth_manager = setup_authentication()
    uploader = DatasetUploader(config, auth_manager)
    
    # Upload each format
    for format_name, file_path in formats.items():
        try:
            repo_id = f"username/sample-{format_name}-dataset"
            print(f"Uploading {format_name} format...")
            
            uploader.upload_dataset(
                data_path=file_path,
                repo_id=repo_id,
                commit_message=f"Upload {format_name} format dataset",
                convert_to_parquet=(format_name != 'parquet')  # Convert others to Parquet
            )
            
            print(f"✅ Uploaded {format_name} dataset")
        except Exception as e:
            print(f"❌ Failed to upload {format_name}: {e}")
        finally:
            # Clean up
            if Path(file_path).exists():
                Path(file_path).unlink()

def example_streaming_dataset():
    """Example: Working with streaming datasets."""
    print("=== Streaming Dataset Example ===")
    
    config, auth_manager = setup_authentication()
    downloader = DatasetDownloader(config, auth_manager)
    
    # Download large dataset in streaming mode
    dataset = downloader.download_dataset(
        repo_id="c4",
        config_name="en",
        split="train",
        streaming=True
    )
    
    print("✅ Opened streaming dataset")
    
    # Process first few samples
    samples_processed = 0
    for sample in dataset:
        if samples_processed >= 5:
            break
        
        print(f"Sample {samples_processed + 1}:")
        print(f"  Text length: {len(sample['text'])} characters")
        print(f"  URL: {sample.get('url', 'N/A')}")
        print()
        
        samples_processed += 1
    
    print(f"✅ Processed {samples_processed} samples in streaming mode")

if __name__ == "__main__":
    try:
        example_upload_dataset()
        example_download_dataset()
        example_manage_datasets()
        example_dataset_processing()
        example_streaming_dataset()
    except KeyboardInterrupt:
        print("\nOperations cancelled by user")
    except Exception as e:
        print(f"Error: {e}")