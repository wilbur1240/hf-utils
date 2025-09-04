# src/cli/dataset_cli.py
import argparse
from pathlib import Path
from typing import Optional

from ..common.config import HFConfig
from ..common.logger import setup_logger
from ..auth.manager import AuthManager
from ..datasets.upload import DatasetUploader
from ..datasets.download import DatasetDownloader
from ..datasets.manage import DatasetManager

logger = setup_logger(__name__)

class DatasetCLI:
    """CLI interface for dataset operations."""
    
    def __init__(self, config: HFConfig, auth_manager: AuthManager):
        self.config = config
        self.auth_manager = auth_manager
        self.uploader = DatasetUploader(config, auth_manager)
        self.downloader = DatasetDownloader(config, auth_manager)
        self.manager = DatasetManager(config, auth_manager)
    
    def add_commands(self, subparsers):
        """Add dataset commands to parser."""
        dataset_parser = subparsers.add_parser('dataset', help='Dataset operations')
        dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_action')
        
        # Upload
        upload_parser = dataset_subparsers.add_parser('upload', help='Upload dataset')
        upload_parser.add_argument('path', help='Path to dataset file or directory')
        upload_parser.add_argument('repo_id', help='Repository ID (username/dataset-name)')
        upload_parser.add_argument('--destination', help='Destination path in repo')
        upload_parser.add_argument('--private', action='store_true', help='Make repository private')
        upload_parser.add_argument('--message', help='Commit message')
        upload_parser.add_argument('--description', help='Commit description')
        upload_parser.add_argument('--no-convert-parquet', action='store_true', 
                                 help='Skip conversion to Parquet format')
        
        # Download
        download_parser = dataset_subparsers.add_parser('download', help='Download dataset')
        download_parser.add_argument('repo_id', help='Repository ID')
        download_parser.add_argument('--destination', help='Local destination directory')
        download_parser.add_argument('--config', help='Dataset configuration name')
        download_parser.add_argument('--split', help='Specific split to download')
        download_parser.add_argument('--revision', help='Specific revision to download')
        download_parser.add_argument('--streaming', action='store_true', help='Enable streaming mode')
        
        # Download as DataFrame
        download_df_parser = dataset_subparsers.add_parser('download-df', 
                                                          help='Download dataset as pandas DataFrame')
        download_df_parser.add_argument('repo_id', help='Repository ID')
        download_df_parser.add_argument('--config', help='Dataset configuration name')
        download_df_parser.add_argument('--split', default='train', help='Dataset split')
        download_df_parser.add_argument('--output', help='Output file path (CSV/Parquet)')
        
        # List
        list_parser = dataset_subparsers.add_parser('list', help='List user datasets')
        list_parser.add_argument('--author', help='Author/organization name')
        list_parser.add_argument('--limit', type=int, help='Maximum results')
        
        # Delete
        delete_parser = dataset_subparsers.add_parser('delete', help='Delete dataset')
        delete_parser.add_argument('repo_id', help='Repository ID to delete')
        delete_parser.add_argument('--force', action='store_true', help='Skip confirmation')
        
        # Info
        info_parser = dataset_subparsers.add_parser('info', help='Get dataset info')
        info_parser.add_argument('repo_id', help='Repository ID')
        info_parser.add_argument('--revision', help='Specific revision')
        
        # Search
        search_parser = dataset_subparsers.add_parser('search', help='Search datasets')
        search_parser.add_argument('query', nargs='?', help='Search query')
        search_parser.add_argument('--author', help='Filter by author')
        search_parser.add_argument('--tags', nargs='*', help='Filter by tags')
        search_parser.add_argument('--language', help='Filter by language')
        search_parser.add_argument('--limit', type=int, default=20, help='Maximum results')
    
    def handle_command(self, args) -> int:
        """Handle dataset CLI commands."""
        try:
            if args.dataset_action == 'upload':
                return self._handle_upload(args)
            elif args.dataset_action == 'download':
                return self._handle_download(args)
            elif args.dataset_action == 'download-df':
                return self._handle_download_df(args)
            elif args.dataset_action == 'list':
                return self._handle_list(args)
            elif args.dataset_action == 'delete':
                return self._handle_delete(args)
            elif args.dataset_action == 'info':
                return self._handle_info(args)
            elif args.dataset_action == 'search':
                return self._handle_search(args)
            else:
                print("Unknown dataset command")
                return 1
        except Exception as e:
            logger.error(f"Dataset command failed: {str(e)}")
            return 1
    
    def _handle_upload(self, args) -> int:
        """Handle dataset upload."""
        print(f"Uploading {args.path} to {args.repo_id}...")
        
        repo_url = self.uploader.upload_dataset(
            data_path=args.path,
            repo_id=args.repo_id,
            destination=args.destination,
            private=args.private,
            commit_message=args.message,
            commit_description=args.description,
            convert_to_parquet=not args.no_convert_parquet
        )
        
        print(f"✅ Dataset uploaded successfully!")
        print(f"Repository: {repo_url}")
        return 0
    
    def _handle_download(self, args) -> int:
        """Handle dataset download."""
        print(f"Downloading {args.repo_id}...")
        
        dataset = self.downloader.download_dataset(
            repo_id=args.repo_id,
            destination=args.destination,
            config_name=args.config,
            split=args.split,
            revision=args.revision,
            streaming=args.streaming
        )
        
        print(f"✅ Dataset downloaded successfully!")
        
        # Show dataset info
        if hasattr(dataset, 'info'):
            print(f"Dataset info: {dataset.info}")
        elif hasattr(dataset, '__len__'):
            print(f"Dataset size: {len(dataset):,} samples")
        
        return 0
    
    def _handle_download_df(self, args) -> int:
        """Handle download as DataFrame."""
        print(f"Downloading {args.repo_id} as DataFrame...")
        
        df = self.downloader.download_as_dataframe(
            repo_id=args.repo_id,
            config_name=args.config,
            split=args.split
        )
        
        print(f"✅ Dataset downloaded as DataFrame!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Save to file if specified
        if args.output:
            output_path = Path(args.output)
            if output_path.suffix.lower() == '.csv':
                df.to_csv(output_path, index=False)
            elif output_path.suffix.lower() == '.parquet':
                df.to_parquet(output_path, index=False)
            else:
                print(f"Unsupported output format: {output_path.suffix}")
                return 1
            
            print(f"Saved to: {output_path}")
        
        return 0
    
    def _handle_list(self, args) -> int:
        """Handle list datasets."""
        datasets = self.manager.list_user_datasets(
            author=args.author,
            limit=args.limit
        )
        
        if not datasets:
            print("No datasets found")
            return 0
        
        print(f"Found {len(datasets)} datasets:")
        print()
        
        for dataset in datasets:
            print(f"📊 {dataset['id']}")
            print(f"   Downloads: {dataset['downloads']:,}")
            print(f"   Likes: {dataset['likes']}")
            if dataset['tags']:
                print(f"   Tags: {', '.join(dataset['tags'])}")
            print()
        
        return 0
    
    def _handle_delete(self, args) -> int:
        """Handle dataset deletion."""
        if not args.force:
            response = input(f"Are you sure you want to delete '{args.repo_id}'? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Deletion cancelled")
                return 0
        
        self.manager.delete_dataset(args.repo_id)
        print(f"✅ Dataset '{args.repo_id}' deleted successfully!")
        return 0
    
    def _handle_info(self, args) -> int:
        """Handle dataset info."""
        info = self.downloader.get_dataset_info(args.repo_id, args.revision)
        
        print(f"📊 {info['id']}")
        print(f"Downloads: {info['downloads']:,}")
        print(f"Likes: {info['likes']}")
        print(f"Created: {info['created_at']}")
        print(f"Last Modified: {info['last_modified']}")
        print(f"Private: {info['private']}")
        
        if info['tags']:
            print(f"Tags: {', '.join(info['tags'])}")
        
        print(f"\nFiles ({len(info['siblings'])}):")
        for file_info in info['siblings'][:10]:  # Show first 10 files
            size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] else 0
            print(f"  📄 {file_info['filename']} ({size_mb:.1f} MB)")
        
        if len(info['siblings']) > 10:
            print(f"  ... and {len(info['siblings']) - 10} more files")
        
        return 0
    
    def _handle_search(self, args) -> int:
        """Handle dataset search."""
        results = self.manager.search_datasets(
            query=args.query,
            author=args.author,
            tags=args.tags,
            language=args.language,
            limit=args.limit
        )
        
        if not results:
            print("No datasets found")
            return 0
        
        print(f"Found {len(results)} datasets:")
        print()
        
        for dataset in results:
            print(f"📊 {dataset['id']}")
            print(f"   Downloads: {dataset['downloads']:,}")
            print(f"   Likes: {dataset['likes']}")
            if dataset['tags']:
                print(f"   Tags: {', '.join(dataset['tags'][:5])}")  # Show first 5 tags
            print()
        
        return 0