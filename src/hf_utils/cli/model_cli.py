# src/cli/model_cli.py
import argparse
from pathlib import Path
from typing import Optional

from ..common.config import HFConfig
from ..common.logger import setup_logger
from ..auth.manager import AuthManager
from ..models.upload import ModelUploader
from ..models.download import ModelDownloader
from ..models.manage import ModelManager

logger = setup_logger(__name__)

class ModelCLI:
    """CLI interface for model operations."""
    
    def __init__(self, config: HFConfig, auth_manager: AuthManager):
        self.config = config
        self.auth_manager = auth_manager
        self.uploader = ModelUploader(config, auth_manager)
        self.downloader = ModelDownloader(config, auth_manager)
        self.manager = ModelManager(config, auth_manager)
    
    def add_commands(self, subparsers):
        """Add model commands to parser."""
        model_parser = subparsers.add_parser('model', help='Model operations')
        model_subparsers = model_parser.add_subparsers(dest='model_action')
        
        # Upload
        upload_parser = model_subparsers.add_parser('upload', help='Upload model')
        upload_parser.add_argument('path', help='Path to model directory or file')
        upload_parser.add_argument('repo_id', help='Repository ID (username/model-name)')
        upload_parser.add_argument('--destination', help='Destination path in repo')
        upload_parser.add_argument('--private', action='store_true', help='Make repository private')
        upload_parser.add_argument('--message', help='Commit message')
        upload_parser.add_argument('--description', help='Commit description')
        upload_parser.add_argument('--create-pr', action='store_true', help='Create pull request')
        upload_parser.add_argument('--revision', help='Target branch/revision')
        
        # Download
        download_parser = model_subparsers.add_parser('download', help='Download model')
        download_parser.add_argument('repo_id', help='Repository ID')
        download_parser.add_argument('--destination', help='Local destination directory')
        download_parser.add_argument('--revision', help='Specific revision to download')
        download_parser.add_argument('--cache-dir', help='Cache directory')
        download_parser.add_argument('--local-files-only', action='store_true', help='Use local cache only')
        download_parser.add_argument('--force-download', action='store_true', help='Force re-download')
        
        # Download file
        download_file_parser = model_subparsers.add_parser('download-file', help='Download specific file')
        download_file_parser.add_argument('repo_id', help='Repository ID')
        download_file_parser.add_argument('filename', help='File to download')
        download_file_parser.add_argument('--destination', help='Local destination path')
        download_file_parser.add_argument('--revision', help='Specific revision')
        
        # List
        list_parser = model_subparsers.add_parser('list', help='List user models')
        list_parser.add_argument('--author', help='Author/organization name')
        list_parser.add_argument('--limit', type=int, help='Maximum results')
        
        # Delete
        delete_parser = model_subparsers.add_parser('delete', help='Delete model')
        delete_parser.add_argument('repo_id', help='Repository ID to delete')
        delete_parser.add_argument('--force', action='store_true', help='Skip confirmation')
        
        # Info
        info_parser = model_subparsers.add_parser('info', help='Get model info')
        info_parser.add_argument('repo_id', help='Repository ID')
        info_parser.add_argument('--revision', help='Specific revision')
        
        # Search
        search_parser = model_subparsers.add_parser('search', help='Search models')
        search_parser.add_argument('query', nargs='?', help='Search query')
        search_parser.add_argument('--author', help='Filter by author')
        search_parser.add_argument('--tags', nargs='*', help='Filter by tags')
        search_parser.add_argument('--pipeline-tag', help='Filter by pipeline tag')
        search_parser.add_argument('--limit', type=int, default=20, help='Maximum results')
    
    def handle_command(self, args) -> int:
        """Handle model CLI commands."""
        try:
            if args.model_action == 'upload':
                return self._handle_upload(args)
            elif args.model_action == 'download':
                return self._handle_download(args)
            elif args.model_action == 'download-file':
                return self._handle_download_file(args)
            elif args.model_action == 'list':
                return self._handle_list(args)
            elif args.model_action == 'delete':
                return self._handle_delete(args)
            elif args.model_action == 'info':
                return self._handle_info(args)
            elif args.model_action == 'search':
                return self._handle_search(args)
            else:
                print("Unknown model command")
                return 1
        except Exception as e:
            logger.error(f"Model command failed: {str(e)}")
            return 1
    
    def _handle_upload(self, args) -> int:
        """Handle model upload."""
        print(f"Uploading {args.path} to {args.repo_id}...")
        
        repo_url = self.uploader.upload_model(
            model_path=args.path,
            repo_id=args.repo_id,
            destination=args.destination,
            private=args.private,
            commit_message=args.message,
            commit_description=args.description,
            create_pr=args.create_pr,
            revision=args.revision
        )
        
        print(f"✅ Model uploaded successfully!")
        print(f"Repository: {repo_url}")
        return 0
    
    def _handle_download(self, args) -> int:
        """Handle model download."""
        print(f"Downloading {args.repo_id}...")
        
        path = self.downloader.download_model(
            repo_id=args.repo_id,
            destination=args.destination,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            force_download=args.force_download
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"Location: {path}")
        return 0
    
    def _handle_download_file(self, args) -> int:
        """Handle file download."""
        print(f"Downloading {args.filename} from {args.repo_id}...")
        
        path = self.downloader.download_file(
            repo_id=args.repo_id,
            filename=args.filename,
            destination=args.destination,
            revision=args.revision
        )
        
        print(f"✅ File downloaded successfully!")
        print(f"Location: {path}")
        return 0
    
    def _handle_list(self, args) -> int:
        """Handle list models."""
        models = self.manager.list_user_models(
            author=args.author,
            limit=args.limit
        )
        
        if not models:
            print("No models found")
            return 0
        
        print(f"Found {len(models)} models:")
        print()
        
        for model in models:
            print(f"📦 {model['id']}")
            print(f"   Downloads: {model['downloads']:,}")
            print(f"   Likes: {model['likes']}")
            if model['tags']:
                print(f"   Tags: {', '.join(model['tags'])}")
            print()
        
        return 0
    
    def _handle_delete(self, args) -> int:
        """Handle model deletion."""
        if not args.force:
            response = input(f"Are you sure you want to delete '{args.repo_id}'? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Deletion cancelled")
                return 0
        
        self.manager.delete_model(args.repo_id)
        print(f"✅ Model '{args.repo_id}' deleted successfully!")
        return 0
    
    def _handle_info(self, args) -> int:
        """Handle model info."""
        info = self.downloader.get_model_info(args.repo_id, args.revision)
        
        print(f"📦 {info['id']}")
        print(f"Downloads: {info['downloads']:,}")
        print(f"Likes: {info['likes']}")
        print(f"Created: {info['created_at']}")
        print(f"Last Modified: {info['last_modified']}")
        print(f"Private: {info['private']}")
        
        if info['tags']:
            print(f"Tags: {', '.join(info['tags'])}")
        
        if info['pipeline_tag']:
            print(f"Pipeline: {info['pipeline_tag']}")
        
        print(f"\nFiles ({len(info['siblings'])}):")
        for file_info in info['siblings'][:10]:  # Show first 10 files
            size_mb = file_info['size'] / (1024 * 1024) if file_info['size'] else 0
            print(f"  📄 {file_info['filename']} ({size_mb:.1f} MB)")
        
        if len(info['siblings']) > 10:
            print(f"  ... and {len(info['siblings']) - 10} more files")
        
        return 0
    
    def _handle_search(self, args) -> int:
        """Handle model search."""
        results = self.manager.search_models(
            query=args.query,
            author=args.author,
            tags=args.tags,
            pipeline_tag=args.pipeline_tag,
            limit=args.limit
        )
        
        if not results:
            print("No models found")
            return 0
        
        print(f"Found {len(results)} models:")
        print()
        
        for model in results:
            print(f"📦 {model['id']}")
            print(f"   Downloads: {model['downloads']:,}")
            print(f"   Likes: {model['likes']}")
            if model['tags']:
                print(f"   Tags: {', '.join(model['tags'][:5])}")  # Show first 5 tags
            print()
        
        return 0
