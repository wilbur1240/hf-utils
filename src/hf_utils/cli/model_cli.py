# src/cli/model_cli.py
import argparse
import json
from pathlib import Path
from typing import Optional, List

from ..common.config import HFConfig
from ..common.logger import setup_logger
from ..auth.manager import AuthManager
from ..models.upload import ModelUploader
from ..models.download import ModelDownloader
from ..models.manage import ModelManager
from ..models.selective_upload import (
    FileUploadConfig, 
    SelectiveUploader,
    upload_single_file,
    upload_files_by_pattern,
    update_model_files
)

logger = setup_logger(__name__)

class ModelCLI:
    """CLI interface for model operations."""
    
    def __init__(self, config: HFConfig, auth_manager: AuthManager):
        self.config = config
        self.auth_manager = auth_manager
        self.uploader = ModelUploader(config, auth_manager)
        self.downloader = ModelDownloader(config, auth_manager)
        self.manager = ModelManager(config, auth_manager)
        self.selective_uploader = SelectiveUploader(config, auth_manager)
    
    def add_commands(self, subparsers):
        """Add model commands to parser."""
        model_parser = subparsers.add_parser('model', help='Model operations')
        model_subparsers = model_parser.add_subparsers(dest='model_action')
        
        # Upload (original full upload)
        upload_parser = model_subparsers.add_parser('upload', help='Upload entire model directory')
        upload_parser.add_argument('path', help='Path to model directory or file')
        upload_parser.add_argument('repo_id', help='Repository ID (username/model-name)')
        upload_parser.add_argument('--destination', help='Destination path in repo')
        upload_parser.add_argument('--private', action='store_true', help='Make repository private')
        upload_parser.add_argument('--message', help='Commit message')
        upload_parser.add_argument('--description', help='Commit description')
        upload_parser.add_argument('--create-pr', action='store_true', help='Create pull request')
        upload_parser.add_argument('--revision', help='Target branch/revision')
        
        # NEW: Upload single file
        upload_file_parser = model_subparsers.add_parser('upload-file', help='Upload a single file')
        upload_file_parser.add_argument('repo_id', help='Repository ID')
        upload_file_parser.add_argument('local_file', help='Path to local file')
        upload_file_parser.add_argument('--remote-path', help='Remote path in repository')
        upload_file_parser.add_argument('--message', help='Commit message')
        upload_file_parser.add_argument('--revision', default='main', help='Target branch/revision')
        upload_file_parser.add_argument('--create-pr', action='store_true', help='Create pull request')
        upload_file_parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded')
        upload_file_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # NEW: Upload files by pattern
        upload_pattern_parser = model_subparsers.add_parser('upload-pattern', help='Upload files matching patterns')
        upload_pattern_parser.add_argument('repo_id', help='Repository ID')
        upload_pattern_parser.add_argument('patterns', nargs='+', help='Glob patterns to match files')
        upload_pattern_parser.add_argument('--local-base', default='.', help='Base directory for local files')
        upload_pattern_parser.add_argument('--remote-base', default='', help='Base directory in repository')
        upload_pattern_parser.add_argument('--exclude', action='append', help='Patterns to exclude')
        upload_pattern_parser.add_argument('--message', help='Commit message')
        upload_pattern_parser.add_argument('--revision', default='main', help='Target branch/revision')
        upload_pattern_parser.add_argument('--create-pr', action='store_true', help='Create pull request')
        upload_pattern_parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded')
        upload_pattern_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # NEW: Upload specific files
        upload_files_parser = model_subparsers.add_parser('upload-files', help='Upload specific files')
        upload_files_parser.add_argument('repo_id', help='Repository ID')
        upload_files_parser.add_argument('files', nargs='+', help='Files to upload')
        upload_files_parser.add_argument('--local-base', default='.', help='Base directory for local files')
        upload_files_parser.add_argument('--remote-base', default='', help='Base directory in repository')
        upload_files_parser.add_argument('--message', help='Commit message')
        upload_files_parser.add_argument('--revision', default='main', help='Target branch/revision')
        upload_files_parser.add_argument('--create-pr', action='store_true', help='Create pull request')
        upload_files_parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded')
        upload_files_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # NEW: Upload from config
        upload_config_parser = model_subparsers.add_parser('upload-config', help='Upload using configuration file')
        upload_config_parser.add_argument('config_file', help='Path to configuration file (.json or .yaml)')
        upload_config_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # NEW: Quick update (convenience command)
        update_parser = model_subparsers.add_parser('update', help='Quick update specific model files')
        update_parser.add_argument('repo_id', help='Repository ID')
        update_parser.add_argument('files', nargs='+', help='Files to update')
        update_parser.add_argument('--message', help='Commit message')
        update_parser.add_argument('--dry-run', action='store_true', help='Show what would be updated')
        update_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

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
        
        # Delete file command
        delete_file_parser = model_subparsers.add_parser('delete-file', help='Delete specific file from model')
        delete_file_parser.add_argument('repo_id', help='Repository ID')
        delete_file_parser.add_argument('file_path', help='Path of file to delete')
        delete_file_parser.add_argument('--message', help='Commit message')
        delete_file_parser.add_argument('--force', action='store_true', help='Skip confirmation')
        
        # List files command
        list_files_parser = model_subparsers.add_parser('list-files', help='List files in model repository')
        list_files_parser.add_argument('repo_id', help='Repository ID')
        list_files_parser.add_argument('--revision', help='Specific revision')

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
            elif args.model_action == 'upload-file':
                return self._handle_upload_file(args)
            elif args.model_action == 'upload-pattern':
                return self._handle_upload_pattern(args)
            elif args.model_action == 'upload-files':
                return self._handle_upload_files(args)
            elif args.model_action == 'upload-config':
                return self._handle_upload_config(args)
            elif args.model_action == 'update':
                return self._handle_update(args)
            elif args.model_action == 'download':
                return self._handle_download(args)
            elif args.model_action == 'download-file':
                return self._handle_download_file(args)
            elif args.model_action == 'delete-file':
                return self._handle_delete_file(args)
            elif args.model_action == 'list-files':
                return self._handle_list_files(args)
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
            print(f"❌ Error: {str(e)}")
            return 1
    
    # NEW: Selective upload handlers
    def _handle_upload_file(self, args) -> int:
        """Handle single file upload."""
        if args.verbose:
            logger.setLevel("DEBUG")
        
        print(f"🔄 Uploading {args.local_file} to {args.repo_id}...")
        
        try:
            result = upload_single_file(
                repo_id=args.repo_id,
                local_file=args.local_file,
                remote_path=args.remote_path,
                repo_type="model",
                commit_message=args.message,
                revision=args.revision,
                create_pr=args.create_pr,
                dry_run=args.dry_run
            )
            
            if result["status"] == "success":
                print(f"✅ Successfully uploaded {args.local_file}")
                if "commit_info" in result:
                    print(f"🔗 Commit: {result['commit_info'].commit_url}")
            elif result["status"] == "dry_run":
                print(f"🔍 Dry run: {result['message']}")
            else:
                print(f"⚠️  {result['message']}")
                
            return 0
            
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            return 1
    
    def _handle_upload_pattern(self, args) -> int:
        """Handle pattern-based upload."""
        if args.verbose:
            logger.setLevel("DEBUG")
        
        print(f"🔄 Uploading files matching patterns to {args.repo_id}...")
        print(f"📋 Patterns: {', '.join(args.patterns)}")
        if args.exclude:
            print(f"🚫 Excluding: {', '.join(args.exclude)}")
        
        try:
            result = upload_files_by_pattern(
                repo_id=args.repo_id,
                patterns=args.patterns,
                local_base_path=args.local_base,
                remote_base_path=args.remote_base,
                exclude_patterns=args.exclude,
                repo_type="model",
                commit_message=args.message,
                revision=args.revision,
                create_pr=args.create_pr,
                dry_run=args.dry_run
            )
            
            if result["status"] == "success":
                print(f"✅ Successfully uploaded {result['files_uploaded']} files")
                if args.verbose and "files" in result:
                    for local_path, remote_path in result["files"]:
                        print(f"  📁 {local_path} -> {remote_path}")
                if "commit_info" in result:
                    print(f"🔗 Commit: {result['commit_info'].commit_url}")
            elif result["status"] == "dry_run":
                print(f"🔍 Dry run: {result['message']}")
                if args.verbose and "files" in result:
                    for local_path, remote_path in result["files"]:
                        print(f"  📁 {local_path} -> {remote_path}")
            else:
                print(f"⚠️  {result['message']}")
                
            return 0
            
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            return 1
    
    def _handle_upload_files(self, args) -> int:
        """Handle specific files upload."""
        if args.verbose:
            logger.setLevel("DEBUG")
        
        print(f"🔄 Uploading {len(args.files)} files to {args.repo_id}...")
        
        # Convert relative paths to absolute from local_base
        file_paths = []
        local_base_path = Path(args.local_base)
        
        for file in args.files:
            file_path = Path(file)
            if not file_path.is_absolute():
                file_path = local_base_path / file_path
            file_paths.append(file_path)
        
        try:
            config = FileUploadConfig(
                repo_id=args.repo_id,
                repo_type="model",
                files=file_paths,
                local_base_path=local_base_path,
                remote_base_path=args.remote_base,
                commit_message=args.message,
                revision=args.revision,
                create_pr=args.create_pr,
                dry_run=args.dry_run
            )
            
            result = self.selective_uploader.upload_files(config)
            
            if result["status"] == "success":
                print(f"✅ Successfully uploaded {result['files_uploaded']} files")
                if args.verbose and "files" in result:
                    for local_path, remote_path in result["files"]:
                        print(f"  📁 {local_path} -> {remote_path}")
                if "commit_info" in result:
                    print(f"🔗 Commit: {result['commit_info'].commit_url}")
            elif result["status"] == "dry_run":
                print(f"🔍 Dry run: {result['message']}")
                if args.verbose and "files" in result:
                    for local_path, remote_path in result["files"]:
                        print(f"  📁 {local_path} -> {remote_path}")
            else:
                print(f"⚠️  {result['message']}")
                
            return 0
            
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            return 1
    
    def _handle_upload_config(self, args) -> int:
        """Handle config file upload."""
        if args.verbose:
            logger.setLevel("DEBUG")
        
        print(f"🔄 Uploading files using config: {args.config_file}")
        
        try:
            # Load configuration from JSON or YAML file
            config_path = Path(args.config_file)
            
            if not config_path.exists():
                print(f"❌ Config file not found: {args.config_file}")
                return 1
            
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                except ImportError:
                    print("❌ PyYAML is required for YAML config files. Install with: pip install pyyaml")
                    return 1
            else:
                print("❌ Config file must be JSON (.json) or YAML (.yml/.yaml)")
                return 1
            
            # Ensure repo_type is model for this CLI
            config_data['repo_type'] = 'model'
            
            # Create config object
            config = FileUploadConfig(**config_data)
            
            result = self.selective_uploader.upload_files(config)
            
            if result["status"] == "success":
                print(f"✅ Successfully uploaded {result['files_uploaded']} files")
                if args.verbose and "files" in result:
                    for local_path, remote_path in result["files"]:
                        print(f"  📁 {local_path} -> {remote_path}")
                if "commit_info" in result:
                    print(f"🔗 Commit: {result['commit_info'].commit_url}")
            elif result["status"] == "dry_run":
                print(f"🔍 Dry run: {result['message']}")
                if args.verbose and "files" in result:
                    for local_path, remote_path in result["files"]:
                        print(f"  📁 {local_path} -> {remote_path}")
            else:
                print(f"⚠️  {result['message']}")
                
            return 0
            
        except Exception as e:
            print(f"❌ Config upload failed: {str(e)}")
            return 1
    
    def _handle_update(self, args) -> int:
        """Handle quick model update."""
        if args.verbose:
            logger.setLevel("DEBUG")
        
        print(f"🔄 Updating {len(args.files)} files in {args.repo_id}...")
        
        try:
            result = update_model_files(
                repo_id=args.repo_id,
                files=args.files,
                commit_message=args.message,
                dry_run=args.dry_run
            )
            
            if result["status"] == "success":
                print(f"✅ Model updated successfully")
                if args.verbose and "files" in result:
                    for local_path, remote_path in result["files"]:
                        print(f"  📁 {local_path} -> {remote_path}")
                if "commit_info" in result:
                    print(f"🔗 Commit: {result['commit_info'].commit_url}")
            elif result["status"] == "dry_run":
                print(f"🔍 {result['message']}")
                if args.verbose and "files" in result:
                    for local_path, remote_path in result["files"]:
                        print(f"  📁 {local_path} -> {remote_path}")
            else:
                print(f"⚠️  {result['message']}")
                
            return 0
            
        except Exception as e:
            print(f"❌ Update failed: {str(e)}")
            return 1

    # Original handlers (unchanged)
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
    
    def _handle_delete_file(self, args):
        """Handle file deletion."""
        from huggingface_hub import delete_file
        
        if not args.force:
            response = input(f"Delete '{args.file_path}' from {args.repo_id}? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Deletion cancelled")
                return 0
        
        try:
            delete_file(
                path_in_repo=args.file_path,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=args.message or f"Delete {args.file_path}",
                token=self.config.token
            )
            print(f"✅ Successfully deleted: {args.file_path}")
            return 0
        except Exception as e:
            print(f"❌ Failed to delete file: {e}")
            return 1

    def _handle_list_files(self, args):
        """Handle file listing."""
        from huggingface_hub import list_repo_files
        
        try:
            files = list_repo_files(
                args.repo_id,
                repo_type="model",
                revision=args.revision,
                token=self.config.token
            )
            
            print(f"📁 Files in {args.repo_id}:")
            for file_path in sorted(files):
                print(f"  📄 {file_path}")
            
            return 0
        except Exception as e:
            print(f"❌ Failed to list files: {e}")
            return 1

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