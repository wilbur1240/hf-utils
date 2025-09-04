# src/cli/main.py
import sys
import argparse
from pathlib import Path
from typing import Optional

from ..common.config import HFConfig
from ..common.logger import setup_logger
from ..auth.manager import AuthManager
from .model_cli import ModelCLI
from .dataset_cli import DatasetCLI

logger = setup_logger(__name__)

class HFUtilsCLI:
    """Main CLI interface for Hugging Face utilities."""
    
    def __init__(self):
        self.config = HFConfig.from_env()
        self.auth_manager = AuthManager(self.config)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create main argument parser."""
        parser = argparse.ArgumentParser(
            prog='hf-utils',
            description='Hugging Face utilities for models, datasets',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  hf-utils auth login
  hf-utils model upload ./my-model username/my-model
  hf-utils model download bert-base-uncased
  hf-utils dataset upload ./data.csv username/my-dataset
  hf-utils dataset download squad
            """
        )
        
        # Global arguments
        parser.add_argument('--token', help='Hugging Face token')
        parser.add_argument('--cache-dir', help='Cache directory')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Auth commands
        self._add_auth_commands(subparsers)
        
        # Model commands
        model_cli = ModelCLI(self.config, self.auth_manager)
        model_cli.add_commands(subparsers)
        
        # Dataset commands
        dataset_cli = DatasetCLI(self.config, self.auth_manager)
        dataset_cli.add_commands(subparsers)
        
        return parser
    
    def _add_auth_commands(self, subparsers):
        """Add authentication commands."""
        auth_parser = subparsers.add_parser('auth', help='Authentication commands')
        auth_subparsers = auth_parser.add_subparsers(dest='auth_action')
        
        # Login
        login_parser = auth_subparsers.add_parser('login', help='Login to Hugging Face')
        login_parser.add_argument('--token', help='Authentication token')
        login_parser.add_argument('--add-to-git-credential', action='store_true',
                                 help='Add token to git credential store')
        
        # Logout
        auth_subparsers.add_parser('logout', help='Logout from Hugging Face')
        
        # Whoami
        auth_subparsers.add_parser('whoami', help='Show current user info')
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the CLI with given arguments."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Configure logging
        if parsed_args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        elif parsed_args.quiet:
            import logging
            logging.getLogger().setLevel(logging.ERROR)
        
        # Update config with CLI arguments
        if parsed_args.token:
            self.config.token = parsed_args.token
        if parsed_args.cache_dir:
            self.config.cache_dir = parsed_args.cache_dir
        
        try:
            if parsed_args.command == 'auth':
                return self._handle_auth_commands(parsed_args)
            elif parsed_args.command == 'model':
                model_cli = ModelCLI(self.config, self.auth_manager)
                return model_cli.handle_command(parsed_args)
            elif parsed_args.command == 'dataset':
                dataset_cli = DatasetCLI(self.config, self.auth_manager)
                return dataset_cli.handle_command(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 1
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _handle_auth_commands(self, args) -> int:
        """Handle authentication commands."""
        if args.auth_action == 'login':
            try:
                success = self.auth_manager.authenticate(
                    token=args.token,
                    add_to_git_credential=args.add_to_git_credential
                )
                if success:
                    print("Successfully logged in!")
                    return 0
                else:
                    print("Login failed")
                    return 1
            except Exception as e:
                print(f"Login failed: {str(e)}")
                return 1
        
        elif args.auth_action == 'logout':
            try:
                self.auth_manager.logout()
                print("Successfully logged out!")
                return 0
            except Exception as e:
                print(f"Logout failed: {str(e)}")
                return 1
        
        elif args.auth_action == 'whoami':
            try:
                if not self.auth_manager.is_authenticated():
                    print("Not logged in")
                    return 1
                
                user_info = self.auth_manager.get_user_info()
                print(f"Logged in as: {user_info.get('name', 'Unknown')}")
                print(f"Email: {user_info.get('email', 'N/A')}")
                
                orgs = self.auth_manager.get_organizations()
                if orgs:
                    print(f"Organizations: {', '.join(org['name'] for org in orgs)}")
                
                return 0
            except Exception as e:
                print(f"Failed to get user info: {str(e)}")
                return 1
        
        else:
            print("Unknown auth command")
            return 1


def main():
    """Main entry point for CLI."""
    cli = HFUtilsCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())