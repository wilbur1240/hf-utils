# tests/test_cli/test_main_cli.py  
# Tests for CLI interface

import pytest
from unittest.mock import patch, Mock
from hf_utils.cli.main import HFUtilsCLI

@pytest.mark.unit
class TestCLI:
    """Test cases for CLI interface."""
    
    def test_cli_init(self):
        """Test CLI initialization."""
        cli = HFUtilsCLI()
        assert cli.config is not None
        assert cli.auth_manager is not None
    
    def test_create_parser(self):
        """Test argument parser creation."""
        cli = HFUtilsCLI()
        parser = cli.create_parser()
        
        # Test that parser was created successfully
        assert parser is not None
        assert parser.prog == 'hf-utils'
        
        # Test basic argument parsing (without --help which exits)
        args = parser.parse_args(['auth', 'whoami'])
        assert args.command == 'auth'
        assert args.auth_action == 'whoami'
        
        # Test model command parsing
        args = parser.parse_args(['model', 'list', '--limit', '10'])
        assert args.command == 'model'
        assert args.model_action == 'list'
        assert args.limit == 10
        
        # Test dataset command parsing
        args = parser.parse_args(['dataset', 'search', 'test'])
        assert args.command == 'dataset'
        assert args.dataset_action == 'search'
        assert args.query == 'test'
    
    def test_help_command_handling(self):
        """Test that help commands are handled properly."""
        cli = HFUtilsCLI()
        parser = cli.create_parser()
        
        # Test help command raises SystemExit as expected
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--help'])
        
        # SystemExit with code 0 indicates successful help display
        assert exc_info.value.code == 0
    
    def test_auth_login_command(self):
        """Test auth login command with mocked auth manager."""
        # Create a mock auth manager
        mock_auth = Mock()
        mock_auth.authenticate.return_value = True
        
        # Create CLI and replace auth manager
        cli = HFUtilsCLI()
        cli.auth_manager = mock_auth
        
        result = cli.run(['auth', 'login', '--token', 'test_token'])
        
        assert result == 0
        mock_auth.authenticate.assert_called_once()
    
    def test_auth_whoami_command(self):
        """Test auth whoami command with mocked auth manager."""
        # Create a mock auth manager
        mock_auth = Mock()
        mock_auth.is_authenticated.return_value = True
        mock_auth.get_user_info.return_value = {'name': 'testuser', 'email': 'test@example.com'}
        mock_auth.get_organizations.return_value = []
        
        # Create CLI and replace auth manager
        cli = HFUtilsCLI()
        cli.auth_manager = mock_auth
        
        result = cli.run(['auth', 'whoami'])
        
        assert result == 0
        mock_auth.is_authenticated.assert_called_once()
        mock_auth.get_user_info.assert_called_once()
        mock_auth.get_organizations.assert_called_once()
    
    def test_auth_whoami_not_authenticated(self):
        """Test auth whoami when not authenticated."""
        # Create a mock auth manager that's not authenticated
        mock_auth = Mock()
        mock_auth.is_authenticated.return_value = False
        
        # Create CLI and replace auth manager
        cli = HFUtilsCLI()
        cli.auth_manager = mock_auth
        
        result = cli.run(['auth', 'whoami'])
        
        assert result == 1  # Should return 1 for "not logged in"
        mock_auth.is_authenticated.assert_called_once()
    
    def test_auth_logout_command(self):
        """Test auth logout command."""
        # Create a mock auth manager
        mock_auth = Mock()
        mock_auth.logout.return_value = None
        
        # Create CLI and replace auth manager
        cli = HFUtilsCLI()
        cli.auth_manager = mock_auth
        
        result = cli.run(['auth', 'logout'])
        
        assert result == 0
        mock_auth.logout.assert_called_once()
    
    def test_global_arguments(self):
        """Test global argument parsing."""
        cli = HFUtilsCLI()
        parser = cli.create_parser()
        
        # Test verbose flag
        args = parser.parse_args(['--verbose', 'auth', 'whoami'])
        assert args.verbose is True
        assert args.command == 'auth'
        
        # Test quiet flag
        args = parser.parse_args(['--quiet', 'auth', 'whoami'])
        assert args.quiet is True
        
        # Test token argument
        args = parser.parse_args(['--token', 'test123', 'auth', 'whoami'])
        assert args.token == 'test123'
    
    def test_invalid_command(self):
        """Test handling of invalid commands."""
        cli = HFUtilsCLI()
        
        # Invalid commands cause argparse to exit with code 2
        with pytest.raises(SystemExit) as exc_info:
            cli.run(['invalid_command'])
        
        # SystemExit with code 2 indicates argument parsing error
        assert exc_info.value.code == 2
    
    def test_invalid_auth_action(self):
        """Test handling of invalid auth actions."""
        cli = HFUtilsCLI()
        
        # Invalid subcommands cause argparse to exit with code 2
        with pytest.raises(SystemExit) as exc_info:
            cli.run(['auth', 'invalid_action'])
        
        # SystemExit with code 2 indicates argument parsing error
        assert exc_info.value.code == 2
    
    def test_keyboard_interrupt(self):
        """Test handling of KeyboardInterrupt."""
        # Create a mock auth manager that raises KeyboardInterrupt
        mock_auth = Mock()
        mock_auth.authenticate.side_effect = KeyboardInterrupt()
        
        # Create CLI and replace auth manager
        cli = HFUtilsCLI()
        cli.auth_manager = mock_auth
        
        result = cli.run(['auth', 'login'])
        assert result == 1
    
    def test_exception_handling(self):
        """Test general exception handling."""
        # Create a mock auth manager that raises an exception
        mock_auth = Mock()
        mock_auth.authenticate.side_effect = Exception("Test error")
        
        # Create CLI and replace auth manager
        cli = HFUtilsCLI()
        cli.auth_manager = mock_auth
        
        result = cli.run(['auth', 'login'])
        assert result == 1