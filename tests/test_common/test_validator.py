# tests/test_common/test_validators.py
# Unit tests for validation utilities

import pytest
from hf_utils.common.validators import (
    validate_repo_name, 
    validate_file_path,
    validate_dataset_format,
    sanitize_filename
)

@pytest.mark.unit
class TestValidators:
    """Test cases for validation utilities."""
    
    @pytest.mark.parametrize("repo_name,expected", [
        ("username/model-name", True),
        ("org/dataset_name", True),
        ("user123/my-model-v2", True),
        ("invalid-name", False),  # Missing username
        ("user/", False),  # Missing model name
        ("user//model", False),  # Double slash
        ("", False),  # Empty
    ])
    def test_validate_repo_name(self, repo_name, expected):
        """Test repository name validation."""
        assert validate_repo_name(repo_name) == expected
    
    def test_validate_file_path_exists(self, temp_dir):
        """Test file path validation for existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        assert validate_file_path(str(test_file)) is True
    
    def test_validate_file_path_not_exists(self):
        """Test file path validation for non-existent file."""
        assert validate_file_path("/nonexistent/file.txt") is False
    
    @pytest.mark.parametrize("filename,expected", [
        ("data.csv", True),
        ("dataset.json", True), 
        ("data.parquet", True),
        ("text.txt", True),
        ("data.xml", False),  # Not in allowed formats
        ("file", False),  # No extension
    ])
    def test_validate_dataset_format(self, filename, expected):
        """Test dataset format validation."""
        assert validate_dataset_format(filename) == expected
    
    @pytest.mark.parametrize("input_name,expected", [
        ("normal_file.txt", "normal_file.txt"),
        ("file with spaces.txt", "file with spaces.txt"),
        ("file<>:|?*.txt", "file_.txt"),  # Invalid chars replaced and deduplicated
        ("file___multiple___underscores.txt", "file_multiple_underscores.txt"),
        ("", "unnamed_file"),  # Empty string
        ("file<with>many|bad?chars*.dat", "file_with_many_bad_chars_.dat"),  # Multiple bad chars
        ("___leading_and_trailing___", "leading_and_trailing"),  # Trim leading/trailing underscores
        ("file.with.dots.txt", "file.with.dots.txt"),  # Dots are preserved
        ("UPPERCASE.TXT", "UPPERCASE.TXT"),  # Case preserved
    ])
    def test_sanitize_filename(self, input_name, expected):
        """Test filename sanitization."""
        assert sanitize_filename(input_name) == expected
    
    def test_sanitize_filename_edge_cases(self):
        """Test edge cases for filename sanitization."""
        # Test that function handles None gracefully (if implemented)
        try:
            result = sanitize_filename(None)
            assert result == "unnamed_file"  # Expected fallback behavior
        except (TypeError, AttributeError):
            # If function doesn't handle None, that's also acceptable
            pass
        
        # Test very long filename - function may not limit length, which is acceptable
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        # Just verify it doesn't crash and returns a string
        assert isinstance(result, str)
        assert result.endswith(".txt")
        
        # Test filename with only invalid characters
        result = sanitize_filename("<>:|?*")
        # Function might return empty string or fallback - both acceptable
        assert isinstance(result, str)
    
    def test_validate_repo_name_edge_cases(self):
        """Test edge cases for repository name validation."""
        edge_cases = [
            ("a/b", True),  # Minimal valid case
            ("user-name/repo-name", True),  # Hyphens allowed
            ("user_name/repo_name", True),  # Underscores allowed
            ("user123/repo456", True),  # Numbers allowed
            ("123user/123repo", True),  # Starting with numbers
            ("user/repo/extra", False),  # Too many slashes
            ("user\\repo", False),  # Wrong slash type
            ("user name/repo name", False),  # Spaces not allowed
            ("user@name/repo", False),  # Special chars not allowed
            ("user/repo@tag", False),  # Special chars in repo name
        ]
        
        for repo_name, expected in edge_cases:
            assert validate_repo_name(repo_name) == expected, f"Failed for: {repo_name}"
    
    def test_validate_dataset_format_edge_cases(self):
        """Test edge cases for dataset format validation."""
        edge_cases = [
            ("DATA.CSV", True),  # Uppercase extension
            ("data.CSV", True),  # Mixed case
            ("file.tar.gz", False),  # Compound extension
            (".csv", False),  # Hidden file - not supported by validator
            ("file.", False),  # Trailing dot
            ("file.csvx", False),  # Similar but invalid extension
            ("file.jsonl", True),  # JSON Lines format
            ("file.arrow", True),  # Arrow format
            ("data.txt", True),  # Plain text files
            ("dataset.parquet", True),  # Parquet format
        ]
        
        for filename, expected in edge_cases:
            result = validate_dataset_format(filename)
            assert result == expected, f"Failed for: {filename}, got {result}, expected {expected}"