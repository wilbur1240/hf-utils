# src/common/validators.py
import re
from typing import List, Optional
from pathlib import Path

def validate_repo_name(repo_name: str) -> bool:
    """Validate Hugging Face repository name format."""
    # HF repo names should be: username/repo-name or org/repo-name
    pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?/[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$'
    return bool(re.match(pattern, repo_name))

def validate_file_path(file_path: str) -> bool:
    """Validate that file path exists and is readable."""
    path = Path(file_path)
    return path.exists() and path.is_file() and path.stat().st_size > 0

def validate_directory_path(dir_path: str) -> bool:
    """Validate that directory path exists."""
    path = Path(dir_path)
    return path.exists() and path.is_dir()

def validate_model_files(model_dir: str, required_files: Optional[List[str]] = None) -> List[str]:
    """Validate model directory contains required files."""
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        raise ValidationError(f"Model directory {model_dir} does not exist")
    
    if required_files is None:
        # Common model files to look for
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    
    missing_files = []
    for file_name in required_files:
        if not (path / file_name).exists():
            missing_files.append(file_name)
    
    return missing_files

def validate_dataset_format(file_path: str, allowed_formats: Optional[List[str]] = None) -> bool:
    """Validate dataset file format."""
    if allowed_formats is None:
        allowed_formats = [".csv", ".json", ".jsonl", ".parquet", ".arrow", ".txt"]
    
    path = Path(file_path)
    return path.suffix.lower() in allowed_formats

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe usage."""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')
    
    return sanitized or "unnamed_file"