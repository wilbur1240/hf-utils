# src/common/config.py
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class HFConfig:
    """Configuration class for Hugging Face utilities."""
    
    # Default settings
    cache_dir: Optional[str] = None
    token: Optional[str] = None
    endpoint: str = "https://huggingface.co"
    max_retries: int = 3
    chunk_size: int = 8192
    timeout: int = 300
    
    # Destinations
    default_model_destination: str = "models"
    default_dataset_destination: str = "datasets" 
    default_space_destination: str = "spaces"
    
    def __post_init__(self):
        """Initialize configuration with environment variables and defaults."""
        # Set cache directory
        if self.cache_dir is None:
            self.cache_dir = os.environ.get(
                "HF_HOME", 
                str(Path.home() / ".cache" / "huggingface")
            )
        
        # Set token from environment if not provided
        if self.token is None:
            # Try environment variables first
            self.token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
            
            # If still no token, try to load from standard HF token file
            if self.token is None:
                self.token = self._load_token_from_standard_location()
    
    def _load_token_from_standard_location(self) -> Optional[str]:
        """Load token from standard Hugging Face token locations."""
        # Standard HF token locations (in order of preference)
        token_locations = [
            Path.home() / ".huggingface" / "token",  # New default location
            Path.home() / ".cache" / "huggingface" / "token",  # Alternative location
        ]
        
        for token_path in token_locations:
            if token_path.exists() and token_path.is_file():
                try:
                    with open(token_path, "r", encoding="utf-8") as f:
                        token = f.read().strip()
                    if token:
                        return token
                except (IOError, OSError):
                    continue
        
        return None
    
    @classmethod
    def from_env(cls) -> "HFConfig":
        """Create configuration from environment variables."""
        return cls(
            cache_dir=os.environ.get("HF_CACHE_DIR"),
            token=os.environ.get("HF_TOKEN"),
            endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            max_retries=int(os.environ.get("HF_MAX_RETRIES", "3")),
            timeout=int(os.environ.get("HF_TIMEOUT", "300"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "cache_dir": self.cache_dir,
            "token": "***" if self.token else None,  # Hide token in output
            "endpoint": self.endpoint,
            "max_retries": self.max_retries,
            "chunk_size": self.chunk_size,
            "timeout": self.timeout,
        }