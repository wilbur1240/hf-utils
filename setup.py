# setup.py
from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from a file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    return []

# Main setup configuration
setup(
    # Basic package information
    name="hf-utils",
    version="0.1.0",
    author="Wilbur",
    author_email="your.email@example.com",
    description="Comprehensive utilities for Hugging Face Hub operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wilbur1240/hf-utils",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Runtime dependencies
    install_requires=[
        "huggingface-hub>=0.19.0",
        "datasets>=2.14.0", 
        "pandas>=1.3.0",
        "pyarrow>=10.0.0",
        "tqdm>=4.64.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
    ],
    
    # Optional dependencies (extras)
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=2.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
            "responses>=0.23.0",
        ],
        # You can install all extras with: pip install -e ".[all]"
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=2.0.0",
            "responses>=0.23.0",
        ],
    },
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "hf-utils=hf_utils.cli.main:main",
        ],
    },
    
    # Include additional files
    include_package_data=True,
    
    # Package data (files to include in package)
    package_data={
        "hf_utils": ["py.typed"],  # Type hints marker file
    },
    
    # Zip safe (can be imported from zip file)
    zip_safe=False,
)