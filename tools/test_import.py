#!/usr/bin/env python3
"""
Test script to verify hf_utils imports work correctly.
Run this after installing the package to check for import issues.
"""

def test_basic_imports():
    """Test basic imports from hf_utils."""
    print("Testing basic imports...")
    
    try:
        import hf_utils
        print("✅ hf_utils imported successfully")
    except Exception as e:
        print(f"❌ Failed to import hf_utils: {e}")
        return False
    
    try:
        from hf_utils import HFConfig, AuthManager
        print("✅ Core classes imported successfully")
    except Exception as e:
        print(f"❌ Failed to import core classes: {e}")
        return False
    
    try:
        from hf_utils.models import ModelUploader, ModelDownloader, ModelManager
        print("✅ Model classes imported successfully")
    except Exception as e:
        print(f"❌ Failed to import model classes: {e}")
        return False
    
    try:
        from hf_utils.datasets import DatasetUploader, DatasetDownloader, DatasetManager
        print("✅ Dataset classes imported successfully")
    except Exception as e:
        print(f"❌ Failed to import dataset classes: {e}")
        return False
    
    try:
        from hf_utils.common.config import HFConfig
        from hf_utils.common.exceptions import HFUtilsError, ValidationError
        from hf_utils.common.validators import validate_repo_name
        print("✅ Common utilities imported successfully")
    except Exception as e:
        print(f"❌ Failed to import common utilities: {e}")
        return False
    
    return True

def test_class_instantiation():
    """Test that classes can be instantiated."""
    print("\nTesting class instantiation...")
    
    try:
        from hf_utils import HFConfig, AuthManager
        
        # Test config creation
        config = HFConfig()
        print(f"✅ HFConfig created: cache_dir={config.cache_dir}")
        
        # Test auth manager creation
        auth_manager = AuthManager(config)
        print("✅ AuthManager created successfully")
        
        # Test model classes
        from hf_utils.models import ModelUploader, ModelDownloader, ModelManager
        
        model_uploader = ModelUploader(config, auth_manager)
        model_downloader = ModelDownloader(config, auth_manager)
        model_manager = ModelManager(config, auth_manager)
        print("✅ Model classes instantiated successfully")
        
        # Test dataset classes
        from hf_utils.datasets import DatasetUploader, DatasetDownloader, DatasetManager
        
        dataset_uploader = DatasetUploader(config, auth_manager)
        dataset_downloader = DatasetDownloader(config, auth_manager)
        dataset_manager = DatasetManager(config, auth_manager)
        print("✅ Dataset classes instantiated successfully")
        
    except Exception as e:
        print(f"❌ Failed to instantiate classes: {e}")
        return False
    
    return True

def test_validators():
    """Test validation functions."""
    print("\nTesting validators...")
    
    try:
        from hf_utils.common.validators import validate_repo_name, sanitize_filename
        
        # Test repo name validation
        assert validate_repo_name("username/model-name") == True
        assert validate_repo_name("invalid-name") == False
        print("✅ Repository name validation works")
        
        # Test filename sanitization
        clean_name = sanitize_filename("file<>name.txt")
        assert "<" not in clean_name and ">" not in clean_name
        print("✅ Filename sanitization works")
        
    except Exception as e:
        print(f"❌ Failed validator tests: {e}")
        return False
    
    return True

def test_cli_import():
    """Test CLI imports."""
    print("\nTesting CLI imports...")
    
    try:
        from hf_utils.cli.main import HFUtilsCLI
        
        cli = HFUtilsCLI()
        parser = cli.create_parser()
        print("✅ CLI classes imported and parser created successfully")
        
    except Exception as e:
        print(f"❌ Failed CLI import test: {e}")
        return False
    
    return True

def main():
    """Run all import tests."""
    print("🧪 Testing hf_utils imports...")
    print("=" * 50)
    
    all_passed = True
    
    # Run all tests
    tests = [
        test_basic_imports,
        test_class_instantiation, 
        test_validators,
        test_cli_import
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            all_passed = False
        print()
    
    # Final result
    print("=" * 50)
    if all_passed:
        print("🎉 All import tests passed! hf_utils is ready to use.")
        
        # Show package info
        try:
            import hf_utils
            print(f"📦 Package version: {hf_utils.__version__}")
            print(f"👤 Author: {hf_utils.__author__}")
            
            # Show available classes
            print("\n📋 Available classes:")
            print("  • HFConfig - Configuration management")
            print("  • AuthManager - Authentication handling")
            print("  • ModelUploader/Downloader/Manager - Model operations") 
            print("  • DatasetUploader/Downloader/Manager - Dataset operations")
            
        except Exception as e:
            print(f"⚠️  Could not get package info: {e}")
            
    else:
        print("❌ Some import tests failed. Check the errors above.")
        print("\n🔧 Common fixes:")
        print("  1. Install the package: pip install -e .")
        print("  2. Install dependencies: pip install -e '.[dev]'")
        print("  3. Check Python path: echo $PYTHONPATH")
        print("  4. Verify you're in the right environment: which python")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())


# Additional diagnostic script for troubleshooting

def diagnose_import_issues():
    """Diagnose common import issues."""
    print("🔍 Diagnosing import issues...")
    print("=" * 50)
    
    import sys
    import os
    from pathlib import Path
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check current working directory
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Check Python path
    print("🛤️  Python path:")
    for path in sys.path:
        print(f"   {path}")
    
    # Check if src directory exists
    src_path = Path("src")
    if src_path.exists():
        print(f"✅ Found src directory: {src_path.absolute()}")
        
        # Check hf_utils directory
        hf_utils_path = src_path / "hf_utils"
        if hf_utils_path.exists():
            print(f"✅ Found hf_utils package: {hf_utils_path.absolute()}")
            
            # List package contents
            print("📦 Package contents:")
            for item in hf_utils_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    print(f"   📁 {item.name}/")
                elif item.suffix == '.py':
                    print(f"   📄 {item.name}")
        else:
            print(f"❌ hf_utils package not found in {src_path.absolute()}")
    else:
        print("❌ src directory not found")
    
    # Check if package is installed
    try:
        import pkg_resources
        try:
            pkg_resources.get_distribution("hf-utils")
            print("✅ hf-utils package is installed")
        except pkg_resources.DistributionNotFound:
            print("❌ hf-utils package is not installed")
            print("   Try: pip install -e .")
    except ImportError:
        print("⚠️  pkg_resources not available")
    
    # Check for __init__.py files
    init_files = [
        "src/hf_utils/__init__.py",
        "src/hf_utils/auth/__init__.py", 
        "src/hf_utils/models/__init__.py",
        "src/hf_utils/datasets/__init__.py",
        "src/hf_utils/common/__init__.py",
        "src/hf_utils/cli/__init__.py"
    ]
    
    print("\n📄 Checking __init__.py files:")
    for init_file in init_files:
        if Path(init_file).exists():
            print(f"   ✅ {init_file}")
        else:
            print(f"   ❌ {init_file}")

if __name__ == "__main__":
    main()
    print("\n" + "=" * 50)
    diagnose_import_issues()