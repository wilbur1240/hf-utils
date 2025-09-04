#!/usr/bin/env python3
"""
Test script to verify authentication persistence works correctly.
"""

from pathlib import Path
import os

def test_token_persistence():
    """Test that tokens are saved and loaded correctly."""
    print("🔐 Testing authentication persistence...")
    
    # Check current token status
    from hf_utils.common.config import HFConfig
    from hf_utils.auth.manager import AuthManager
    
    config = HFConfig()
    auth_manager = AuthManager(config)
    
    print(f"📁 Cache directory: {config.cache_dir}")
    print(f"🔑 Token from config: {'***' if config.token else 'None'}")
    
    # Check if token file exists
    token_path = Path.home() / ".huggingface" / "token"
    print(f"💾 Token file exists: {token_path.exists()}")
    
    if token_path.exists():
        try:
            with open(token_path, 'r') as f:
                token_content = f.read().strip()
            print(f"📄 Token file has content: {bool(token_content)}")
            print(f"🔢 Token length: {len(token_content) if token_content else 0} characters")
        except Exception as e:
            print(f"❌ Error reading token file: {e}")
    
    # Test authentication status
    try:
        is_authenticated = auth_manager.is_authenticated()
        print(f"✅ Authentication status: {is_authenticated}")
        
        if is_authenticated:
            user_info = auth_manager.get_user_info()
            print(f"👤 User: {user_info.get('name', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error checking authentication: {e}")
    
    return True

def check_environment():
    """Check environment variables and paths."""
    print("\n🌍 Environment check:")
    
    env_vars = [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN", 
        "HF_HOME",
        "HF_CACHE_DIR"
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            # Hide token values for security
            if "TOKEN" in var:
                display_value = "***"
            else:
                display_value = value
            print(f"  {var}: {display_value}")
        else:
            print(f"  {var}: (not set)")
    
    # Check standard paths
    paths_to_check = [
        Path.home() / ".huggingface",
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface",
    ]
    
    print("\n📁 Path check:")
    for path in paths_to_check:
        status = "✅ exists" if path.exists() else "❌ missing"
        path_type = "📁 dir" if path.is_dir() else "📄 file" if path.is_file() else "❓ unknown"
        print(f"  {path}: {status} {path_type}")

def main():
    """Run authentication persistence tests."""
    print("🧪 Authentication Persistence Test")
    print("=" * 50)
    
    try:
        check_environment()
        print()
        test_token_persistence()
        
        print("\n" + "=" * 50)
        print("💡 Tips for troubleshooting:")
        print("1. Make sure you ran 'hf-utils auth login' successfully")
        print("2. Check that ~/.huggingface/token file exists and has content")
        print("3. Verify file permissions are correct (readable)")
        print("4. Try setting HF_TOKEN environment variable as backup")
        
        print("\n🔧 Manual verification:")
        print("  ls -la ~/.huggingface/")
        print("  cat ~/.huggingface/token")
        print("  hf-utils auth whoami")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    exit(main())