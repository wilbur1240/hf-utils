#!/usr/bin/env python3
"""
Upload HRL-PPO-USV model to Hugging Face Hub.
Uploads only model_weight.zip and vecnormalize.pkl files.
"""

import shutil
from pathlib import Path

from hf_utils import HFConfig, AuthManager
from hf_utils.models.upload import ModelUploader

def create_structured_directories(work_dir: Path, source_files: dict):
    """Create organized directory structure."""

    # define directory structure
    directories = {
        "model": work_dir / "model",
    }

    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # File organization mapping
    file_organization = {
        # Model files
        "policy-weight": directories["model"] / "hrl-v1-policy-weights.zip",
        "vecnormalize": directories["model"] / "hrl-v1-vecnormalize.pkl",
    }

    return directories, file_organization

def create_readme() -> str:
    """Create README for HRL-PPO-USV model."""
    
    return """---
license: apache-2.0
tags:
- reinforcement-learning
- ppo
- hierarchical-rl
- usv
- navigation
- collision-avoidance
- continuous-control
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
---

# HRL-PPO-USV Model

## Model Description

This is a Hierarchical Reinforcement Learning (HRL) model using Proximal Policy Optimization (PPO) trained for Unmanned Surface Vehicle (USV) control tasks.

## Repository Structure

```
├── model/                          # Trained model files
│   ├── hrl-v1-policy-weight.zip    # model weights and configuration
│   └── hrl-v1-vecnormalize.pkl     # VecNormalize wrapper for observation normalization
└── README.md                       # This README file
```

## Model Performance

### Simulation Results

### Real-World Deployment

---
## Inferencing

---
## Training 

- **Algorithm**: PPO
- **Framework**: Stable Baselines3
- **Task**: USV Collidsion Avoidance

"""


def main():
    """Main upload function."""
    
    # =================================================================
    # CONFIGURATION - MODIFY THESE
    # =================================================================
    
    REPO_ID = "Wilbur1240/hrl-ppo-usv"  # Change to your HuggingFace username
    
    # Source files - update these paths to your actual files
    source_files = {
        "policy-weight": "./models/hrl_policy/hrl-v1-policy-weights.zip",  # UPDATE THIS PATH
        "vecnormalize": "./models/hrl_policy/hrl-v1-vecnormalize.pkl",  # UPDATE THIS PATH
    }
    
    # =================================================================
    # SETUP
    # =================================================================
    
    print("=" * 60)
    print("HRL-PPO-USV Model Upload to Hugging Face")
    print("=" * 60)
    
    # Create working directory
    work_dir = Path("./models/hrl_ppo_usv")
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Working directory: {work_dir.absolute()}")
    
    # Create directory structure
    directories, file_organization = create_structured_directories(work_dir, source_files)
    print("📁 Created directory structure:")
    for name, path in directories.items():
        print(f"   📁 {name}: {path.relative_to(work_dir)}")

    # Verify authentication
    config = HFConfig()
    auth_manager = AuthManager(config)
    uploader = ModelUploader(config, auth_manager)
    if not auth_manager.is_authenticated():
        print("❌ Not authenticated with Hugging Face.")
        print("   Please run: hf-utils auth login")
        return 1
    
    print(f"👤 Authenticated as: {auth_manager.get_username()}")
    
    # =================================================================
    # ORGANIZE FILES INTO STRUCTURE
    # =================================================================
    
    print("\n📋 Organizing files into structured directories...")
    for source_key, dest_path in file_organization.items():
        source_path = source_files[source_key]
        source = Path(source_path)
        
        if source.exists():
            shutil.copy2(source, dest_path)
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            rel_dest = dest_path.relative_to(work_dir)
            print(f"✅ {rel_dest} ({size_mb:.2f} MB)")
        else:
            print(f"❌ File not found: {source}")
            return 1
    
    # =================================================================
    # CREATE README
    # =================================================================
    
    print("\n📝 Creating README...")
    readme_content = create_readme()
    readme_path = work_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"✅ README.md created")
    
    # =================================================================
    # SHOW FINAL STRUCTURE
    # =================================================================
    
    print(f"\n📁 Final repository structure:")
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(item, next_prefix, max_depth, current_depth + 1)
    
    show_tree(work_dir)
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in work_dir.rglob("*") if f.is_file())
    print(f"\n📊 Total repository size: {total_size / (1024 * 1024):.2f} MB")

    # =================================================================
    # UPLOAD TO HUGGING FACE
    # =================================================================
    
    print("\n🚀 Uploading to Hugging Face Hub...")
    print(f"📦 Repository: {REPO_ID}")
    
    try:
        repo_url = uploader.upload_model(
            model_path=str(work_dir),
            repo_id=REPO_ID,
            commit_message="Upload hrl-ppo-usv model",
            commit_description="Initial upload of HRL-PPO-USV model files and README",
            model_card={
                "license": "apache-2.0",
                "tags": [
                    "reinforcement-learning",
                    "ppo", 
                    "navigation",
                    "usv",
                    "collision-avoidance",
                    "residual-learning",
                    "maritime-robotics"
                ]
            }
        )
        
        print("\n" + "=" * 60)
        print("✅ Upload completed successfully!")
        print("=" * 60)
        print(f"\n🔗 Repository uploaded to:")
        print(f"   {repo_url}")
        
        # Cleanup option
        cleanup = input(f"\nDelete local files in {work_dir}? (y/N): ")
        if cleanup.lower() in ['y', 'yes']:
            shutil.rmtree(work_dir)
            print("🗑️ Local files deleted")
        else:
            print(f"📁 Files kept at: {work_dir.absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"📁 Files are available at: {work_dir.absolute()}")
        return 1


if __name__ == "__main__":
    exit(main())