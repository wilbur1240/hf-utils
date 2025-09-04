#!/usr/bin/env python3
"""
Simple PPO USV model upload script - no temporary directories.
Files are organized in current directory for easy inspection.
"""

import json
import shutil
from pathlib import Path

from hf_utils import HFConfig, AuthManager
from hf_utils.models.upload import ModelUploader

def main():
    print("🚀 PPO USV Model Upload (Simple Version)")
    print("=" * 50)
    
    # =================================================================
    # CONFIGURE YOUR PATHS HERE
    # =================================================================
    
    source_files = {
        "ppo_usv.zip": str(Path("~/local_collision_avoidance/rl_model/ppo/ppo_usv.zip").expanduser()),
        "residual_and_prediction.png": str(Path("~/local_collision_avoidance/figs/residual_and_prediction.png").expanduser()),  # Update path
        "residual_wo_obs.png": str(Path("~/local_collision_avoidance/figs/residual_wo_obs.png").expanduser()),                  # Update path
        "residual_w_prediction_best.png": str(Path("~/local_collision_avoidance/figs/residual_w_prediction_best.png").expanduser())  # Update path
    }
    
    repo_id = "Wilbur1240/ppo-usv"  # ← UPDATE WITH YOUR USERNAME
    
    # =================================================================
    # SETUP
    # =================================================================
    
    # Create working directory (you can see this!)
    work_dir = Path("./models/ppo_usv")
    work_dir.mkdir(exist_ok=True)
    
    print(f"📁 Working directory: {work_dir.absolute()}")
    
    # Setup authentication and uploader
    config = HFConfig()
    auth_manager = AuthManager(config)
    uploader = ModelUploader(config, auth_manager)
    
    if not auth_manager.is_authenticated():
        print("❌ Not authenticated. Please run: hf-utils auth login")
        return 1
    
    print(f"👤 Authenticated as: {auth_manager.get_username()}")
    
    # =================================================================
    # COPY FILES
    # =================================================================
    
    print("\n📋 Copying files...")
    for dest_name, source_path in source_files.items():
        source = Path(source_path)
        dest = work_dir / dest_name
        
        if source.exists():
            shutil.copy2(source, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"✅ {dest_name} ({size_mb:.2f} MB)")
        else:
            print(f"❌ File not found: {source}")
            print(f"   Please update the path in the script")
            return 1
    
    # =================================================================
    # CREATE README WITH EMBEDDED IMAGES
    # =================================================================
    
    readme_content = """---
license: apache-2.0
tags:
- reinforcement-learning
- ppo
- navigation
- usv
- autonomous-vehicle
library_name: stable-baselines3
---

# PPO USV Navigation Model

## Model Description

This is a Proximal Policy Optimization (PPO) reinforcement learning model trained for Unmanned Surface Vehicle (USV) navigation. The model learns to navigate in maritime environments while avoiding obstacles and reaching target destinations.

## Model Details

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Environment**: USV Navigation Simulation
- **Framework**: Stable Baselines3
- **Training Episodes**: [Add your training details]
- **Performance**: [Add your performance metrics]

## Files Included

- `ppo_usv.zip`: Trained PPO model weights and configuration
- `residual_and_prediction.png` : Residual RL navigation with obs prediction result
- `residual_wo_obs.png` : Residual RL navigation without obs result
- `residual_w_prediction_best.png` : Residual RL navigation with obs prediction best result

## Results Visualization

### Residual RL Navigation with Obstacle Prediction
![Residual RL with Prediction](residual_and_prediction.png)

This visualization shows the navigation performance when using residual reinforcement learning combined with obstacle prediction. The model demonstrates improved path planning and obstacle avoidance capabilities.

### Residual RL Navigation without Obstacle Prediction
![Residual RL without Obstacles](residual_wo_obs.png)

Comparison results showing the model's performance when obstacle prediction is disabled. This baseline helps demonstrate the value of the prediction component.

### Best Performance Results
![Best Performance](residual_w_prediction_best.png)

The best achieved performance combining residual learning with obstacle prediction, showcasing optimal navigation trajectories and collision avoidance behaviors.

## Usage

see [ARG-NCTU/local_collision_avoidance](https://github.com/ARG-NCTU/local_collision_avoidance) for more details.

## Training Details

[Add details about your training process, hyperparameters, etc.]

## Results

The model achieved the following performance:

- Average Episode Reward: [Your metric]
- Success Rate: [Your metric]
- Navigation Efficiency: [Your metric]

See the included PNG files for detailed performance visualizations.
"""
    
    readme_path = work_dir / "README.md"
    readme_path.write_text(readme_content, encoding='utf-8')
    print("✅ Created README.md with embedded images")
    
    # =================================================================
    # CREATE CONFIG.JSON
    # =================================================================
    
    config_data = {
        "model_type": "ppo",
        "framework": "reinforcement_learning", 
        "library_name": "stable-baselines3",
        "environment": "usv_navigation",
        "approach": "residual_learning",
        "features": ["obstacle_prediction", "collision_avoidance", "maritime_navigation"],
        "tags": [
            "reinforcement-learning", 
            "ppo", 
            "navigation", 
            "usv", 
            "autonomous-vehicle",
            "collision-avoidance",
            "residual-learning"
        ]
    }
    
    config_path = work_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print("✅ Created config.json")
    
    # =================================================================
    # SHOW FILES READY FOR UPLOAD
    # =================================================================
    
    print(f"\n📋 Files ready for upload in {work_dir}:")
    total_size = 0
    for file_path in work_dir.iterdir():
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"   📄 {file_path.name} ({size_mb:.2f} MB)")
    
    print(f"   📊 Total size: {total_size:.2f} MB")
    
    # =================================================================
    # UPLOAD
    # =================================================================
    
    print(f"\n🚀 Uploading to {repo_id}...")
    
    try:
        repo_url = uploader.upload_model(
            model_path=str(work_dir),
            repo_id=repo_id,
            commit_message="Upload PPO USV model with residual learning and embedded visualizations",
            commit_description="Complete PPO model for USV autonomous navigation with obstacle prediction",
            model_card={
                "license": "apache-2.0",
                "tags": [
                    "reinforcement-learning", 
                    "ppo", 
                    "navigation", 
                    "usv", 
                    "collision-avoidance",
                    "residual-learning"
                ]
            }
        )
        
        print(f"✅ SUCCESS! Model uploaded to:")
        print(f"🔗 {repo_url}")
        print(f"📊 Images will display automatically on the model page")
        
        # Cleanup option
        cleanup = input(f"\nDelete local files in {work_dir}? (y/N): ")
        if cleanup.lower() in ['y', 'yes']:
            shutil.rmtree(work_dir)
            print("🗑️  Local files deleted")
        else:
            print(f"📁 Files kept at: {work_dir.absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"📁 Files are still available at: {work_dir.absolute()}")
        return 1

if __name__ == "__main__":
    exit(main())