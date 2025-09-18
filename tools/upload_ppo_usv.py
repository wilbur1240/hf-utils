#!/usr/bin/env python3
"""
Structured PPO USV model upload script with organized directories.
Creates clean directory structure on Hugging Face Hub.
"""

import json
import shutil
from pathlib import Path

from hf_utils import HFConfig, AuthManager
from hf_utils.models.upload import ModelUploader

def create_structured_directories(work_dir: Path, source_files: dict):
    """Create organized directory structure."""
    
    # Define directory structure
    directories = {
        "model": work_dir / "model",
        "results": work_dir / "results" / "simulation",
        "trajectories": work_dir / "results" / "real_world",
        "rosbags": work_dir / "results" / "rosbags",
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # File organization mapping
    file_organization = {
        # Model files
        "ppo_usv.zip": directories["model"] / "ppo_usv.zip",
        
        # Simulation results
        "residual_and_prediction.png": directories["results"] / "residual_with_prediction.png",
        "residual_wo_obs.png": directories["results"] / "residual_wo_obs.png", 
        "residual_w_prediction_best.png": directories["results"] / "best_performance.png",
        
        # Real-world trajectories
        "real_collision_traj_1.png": directories["trajectories"] / "trajectory_trial_1.png",
        "real_collision_traj_2.png": directories["trajectories"] / "trajectory_trial_2.png",
        "real_collision_traj_3.png": directories["trajectories"] / "trajectory_trial_3.png",
        "real_collision_traj_4.png": directories["trajectories"] / "trajectory_trial_4.png",
        "real_collision_traj_5.png": directories["trajectories"] / "trajectory_trial_5.png",
        "real_collision_traj_6.png": directories["trajectories"] / "trajectory_trial_6.png",
        "real_collision_traj_7.png": directories["trajectories"] / "trajectory_trial_7.png",
        "real_collision_traj_8.png": directories["trajectories"] / "trajectory_trial_8.png",
    
        # Rosbags
        "trial_1.bag": directories["rosbags"] / "_2025-07-17-14-50-47_0.bag",
        "trial_2.bag": directories["rosbags"] / "_2025-07-17-14-56-45_0.bag",
        "trial_3.bag": directories["rosbags"] / "_2025-07-17-15-02-19_0.bag",
        "trial_4.bag": directories["rosbags"] / "_2025-07-17-15-11-32_0.bag",
        "trial_5.bag": directories["rosbags"] / "_2025-07-17-15-16-48_0.bag",
        "trial_6.bag": directories["rosbags"] / "_2025-07-17-15-20-29_0.bag",
        "trial_7.bag": directories["rosbags"] / "_2025-07-17-15-29-50_0.bag",
        "trial_8.bag": directories["rosbags"] / "_2025-07-17-15-49-04_0.bag",
    }
    
    return directories, file_organization

def create_structured_readme(directories: dict) -> str:
    """Create README with structured file references."""
    
    return """---
license: apache-2.0
tags:
- reinforcement-learning
- ppo
- navigation
- usv
- collision-avoidance
- residual-learning
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
---

# PPO USV Navigation Model

## Model Description

This is a Proximal Policy Optimization (PPO) reinforcement learning model trained for Unmanned Surface Vehicle (USV) navigation. The model learns to navigate in maritime environments while avoiding obstacles and reaching target destinations using residual learning techniques.

## Repository Structure

```
├── model/                          # Trained model files
│   └── ppo_usv.zip                # PPO model weights and configuration
├── results/                       # Experimental results
│   └── simulation/                # Simulation environment results
│       ├── residual_with_prediction.png     # With obstacle prediction
│       ├── residual_wo_obs.png  # Without obstacle prediction  
│       └── best_performance.png             # Best achieved performance
│   └── real_world/               # Real-world deployment results
│       ├── trajectory_trial_1.png # USV trajectory trial 1
|       |   ...        
│       └── trajectory_trial_8.png # USV trajectory trial 8
│   └── rosbags/                  # ROS bag files from real-world trials
|       └── ...
├── configs/                      # Configuration files
│   └── model_config.json        # Model hyperparameters and settings
└── docs/                        # Additional documentation
    └── training_details.md      # Detailed training information
```

## Model Performance

### Simulation Results

#### Residual RL without Obstacle Prediction (Baseline)
![Residual RL without Obstacles](results/simulation/residual_wo_obs.png)

**Performance Metrics:**
- Success Rate (Episodes 1-100): 100.00%
- Success Rate (Episodes 1-200): 100.00%

This baseline demonstrates the model's performance when obstacle prediction is disabled.

#### Residual RL with Obstacle Prediction
![Residual RL with Prediction](results/simulation/residual_with_prediction.png)

**Performance Metrics:**
- Best Success Rate: 89.20% (Episodes 1-213)
- Success Rate (Episodes 1-200): 88.50%

This visualization shows the navigation performance when using residual reinforcement learning combined with obstacle prediction. The model demonstrates improved path planning and obstacle avoidance capabilities.

The best achieved performance combining residual learning with obstacle prediction, showcasing optimal navigation trajectories and collision avoidance behaviors.

---

### Real-World Deployment

The model was successfully deployed on Jong Shyn No.5 USV for real-world collision avoidance validation.

#### Navigation Trajectories Examples
![Real Trajectory 1](results/real_world/trajectory_trial_1.png)
*Trial 1: Successful obstacle avoidance maneuver*

![Real Trajectory 5](results/real_world/trajectory_trial_5.png)
*Trial 5: Complex multi-obstacle navigation scenario*

These results demonstrate successful **simulation-to-real transfer** of the trained policy. See more trajectories in the 'results/real_world/' directory.

### Experiment Metadata

| Trial | Date       | Model Weights   | Method                     | ROSBag Info              |
|-------|------------|-----------------|----------------------------|--------------------------|
| 1     | 2025-07-17 | `ppo_usv.zip` | Residual RL + Prediction   | [`trial_1 bag`](results/rosbags/_2025-07-17-14-50-47_0.bag) |
| 2     | 2025-07-17 | `ppo_usv.zip` | Residual RL + Prediction   | [`trial_2 bag`](results/rosbags/_2025-07-17-14-56-45_0.bag) |
| 3     | 2025-07-17 | `ppo_usv.zip` | Residual RL + Prediction   | [`trial_3 bag`](results/rosbags/_2025-07-17-15-02-19_0.bag) |
| 4     | 2025-07-17 | `ppo_usv.zip` | Residual RL + Prediction   | [`trial_4 bag`](results/rosbags/_2025-07-17-15-11-32_0.bag) |
| 5     | 2025-07-17 | `ppo_usv.zip` | Residual RL + Prediction   | [`trial_5 bag`](results/rosbags/_2025-07-17-15-16-48_0.bag) |
| 6     | 2025-07-17 | `ppo_usv.zip` | Residual RL + Prediction   | [`trial_6 bag`](results/rosbags/_2025-07-17-15-20-29_0.bag) |
| 7     | 2025-07-17 | `ppo_usv.zip` | Residual RL + Prediction   | [`trial_7 bag`](results/rosbags/_2025-07-17-15-29-50_0.bag) |
| 8     | 2025-07-17 | `ppo_usv.zip` | Residual RL + Prediction   | [`trial_8 bag`](results/rosbags/_2025-07-17-15-49-04_0.bag) |

---
## Quick Start Inferencing

### Download and Setup
```bash
# Download the model
wget https://huggingface.co/Wilbur1240/ppo-usv/resolve/main/model/ppo_usv.zip

# move to your local collision avoidance directory
mv ppo_usv.zip -d local_collision_avoidance/rl_model/ppo/

# launch the sim lidar crop launch file
roslaunch js_lidar_crop dynamic_lidar_crop_sim.launch # for simulation world inferencing

# start rl model 
source ~/local_collision_avoidance/script/ppo.sh

# run inferencing script in ubuntu terminal
python3 ~/local_collision_avoidance/script/inference.py
```


## Implementation Details

For complete implementation, training scripts, and deployment code:
- **Repository**: [ARG-NCTU/local_collision_avoidance](https://github.com/ARG-NCTU/local_collision_avoidance)
- **Training Script**: [train_usv.py](https://github.com/ARG-NCTU/local_collision_avoidance/blob/main/rl/ppo/train_usv.py)

### Key Hyperparameters
- `gamma=0.999` (discount factor)
- `gae_lambda=0.98` (GAE lambda)
- `batch_size=64` (training batch size)

"""

def create_additional_docs(directories: dict):
    """Create additional documentation files."""
    
    # Training details
    training_details = """# Training Details

## Environment Setup
- **Simulator**: Custom USV navigation environment with ROS integration
- **Observation Space**: Multi-modal sensor data (LiDAR, GPS, IMU, camera)
- **Action Space**: Continuous control (thrust, steering angle)
- **Reward Function**: Distance-based with collision penalties and efficiency bonuses

## Training Configuration
```python
# PPO Hyperparameters
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5
}
```

## Residual Learning Architecture
The model uses a residual learning approach where:
1. **Base Controller**: Traditional PID-based navigation
2. **Learned Residual**: Neural network corrections
3. **Final Action**: Base action + learned residual

## Training Results
- **Total Timesteps**: 1,000,000
- **Training Duration**: ~12 hours on RTX 3080
- **Convergence**: Achieved stable performance after 800k steps
"""
    
    docs_dir = directories["docs"]
    (docs_dir / "training_details.md").write_text(training_details)
    
    # Model configuration
    model_config = {
        "model_architecture": {
            "policy_type": "MlpPolicy",
            "feature_extractor": "CustomUSVExtractor",
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256]
            }
        },
        "training_hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.999,
            "gae_lambda": 0.98,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5
        },
        "environment_config": {
            "observation_space": "multi_modal_sensors",
            "action_space": "continuous_control",
            "reward_function": "distance_based_with_penalties",
            "episode_length": 1000
        },
        "deployment_info": {
            "platform": "Jong Shyn No.5 USV",
            "ros_version": "ROS Noetic",
            "deployment_date": "2025-07-17",
            "validation_trials": 10
        }
    }
    
    config_path = directories["configs"] / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)

def main():
    print("🚀 PPO USV Model Upload (Structured Version)")
    print("=" * 50)
    
    # =================================================================
    # CONFIGURE YOUR PATHS HERE  
    # =================================================================
    
    source_files = {
        "ppo_usv.zip": str(Path("~/hf-utils/models/ppo_usv/ppo_usv.zip").expanduser()),
        "residual_and_prediction.png": str(Path("~/hf-utils/models/ppo_usv/residual_and_prediction.png").expanduser()),
        "residual_wo_obs.png": str(Path("~/hf-utils/models/ppo_usv/residual_wo_obs.png").expanduser()),
        "residual_w_prediction_best.png": str(Path("~/hf-utils/models/ppo_usv/residual_w_prediction_best.png").expanduser()),
        "real_collision_traj_1.png": str(Path("~/hf-utils/models/ppo_usv/real_collision_traj_1.png").expanduser()),
        "real_collision_traj_2.png": str(Path("~/hf-utils/models/ppo_usv/real_collision_traj_2.png").expanduser()),
        "real_collision_traj_3.png": str(Path("~/hf-utils/models/ppo_usv/real_collision_traj_3.png").expanduser()),
        "real_collision_traj_4.png": str(Path("~/hf-utils/models/ppo_usv/real_collision_traj_4.png").expanduser()),
        "real_collision_traj_5.png": str(Path("~/hf-utils/models/ppo_usv/real_collision_traj_5.png").expanduser()),
        "real_collision_traj_6.png": str(Path("~/hf-utils/models/ppo_usv/real_collision_traj_6.png").expanduser()),
        "real_collision_traj_7.png": str(Path("~/hf-utils/models/ppo_usv/real_collision_traj_7.png").expanduser()),
        "real_collision_traj_8.png": str(Path("~/hf-utils/models/ppo_usv/real_collision_traj_8.png").expanduser()),
        "trial_1.bag": str(Path("~/local_collision_avoidance/0717/0717_1450/").expanduser()) + "/_2025-07-17-14-50-47_0.bag",
        "trial_2.bag": str(Path("~/local_collision_avoidance/0717/0717_1456/").expanduser()) + "/_2025-07-17-14-56-45_0.bag",
        "trial_3.bag": str(Path("~/local_collision_avoidance/0717/0717_1502/").expanduser()) + "/_2025-07-17-15-02-19_0.bag",
        "trial_4.bag": str(Path("~/local_collision_avoidance/0717/0717_1511/").expanduser()) + "/_2025-07-17-15-11-32_0.bag",
        "trial_5.bag": str(Path("~/local_collision_avoidance/0717/0717_1516/").expanduser()) + "/_2025-07-17-15-16-48_0.bag",
        "trial_6.bag": str(Path("~/local_collision_avoidance/0717/0717_1520/").expanduser()) + "/_2025-07-17-15-20-29_0.bag",
        "trial_7.bag": str(Path("~/local_collision_avoidance/0717/0717_1529/").expanduser()) + "/_2025-07-17-15-29-50_0.bag",
        "trial_8.bag": str(Path("~/local_collision_avoidance/0717/0717_1548/").expanduser()) + "/_2025-07-17-15-49-04_0.bag",
    }
    
    repo_id = "Wilbur1240/ppo-usv"
    
    # =================================================================
    # SETUP STRUCTURED DIRECTORIES
    # =================================================================
    
    work_dir = Path("./models/ppo_usv")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Working directory: {work_dir.absolute()}")
    
    # Create directory structure
    directories, file_organization = create_structured_directories(work_dir, source_files)
    
    print("📂 Created directory structure:")
    for name, path in directories.items():
        print(f"   📁 {name}: {path.relative_to(work_dir)}")
    
    # Setup authentication
    config = HFConfig()
    auth_manager = AuthManager(config)
    uploader = ModelUploader(config, auth_manager)
    
    if not auth_manager.is_authenticated():
        print("❌ Not authenticated. Please run: hf-utils auth login")
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
    # CREATE DOCUMENTATION
    # =================================================================
    
    print("\n📝 Creating documentation...")
    
    # Create structured README
    readme_content = create_structured_readme(directories)
    readme_path = work_dir / "README.md"
    readme_path.write_text(readme_content, encoding='utf-8')
    print("✅ Created structured README.md")
    
    # Create additional documentation
    # create_additional_docs(directories)
    # print("✅ Created training details and configuration files")
    
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
    # UPLOAD
    # =================================================================
    
    print(f"\n🚀 Uploading structured repository to {repo_id}...")
    
    try:
        repo_url = uploader.upload_model(
            model_path=str(work_dir),
            repo_id=repo_id,
            commit_message="Upload structured PPO USV model with organized directories",
            commit_description="Well-organized repository with model files, results, documentation, and configurations in structured directories",
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
        
        print(f"✅ SUCCESS! Structured repository uploaded to:")
        print(f"🔗 {repo_url}")
        print(f"📁 Browse the organized structure on the Files tab")
        
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