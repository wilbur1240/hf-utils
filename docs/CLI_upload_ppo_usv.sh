# Do not run this script directly. It is a documentation file.

# =============================================================================
# CLI METHOD: Upload RL Model with Files and Description
# =============================================================================

# Step 1: Prepare your model directory structure
# Create a directory for your model files
mkdir -p ~/hf-utils/models/ppo_usv
cd ~/hf-utils/models/ppo_usv

# Copy your files to the model directory
cp ~/local_collision_avoidance/rl_model/ppo/ppo_usv.zip .
cp ~/local_collision_avoidance/figs/residual_and_prediction.png .
cp ~/local_collision_avoidance/figs/residual_wo_obs.png . 
cp ~/local_collision_avoidance/figs/residual_w_prediction_best.png .

# Step 2: Create a model configuration file (optional but recommended)
cat > config.json << 'EOF'
{
  "model_type": "ppo",
  "framework": "reinforcement_learning",
  "environment": "usv_navigation",
  "algorithm": "proximal_policy_optimization",
  "library_name": "stable-baselines3",
  "tags": ["reinforcement-learning", "ppo", "navigation", "usv"]
}
EOF

# Step 3: Create a detailed README.md with model description
cat > README.md << 'EOF'
---
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
EOF

# Step 4: Verify your directory structure
echo "📁 Model directory contents:"
ls -la

# Should show:
# ppo_usv.zip
# result1.png, result2.png, result3.png
# config.json
# README.md

# Step 5: Upload using hf-utils CLI
# Make sure you're authenticated first
hf-utils auth whoami

# Upload the model (replace 'username' with your HF username)
hf-utils model upload \
    . \
    username/ppo-usv-model \
    --private \
    --message "Upload PPO USV navigation model with training results" \
    --description "Reinforcement learning model for autonomous USV navigation using PPO algorithm"

# Alternative: Upload as public model
hf-utils model upload \
    . \
    username/ppo-usv-model \
    --message "Upload PPO USV navigation model" \
    --description "RL model for USV navigation with training visualizations"

# Step 6: Verify upload
echo "✅ Model uploaded! Check your model at:"
echo "https://huggingface.co/username/ppo-usv-model"

# =============================================================================
# ADVANCED CLI OPTIONS
# =============================================================================

# Upload with specific tags and metadata
hf-utils model upload \
    ~/my-ppo-usv-model \
    username/ppo-usv-model \
    --message "Initial upload of PPO USV model" \
    --description "PPO-based reinforcement learning model for USV autonomous navigation" \
    --private false

# Upload to a specific branch/revision
hf-utils model upload \
    ~/my-ppo-usv-model \
    username/ppo-usv-model \
    --revision "v1.0" \
    --message "Version 1.0 release"

# Create a pull request instead of direct upload
hf-utils model upload \
    ~/my-ppo-usv-model \
    username/ppo-usv-model \
    --create-pr \
    --message "Propose PPO USV model upload"

# =============================================================================
# TROUBLESHOOTING COMMON ISSUES
# =============================================================================

# If upload fails due to file size, check file sizes
echo "📊 File sizes:"
du -h ~/my-ppo-usv-model/*

# If you need to exclude certain files, create .gitignore
cat > ~/my-ppo-usv-model/.gitignore << 'EOF'
*.log
*.tmp
__pycache__/
.pytest_cache/
EOF

# Check if you have write access to the repository
hf-utils model info username/ppo-usv-model

# List your existing models
hf-utils model list --author username --limit 10