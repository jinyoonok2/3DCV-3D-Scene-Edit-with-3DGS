# Configuration Files

This directory contains all configuration files for the 3DGS Scene Editing pipeline.

## Files

### `environment.yml`
Conda environment specification for the entire pipeline.
- Defines Python version, CUDA toolkit, PyTorch, and all dependencies
- Used by `setup.sh` to create the `3dgs-scene-edit` conda environment
- Single unified environment for all pipeline steps (00-07)

**Usage:**
```bash
# Create environment
conda env create -f configs/environment.yml

# Update existing environment
conda env update -f configs/environment.yml --prune
```

### `garden_config.yaml`
Default configuration for the garden scene brown plant removal project.
- Project settings (name, scene, task)
- Dataset paths and parameters
- Training hyperparameters
- Segmentation settings (GroundingDINO, SAM2)
- ROI extraction parameters
- Inpainting and optimization settings
- Object placement configuration

**Usage:**
```bash
# All pipeline scripts use this by default
python 00_check_dataset.py
python 01_train_gs_initial.py
# ... etc

# Or explicitly specify
python 00_check_dataset.py --config configs/garden_config.yaml
```

## Creating Custom Configs

### For Different Scenes

Copy the garden config and modify for your scene:

```bash
# Copy template
cp configs/garden_config.yaml configs/bicycle_config.yaml

# Edit the new config
nano configs/bicycle_config.yaml
```

Update these key fields:
```yaml
project:
  name: "bicycle_object_removal"
  scene: "bicycle"
  description: "Remove object from bicycle scene"

paths:
  dataset_root: "datasets/360_v2/bicycle"

segmentation:
  text_prompt: "kickstand"  # Your target object
```

Run with custom config:
```bash
python 00_check_dataset.py --config configs/bicycle_config.yaml
python 01_train_gs_initial.py --config configs/bicycle_config.yaml
# ... etc
```

### Using init_project.py

Alternatively, use the initialization script:

```bash
# Creates/updates configs/garden_config.yaml
python init_project.py --scene garden --dataset_root datasets/360_v2/garden

# Create custom config
python init_project.py --scene bicycle \
  --dataset_root datasets/360_v2/bicycle \
  --config configs/bicycle_config.yaml
```

## Directory Structure

```
configs/
├── README.md              # This file
├── environment.yml        # Conda environment specification
├── garden_config.yaml     # Default garden scene config
└── (your custom configs)  # bicycle_config.yaml, etc.
```

## Best Practices

1. **One config per project**: Each scene/experiment should have its own config file
2. **Descriptive names**: Use clear names like `garden_brownplant_removal`, `bicycle_kickstand_removal`
3. **Version control**: Commit config files to track experimental parameters
4. **Don't modify garden_config.yaml**: Create copies for new experiments
5. **Document changes**: Add comments in your config files to explain parameter choices

## Configuration Hierarchy

All pipeline scripts follow this precedence:

1. **Command-line arguments** (highest priority)
2. **Config file values**
3. **Default values** (lowest priority)

Example:
```bash
# Uses factor from configs/garden_config.yaml
python 01_train_gs_initial.py

# Overrides factor to 2
python 01_train_gs_initial.py --factor 2

# Uses custom config
python 01_train_gs_initial.py --config configs/bicycle_config.yaml --factor 2
```

## See Also

- Main README: `../README.md`
- Example configs: `garden_config.yaml`
- Init script: `../init_project.py`
