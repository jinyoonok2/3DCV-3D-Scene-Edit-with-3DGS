# Unified Output Structure Guide

## Overview

The new pipeline uses a **unified, config-driven output structure** that organizes all project files in one place.

**Key principle:** 
- `config.yaml` lives at **ROOT** (you edit this)
- `outputs/` contains **results only** (auto-generated)

## Old Structure (Distributed)
```
outputs/
├── garden/
│   ├── 00_dataset/
│   ├── 01_gs_base/
│   └── round_001/
│       ├── pre_edit/
│       ├── roi.pt
│       └── 05a_holed/
```

**Problems:**
- Scattered across multiple directories
- Hard to track what belongs together
- Difficult to manage multiple experiments
- No clear project configuration

## New Structure (Unified)
```
ROOT/
├── config.yaml                       # ← YOU EDIT THIS (at root!)
├── datasets/
├── models/
└── outputs/
    └── garden_brownplant_removal/    # ← Auto-created from config
        ├── 00_dataset/               # Module outputs
        ├── 00_dataset/               # Dataset validation
        ├── 01_initial_gs/            # Initial 3DGS
        │   └── ckpt_initial.pt
        ├── 02_renders/               # Pre-edit renders
        ├── 03_masks/                 # Segmentation masks
        ├── 04_roi/                   # 3D ROI weights
        │   └── roi.pt
        ├── 05_inpainting/            # Object removal
        │   ├── holed/                # 05a output
        │   ├── inpainted/            # 05b output
        │   └── optimized/            # 05c output (final)
        ├── 06_object_gen/            # (Replacement only)
        ├── 07_merge/                 # (Replacement only)
        ├── 08_evaluation/            # (Replacement only)
        └── logs/                     # Manifests & logs
```

**Benefits:**
- ✅ Config at root (easy to find and edit)
- ✅ Outputs unified in one directory
- ✅ Self-contained projects
- ✅ Easy to share/archive
- ✅ Configuration-driven
- ✅ Clear workflow progression

## Quick Start

### 1. Edit Config (Required)

The `config.yaml` file already exists at the **root directory**. Edit it:

```bash
# Edit the main config file
nano config.yaml

# Key sections to edit:
# - project.name: Your project identifier
# - segmentation.text_prompt: Object to remove/replace
# - inpainting settings: SDXL prompts, thresholds
```

**Or use the helper tool (optional):**
```bash
# Quick setup: creates/overwrites config.yaml
python init_project.py --scene garden --text "brown plant" --task removal

# For a second experiment: create separate config
python init_project.py --scene garden --text "bench" --config experiment2.yaml
```

### 2. Run Pipeline

**Default: Uses config.yaml automatically**
```bash
# These read config.yaml by default
python 00_check_dataset.py
python 01_train_gs_initial.py
python 02_render_training_views.py
python 03_ground_text_to_masks.py
python 04a_lift_masks_to_roi3d.py
python 05a_remove_and_render_holes.py
python 05b_inpaint_holes.py
python 05c_optimize_to_targets.py
```

**Or specify a different config:**
```bash
python 00_check_dataset.py --config experiment2.yaml
python 01_train_gs_initial.py --config experiment2.yaml
# ... etc
```

**Legacy: Command-line args still work**
```bash
# Old way - for backward compatibility
python 05a_remove_and_render_holes.py \
  --ckpt outputs/garden/01_gs_base/ckpt_initial.pt \
  --roi outputs/garden/round_001/roi.pt \
  --data_root datasets/360_v2/garden \
  --roi_thresh 0.3
```

## Config File Structure

### Essential Sections

**Project Info**
```yaml
project:
  name: "garden_brownplant_removal"
  scene: "garden"
  task: "removal"  # or "replacement"
  description: "Remove brown plant from garden"
```

**Paths (Auto-Generated)**
```yaml
paths:
  dataset_root: "datasets/360_v2/garden"
  output_root: "outputs/${project.name}"
  # All other paths are relative to output_root
```

**Segmentation Settings**
```yaml
segmentation:
  text_prompt: "brown plant"
  box_threshold: 0.35
  text_threshold: 0.25
```

**ROI Settings**
```yaml
roi:
  threshold: 0.3          # Lower = more aggressive
  min_views: 3
```

**Inpainting Settings**
```yaml
inpainting:
  removal:
    roi_threshold: 0.3
  sdxl:
    prompt: "natural garden scene"
    negative_prompt: "brown plant, object"
    strength: 0.99
  optimization:
    iterations: 1000
```

## Multiple Projects

You can run multiple experiments by creating separate config files:

```bash
# Method 1: Use init_project helper
python init_project.py --scene garden --text "brown plant" --config exp1.yaml
python init_project.py --scene garden --text "bench" --config exp2.yaml
python init_project.py --scene garden --text "old flower" \
  --task replacement --new-text "red rose" --config exp3.yaml

# Method 2: Manually copy and edit
cp config.yaml exp1.yaml
nano exp1.yaml  # Edit project.name and other settings
```

Directory structure in your root:
```
ROOT/
├── config.yaml              # Default config
├── exp1.yaml                # Experiment 1
├── exp2.yaml                # Experiment 2
├── exp3.yaml                # Experiment 3
└── outputs/
    ├── garden_brown_plant_removal/
    ├── garden_bench_removal/
    └── garden_old_flower_replacement/
```

Run each experiment:
```bash
python 00_check_dataset.py --config exp1.yaml
python 00_check_dataset.py --config exp2.yaml
python 00_check_dataset.py --config exp3.yaml
```

## Migration from Old Structure

If you have existing outputs in the old structure:

```bash
# 1. Create new config
python init_project.py --scene garden --text "brown plant" --config config.yaml

# 2. Copy files to new structure
OLD=outputs/garden
NEW=outputs/garden_brown_plant_removal

cp $OLD/01_gs_base/ckpt_initial.pt $NEW/01_initial_gs/
cp $OLD/round_001/roi.pt $NEW/04_roi/
cp -r $OLD/round_001/pre_edit/* $NEW/02_renders/

# 3. Continue pipeline from where you left off
python 05a_remove_and_render_holes.py --config config.yaml
```

## Advantages

### 1. **Reproducibility**
```bash
# Share project: config + outputs
tar -czf garden_project.tar.gz config.yaml outputs/garden_brownplant_removal/

# On another machine
tar -xzf garden_project.tar.gz
# Config is at root, outputs in outputs/
# All settings preserved in config.yaml
```

### 2. **Experiment Tracking**
```bash
# Each module saves manifest
outputs/garden_brownplant_removal/logs/
├── 01_initial_gs_manifest.json
├── 03_masks_manifest.json
├── 04_roi_manifest.json
└── 05a_removal_manifest.json
```

### 3. **Easy Debugging**
```bash
# All files for one project in one place
ls outputs/garden_brownplant_removal/
# vs hunting through: outputs/garden/, outputs/garden/round_001/, etc.
```

### 4. **Config Management**
```bash
# Version control your configs
git add config.yaml
git commit -m "Tuned ROI threshold to 0.3"

# Try different settings without code changes
sed -i 's/roi_threshold: 0.3/roi_threshold: 0.5/' config.yaml
python 05a_remove_and_render_holes.py --config config.yaml
```

## Best Practices

1. **Keep config.yaml at root** - Easy to find and edit
2. **Version control your configs** - Track in git (but not outputs/)
3. **Use descriptive project names** - scene_object_task format
4. **One config per experiment** - exp1.yaml, exp2.yaml, etc.
5. **Archive with config** - Always include config.yaml when sharing outputs

## FAQ

**Q: Where is config.yaml?**  
A: At the root directory of the project, not inside outputs/

**Q: Can I still use command-line args?**  
A: Yes, but config is recommended for reproducibility

**Q: Can I customize paths?**  
A: Yes, edit `paths:` section in config.yaml

**Q: How do I share my project?**  
A: Share config.yaml + outputs/project_name/ directory together

**Q: Do I need init_project.py?**  
A: No, it's optional. You can manually edit config.yaml

**Q: What about old outputs?**  
A: They still work, but migrate to new structure for better organization
