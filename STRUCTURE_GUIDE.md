# Unified Output Structure Guide

## Overview

The new pipeline uses a **unified, config-driven output structure** that organizes all project files in one place.

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
outputs/
└── garden_brownplant_removal/        # ← Single project directory
    ├── config.yaml                   # ← Project configuration
    ├── 00_dataset/                   # Dataset validation
    ├── 01_initial_gs/                # Initial 3DGS
    │   └── ckpt_initial.pt
    ├── 02_renders/                   # Pre-edit renders
    ├── 03_masks/                     # Segmentation masks
    ├── 04_roi/                       # 3D ROI weights
    │   └── roi.pt
    ├── 05_inpainting/               # Object removal
    │   ├── holed/                    # 05a output
    │   ├── inpainted/                # 05b output
    │   └── optimized/                # 05c output (final)
    ├── 06_object_gen/               # (Replacement only)
    ├── 07_merge/                    # (Replacement only)
    ├── 08_evaluation/               # (Replacement only)
    └── logs/                         # Manifests & logs
```

**Benefits:**
- ✅ Everything in one place
- ✅ Self-contained projects
- ✅ Easy to share/archive
- ✅ Configuration-driven
- ✅ Clear workflow progression

## Quick Start

### 1. Initialize Project

```bash
# Create new project config
python init_project.py \
  --scene garden \
  --text "brown plant" \
  --task removal
```

This creates:
- `config.yaml` - Project configuration
- `outputs/garden_brown_plant_removal/` - Directory structure

### 2. Review/Edit Config

```bash
# Edit config file
nano config.yaml

# Key sections:
# - project: Name, description
# - paths: Input/output directories
# - segmentation: Text prompt, thresholds
# - inpainting: SDXL settings, optimization
```

### 3. Run Pipeline with Config

**Option A: Use config file (recommended)**
```bash
python 00_check_dataset.py --config config.yaml
python 01_train_gs_initial.py --config config.yaml
python 02_render_training_views.py --config config.yaml
python 03_ground_text_to_masks.py --config config.yaml
python 04a_lift_masks_to_roi3d.py --config config.yaml
python 05a_remove_and_render_holes.py --config config.yaml
python 05b_inpaint_holes.py --config config.yaml
python 05c_optimize_to_targets.py --config config.yaml
```

**Option B: Use command-line args (manual)**
```bash
# Old way - still works but not recommended
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

You can run multiple experiments easily:

```bash
# Experiment 1: Remove brown plant
python init_project.py --scene garden --text "brown plant" \
  --config exp1_brownplant.yaml

# Experiment 2: Remove bench
python init_project.py --scene garden --text "bench" \
  --config exp2_bench.yaml

# Experiment 3: Replace flower
python init_project.py --scene garden --text "old flower" \
  --task replacement --new-text "red rose" \
  --config exp3_flower.yaml
```

Each creates its own isolated directory:
```
outputs/
├── garden_brown_plant_removal/
├── garden_bench_removal/
└── garden_old_flower_replacement/
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
# Share entire project
tar -czf garden_project.tar.gz outputs/garden_brownplant_removal/

# On another machine
tar -xzf garden_project.tar.gz
cd outputs/garden_brownplant_removal
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

1. **Always use init_project.py** to start new work
2. **Version control your config files** (but not outputs/)
3. **Use descriptive project names** (scene_object_task)
4. **Archive completed projects** to save disk space
5. **Keep configs with outputs** for reproducibility

## FAQ

**Q: Can I still use command-line args?**  
A: Yes, but config is recommended for reproducibility.

**Q: Can I customize paths?**  
A: Yes, edit `paths:` section in config.yaml

**Q: How do I share my project?**  
A: Share the entire `outputs/project_name/` directory + config

**Q: What about old outputs?**  
A: They still work, but migrate to new structure for better organization
