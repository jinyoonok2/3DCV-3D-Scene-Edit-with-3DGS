# 3D Scene Editing with 3D Gaussian Splatting

Text-guided 3D scene editing: **gsplat + SAM2 + GroundingDINO + SDXL Inpainting**

**Pipeline**: Train 3DGS → Segment objects → Compute 3D ROI → Remove & inpaint → Optimize

---

## Quick Start

```bash
# 1. Setup environment
git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS
cd 3DCV-3D-Scene-Edit-with-3DGS
chmod +x setup.sh && ./setup.sh
source activate.sh

# 2. Initialize config
python init_project.py --scene garden --dataset_root datasets/360_v2/garden

# 3. Run pipeline (all from config.yaml)
python 00_check_dataset.py
python 01_train_gs_initial.py
python 02_render_training_views.py
python 03_ground_text_to_masks.py
python 04a_lift_masks_to_roi3d.py
python 05a_remove_and_render_holes.py
python 05b_inpaint_holes.py
python 05c_optimize_to_targets.py
```

**Override parameters:**
```bash
python 01_train_gs_initial.py --iters 50000
python 03_ground_text_to_masks.py --text "red flowers" --dino_thresh 0.25
```

---

## Configuration System

All modules support dual-mode operation: config file + CLI overrides.

**Initialize:**
```bash
python init_project.py --scene garden --dataset_root datasets/360_v2/garden
```

**Config structure** (`config.yaml`):
```yaml
scene: garden
paths:
  dataset_root: datasets/360_v2/garden
dataset:
  factor: 4          # Image downsample (1/2/4/8)
  seed: 42
  test_every: 8
training:
  iterations: 30000
  sh_degree: 3
segmentation:
  text_prompt: "brown plant"
  dino_threshold: 0.3
roi:
  threshold: 0.7
inpainting:
  prompt: "natural outdoor scene"
  strength: 0.99
```

**Usage modes:**
- Config-only: `python module.py`
- Override: `python module.py --param value`
- Custom config: `python module.py --config exp2.yaml`

---

## Pipeline Modules

### 00. Check Dataset
Validates dataset structure and camera poses.
```bash
python 00_check_dataset.py
```
**Outputs**: `summary.txt`, thumbnails

---

### 01. Train Initial 3DGS
Train baseline 3D Gaussian Splatting model.
```bash
python 01_train_gs_initial.py
python 01_train_gs_initial.py --iters 50000  # Override
```
**Outputs**: `ckpt_*.pt`, renders, `metrics.json`
**Time**: ~30-60 min (RTX 4090)

---

### 02. Render Training Views
Render 3DGS from all training viewpoints.
```bash
python 02_render_training_views.py
```
**Outputs**: Rendered PNGs for each view

---

### 03. Ground Text to Masks
Segment objects using text prompts (GroundingDINO + SAM2).
```bash
python 03_ground_text_to_masks.py
python 03_ground_text_to_masks.py --text "wooden table" --dino_thresh 0.25
```

**Selection modes:**
- `--dino_selection confidence`: Pick highest confidence box (default)
- `--dino_selection largest`: Pick largest bounding box
- `--sam_selection confidence`: Pick highest confidence mask (default)
- `--sam_selection None`: Save all masks as `view_000_0.png`, `view_000_1.png`, etc.

**Spatial filtering:**
```bash
python 03_ground_text_to_masks.py \
    --text "plant" \
    --reference_box "[450,100,800,530]" \
    --reference_overlap_thresh 0.8  # 80% overlap required
```

**Outputs**: 
- `sam_masks/`: Binary masks (PNG + NPY)
- `boxes/`: Detection visualizations
- `overlays/`: Mask overlays
- `coverage.csv`: Statistics per view

---

### 04a. Lift Masks to 3D ROI
Convert 2D masks to per-Gaussian ROI weights.
```bash
python 04a_lift_masks_to_roi3d.py
python 04a_lift_masks_to_roi3d.py --roi_thresh 0.5
```

**Algorithm**: Projects Gaussians to each view, checks mask overlap, computes `roi_weight = masked_views / visible_views`

**Outputs**: `roi.pt` (per-Gaussian weights [0,1])
**Time**: ~2-15 min depending on GPU (RTX 4090: ~3 min, RTX 2060: ~10 min)

---

### 04b. Visualize ROI (Optional)
Render full scene with ROI highlighting.
```bash
python 04b_visualize_roi.py
```
**Outputs**: Projection visualizations, IoU metrics

---

### 05a. Remove and Render Holes
Delete ROI Gaussians and render scene with holes.
```bash
python 05a_remove_and_render_holes.py
python 05a_remove_and_render_holes.py --roi_thresh 0.7
```
**Outputs**: 
- `ckpt_holed.pt`: Checkpoint with ROI removed
- `renders/`: Holed scene images
- `masks/`: Hole masks (white = hole)

---

### 05b. Inpaint Holes
Fill holes using SDXL Inpainting.
```bash
python 05b_inpaint_holes.py
python 05b_inpaint_holes.py --prompt "lush garden" --strength 0.95
```
**Outputs**: `targets/train/`: Inpainted images
**Time**: ~30-60 sec per view (depends on GPU and num_steps)

---

### 05c. Optimize to Targets
Fine-tune 3DGS to match inpainted targets.
```bash
python 05c_optimize_to_targets.py
python 05c_optimize_to_targets.py --iters 1000 --lr_means 1.6e-4
```
**Outputs**: 
- `ckpt_patched.pt`: Final edited checkpoint
- `renders/`: Final renders
- `loss_curve.png`: Training loss plot

**Time**: ~5-15 min for 1000 iterations

---

## Typical Workflow

**1. Full pipeline with defaults:**
```bash
source activate.sh
python init_project.py --scene garden
python 00_check_dataset.py
python 01_train_gs_initial.py
python 02_render_training_views.py
python 03_ground_text_to_masks.py
python 04a_lift_masks_to_roi3d.py
python 05a_remove_and_render_holes.py
python 05b_inpaint_holes.py
python 05c_optimize_to_targets.py
```

**2. Experiment with different prompts:**
```bash
# Edit config.yaml:
segmentation:
  text_prompt: "red flowers"
  
# Or override:
python 03_ground_text_to_masks.py --text "red flowers"
python 05b_inpaint_holes.py --prompt "garden with green grass"
```

**3. Fine-tune parameters:**
```bash
python 01_train_gs_initial.py --iters 50000
python 03_ground_text_to_masks.py --dino_thresh 0.25 --sam_thresh 0.3
python 04a_lift_masks_to_roi3d.py --roi_thresh 0.6
python 05c_optimize_to_targets.py --iters 2000 --lr_means 2e-4
```

---

## File Structure

```
project/
├── config.yaml              # Generated by init_project.py
├── datasets/
│   └── 360_v2/garden/
│       ├── images/          # Training images
│       ├── sparse/          # COLMAP data
│       └── poses_bounds.npy
├── outputs/
│   └── garden/
│       ├── 00_dataset/      # Dataset validation
│       ├── 01_gs_base/      # Initial 3DGS checkpoint
│       ├── 02_renders/      # Training view renders
│       ├── 03_masks/        # Segmentation masks
│       ├── roi/             # 3D ROI weights
│       │   ├── roi.pt
│       │   ├── 05a_holed/   # Scene with holes
│       │   ├── 05b_inpainted/ # Inpainted targets
│       │   └── 05c_optimized/ # Final result
│       └── logs/            # Module manifests
```

---

## Advanced Usage

### Custom Config Files
```bash
python init_project.py --scene experiment1 --config exp1.yaml
python 01_train_gs_initial.py --config exp1.yaml
```

### Multi-round Editing
```bash
# Round 1: Remove plant
python init_project.py --scene garden_round1
# ... run pipeline ...

# Round 2: Further refinement
python init_project.py --scene garden_round2 \
    --ckpt outputs/garden_round1/roi/05c_optimized/ckpt_patched.pt
# ... run pipeline with new prompts ...
```

### Vast.ai Cloud Setup
```bash
# Use vastai-specific setup
./setup_vastai.sh

# Upload your data, then run
python 04a_lift_masks_to_roi3d.py  # Heavy computation
./zip_outputs.sh  # Download results
```

---

## Parameters Reference

### Training (Module 01)
- `iterations`: Training iterations (default: 30000)
- `sh_degree`: Spherical harmonics degree (default: 3)
- `seed`: Random seed (default: 42)

### Segmentation (Module 03)
- `text_prompt`: GroundingDINO text query
- `dino_threshold`: Detection confidence threshold (0.2-0.4)
- `sam_threshold`: SAM2 mask quality threshold (0.3-0.5)
- `dino_selection`: Box selection mode (confidence/largest/None)
- `sam_selection`: Mask selection mode (confidence/None)
- `reference_box`: Spatial filter [x1,y1,x2,y2]

### ROI (Module 04a)
- `roi_thresh`: Threshold for binary ROI (0.5-0.8)
- `min_views`: Minimum views for Gaussian inclusion (default: 3)

### Inpainting (Module 05b)
- `prompt`: SDXL inpainting prompt
- `negative_prompt`: Negative guidance
- `strength`: Denoising strength (0.9-1.0)
- `guidance_scale`: CFG scale (7.0-10.0)
- `num_steps`: Diffusion steps (30-50)

### Optimization (Module 05c)
- `iters`: Optimization iterations (500-2000)
- `lr_means`: Learning rate for positions (1e-4 to 2e-4)
- `lr_scales`: Learning rate for scales (5e-3)
- `densify_from_iter`/`until_iter`: Densification window

---

## Hardware Requirements

**Minimum**: RTX 2060 (6GB VRAM) - slow but functional
**Recommended**: RTX 3090/4090 (24GB VRAM)
**Cloud**: Vast.ai with RTX 4090 (~$0.30-0.50/hr)

**Typical VRAM usage:**
- Training (01): ~6-12 GB
- ROI computation (04a): ~4-8 GB
- Inpainting (05b): ~8-14 GB
- Optimization (05c): ~6-10 GB

---

## Troubleshooting

**Import errors:**
```bash
pip install tyro  # If missing
source activate.sh  # Make sure venv is activated
```

**CUDA out of memory:**
```bash
# Reduce batch size
python 01_train_gs_initial.py --batch_size 4

# Or use smaller image resolution
python init_project.py --scene garden --factor 8  # More downsampling
```

**No masks detected:**
```bash
# Lower thresholds
python 03_ground_text_to_masks.py --dino_thresh 0.2

# Try different prompts
python 03_ground_text_to_masks.py --text "plant . potted plant . planter"
```

**ROI too large/small:**
```bash
# Adjust threshold
python 04a_lift_masks_to_roi3d.py --roi_thresh 0.6  # Higher = smaller ROI
```

---

## Citation

Based on research combining:
- **gsplat**: 3D Gaussian Splatting rendering
- **GroundingDINO**: Open-vocabulary object detection
- **SAM2**: Segment Anything Model 2
- **SDXL**: Stable Diffusion XL Inpainting

---

## License

MIT License - See LICENSE file for details
