# 3D Scene Editing with 3D Gaussian Splatting

Text-guided 3D scene editing: **gsplat + SAM2 + GroundingDINO + TripoSR + SDXL**

**Pipelines:**
- **Object Removal** (00-05c): Train → Segment → Remove → Inpaint → Optimize
- **Object Replacement** (06-09): Generate mesh → Convert to GS → Place at ROI → Optimize

---

## Quick Start

```bash
# Setup
git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS
cd 3DCV-3D-Scene-Edit-with-3DGS
./setup.sh && source activate.sh

# Initialize config
python init_project.py --scene garden --dataset_root datasets/360_v2/garden

# Object Removal Pipeline
python 00_check_dataset.py
python 01_train_gs_initial.py
python 02_render_training_views.py
python 03_ground_text_to_masks.py --text "brown plant"
python 04a_lift_masks_to_roi3d.py
python 05a_remove_and_render_holes.py
python 05b_inpaint_holes.py
python 05c_optimize_to_targets.py

# Object Replacement Pipeline (optional)
python 06_object_generation.py --image flower.png
python 07_mesh_to_gaussians.py --mesh outputs/.../mesh.ply
python 08_place_object_at_roi.py --object_gaussians outputs/.../gaussians.pt --roi outputs/.../roi.pt --scene_gaussians outputs/.../05c_optimized.pt
python 05c_optimize_to_targets.py --init_ckpt outputs/.../merged_with_object.pt  # Reuse 05c for final optimization
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
  threshold: 0.01    # ROI weight threshold (lower = more permissive)
inpainting:
  removal:
    roi_threshold: 0.01
  optimization:
    iterations: 1000
    learning_rates:
      means: 0.00016
      scales: 0.005
      quats: 0.001
      opacities: 0.05
      sh0: 0.0025
```

**Note**: SDXL inpainting parameters are NOT in config - pass via CLI to `05b_inpaint_holes_sdxl.py`

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
Convert 2D masks to per-Gaussian ROI weights using render-based voting with occlusion filtering.
```bash
python 04a_lift_masks_to_roi3d.py
python 04a_lift_masks_to_roi3d.py --roi_thresh 0.5
```

**Algorithm**: 
1. For each training view with a mask:
   - Render depth map to get visible surface
   - Project all Gaussians to 2D pixel coordinates
   - Filter to only in-bounds Gaussians (reduces computation)
   - Perform occlusion test: depth_diff < 5% of rendered depth (relative tolerance)
   - Accumulate mask values only for visible (non-occluded) Gaussians
2. Normalize: `roi_weight = sum(mask_values) / visible_views`

**Occlusion Handling**: Uses relative depth tolerance (5% of scene depth) to ensure only Gaussians on the visible surface are selected, excluding background Gaussians behind the target object.

**Outputs**: `roi.pt` (per-Gaussian weights [0,1]), `roi_binary.pt`, `metrics.json`
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
Delete ROI Gaussians and generate hole masks using multi-pass depth comparison.
```bash
python 05a_remove_and_render_holes.py
python 05a_remove_and_render_holes.py --roi_thresh 0.7
```

**Multi-Pass Depth Comparison** (Gaussian Editor Method):
This module uses a robust 3-pass rendering approach to identify holes:
1. **Pass 1**: Render full scene with depth mode → `depth_full` (closest object at each pixel)
2. **Pass 2**: Render ROI-only scene with depth mode → `depth_roi`, `alpha_roi` (where ROI exists)
3. **Pass 3**: Render holed scene (ROI deleted) → visualization

**Mask Generation**: `mask = (alpha_roi > 0.1) & (|depth_full - depth_roi| < 0.01)`
- Identifies pixels where ROI was the **frontmost visible object**
- Robust to "artichoke problem" (incomplete ROI selection with inner core remaining)
- Only marks pixels that need inpainting (not occluded regions)

**Why This Works**: Instead of comparing opacity before/after deletion (fails if inner core remains), we ask "Is the ROI the closest visible object?" This approach handles volumetric 3D Gaussian clouds correctly.

**Outputs**: 
- `holed/train/`: Holed scene renders (pot removed)
- `masks/train/`: Hole masks for inpainting (white = needs inpainting)
- Per-view depth maps and comparison visualizations

---

### 05b. Inpaint Holes
Fill holes using LaMa (default) or SDXL Inpainting.

**LaMa Inpainting** (Default - Object Removal):
- Clean texture continuation without hallucination
- No prompts needed, fast (~0.5 sec/image)
- Best for object removal tasks

```bash
# Config-based (uses defaults from config.yaml)
python 05b_inpaint_holes.py

# With parameters (mask processing options)
python 05b_inpaint_holes.py --mask_blur 8 --mask_dilate 0 --mask_erode 0
```

**SDXL Inpainting** (Ablation Study - Creative Inpainting):
- Text-guided inpainting with creative generation
- Good for adding new content
- Slower (~30-60 sec/image), may hallucinate objects
- **All parameters via CLI** (not in config)

```bash
python 05b_inpaint_holes_sdxl.py \
    --model sdxl \
    --prompt "natural outdoor garden scene with grass and plants" \
    --negative_prompt "brown plant, dead plant, object, artifact, blur" \
    --strength 0.99 \
    --guidance_scale 7.5 \
    --num_steps 50
```

**Model Comparison**:
| Model | Speed | Quality | Hallucination | Use Case |
|-------|-------|---------|---------------|----------|
| LaMa | ⚡ Fast (~0.5s) | Clean texture | ✅ None | Object removal |
| SDXL | 🐌 Slow (~40s) | Creative | ❌ Possible | Content generation |

**Outputs**: `targets/train/`: Inpainted images
**Time**: 
- LaMa: ~1-2 min for 161 images
- SDXL: ~90-120 min for 161 images

---

### 05c. Optimize to Targets
Fine-tune 3DGS to match inpainted targets using L1 loss optimization.

**Algorithm**:
1. Load holed checkpoint (from 05a) and inpainted targets (from 05b)
2. Optimize all Gaussian parameters (means, scales, rotations, opacities, SH colors)
3. Use same learning rates as initial training for stability
4. **No densification** (6.4M Gaussians sufficient, avoids CUDA errors)
5. Render final views after optimization

```bash
# Config-based (recommended)
python 05c_optimize_to_targets.py

# Override parameters
python 05c_optimize_to_targets.py --iters 1000 --lr_means 1.6e-4 --lr_sh0 2.5e-3
```

**Implementation Details**:
- Separate Adam optimizers for each parameter type (matches `01_train_gs_initial.py`)
- Learning rates from config: `means=1.6e-4`, `scales=5e-3`, `quats=1e-3`, `opacities=5e-2`, `sh0=2.5e-3`, `shN=sh0/20`
- Densification disabled (`use_strategy=False`) for simple parameter optimization
- Renders all 161 training views at end for verification

**Outputs**: 
- `ckpt_patched.pt`: Final edited checkpoint (6.4M Gaussians)
- `renders/train/`: Final rendered views (00000.png - 00160.png)
- `loss_curve.png`: Training loss plot
- `manifest.json`: Optimization metadata

**Time**: ~2-3 min for 1000 iterations (RTX 4090)

---

## Typical Workflow

**1. Full pipeline with defaults (LaMa):**
```bash
source activate.sh
python init_project.py --scene garden
python 00_check_dataset.py
python 01_train_gs_initial.py
python 02_render_training_views.py
python 03_ground_text_to_masks.py
python 04a_lift_masks_to_roi3d.py
python 05a_remove_and_render_holes.py
python 05b_inpaint_holes.py          # LaMa (default)
python 05c_optimize_to_targets.py
```

**2. SDXL inpainting (alternative):**
```bash
python 05b_inpaint_holes_sdxl.py --model sdxl --prompt "garden scene" --strength 0.99
python 05c_optimize_to_targets.py
```

**3. Object Replacement (Remove old → Add new):**
```bash
# Remove object (Modules 00-05c)
python 00_check_dataset.py
python 01_train_gs_initial.py
python 02_render_training_views.py
python 03_ground_text_to_masks.py --text "brown plant"
python 04a_lift_masks_to_roi3d.py
python 05a_remove_and_render_holes.py
python 05b_inpaint_holes.py
python 05c_optimize_to_targets.py

# Generate and place new object (Modules 06-09)
python 06_object_generation.py --image flower.png
python 07_mesh_to_gaussians.py --mesh outputs/garden/06_object_gen/mesh.ply --num_points 10000
python 08_place_object_at_roi.py \
  --object_gaussians outputs/garden/06_object_gen/gaussians.pt \
  --roi outputs/garden/round_001/roi.pt \
  --scene_gaussians outputs/garden/round_001/05c_optimized.pt \
  --placement bottom --scale_factor 0.8
python 05c_optimize_to_targets.py --init_ckpt outputs/garden/round_001/merged_with_object.pt
```

---

## Module Details

### 00-05c: Object Removal
See original sections above for details.

---

### 06. Object Generation
Generate 3D mesh from image using TripoSR.
```bash
python 06_object_generation.py --image flower.png
```
**Outputs**: `mesh.obj`, `mesh.ply`, `input_image.png`, `preview.png`

---

### 07. Mesh to Gaussians
Convert mesh to Gaussian splats by sampling surface points.
```bash
python 07_mesh_to_gaussians.py --mesh mesh.ply --num_points 10000
```
**Outputs**: `gaussians.pt` (Gaussian parameters)

---

### 08. Place Object at ROI
Transform and merge generated object with scene at ROI location.
```bash
python 08_place_object_at_roi.py \
  --object_gaussians gaussians.pt \
  --roi roi.pt \
  --scene_gaussians scene.pt \
  --placement bottom  # or center/top
```
**Options**:
- `--placement bottom`: Sits on surface (default)
- `--scale_factor 0.8`: Size relative to ROI (default 80%)
- `--z_offset 0.05`: Adjust height
- `--rotation_degrees 45`: Rotate object

**Outputs**: `merged_with_object.pt`

---

### 09. Final Optimization
Use Module 05c to optimize merged scene:
```bash
python 05c_optimize_to_targets.py --init_ckpt merged_with_object.pt
```

---

## Typical Workflows

**Remove only:**
```bash
00 → 01 → 02 → 03 → 04a → 05a → 05b → 05c
```

**Replace:**
```bash
00 → 01 → 02 → 03 → 04a → 05a → 05b → 05c → 06 → 07 → 08 → 05c
```

---

## File Structure

```
project/
├── config.yaml
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

## Technical Improvements

### Module 04a: Occlusion-Aware ROI Computation
**Problem**: Original approach included background Gaussians behind target objects in the ROI.

**Solution**: Render-based voting with depth filtering:
- Projects all Gaussians to 2D for each view
- Renders depth map to identify visible surface
- Filters Gaussians using relative depth tolerance: `depth_diff < 5% * scene_depth`
- Only visible (non-occluded) Gaussians receive mask votes
- Results in precise ROI selection excluding background

**Benefits**:
- ✅ Accurate ROI boundaries
- ✅ No background leakage
- ✅ Stable across different scene scales (relative threshold)

### Module 05a: Multi-Pass Depth Comparison for Hole Masks
**Problem**: Original alpha-difference method failed with volumetric Gaussian clouds ("artichoke problem"):
- Deleting outer shell left inner core
- Inner core maintained opacity → `alpha_diff ≈ 0` → empty masks

**Solution**: Three-pass rendering with depth comparison (Gaussian Editor approach):
1. Render full scene → `depth_full` (what's closest?)
2. Render ROI-only → `depth_roi`, `alpha_roi` (where is ROI?)
3. Compare: `mask = (alpha_roi > 0.1) & (|depth_full - depth_roi| < 0.01)`

**Key Insight**: Instead of asking "Did opacity change?" (fails if inner core remains), we ask "Was the ROI the frontmost visible object?" (works regardless of deletion completeness)

**Benefits**:
- ✅ Robust to incomplete ROI selection
- ✅ Correctly identifies visible surfaces only
- ✅ Handles occlusions automatically
- ✅ Precise hole masks for inpainting

**Comparison**:
| Approach | Robustness | Occlusion Handling | Inner Core Problem |
|----------|------------|--------------------|--------------------|
| Alpha difference (old) | ❌ Fragile | ❌ No | ❌ Fails |
| Depth comparison (new) | ✅ Robust | ✅ Yes | ✅ Works |

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
- `roi_thresh`: Threshold for binary ROI (0.01-0.8, default: 0.01 for maximum inclusiveness)
- `min_views`: Minimum views for Gaussian inclusion (default: 3)
- `depth_threshold`: Relative depth tolerance for occlusion filtering (default: 0.05 = 5%)

### Hole Masking (Module 05a)
- `roi_thresh`: ROI threshold for deletion (should match 04a setting)
- `depth_threshold`: Depth comparison tolerance in scene units (default: 0.01m)
- `alpha_threshold`: Minimum alpha for ROI visibility (default: 0.1)

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
# Adjust threshold (higher = smaller ROI, more selective)
python 04a_lift_masks_to_roi3d.py --roi_thresh 0.6  # For smaller, precise ROI
python 04a_lift_masks_to_roi3d.py --roi_thresh 0.01 # For maximum inclusiveness
```

**Empty or incorrect hole masks (Module 05a):**
- ✅ **Solved**: Now uses multi-pass depth comparison instead of alpha difference
- If masks still look wrong:
  ```bash
  # Adjust depth comparison threshold
  python 05a_remove_and_render_holes.py --depth_threshold 0.02  # More permissive
  
  # Or adjust alpha threshold for ROI visibility
  python 05a_remove_and_render_holes.py --alpha_threshold 0.05  # Lower = more sensitive
  ```

**"Artichoke problem" (inner core remaining after deletion):**
- ✅ **Solved**: Module 05a's depth comparison method is robust to incomplete ROI selection
- The new approach asks "Was the ROI frontmost?" instead of "Did opacity change?"
- Works correctly even if some inner Gaussians remain after deletion

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
