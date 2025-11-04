# 3D Scene Editing with 3D Gaussian Splatting# 3D Scene Editing with 3D Gaussian Splatting



**Text-guided 3D scene editing using GaussianEditor's 3D Inpainting pipeline.**Text-guided 3D scene editing using **gsplat + Stable Diffusion Inpainting + GroundingDINO + SAM2**.  

Each module is a **single Python file** with clear inputs/outputs and verification steps.

Combines **gsplat** (3D Gaussian Splatting) + **SDXL Inpainting** + **GroundingDINO** + **SAM2** for semantic object removal and replacement in 3D scenes.

**Example task**: Edit the brown plant in the Mip-NeRF 360 garden scene.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

---

## Quick Start

## Overview

### Local Setup (WSL/Linux)

This project implements a **config-driven 3D scene editing pipeline** that allows you to:```bash

- 🎯 **Remove objects** from 3D scenes using text prompts# Clone and setup

- 🔄 **Replace objects** with AI-generated alternativesgit clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS

- 🎨 **Edit scenes** semantically using natural languagecd 3DCV-3D-Scene-Edit-with-3DGS

- 📦 **Organize projects** in a clean, unified structurechmod +x setup.sh

./setup.sh

**Key Features:**

- Single `config.yaml` at root for all settings# Activate environment

- Unified output structure (`outputs/project_name/`)source activate.sh

- Modular pipeline (8 independent Python scripts)

- Support for local machines and cloud GPUs (Vast.ai)# Run modules

- Based on GaussianEditor's 3D Inpainting methodologypython 00_check_dataset.py --data_root datasets/360_v2/garden --factor 4

python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000

---# ... continue with other modules

```

## Quick Start

### Vast.ai Setup (Cloud GPU)

### 1. Setup```bash

# Clone and setup (auto-downloads dataset and models)

**Local/WSL:**git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS

```bashcd 3DCV-3D-Scene-Edit-with-3DGS

git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGSchmod +x setup_vastai.sh

cd 3DCV-3D-Scene-Edit-with-3DGS./setup_vastai.sh

chmod +x setup.sh

./setup.sh# If tyro is missing, install manually

source activate.shpip install tyro

```

# Activate and run

**Vast.ai (Cloud GPU):**source activate.sh

```bash# Upload your checkpoint and masks, then run Module 04

git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS```

cd 3DCV-3D-Scene-Edit-with-3DGS

chmod +x setup_vastai.sh**Note**: RTX 4090 recommended for Vast.ai. RTX 5090 not yet supported by PyTorch stable (use RTX 4090, 3090, or A100).

./setup_vastai.sh

source activate.sh---

```

## Setup Scripts

### 2. Configure Your Project

- **`setup.sh`**: Universal setup for local/WSL/cloud (auto-detects environment)

Edit `config.yaml` at the root:- **`setup_vastai.sh`**: Optimized for Vast.ai with CUDA 12.6, automatic dataset download

```yaml- **`activate.sh`**: Activate virtual environment

project:- **`zip_outputs.sh`**: Package outputs folder for download

  name: "garden_brownplant_removal"- **`unzip_outputs.sh`**: Extract outputs.zip

  scene: "garden"- **`download_models.sh`**: Download GroundingDINO and SAM2 weights

  task: "removal"

> No code here—only responsibilities, I/O, and verification steps.  

segmentation:> You can run each file as a standalone script with simple CLI args.

  text_prompt: "brown plant"

---

inpainting:

  sdxl:## Global Conventions

    prompt: "natural outdoor garden scene"

    negative_prompt: "brown plant, dead plant"- **Dataset:** `datasets/360_v2/garden/` (images and poses prepared for Mip-NeRF 360 “garden”)  

```- **Experiment root:** `outputs/garden_pot_to_black_cup/`  

- **View naming:** `view_000.png`, `view_001.png`, … (match your dataset’s ordering)  

Or use the helper tool:- **JSON manifest:** every module that produces outputs creates a `manifest.json` with parameters, seeds, and version strings used  

```bash- **Round naming:** `round_001/`, `round_002/`, … (even if you start with one round)

python init_project.py --scene garden --text "brown plant" --task removal

```---



### 3. Run the Pipeline## 00_check_dataset.py — Validate Inputs



```bash**Goal**: Confirm dataset paths, list images/poses, and preview a few frames to ensure the **plant pot** is visible.

# Object Removal Pipeline (Modules 00-05)

python 00_check_dataset.py --data_root datasets/360_v2/garden --factor 4**Inputs**

python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000- `--data_root datasets/360_v2/garden/`

python 02_render_training_views.py --ckpt outputs/garden/01_gs_base/ckpt_initial.pt

python 03_ground_text_to_masks.py --images_root outputs/garden/round_001/pre_edit --text "brown plant"**Saves (in `outputs/garden_pot_to_black_cup/00_dataset/`)**

python 04a_lift_masks_to_roi3d.py --ckpt outputs/garden/01_gs_base/ckpt_initial.pt --masks_root outputs/garden/round_001/masks/sam_masks- `summary.txt`: counts of images, intrinsics/extrinsics stats, min/max focal, etc.

python 05a_remove_and_render_holes.py --ckpt outputs/garden/01_gs_base/ckpt_initial.pt --roi outputs/garden/round_001/roi.pt- `thumbs/` (e.g., 6 images sampling the sequence): `thumb_000.png`, …

python 05b_inpaint_holes.py --holed_dir outputs/garden/round_001/05a_holed --prompt "natural garden scene"- `manifest.json`: paths, data_root, timestamp

python 05c_optimize_to_targets.py --ckpt outputs/garden/round_001/05a_holed/ckpt_holed.pt --targets_dir outputs/garden/round_001/05b_inpainted

```**Verify**

- `summary.txt` matches expected counts (images > 50 for mip360 garden)

*Note: Config system integration coming soon - modules will read from `config.yaml` by default*- Thumbnails show the table and **plant pot**

- Any mismatch or missing poses is surfaced here before training

---

---

## Project Structure

## 01_train_gs_initial.py — Train/Load Initial 3DGS Scene (gsplat)

### **Root Directory** (what you edit)

```**Goal**: Produce a baseline **3D Gaussian** scene and sanity-check rendering quality.

3DCV-3D-Scene-Edit-with-3DGS/

├── config.yaml                    # ← EDIT THIS (main configuration)**Inputs**

├── init_project.py                # Optional helper to create configs- `--data_root datasets/360_v2/garden/`

├── datasets/                      # Input datasets- `--iters 30000` (example), `--seed 42`

├── models/                        # Model weights (GroundingDINO, SAM2)

├── 00_check_dataset.py            # Module scripts**Saves (in `outputs/garden_pot_to_black_cup/01_gs_base/`)**

├── 01_train_gs_initial.py- `ckpt_initial.pt` (gsplat model checkpoint)

├── 02_render_training_views.py- `renders/` pretrain renders for all training views (`view_000.png`, …)

├── 03_ground_text_to_masks.py- `metrics.json`: PSNR/SSIM over training views

├── 04a_lift_masks_to_roi3d.py- `manifest.json`: config, training iters, seed, gsplat version

├── 04b_visualize_roi.py

├── 05a_remove_and_render_holes.py**Verify**

├── 05b_inpaint_holes.py- Renders are sharp and align with photos

├── 05c_optimize_to_targets.py- Metrics stabilize or improve over time

└── outputs/                       # ← Auto-generated results- Visually confirm the **plant pot** reconstructs reasonably

    └── garden_brownplant_removal/ # One directory per project

```---



### **Output Structure** (auto-created from config)## 02_render_training_views.py — Pre-Edit Renders for a Round

```

outputs/garden_brownplant_removal/**Goal**: Render the **current** 3D scene (from checkpoint) for the chosen views; these are the images sent to diffusion.

├── 00_dataset/                    # Dataset validation

├── 01_initial_gs/                 # Initial 3DGS training**Inputs**

│   └── ckpt_initial.pt- `--ckpt outputs/garden_pot_to_black_cup/01_gs_base/ckpt_initial.pt`

├── 02_renders/                    # Pre-edit renders- `--views all` (or a subset list), `--seed 42`

│   └── train/*.png

├── 03_masks/                      # GroundingDINO + SAM2 masks**Saves (in `outputs/garden_pot_to_black_cup/round_001/pre_edit/`)**

│   ├── boxes/- `pre_edit_view_000.png`, `pre_edit_view_001.png`, …

│   ├── sam_masks/- `manifest.json`: which checkpoint, which poses, resolution, seed

│   └── overlays/

├── 04_roi/                        # 3D ROI weights**Verify**

│   └── roi.pt- Renders match 01’s visual quality (no regressions)

├── 05_inpainting/                 # Object removal & inpainting- Dimensions and naming match later modules’ expectations

│   ├── holed/                     # 05a: Object removed

│   │   ├── ckpt_holed.pt---

│   │   ├── renders/train/*.png

│   │   └── masks/train/*.png## 03_ground_text_to_masks.py — Text → Boxes → Masks (GroundingDINO + SAM2)

│   ├── inpainted/                 # 05b: SDXL inpainted

│   │   └── targets/train/*.png**Goal**: From **text**, produce **2D masks** isolating objects on each rendered view using two-stage selection.

│   └── optimized/                 # 05c: Final 3DGS

│       └── ckpt_final.pt**Inputs**

└── logs/                          # Manifests & metadata- `--images_root outputs/garden/round_001/pre_edit/train/`

    └── *_manifest.json- `--text "brown plant"` (text prompt for GroundingDINO)

```- `--dino_thresh 0.3` (detection threshold)

- `--sam_thresh 0.3` (SAM2 confidence threshold)

**Benefits:**- `--dino_selection confidence` (box selection: confidence/largest/None)

- ✅ Config at root (easy to find and version control)- `--sam_selection confidence` (mask selection: confidence/None)

- ✅ Outputs unified in one directory (easy to share/archive)- `--reference_box "[450,100,800,530]"` (optional spatial filter)

- ✅ Self-contained projects (no scattered files)- `--reference_overlap_thresh 0.8` (overlap threshold for spatial filter)

- ✅ Clear workflow progression (00 → 01 → 02 → ...)

**Two-Stage Selection System**:

---1. **DINO Stage**: Select boxes based on detection confidence or size

   - `confidence`: Highest DINO score (default)

## Pipeline Overview   - `largest`: Biggest bounding box

   - `None`: Keep all detections

### **Workflow A: Object Removal** (Modules 00-05)2. **SAM2 Stage**: Select masks from SAM2 output

   - `confidence`: Highest SAM2 score (default)

```   - `None`: Save all masks (creates `_0.png`, `_1.png`, etc.)

[00] Validate Dataset

  ↓ datasets/360_v2/garden/**Spatial Filtering** (optional):

[01] Train Initial 3DGS (30K iters)- `--reference_box`: Define region of interest [x1,y1,x2,y2]

  ↓ ckpt_initial.pt- `--reference_overlap_thresh`: Fraction of detection box that must overlap (0.8 = 80%)

[02] Render Training Views- Boxes not meeting threshold are rejected before SAM2

  ↓ pre_edit/*.png

[03] Text → Masks (GroundingDINO + SAM2)**Saves (in `outputs/garden/round_001/masks_brown_plant/`)**

  ↓ sam_masks/*.png- `boxes/` (annotated detections: blue=reference, green=selected, red=rejected)

[04a] 2D Masks → 3D ROI Weights- `sam_masks/` (binary masks: PNG + NPY, multiple files if sam_selection=None)

  ↓ roi.pt- `overlays/` (mask overlays showing all candidates with selection highlighting)

[05a] Remove ROI Gaussians & Render Holes- `coverage.csv` (includes: num_saved_masks, mask_area, dino_score, sam_score)

  ↓ ckpt_holed.pt, holed renders + masks- `manifest.json` (records dino_selection, sam_selection, sam_thresh, thresholds)

[05b] SDXL Inpainting (Fill Holes)

  ↓ inpainted targets**Verify**

[05c] Optimize 3DGS to Match Targets- Box visualizations show correct selection (green) vs rejection (red)

  ↓ ckpt_final.pt (object removed!)- Overlays show the target region highlighted

```- Coverage CSV shows reasonable mask counts and sizes

- Manifest records all parameters for reproducibility

### **Workflow B: Object Replacement** (Modules 00-08)

*Coming soon - extends Workflow A with:***Example Commands**:

- [06] Generate New Object (TripoSR image-to-3D)```bash

- [07] Merge Scenes (Depth-based alignment)# Select highest confidence box and mask

- [08] Evaluate Result (CLIP, LPIPS, SSIM)python 03_ground_text_to_masks.py \

    --images_root outputs/garden/round_001/pre_edit/train \

---    --text "brown plant" \

    --dino_thresh 0.3 --sam_thresh 0.3 \

## Module Documentation    --dino_selection confidence --sam_selection confidence \

    --output_dir outputs/garden/round_001/masks_brown_plant

### **Module 00: Validate Dataset**

```bash# Save all masks from largest box

python 00_check_dataset.py \python 03_ground_text_to_masks.py \

  --data_root datasets/360_v2/garden \    --images_root outputs/garden/round_001/pre_edit/train \

  --factor 4    --text "wooden pot" \

```    --dino_selection largest --sam_selection None \

**Purpose:** Verify dataset structure, count images, check poses      --reference_box "[500,150,800,500]" --reference_overlap_thresh 0.9 \

**Output:** `00_dataset/summary.txt`, thumbnails      --output_dir outputs/garden/round_001/masks_pot

**Time:** <1 minute```



------



### **Module 01: Train Initial 3DGS**## 04_lift_masks_to_roi3d.py — 2D Masks → 3D ROI (Per-Gaussian Weights)

```bash

python 01_train_gs_initial.py \**Goal**: Convert per-view 2D masks into **per-Gaussian ROI weights** `roi ∈ [0,1]` by projecting masks to 3D and aggregating visibility.

  --data_root datasets/360_v2/garden \

  --result_dir outputs/garden/01_gs_base \**Algorithm**:

  --iters 300001. Load 3D Gaussians from checkpoint (handles nested 'splats' key format)

```2. For each training view with a mask:

**Purpose:** Train 3D Gaussian Splatting scene representation     - Render Gaussians to get per-pixel Gaussian IDs

**Output:** `01_initial_gs/ckpt_initial.pt`, renders, metrics     - For each Gaussian visible in view:

**Time:** ~3-4 hours (RTX 2060), ~30-45 min (RTX 4090)       - Check if it projects inside the mask

**Key Settings:** Uses gsplat with automatic densification     - Track: `num_masked_views` and `num_visible_views`

3. Compute per-Gaussian weight: `weight = num_masked_views / num_visible_views`

---4. Threshold at `roi_thresh` to get binary ROI



### **Module 02: Render Training Views****Inputs**

```bash- `--ckpt outputs/garden/01_gs_base/ckpt_initial.pt` (nested format supported)

python 02_render_training_views.py \- `--masks_root outputs/garden/round_001/masks_brown_plant/sam_masks/`

  --ckpt outputs/garden/01_gs_base/ckpt_initial.pt \- `--data_root datasets/360_v2/garden` (for camera poses and images)

  --data_root datasets/360_v2/garden \- `--roi_thresh 0.5` (threshold for binary ROI)

  --output_dir outputs/garden/round_001/pre_edit- `--sh_degree 3` (must match training)

```

**Purpose:** Render scene from trained checkpoint  **Checkpoint Format**: Automatically handles both formats:

**Output:** `02_renders/train/*.png` (161 views for garden)  - Flat: `ckpt['means']`, `ckpt['scales']`, etc.

**Time:** ~5-10 minutes- Nested: `ckpt['splats']['means']`, `ckpt['splats']['scales']`, etc.



---**Saves (in `outputs/garden/round_001/`)**

- `roi.pt` (per-Gaussian weights in [0,1], shape: [N_gaussians])

### **Module 03: Text-to-Masks (GroundingDINO + SAM2)**- Statistics printed: mean weight, std, min, max, num above threshold

```bash

python 03_ground_text_to_masks.py \**Performance**:

  --images_root outputs/garden/round_001/pre_edit \- RTX 2060: ~5-15 minutes for 161 views × 245K Gaussians

  --text "brown plant" \- RTX 4090: ~2-5 minutes

  --output_dir outputs/garden/round_001/masks_brownplant- RTX 3090: ~3-7 minutes

```

**Purpose:** Segment target object using text prompt  **Verify**

**Output:** `03_masks/sam_masks/*.png`, boxes, overlays  - ROI statistics show reasonable distribution (not all 0 or all 1)

**Time:** ~10-15 minutes  - Number of Gaussians above threshold is reasonable (not empty, not everything)

- Check console output for per-view processing progress

**Key Features:**

- Two-stage selection (DINO → SAM2)**Example Command**:

- Spatial filtering with reference boxes```bash

- Configurable confidence thresholds# Local (may be slow on RTX 2060)

python 04_lift_masks_to_roi3d.py \

**Selection Options:**    --ckpt outputs/garden/01_gs_base/ckpt_initial.pt \

- `--dino_selection confidence|largest|None`    --masks_root outputs/garden/round_001/masks_brown_plant/sam_masks \

- `--sam_selection confidence|None`    --data_root datasets/360_v2/garden \

- `--reference_box "[x1,y1,x2,y2]"` (optional spatial filter)    --roi_thresh 0.5 \

    --sh_degree 3 \

---    --output_dir outputs/garden/round_001



### **Module 04a: Lift 2D Masks to 3D ROI**# Vast.ai (recommended for speed)

```bash# 1. Upload checkpoint and masks

python 04a_lift_masks_to_roi3d.py \# 2. Run same command on faster GPU

  --ckpt outputs/garden/01_gs_base/ckpt_initial.pt \# 3. Download roi.pt back to local machine

  --masks_root outputs/garden/round_001/masks_brownplant/sam_masks \```

  --data_root datasets/360_v2/garden \

  --output_dir outputs/garden/round_001 \**Workflow for Vast.ai**:

  --roi_thresh 0.5```bash

```# On Vast.ai: Package outputs

**Purpose:** Convert 2D masks to per-Gaussian 3D weights  ./zip_outputs.sh

**Output:** `04_roi/roi.pt` (weights ∈ [0,1] for each Gaussian)  

**Time:** ~5-15 minutes (RTX 2060), ~2-5 min (RTX 4090)  # On local machine: Download and extract

scp root@<vastai-ip>:/workspace/3DCV-3D-Scene-Edit-with-3DGS/outputs.zip .

**Algorithm:**./unzip_outputs.sh

1. Load 3D Gaussians from checkpoint```

2. For each training view with a mask:

   - Render Gaussians to get per-pixel Gaussian IDs---

   - Track which Gaussians project inside masks

3. Compute per-Gaussian weight: `num_masked_views / num_visible_views`## 05_remove_and_patch.py — Remove Object & Patch Hole (Object-Inpainting Workflow)



---Goal: Remove the target object (identified by `roi.pt`) and use 2D inpainting models to generate realistic patched targets that guide a short 3DGS patching optimization. This replaces the IP2P-first workflow when the goal is object replacement/inpainting rather than stylistic edits.



### **Module 05a: Remove ROI and Render Holes**Inputs

```bash

python 05a_remove_and_render_holes.py \-- `--ckpt_in outputs/garden_pot_to_black_cup/01_gs_base/ckpt_initial.pt`

  --ckpt outputs/garden/01_gs_base/ckpt_initial.pt \

  --roi outputs/garden/round_001/roi.pt \-- `--roi outputs/garden_pot_to_black_cup/round_001/roi/roi.pt`

  --data_root datasets/360_v2/garden \

  --roi_thresh 0.3-- `--data_root datasets/360_v2/garden` (for camera poses)

```

**Purpose:** Delete ROI Gaussians and render scene with holes  -- `--iters 1000` (short optimization for hole-patching)

**Output:**

- `05_inpainting/holed/ckpt_holed.pt` (reduced 3DGS)Process:

- `renders/train/*.png` (scene with holes)

- `masks/train/*.png` (where to inpaint)1. Load `ckpt_in` and `roi.pt`.

2. Delete or deactivate all Gaussians whose ROI weight is above `--roi_thresh` (creates a scene with a hole where the object lived).

**Time:** ~5-10 minutes  3. Render multiple views of this "holed" scene.

**Key:** Lower `roi_thresh` = more aggressive removal4. Patch targets: for each rendered view, run a 2D Inpainting Diffusion Model (e.g., SDXL-Inpainting) to fill the hole plausibly. Save these patched images as the new supervision targets.

5. Optimize: run a short, ROI-gated optimization loop to train the holed 3DGS model to match the patched 2D targets (only loss inside ROI), producing a patched scene checkpoint.

**ROI Threshold Guide:**

- `0.7`: Conservative (2-5% deleted) - small holesSaves (in `outputs/garden_pot_to_black_cup/round_001/05_patched_scene/`)

- `0.5`: Moderate (4-7% deleted) - balanced

- `0.3`: Aggressive (7-12% deleted) - large holes- `ckpt_patched.pt`: The 3DGS checkpoint with the plant pot removed and the hole patched via short optimization.

- `0.1`: Very aggressive (11-15% deleted) - may over-delete

- `patched_targets/`: The per-view 2D inpainted images used as supervision.

---

- `manifest.json`: parameters, number of iters, diffusion model used for patching.

### **Module 05b: SDXL Inpainting**

```bashVerify

python 05b_inpaint_holes.py \

  --holed_dir outputs/garden/round_001/05a_holed \- Renders from `ckpt_patched.pt` show the table with the pot removed and the surrounding area plausibly filled.

  --prompt "natural outdoor garden scene with grass and plants" \

  --negative_prompt "brown plant, dead plant, object" \- `patched_targets/` images look realistic and coherent with the scene lighting.

  --strength 0.99

```---

**Purpose:** Fill holes using SDXL Inpainting diffusion model  

**Output:** `05_inpainting/inpainted/targets/train/*.png`  ## 06_generate_new_object.py — Generate 3D Object from Text (Object Incorporation)

**Time:** ~15-25 minutes (161 views on RTX 4090), ~45-60 min (RTX 2060)  

**Model:** `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (~6GB, auto-downloads once)Goal: Create a new, separate 3DGS model for the desired replacement object ("empty black cup") using a 2D inpainting → image-to-3D lift pipeline.



**Parameters:**Inputs

- `--prompt`: Describe desired scene (without removed object)

- `--negative_prompt`: What to avoid (include removed object)-- `--instruction "an empty black cup"` (text prompt for the new object)

- `--strength`: 0.99 = maximum change, 0.5 = minimal change

- `--guidance_scale`: 7.5 = balanced, higher = stronger prompt adherence-- `--reference_view datasets/360_v2/garden/images/view_000.png` (a single good view)



----- `--reference_mask outputs/garden/round_001/masks_brown_plant/sam_masks/view_000.png` (mask of the old object to define location/scale)



### **Module 05c: Optimize to Inpainted Targets**Process:

```bash

python 05c_optimize_to_targets.py \1. Generate a high-quality 2D inpainted image for the target object: use an inpainting diffusion model (e.g., SDXL-Inpainting or SDXL+ControlNet) to place the cup into the `reference_view` guided by the `reference_mask` and the `--instruction`.

  --ckpt outputs/garden/round_001/05a_holed/ckpt_holed.pt \

  --targets_dir outputs/garden/round_001/05b_inpainted \2. Lift to 3D: feed this single object-centered 2D image into an Image→3D model (e.g., Wonder3D or a similar single-image reconstruction model) to produce a coarse 3D mesh/sculpt.

  --data_root datasets/360_v2/garden \

  --iters 10003. Convert: convert the mesh into a 3DGS representation (Gaussians). This produces a coarse `ckpt_new_object.pt` containing only the cup.

```

**Purpose:** Optimize 3DGS to match inpainted targets  4. (Optional) Refine: run a refinement optimization on the new 3DGS object (HGS-style hierarchical refinement) to improve detail and shading.

**Output:** `05_inpainting/optimized/ckpt_final.pt` (edited scene)  

**Time:** ~15-30 minutes  Saves (in `outputs/garden_pot_to_black_cup/round_001/06_new_object/`)



**Features:**- `ckpt_new_object.pt`: 3DGS checkpoint containing the generated empty black cup.

- Densification (split/clone/prune) from iter 100-800

- Optimizes: means, quats, scales, opacities, sh0- `generated_2d_image.png`: the 2D inpainted image used to lift to 3D.

- Default 1000 iterations with configurable learning rates

- `manifest.json`: instruction, inpainting model, lift model, seed.

---

Verify

## Configuration System

- `generated_2d_image.png` shows a clear, well-composed empty black cup.

### **Config File (`config.yaml`)**

- Renders from `ckpt_new_object.pt` look coherent from multiple angles (coarse but recognizable).

All settings in one place at the root directory:

---

```yaml

# Project identity## 07_merge_scenes.py — Combine Scene and Object

project:

  name: "garden_brownplant_removal"Goal: Merge the patched scene (`ckpt_patched.pt`) and the new object (`ckpt_new_object.pt`) into a final 3D scene where the cup sits correctly in the hole.

  scene: "garden"

  task: "removal"  # or "replacement"Inputs

  description: "Remove brown plant from garden scene"

-- `--ckpt_scene outputs/garden_pot_to_black_cup/round_001/05_patched_scene/ckpt_patched.pt`

# Paths (auto-resolved)

paths:-- `--ckpt_object outputs/garden_pot_to_black_cup/round_001/06_new_object/ckpt_new_object.pt`

  dataset_root: "datasets/360_v2/garden"

  output_root: "outputs/${project.name}"  # Auto-expands to project name-- `--data_root datasets/360_v2/garden` (for camera poses/depth estimation and coordinate alignment)



# Dataset settingsProcess:

dataset:

  factor: 41. Load both 3DGS checkpoints.

  test_every: 82. Align coordinates: estimate the original object depth/pose (from `ckpt_initial` or depth heuristics) and transform the object Gaussians into the main scene coordinate system, matching scale and position to the original ROI.

  seed: 423. Concatenate Gaussians: merge the two Gaussian sets into a single model. Resolve any duplicate IDs or conflicts.

4. (Optional) Run a short, ROI-gated blending optimization to smooth seams and color transitions between object and scene.

# Training settings

training:Saves (in `outputs/garden_pot_to_black_cup/round_001/07_final_scene/`)

  iterations: 30000

  sh_degree: 3- `ckpt_final.pt`: The final 3DGS checkpoint with the empty black cup inserted.

  eval_steps: [7000, 30000]

  save_steps: [7000, 30000]- `final_renders/`: Renders from multiple viewpoints for verification.



# Segmentation- `manifest.json`: paths to merged checkpoints and any final optimization params.

segmentation:

  text_prompt: "brown plant"Verify

  box_threshold: 0.35

  text_threshold: 0.25- `final_renders/` show the cup correctly positioned on the table where the plant pot used to be, with no obvious floating/scale artifacts.

  nms_threshold: 0.8

  sam_model: "sam2_hiera_large"---



# ROI## 08_evaluate_round.py — Metrics & Reports (Updated for Inpainting Workflow)

roi:

  threshold: 0.3Goal: Summarize the round quantitatively and visually. This module is similar to the original evaluation but adapted to object removal/inpainting and object insertion workflows.

  min_views: 3

Inputs

# Inpainting

inpainting:- Path to original `pre_edit/` renders (from Module 02)

  removal:- Path to `final_renders/` (from Module 07)

    roi_threshold: 0.3- Path to `roi/` (from Module 04)

  sdxl:

    model: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"Saves (in `outputs/garden_pot_to_black_cup/round_001/report/`)

    prompt: "natural outdoor garden scene with grass and plants"

    negative_prompt: "brown plant, dead plant, object, artifact, blur"- `metrics.yaml` with:

    strength: 0.99  - Inside-ROI: CLIP score for the new object (text vs crop), LPIPS to patched targets

    guidance_scale: 7.5  - Outside-ROI: LPIPS/SSIM/PSNR drift comparing `final_renders/` to `pre_edit/`

    num_inference_steps: 50  - ROI IoU: projected ROI overlap if needed

  optimization:

    iterations: 1000- `gallery/`: montage grids of (Original Render) vs (Patched target / Generated object) vs (Final Render)

    densification:

      start_iter: 100- `manifest.json`: full provenance and parameters

      stop_iter: 800

      grad_thresh: 0.0002Verify

      densify_every: 100

    learning_rates:- Inside-ROI CLIP similarity to "empty black cup" is high

      means: 0.00016

      scales: 0.005- Outside-ROI drift is small, demonstrating background preservation

      quats: 0.001

      opacities: 0.05- Gallery images show a coherent replacement across views

      sh0: 0.0025

      shN: 0.0025---



# Hardware settings## File-to-File Dependencies (NEW WORKFLOW)

hardware:

  device: "cuda"- `01_train_gs_initial.py` → `ckpt_initial.pt` used by `02_render_training_views.py`, `04_lift_masks_to_roi3d.py`, `05_remove_and_patch.py`

  num_workers: 4

  mixed_precision: true- `02_render_training_views.py` → `pre_edit/*.png` used by `03_ground_text_to_masks.py`, `06_generate_new_object.py` (as reference view)



# Logging- `03_ground_text_to_masks.py` → `sam_masks/*.png|.npy` used by `04_lift_masks_to_roi3d.py`, `06_generate_new_object.py` (as reference mask)

logging:

  level: "INFO"- `04_lift_masks_to_roi3d.py` → `roi/roi.pt` used by `05_remove_and_patch.py` to identify which Gaussians to delete

  save_renders: true

  save_frequency: 100- `05_remove_and_patch.py` → `ckpt_patched.pt` used by `07_merge_scenes.py` as the base scene

```

- `06_generate_new_object.py` → `ckpt_new_object.pt` used by `07_merge_scenes.py` as the object to add

### **Multiple Experiments**

- `07_merge_scenes.py` → `ckpt_final.pt` used by `08_evaluate_round.py` for final validation

```bash

# Create separate configs---

cp config.yaml exp1_thresh03.yaml

cp config.yaml exp2_thresh05.yaml## Naming the Instruction and Mask Prompts



# Edit each- Edit instruction (Module 06): `"an empty black cup"` (for SDXL/ControlNet)

nano exp1_thresh03.yaml  # Set roi.threshold: 0.3

nano exp2_thresh05.yaml  # Set roi.threshold: 0.5- Mask text (Module 03): `"brown plant . potted plant . planter"` (for GroundingDINO to find the object to remove)



# Run experiments (when config support is added)## Acceptance Criteria Before Connecting Modules

python 05a_remove_and_render_holes.py --config exp1_thresh03.yaml

python 05a_remove_and_render_holes.py --config exp2_thresh05.yaml05: `ckpt_patched.pt` renders a clean scene with the pot fully removed and the hole plausibly patched.

```

06: `ckpt_new_object.pt` renders a high-quality, 3D-consistent "black cup" from multiple angles.

---

07: `ckpt_final.pt` renders show the cup correctly positioned and scaled on the table, with no floating artifacts.

## Setup Details

08: Report shows low drift in the background and high text-image alignment for the new object.

### **Requirements**

- Python 3.8+

- CUDA 11.8+ (12.1+ recommended)## 07_round_driver.py — Orchestrate One Full Round (Optional at First)

- 8GB+ GPU RAM (16GB+ recommended for training)

- 20GB disk space (datasets + models + outputs)**Goal**: Call Modules 02 → 05 → 06 in sequence to run a **complete round** automatically.  

You can postpone this until Modules 02–06 are validated.

### **Key Dependencies**

- PyTorch 2.0+**Inputs**

- gsplat 1.5.3 (3D Gaussian Splatting)- `--ckpt_in …/ckpt_initial.pt` (or latest)

- diffusers (SDXL Inpainting)- `--round_index 1`

- GroundingDINO + SAM2 (segmentation)- Pass-through flags for rendering, diffusion, and optimization

- TripoSR (image-to-3D, for replacement)

- Depth Anything v2 (depth estimation)**Saves (in `outputs/garden_pot_to_black_cup/round_001/`)**

- CLIP (semantic evaluation)- Re-creates the **exact folder structure** used by Modules 02, 03, 04, 05, 06

- `round_manifest.json`: a single provenance file stitching together sub-manifests

### **Model Downloads**

**Automatic (on first run):****Verify**

- SDXL Inpainting: ~6GB- After a single run, all expected subfolders and artifacts exist

- Depth Anything V2: ~1.3GB- Visual check: pre_edit → edited → post_opt is coherent

- CLIP: ~350MB

---

**Manual (via setup scripts):**

- GroundingDINO: ~662MB## 08_evaluate_round.py — Metrics & Reports

- SAM2: ~857MB

**Goal**: Summarize per-round performance numerically and visually.

### **Setup Scripts**

**Inputs**

**`setup.sh`** - Universal setup (local/WSL/cloud)- Paths to `pre_edit/`, `edited_targets/`, `post_opt/`, `roi/`

```bash

chmod +x setup.sh**Saves (in `outputs/garden_pot_to_black_cup/round_001/report/`)**

./setup.sh          # Install all dependencies- `metrics.yaml`:  

./setup.sh --reset  # Clean reinstall  - Inside-ROI: LPIPS/SSIM/Charbonnier to edited targets  

```  - Outside-ROI: drift (ΔRGB, Δopacity)  

  - ROI IoU (recomputed vs projected ROI)

**`setup_vastai.sh`** - Optimized for Vast.ai- `gallery/`: grids of pre_edit vs edited vs post_opt

```bash- `summary.txt`: brief interpretation of metrics

chmod +x setup_vastai.sh- `manifest.json`: which views evaluated, seeds, versions

./setup_vastai.sh   # Auto-downloads dataset + models

```**Verify**

- Inside-ROI numbers improve from pre_edit→post_opt

**`activate.sh`** - Activate virtual environment- Drift outside ROI stays low

```bash- Gallery images show consistent **cup** appearance from multiple views

source activate.sh  # Run before each session

```---



**`download_models.sh`** - Download model weights## 09_round_2_plus.py — Optional Multi-Round Progression

```bash

chmod +x download_models.sh**Goal**: Start **Round 2** from `ckpt_out.pt` of Round 1, **re-render** and **re-edit** to refine the cup’s appearance.

./download_models.sh  # Downloads GroundingDINO + SAM2

```**Inputs**

- `--ckpt_in outputs/garden_pot_to_black_cup/round_001/post_opt/ckpt_out.pt`

---- Reuse or recompute ROI (if the region changed, recompute masks/ROI)

- Same instruction; adjust inpainting parameters or loss weights if needed

## Tips & Best Practices

**Saves (in `outputs/garden_pot_to_black_cup/round_002/`)**

### **ROI Threshold Tuning**- Same structure as Round 1: `pre_edit/`, `edited_targets/`, `post_opt/`, `roi/` (if recomputed), `report/`, `manifest.json`

Module 04a creates ROI weights [0,1]. Module 05a uses a threshold to decide which Gaussians to delete:

**Verify**

1. Start with Module 04a `--roi_thresh 0.5` (creates ROI weights)- Further improvement inside ROI; still low drift outside

2. Experiment with Module 05a `--roi_thresh 0.3-0.5` (uses ROI for deletion)- Progress is visible in side-by-side galleries across rounds

3. Check output masks - should show visible holes where object was

---

**If holes are too small:**

- Lower `--roi_thresh` in Module 05a (e.g., 0.3 → 0.2)## File-to-File Dependencies (What Gets Passed Forward)

- Or re-run Module 04a with lower threshold

- `01_train_gs_initial.py` → produces `ckpt_initial.pt`  

### **SDXL Inpainting Prompts**  used by `02_render_training_views.py`, `04_lift_masks_to_roi3d.py`, `06_optimize_with_roi.py`

- **Positive prompt:** Describe the desired scene *without* mentioning the removed object

  - Good: "natural outdoor garden scene with grass and plants"- `02_render_training_views.py` → `pre_edit/*.png`  

  - Bad: "garden without brown plant" (negative phrasing confuses model)  used by `03_ground_text_to_masks.py` (if grounding on renders), `05_remove_and_patch.py`, `06_generate_new_object.py`

- **Negative prompt:** Explicitly mention the removed object + unwanted artifacts

  - "brown plant, dead plant, object, artifact, blur"- `03_ground_text_to_masks.py` → `sam_masks/*.png|.npy`  

- **Strength:** 0.99 = maximum change, 0.5 = minimal change (use 0.95-0.99 for removal)  used by `04_lift_masks_to_roi3d.py`



### **Performance Optimization**- `04_lift_masks_to_roi3d.py` → `roi/roi.pt`  

- **Module 01 (Training):** Most time-consuming (~3-4 hours on RTX 2060)  used by `05_remove_and_patch.py` to identify which Gaussians to remove

  - Consider using Vast.ai with RTX 4090 (~30-45 min)

- **Module 04a (ROI):** Benefits from faster GPU- `05_remove_and_patch.py` → `ckpt_patched.pt`, `patched_targets/*.png`  

  - RTX 2060: 5-15 min  used by `07_merge_scenes.py` as the scene with the object removed and hole patched

  - RTX 4090: 2-5 min

- **Module 05b (SDXL):** Significantly faster on cloud GPU- `06_generate_new_object.py` → `ckpt_new_object.pt`  

  - RTX 2060: 45-60 min  used by `07_merge_scenes.py` as the new object to insert

  - RTX 4090: 10-15 min

- `06_optimize_with_roi.py` → `post_opt/*.png`, `ckpt_out.pt`  

### **Vast.ai Workflow**  used by `07_round_driver.py` (provenance), and by **Round 2** as new `--ckpt_in`

1. Run Modules 00-03 locally (fast, minimal GPU needed)

2. Upload checkpoint + masks to Vast.ai---

3. Run Modules 04a, 05a, 05b, 05c on cloud GPU

4. Download final checkpoint back to local## Naming the Instruction and Mask Prompts



**Upload to Vast.ai:**- **Edit instruction (Module 05):**  

```bash  “Replace the plant pot on the table with an empty black cup.”

scp -r outputs/garden/01_gs_base/ckpt_initial.pt root@<vastai-ip>:/workspace/3DCV-3D-Scene-Edit-with-3DGS/outputs/garden/01_gs_base/

scp -r outputs/garden/round_001/masks_brownplant root@<vastai-ip>:/workspace/3DCV-3D-Scene-Edit-with-3DGS/outputs/garden/round_001/- **Mask text (Module 03):**  

```  “plant pot . potted plant . planter . flowerpot”



**Download from Vast.ai:**If detection recall is low, add synonyms or try “pot on table” or “table plant pot”.

```bash

scp root@<vastai-ip>:/workspace/3DCV-3D-Scene-Edit-with-3DGS/outputs/garden/round_001/05c_optimized/ckpt_final.pt outputs/garden/round_001/05c_optimized/---

```

## Acceptance Criteria Before Connecting Modules

---

1. **00**: dataset summary and thumbnails look correct; pot visible.  

## Troubleshooting2. **01**: baseline renders are sharp; metrics sane.  

3. **02**: renders reproducible from checkpoint; filenames align.  

### **NumPy Version Error**4. **03**: masks overlay correctly; coverage is reasonable.  

```5. **04**: ROI projects well (IoU > 0.5 typical); ROI not empty.  

OverflowError: Python integer -1 out of bounds for uint646. **05**: edited targets reflect “empty black cup” without global drift.  

```7. **06**: inside-ROI loss decreases; outside-ROI drift small; visuals improve.  

**Solution:** Downgrade NumPy: `pip install "numpy<2.0"`8. **08**: report shows quantitative improvement and consistent visuals.



### **CUDA Out of Memory**Once all pass, you can use **07** (round driver) to automate rounds, and **09** for multi-round refinement.

**Solutions:**

- Reduce batch size in training---

- Use gradient checkpointing

- Run modules separately (05a → 05b → 05c) instead of together## Directory Skeleton (after Round 1)

- Upgrade to GPU with more VRAM


### **Small/No Holes in 05a**
**Problem:** Object not removed properly  
**Solutions:**
1. Lower `--roi_thresh` in Module 05a (try 0.3 instead of 0.7)
2. Check Module 03 masks are accurate (view `overlays/`)
3. Verify Module 04a ROI statistics:
   ```bash
   python -c "import torch; roi=torch.load('outputs/garden/round_001/roi.pt'); print(f'Mean: {roi.mean():.4f}, Max: {roi.max():.4f}, >0.3: {(roi>0.3).sum()}')"
   ```
4. Re-run Module 04a with lower `--roi_thresh`

### **HuggingFace Authentication**
```
401 Client Error: Unauthorized
```
**Solution:** Login to HuggingFace: `huggingface-cli login` (or `hf auth login`)

### **SDXL Model Not Found**
```
404 Client Error: Not Found
```
**Solution:** Model path was updated. Code uses correct path: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`

### **Module 03 No Detections**
**Problem:** GroundingDINO doesn't detect target object  
**Solutions:**
1. Lower `--dino_thresh` (try 0.2 instead of 0.35)
2. Add more synonyms to text prompt: "brown plant . potted plant . planter"
3. Try different phrasing: "plant pot" vs "potted plant"
4. Check `boxes/` visualizations to see what was detected

---

## Citation

If you use this code, please cite:

```bibtex
@software{3dgs_scene_editing,
  title={3D Scene Editing with 3D Gaussian Splatting},
  author={Jin Yoon Ok},
  year={2025},
  url={https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS}
}
```

**Based on:**
- [gsplat](https://github.com/nerfstudio-project/gsplat) - 3D Gaussian Splatting
- [GaussianEditor](https://buaacyw.github.io/gaussian-editor/) - 3D Inpainting methodology
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Text-to-box detection
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Segmentation
- [SDXL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) - Diffusion model

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For major changes, open an issue first to discuss.

---

## Support

- **Issues:** [GitHub Issues](https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS/issues)
- **Discussions:** [GitHub Discussions](https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS/discussions)

---

**Project Status:** Active Development 🚧  
**Last Updated:** November 2025
