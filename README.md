# 3D Scene Editing with 3D Gaussian Splatting

Text-guided 3D scene editing using **gsplat + Stable Diffusion Inpainting + GroundingDINO + SAM2**.  
Each module is a **single Python file** with clear inputs/outputs and verification steps.

**Example task**: Edit the brown plant in the Mip-NeRF 360 garden scene.

---

## Quick Start

### Local Setup (WSL/Linux)
```bash
# Clone and setup
git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS
cd 3DCV-3D-Scene-Edit-with-3DGS
chmod +x setup.sh
./setup.sh

# Activate environment and run modules
source activate.sh
# See module commands below
```

### Vast.ai Setup (Cloud GPU)
```bash
# Clone and setup (auto-downloads dataset and models)
git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS
cd 3DCV-3D-Scene-Edit-with-3DGS
chmod +x setup_vastai.sh
./setup_vastai.sh

# If tyro is missing, install manually
pip install tyro

# Activate and run
source activate.sh
# Upload your checkpoint and masks, then run Module 04
```

**Note**: RTX 4090 recommended for Vast.ai. RTX 5090 not yet supported by PyTorch stable (use RTX 4090, 3090, or A100).

---

## Setup Scripts

- **`setup.sh`**: Universal setup for local/WSL/cloud (auto-detects environment)
- **`setup_vastai.sh`**: Optimized for Vast.ai with CUDA 12.6, automatic dataset download
- **`activate.sh`**: Activate virtual environment
- **`zip_outputs.sh`**: Package outputs folder for download
- **`unzip_outputs.sh`**: Extract outputs.zip
- **`download_models.sh`**: Download GroundingDINO and SAM2 weights

> No code here—only responsibilities, I/O, and verification steps.  
> You can run each file as a standalone script with simple CLI args.

---

## Global Conventions

- **Dataset:** `datasets/360_v2/garden/` (images and poses prepared for Mip-NeRF 360 “garden”)  
- **Experiment root:** `outputs/garden_pot_to_black_cup/`  
- **View naming:** `view_000.png`, `view_001.png`, … (match your dataset’s ordering)  
- **JSON manifest:** every module that produces outputs creates a `manifest.json` with parameters, seeds, and version strings used  
- **Round naming:** `round_001/`, `round_002/`, … (even if you start with one round)

---

## 00_check_dataset.py — Validate Inputs

**Goal**: Confirm dataset paths, list images/poses, and preview a few frames to ensure the **plant pot** is visible.

**Inputs**
- `--data_root datasets/360_v2/garden/`

**Saves (in `outputs/garden_pot_to_black_cup/00_dataset/`)**
- `summary.txt`: counts of images, intrinsics/extrinsics stats, min/max focal, etc.
- `thumbs/` (e.g., 6 images sampling the sequence): `thumb_000.png`, …
- `manifest.json`: paths, data_root, timestamp

**Verify**
- `summary.txt` matches expected counts (images > 50 for mip360 garden)
- Thumbnails show the table and **plant pot**
- Any mismatch or missing poses is surfaced here before training

---

## 01_train_gs_initial.py — Train/Load Initial 3DGS Scene (gsplat)

**Goal**: Produce a baseline **3D Gaussian** scene and sanity-check rendering quality.

**Inputs**
- `--data_root datasets/360_v2/garden/`
- `--iters 30000` (example), `--seed 42`

**Saves (in `outputs/garden_pot_to_black_cup/01_gs_base/`)**
- `ckpt_initial.pt` (gsplat model checkpoint)
- `renders/` pretrain renders for all training views (`view_000.png`, …)
- `metrics.json`: PSNR/SSIM over training views
- `manifest.json`: config, training iters, seed, gsplat version

**Verify**
- Renders are sharp and align with photos
- Metrics stabilize or improve over time
- Visually confirm the **plant pot** reconstructs reasonably

---

## 02_render_training_views.py — Pre-Edit Renders for a Round

**Goal**: Render the **current** 3D scene (from checkpoint) for the chosen views; these are the images sent to diffusion.

**Inputs**
- `--ckpt outputs/garden_pot_to_black_cup/01_gs_base/ckpt_initial.pt`
- `--views all` (or a subset list), `--seed 42`

**Saves (in `outputs/garden_pot_to_black_cup/round_001/pre_edit/`)**
- `pre_edit_view_000.png`, `pre_edit_view_001.png`, …
- `manifest.json`: which checkpoint, which poses, resolution, seed

**Verify**
- Renders match 01’s visual quality (no regressions)
- Dimensions and naming match later modules’ expectations

---

## 03_ground_text_to_masks.py — Text → Boxes → Masks (GroundingDINO + SAM2)

**Goal**: From **text**, produce **2D masks** isolating objects on each rendered view using two-stage selection.

**Inputs**
- `--images_root outputs/garden/round_001/pre_edit/train/`
- `--text "brown plant"` (text prompt for GroundingDINO)
- `--dino_thresh 0.3` (detection threshold)
- `--sam_thresh 0.3` (SAM2 confidence threshold)
- `--dino_selection confidence` (box selection: confidence/largest/None)
- `--sam_selection confidence` (mask selection: confidence/None)
- `--reference_box "[450,100,800,530]"` (optional spatial filter)
- `--reference_overlap_thresh 0.8` (overlap threshold for spatial filter)

**Two-Stage Selection System**:
1. **DINO Stage**: Select boxes based on detection confidence or size
   - `confidence`: Highest DINO score (default)
   - `largest`: Biggest bounding box
   - `None`: Keep all detections
2. **SAM2 Stage**: Select masks from SAM2 output
   - `confidence`: Highest SAM2 score (default)
   - `None`: Save all masks (creates `_0.png`, `_1.png`, etc.)

**Spatial Filtering** (optional):
- `--reference_box`: Define region of interest [x1,y1,x2,y2]
- `--reference_overlap_thresh`: Fraction of detection box that must overlap (0.8 = 80%)
- Boxes not meeting threshold are rejected before SAM2

**Saves (in `outputs/garden/round_001/masks_brown_plant/`)**
- `boxes/` (annotated detections: blue=reference, green=selected, red=rejected)
- `sam_masks/` (binary masks: PNG + NPY, multiple files if sam_selection=None)
- `overlays/` (mask overlays showing all candidates with selection highlighting)
- `coverage.csv` (includes: num_saved_masks, mask_area, dino_score, sam_score)
- `manifest.json` (records dino_selection, sam_selection, sam_thresh, thresholds)

**Verify**
- Box visualizations show correct selection (green) vs rejection (red)
- Overlays show the target region highlighted
- Coverage CSV shows reasonable mask counts and sizes
- Manifest records all parameters for reproducibility

**Example Commands**:
```bash
# Select highest confidence box and mask
python 03_ground_text_to_masks.py \
    --images_root outputs/garden/round_001/pre_edit/train \
    --text "brown plant" \
    --dino_thresh 0.3 --sam_thresh 0.3 \
    --dino_selection confidence --sam_selection confidence \
    --output_dir outputs/garden/round_001/masks_brown_plant

# Save all masks from largest box
python 03_ground_text_to_masks.py \
    --images_root outputs/garden/round_001/pre_edit/train \
    --text "wooden pot" \
    --dino_selection largest --sam_selection None \
    --reference_box "[500,150,800,500]" --reference_overlap_thresh 0.9 \
    --output_dir outputs/garden/round_001/masks_pot
```

---

## 04_lift_masks_to_roi3d.py — 2D Masks → 3D ROI (Per-Gaussian Weights)

**Goal**: Convert per-view 2D masks into **per-Gaussian ROI weights** `roi ∈ [0,1]` by projecting masks to 3D and aggregating visibility.

**Algorithm**:
1. Load 3D Gaussians from checkpoint (handles nested 'splats' key format)
2. For each training view with a mask:
   - Render Gaussians to get per-pixel Gaussian IDs
   - For each Gaussian visible in view:
     - Check if it projects inside the mask
     - Track: `num_masked_views` and `num_visible_views`
3. Compute per-Gaussian weight: `weight = num_masked_views / num_visible_views`
4. Threshold at `roi_thresh` to get binary ROI

**Inputs**
- `--ckpt outputs/garden/01_gs_base/ckpt_initial.pt` (nested format supported)
- `--masks_root outputs/garden/round_001/masks_brown_plant/sam_masks/`
- `--data_root datasets/360_v2/garden` (for camera poses and images)
- `--roi_thresh 0.5` (threshold for binary ROI)
- `--sh_degree 3` (must match training)

**Checkpoint Format**: Automatically handles both formats:
- Flat: `ckpt['means']`, `ckpt['scales']`, etc.
- Nested: `ckpt['splats']['means']`, `ckpt['splats']['scales']`, etc.

**Saves (in `outputs/garden/round_001/`)**
- `roi.pt` (per-Gaussian weights in [0,1], shape: [N_gaussians])
- Statistics printed: mean weight, std, min, max, num above threshold

**Performance**:
- RTX 2060: ~5-15 minutes for 161 views × 245K Gaussians
- RTX 4090: ~2-5 minutes
- RTX 3090: ~3-7 minutes

**Verify**
- ROI statistics show reasonable distribution (not all 0 or all 1)
- Number of Gaussians above threshold is reasonable (not empty, not everything)
- Check console output for per-view processing progress

**Example Command**:
```bash
# Local (may be slow on RTX 2060)
python 04_lift_masks_to_roi3d.py \
    --ckpt outputs/garden/01_gs_base/ckpt_initial.pt \
    --masks_root outputs/garden/round_001/masks_brown_plant/sam_masks \
    --data_root datasets/360_v2/garden \
    --roi_thresh 0.5 \
    --sh_degree 3 \
    --output_dir outputs/garden/round_001

# Vast.ai (recommended for speed)
# 1. Upload checkpoint and masks
# 2. Run same command on faster GPU
# 3. Download roi.pt back to local machine
```

**Workflow for Vast.ai**:
```bash
# On Vast.ai: Package outputs
./zip_outputs.sh

# On local machine: Download and extract
scp root@<vastai-ip>:/workspace/3DCV-3D-Scene-Edit-with-3DGS/outputs.zip .
./unzip_outputs.sh
```

---

## 05_remove_and_patch.py — Remove Object & Patch Hole (Object-Inpainting Workflow)

Goal: Remove the target object (identified by `roi.pt`) and use 2D inpainting models to generate realistic patched targets that guide a short 3DGS patching optimization. This replaces the IP2P-first workflow when the goal is object replacement/inpainting rather than stylistic edits.

Inputs

-- `--ckpt_in outputs/garden_pot_to_black_cup/01_gs_base/ckpt_initial.pt`

-- `--roi outputs/garden_pot_to_black_cup/round_001/roi/roi.pt`

-- `--data_root datasets/360_v2/garden` (for camera poses)

-- `--iters 1000` (short optimization for hole-patching)

Process:

1. Load `ckpt_in` and `roi.pt`.
2. Delete or deactivate all Gaussians whose ROI weight is above `--roi_thresh` (creates a scene with a hole where the object lived).
3. Render multiple views of this "holed" scene.
4. Patch targets: for each rendered view, run a 2D Inpainting Diffusion Model (e.g., SDXL-Inpainting) to fill the hole plausibly. Save these patched images as the new supervision targets.
5. Optimize: run a short, ROI-gated optimization loop to train the holed 3DGS model to match the patched 2D targets (only loss inside ROI), producing a patched scene checkpoint.

Saves (in `outputs/garden_pot_to_black_cup/round_001/05_patched_scene/`)

- `ckpt_patched.pt`: The 3DGS checkpoint with the plant pot removed and the hole patched via short optimization.

- `patched_targets/`: The per-view 2D inpainted images used as supervision.

- `manifest.json`: parameters, number of iters, diffusion model used for patching.

Verify

- Renders from `ckpt_patched.pt` show the table with the pot removed and the surrounding area plausibly filled.

- `patched_targets/` images look realistic and coherent with the scene lighting.

---

## 06_generate_new_object.py — Generate 3D Object from Text (Object Incorporation)

Goal: Create a new, separate 3DGS model for the desired replacement object ("empty black cup") using a 2D inpainting → image-to-3D lift pipeline.

Inputs

-- `--instruction "an empty black cup"` (text prompt for the new object)

-- `--reference_view datasets/360_v2/garden/images/view_000.png` (a single good view)

-- `--reference_mask outputs/garden/round_001/masks_brown_plant/sam_masks/view_000.png` (mask of the old object to define location/scale)

Process:

1. Generate a high-quality 2D inpainted image for the target object: use an inpainting diffusion model (e.g., SDXL-Inpainting or SDXL+ControlNet) to place the cup into the `reference_view` guided by the `reference_mask` and the `--instruction`.

2. Lift to 3D: feed this single object-centered 2D image into an Image→3D model (e.g., Wonder3D or a similar single-image reconstruction model) to produce a coarse 3D mesh/sculpt.

3. Convert: convert the mesh into a 3DGS representation (Gaussians). This produces a coarse `ckpt_new_object.pt` containing only the cup.

4. (Optional) Refine: run a refinement optimization on the new 3DGS object (HGS-style hierarchical refinement) to improve detail and shading.

Saves (in `outputs/garden_pot_to_black_cup/round_001/06_new_object/`)

- `ckpt_new_object.pt`: 3DGS checkpoint containing the generated empty black cup.

- `generated_2d_image.png`: the 2D inpainted image used to lift to 3D.

- `manifest.json`: instruction, inpainting model, lift model, seed.

Verify

- `generated_2d_image.png` shows a clear, well-composed empty black cup.

- Renders from `ckpt_new_object.pt` look coherent from multiple angles (coarse but recognizable).

---

## 07_merge_scenes.py — Combine Scene and Object

Goal: Merge the patched scene (`ckpt_patched.pt`) and the new object (`ckpt_new_object.pt`) into a final 3D scene where the cup sits correctly in the hole.

Inputs

-- `--ckpt_scene outputs/garden_pot_to_black_cup/round_001/05_patched_scene/ckpt_patched.pt`

-- `--ckpt_object outputs/garden_pot_to_black_cup/round_001/06_new_object/ckpt_new_object.pt`

-- `--data_root datasets/360_v2/garden` (for camera poses/depth estimation and coordinate alignment)

Process:

1. Load both 3DGS checkpoints.
2. Align coordinates: estimate the original object depth/pose (from `ckpt_initial` or depth heuristics) and transform the object Gaussians into the main scene coordinate system, matching scale and position to the original ROI.
3. Concatenate Gaussians: merge the two Gaussian sets into a single model. Resolve any duplicate IDs or conflicts.
4. (Optional) Run a short, ROI-gated blending optimization to smooth seams and color transitions between object and scene.

Saves (in `outputs/garden_pot_to_black_cup/round_001/07_final_scene/`)

- `ckpt_final.pt`: The final 3DGS checkpoint with the empty black cup inserted.

- `final_renders/`: Renders from multiple viewpoints for verification.

- `manifest.json`: paths to merged checkpoints and any final optimization params.

Verify

- `final_renders/` show the cup correctly positioned on the table where the plant pot used to be, with no obvious floating/scale artifacts.

---

## 08_evaluate_round.py — Metrics & Reports (Updated for Inpainting Workflow)

Goal: Summarize the round quantitatively and visually. This module is similar to the original evaluation but adapted to object removal/inpainting and object insertion workflows.

Inputs

- Path to original `pre_edit/` renders (from Module 02)
- Path to `final_renders/` (from Module 07)
- Path to `roi/` (from Module 04)

Saves (in `outputs/garden_pot_to_black_cup/round_001/report/`)

- `metrics.yaml` with:
  - Inside-ROI: CLIP score for the new object (text vs crop), LPIPS to patched targets
  - Outside-ROI: LPIPS/SSIM/PSNR drift comparing `final_renders/` to `pre_edit/`
  - ROI IoU: projected ROI overlap if needed

- `gallery/`: montage grids of (Original Render) vs (Patched target / Generated object) vs (Final Render)

- `manifest.json`: full provenance and parameters

Verify

- Inside-ROI CLIP similarity to "empty black cup" is high

- Outside-ROI drift is small, demonstrating background preservation

- Gallery images show a coherent replacement across views

---

## File-to-File Dependencies (NEW WORKFLOW)

- `01_train_gs_initial.py` → `ckpt_initial.pt` used by `02_render_training_views.py`, `04_lift_masks_to_roi3d.py`, `05_remove_and_patch.py`

- `02_render_training_views.py` → `pre_edit/*.png` used by `03_ground_text_to_masks.py`, `06_generate_new_object.py` (as reference view)

- `03_ground_text_to_masks.py` → `sam_masks/*.png|.npy` used by `04_lift_masks_to_roi3d.py`, `06_generate_new_object.py` (as reference mask)

- `04_lift_masks_to_roi3d.py` → `roi/roi.pt` used by `05_remove_and_patch.py` to identify which Gaussians to delete

- `05_remove_and_patch.py` → `ckpt_patched.pt` used by `07_merge_scenes.py` as the base scene

- `06_generate_new_object.py` → `ckpt_new_object.pt` used by `07_merge_scenes.py` as the object to add

- `07_merge_scenes.py` → `ckpt_final.pt` used by `08_evaluate_round.py` for final validation

---

## Naming the Instruction and Mask Prompts

- Edit instruction (Module 06): `"an empty black cup"` (for SDXL/ControlNet)

- Mask text (Module 03): `"brown plant . potted plant . planter"` (for GroundingDINO to find the object to remove)

## Acceptance Criteria Before Connecting Modules

05: `ckpt_patched.pt` renders a clean scene with the pot fully removed and the hole plausibly patched.

06: `ckpt_new_object.pt` renders a high-quality, 3D-consistent "black cup" from multiple angles.

07: `ckpt_final.pt` renders show the cup correctly positioned and scaled on the table, with no floating artifacts.

08: Report shows low drift in the background and high text-image alignment for the new object.


## 07_round_driver.py — Orchestrate One Full Round (Optional at First)

**Goal**: Call Modules 02 → 05 → 06 in sequence to run a **complete round** automatically.  
You can postpone this until Modules 02–06 are validated.

**Inputs**
- `--ckpt_in …/ckpt_initial.pt` (or latest)
- `--round_index 1`
- Pass-through flags for rendering, diffusion, and optimization

**Saves (in `outputs/garden_pot_to_black_cup/round_001/`)**
- Re-creates the **exact folder structure** used by Modules 02, 03, 04, 05, 06
- `round_manifest.json`: a single provenance file stitching together sub-manifests

**Verify**
- After a single run, all expected subfolders and artifacts exist
- Visual check: pre_edit → edited → post_opt is coherent

---

## 08_evaluate_round.py — Metrics & Reports

**Goal**: Summarize per-round performance numerically and visually.

**Inputs**
- Paths to `pre_edit/`, `edited_targets/`, `post_opt/`, `roi/`

**Saves (in `outputs/garden_pot_to_black_cup/round_001/report/`)**
- `metrics.yaml`:  
  - Inside-ROI: LPIPS/SSIM/Charbonnier to edited targets  
  - Outside-ROI: drift (ΔRGB, Δopacity)  
  - ROI IoU (recomputed vs projected ROI)
- `gallery/`: grids of pre_edit vs edited vs post_opt
- `summary.txt`: brief interpretation of metrics
- `manifest.json`: which views evaluated, seeds, versions

**Verify**
- Inside-ROI numbers improve from pre_edit→post_opt
- Drift outside ROI stays low
- Gallery images show consistent **cup** appearance from multiple views

---

## 09_round_2_plus.py — Optional Multi-Round Progression

**Goal**: Start **Round 2** from `ckpt_out.pt` of Round 1, **re-render** and **re-edit** to refine the cup’s appearance.

**Inputs**
- `--ckpt_in outputs/garden_pot_to_black_cup/round_001/post_opt/ckpt_out.pt`
- Reuse or recompute ROI (if the region changed, recompute masks/ROI)
- Same instruction; adjust inpainting parameters or loss weights if needed

**Saves (in `outputs/garden_pot_to_black_cup/round_002/`)**
- Same structure as Round 1: `pre_edit/`, `edited_targets/`, `post_opt/`, `roi/` (if recomputed), `report/`, `manifest.json`

**Verify**
- Further improvement inside ROI; still low drift outside
- Progress is visible in side-by-side galleries across rounds

---

## File-to-File Dependencies (What Gets Passed Forward)

- `01_train_gs_initial.py` → produces `ckpt_initial.pt`  
  used by `02_render_training_views.py`, `04_lift_masks_to_roi3d.py`, `06_optimize_with_roi.py`

- `02_render_training_views.py` → `pre_edit/*.png`  
  used by `03_ground_text_to_masks.py` (if grounding on renders), `05_remove_and_patch.py`, `06_generate_new_object.py`

- `03_ground_text_to_masks.py` → `sam_masks/*.png|.npy`  
  used by `04_lift_masks_to_roi3d.py`

- `04_lift_masks_to_roi3d.py` → `roi/roi.pt`  
  used by `05_remove_and_patch.py` to identify which Gaussians to remove

- `05_remove_and_patch.py` → `ckpt_patched.pt`, `patched_targets/*.png`  
  used by `07_merge_scenes.py` as the scene with the object removed and hole patched

- `06_generate_new_object.py` → `ckpt_new_object.pt`  
  used by `07_merge_scenes.py` as the new object to insert

- `06_optimize_with_roi.py` → `post_opt/*.png`, `ckpt_out.pt`  
  used by `07_round_driver.py` (provenance), and by **Round 2** as new `--ckpt_in`

---

## Naming the Instruction and Mask Prompts

- **Edit instruction (Module 05):**  
  “Replace the plant pot on the table with an empty black cup.”

- **Mask text (Module 03):**  
  “plant pot . potted plant . planter . flowerpot”

If detection recall is low, add synonyms or try “pot on table” or “table plant pot”.

---

## Acceptance Criteria Before Connecting Modules

1. **00**: dataset summary and thumbnails look correct; pot visible.  
2. **01**: baseline renders are sharp; metrics sane.  
3. **02**: renders reproducible from checkpoint; filenames align.  
4. **03**: masks overlay correctly; coverage is reasonable.  
5. **04**: ROI projects well (IoU > 0.5 typical); ROI not empty.  
6. **05**: edited targets reflect “empty black cup” without global drift.  
7. **06**: inside-ROI loss decreases; outside-ROI drift small; visuals improve.  
8. **08**: report shows quantitative improvement and consistent visuals.

Once all pass, you can use **07** (round driver) to automate rounds, and **09** for multi-round refinement.

---

## Directory Skeleton (after Round 1)

