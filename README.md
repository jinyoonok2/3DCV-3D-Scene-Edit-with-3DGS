# Instruct GS Reconstruction — Module-Per-File Build (gsplat)

This README describes a **module-first** plan to implement text-guided 3D editing on **Mip-NeRF 360 / garden** using **gsplat + InstructPix2Pix + GroundingDINO + SAM/SAM2**.  
Each module is a **single Python file** with a clear job, **inputs/outputs**, **where to save artifacts**, and **how to verify** it works before moving on.  
The concrete task: **repaint the plant pot on the table into an empty black cup**.

> No code here—only responsibilities, I/O, and verification steps.  
> You can run each file as a standalone script with simple CLI args you define later.

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

## 03_ground_text_to_masks.py — Text → Boxes → Masks (GroundingDINO + SAM/SAM2)

**Goal**: From **text**, produce **2D masks** isolating the **plant pot** on each rendered view.

**Inputs**
- `--images_root outputs/garden_pot_to_black_cup/round_001/pre_edit/`
- `--text "plant pot . potted plant . planter . flowerpot"`
- Thresholds (e.g., `--dino_thresh 0.30`)
- SAM/SAM2 model selection and weights path

**Saves (in `outputs/garden_pot_to_black_cup/round_001/masks/`)**
- `boxes/box_view_000.png` (visualization with boxes)
- `sam_masks/mask_view_000.png` (binary PNG, 0/255)
- `sam_masks/mask_view_000.npy` (float mask in [0,1])
- `overlays/overlay_view_000.png` (mask overlay on pre_edit image)
- `coverage.csv` (per-view mask area %, detection scores)
- `manifest.json`: text prompt, thresholds, model versions, seed

**Verify**
- Overlays show the **pot region** highlighted (most of the time)
- Coverage is reasonable (e.g., 0.5%–5% of the image, not 0% or 50%+)
- Failures are logged (views without detections); this is acceptable

---

## 04_lift_masks_to_roi3d.py — 2D Masks → 3D ROI (Per-Gaussian Weights)

**Goal**: Convert per-view 2D masks into a **per-Gaussian ROI weight** `roi ∈ [0,1]` that projects back to the masks.

**Inputs**
- `--ckpt outputs/garden_pot_to_black_cup/01_gs_base/ckpt_initial.pt`
- `--masks_root outputs/garden_pot_to_black_cup/round_001/masks/sam_masks/`
- `--poses_from outputs/garden_pot_to_black_cup/round_001/pre_edit/` (to align views)
- `--iters 600`, `--roi_thresh 0.5`, `--seed 42`

**Saves (in `outputs/garden_pot_to_black_cup/round_001/roi/`)**
- `roi.pt` (tensor of size N_gaussians with values in [0,1])
- `roi_binary.pt` (thresholded indices or binary tensor)
- `proj_masks/roi_proj_view_000.png` (projected ROI vs. SAM mask comparison)
- `metrics.json`: mean IoU with SAM masks on firing views, sparsity %, number of ROI Gaussians
- `manifest.json`: thresholds, iters, seed

**Verify**
- Mean IoU with SAM masks > 0.5 (tune thresholds if needed)
- `roi_binary.pt` is **not empty** and is spatially localized to the pot region
- Projected ROI looks consistent across multiple viewpoints

---

## 05_ip2p_edit_targets.py — InstructPix2Pix Edits (Targets for This Round)

**Goal**: Create **edited targets** for training by applying InstructPix2Pix to **pre-edit renders** with the instruction:

> “Replace the plant pot on the table with an empty black cup.”

**Inputs**
- `--images_root outputs/garden_pot_to_black_cup/round_001/pre_edit/`
- `--instruction "Replace the plant pot on the table with an empty black cup"`
- Optional: pass a **2D mask** per view to constrain edits
- IP2P settings (steps, guidance, seed)

**Saves (in `outputs/garden_pot_to_black_cup/round_001/edited_targets/`)**
- `edited_view_000.png`, …
- `side_by_side/view_000_triptych.png` (pre_edit, edited, diff)
- `manifest.json`: instruction, diffusion params, model version, seed

**Verify**
- Visually, pot appearance trends toward **empty black cup** (shape/color cues)
- No large unintended changes outside the pot region (spot-check triptychs)
- Optional: simple CLIP text-image score on crops increases vs pre_edit

---

## 06_optimize_with_roi.py — ROI-Gated 3DGS Optimization (Round 1)

**Goal**: Optimize the 3DGS so rendered images match **edited targets** **inside ROI** and preserve the rest **outside ROI**.

**Inputs**
- `--ckpt_in outputs/garden_pot_to_black_cup/01_gs_base/ckpt_initial.pt`
- `--roi outputs/garden_pot_to_black_cup/round_001/roi/roi.pt`
- `--edited_root outputs/garden_pot_to_black_cup/round_001/edited_targets/`
- `--preedit_root outputs/garden_pot_to_black_cup/round_001/pre_edit/` (for preservation outside ROI)
- `--iters 2500`, loss weights (e.g., `--w_edit 1.0 --w_preserve 0.15`), `--seed 42`

**Saves (in `outputs/garden_pot_to_black_cup/round_001/post_opt/`)**
- `post_view_000.png`, …
- `loss_curves.json`: inside-ROI loss, outside-ROI drift per iter
- `ckpt_out.pt`: updated 3DGS checkpoint after Round 1
- `manifest.json`: all loss weights, iters, seed

**Verify**
- Inside-ROI loss decreases over iterations
- Outside-ROI drift remains low (compare `post_view_x` vs `pre_edit_view_x`)
- Visuals: pot region morphs toward **empty black cup** from multiple views

---

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
- Same instruction; adjust IP2P or loss weights if needed

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
  used by `03_ground_text_to_masks.py` (if grounding on renders), `05_ip2p_edit_targets.py`, `06_optimize_with_roi.py` (preservation)

- `03_ground_text_to_masks.py` → `sam_masks/*.png|.npy`  
  used by `04_lift_masks_to_roi3d.py`

- `04_lift_masks_to_roi3d.py` → `roi/roi.pt`  
  used by `06_optimize_with_roi.py` to gate edit/preserve losses

- `05_ip2p_edit_targets.py` → `edited_targets/*.png`  
  used by `06_optimize_with_roi.py` as the supervision images for this round

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

