# 3D Scene Editing with 3D Gaussian Splatting

**GaussianEditor 3D Inpainting Pipeline**  
Text-guided object removal and replacement using:
- **gsplat** (3D Gaussian Splatting)
- **SDXL Inpainting** (2D hole-filling)
- **GroundingDINO + SAM2** (object segmentation)
- **TripoSR** (Image-to-3D generation)
- **Depth Anything v2** (depth-based alignment)

**Example**: Replace brown plant pot → empty black cup

---

## Quick Start

```bash
# Setup
git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS
cd 3DCV-3D-Scene-Edit-with-3DGS
chmod +x setup.sh && ./setup.sh
source activate.sh
```

---

## Pipeline Overview

### Two Workflows:

**A) Object Removal Only**
```
00-04: Prep → 05: Remove & Patch → Done
```

**B) Object Replacement (3D Inpainting)**
```
00-04: Prep → 05: Remove & Patch → 06: Generate New Object → 07: Merge → 08: Evaluate
```

---

## Modules

### 00-04: Preparation (Same for Both Workflows)

**00_check_dataset.py** - Validate dataset
- Input: Dataset path
- Output: Summary, thumbnails

**01_train_gs_initial.py** - Train baseline 3DGS
- Input: Dataset path
- Output: `ckpt_initial.pt`, baseline renders
- Iterations: 30,000

**02_render_training_views.py** - Render all views
- Input: `ckpt_initial.pt`
- Output: `pre_edit/*.png`

**03_ground_text_to_masks.py** - Segment object
- Input: Renders, text prompt
- Uses: GroundingDINO + SAM2
- Output: `sam_masks/*.png`

**04a_lift_masks_to_roi3d.py** - Create 3D ROI
- Input: Checkpoint, masks, poses
- Output: `roi.pt` (per-Gaussian weights)

---

### 05: Object Removal

**05_remove_and_patch.py** - Remove object & fill hole
- Input: `ckpt_initial.pt`, `roi.pt`
- Process:
  1. Delete Gaussians in ROI
  2. Render holed scene
  3. Run SDXL Inpainting on each view
  4. Optimize to match patched targets
- Output: `ckpt_patched.pt`, `patched_targets/*.png`

**Stop here for removal-only workflow.**

---

### 06-07: Object Replacement

**06_generate_new_object.py** - Create new 3D object
- Input: Text prompt, reference view
- Process:
  1. SDXL Inpainting → 2D reference image
  2. TripoSR → 3D mesh
  3. Mesh → 3DGS representation
- Output: `ckpt_new_object.pt`, `generated_2d_reference.png`

**07_merge_scenes.py** - Position & merge objects
- Input: `ckpt_patched.pt`, `ckpt_new_object.pt`
- Process:
  1. Depth Anything v2 → depth maps
  2. Align object to scene (position + scale)
  3. Concatenate Gaussian sets
  4. Optional: blending optimization
- Output: `ckpt_final.pt`, `final_renders/*.png`

---

### 08: Evaluation

**08_evaluate_final_result.py** - Metrics & visualization
- Metrics:
  - Inside ROI: CLIP score, LPIPS to targets
  - Outside ROI: LPIPS/SSIM/PSNR drift
  - Boundary: seam detection
- Output: `metrics.yaml`, `gallery/`, `summary.txt`

---

## Required Models

| Model | Purpose | Used In |
|-------|---------|---------|
| **GroundingDINO** | Text→bounding boxes | Module 03 |
| **SAM2** | Boxes→masks | Module 03 |
| **SDXL Inpainting** | 2D hole-filling | Modules 05, 06 |
| **TripoSR** | Image→3D mesh | Module 06 |
| **Depth Anything v2** | Depth estimation | Module 07 |

Install: See `setup.sh` or `requirements.txt`

---

## Usage Examples

### Removal Only
```bash
python 00_check_dataset.py --data_root datasets/360_v2/garden
python 01_train_gs_initial.py --data_root datasets/360_v2/garden --iters 30000
python 02_render_training_views.py --ckpt outputs/garden/01_gs_base/ckpt_initial.pt
python 03_ground_text_to_masks.py --images_root outputs/garden/round_001/pre_edit --text "brown plant"
python 04a_lift_masks_to_roi3d.py --ckpt outputs/garden/01_gs_base/ckpt_initial.pt --masks_root outputs/garden/round_001/masks/sam_masks
python 05_remove_and_patch.py --ckpt_in outputs/garden/01_gs_base/ckpt_initial.pt --roi outputs/garden/round_001/roi.pt
```

### Replacement
```bash
# Modules 00-05: Same as above
python 06_generate_new_object.py --prompt "empty black ceramic cup" --reference_view outputs/garden/round_001/pre_edit/view_050.png
python 07_merge_scenes.py --ckpt_scene outputs/garden/round_001/05_patched_scene/ckpt_patched.pt --ckpt_object outputs/garden/round_001/06_new_object/ckpt_new_object.pt
python 08_evaluate_final_result.py --pre_edit_dir outputs/garden/round_001/pre_edit --final_renders_dir outputs/garden/round_001/07_final_scene/final_renders
```

---

## Key Technical Details

### ROI (Region of Interest)
- Per-Gaussian weight ∈ [0,1] indicating object membership
- Computed from multi-view mask projections
- Used to isolate edits to specific regions

### Densification
- Gaussians automatically split/clone/prune during optimization
- New Gaussians inherit parent's ROI weight
- Preserves semantic boundaries during refinement

### Depth Alignment (Module 07)
- Scene depth: Rendered from original 3DGS
- Object depth: Estimated by Depth Anything v2
- Alignment: Match median depths in ROI region
- Result: Correctly positioned and scaled object

---

## Output Structure

```
outputs/garden_pot_to_cup/
├── 00_dataset/
├── 01_gs_base/
│   └── ckpt_initial.pt
└── round_001/
    ├── pre_edit/
    ├── masks/sam_masks/
    ├── roi.pt
    ├── 05_patched_scene/
    │   └── ckpt_patched.pt
    ├── 06_new_object/
    │   └── ckpt_new_object.pt
    ├── 07_final_scene/
    │   └── ckpt_final.pt
    └── report/
```

---

## Acceptance Criteria

1. **Module 00**: Dataset valid, object visible
2. **Module 01**: PSNR > 25, sharp renders
3. **Module 02**: Renders match Module 01 quality
4. **Module 03**: Masks cover object accurately
5. **Module 04**: ROI statistics reasonable (not all 0/1)
6. **Module 05**: Hole-filling looks natural
7. **Module 06**: Object recognizable from multiple angles
8. **Module 07**: Object positioned correctly, no floating
9. **Module 08**: CLIP > 0.75, background LPIPS < 0.05

---

## Troubleshooting

**Masks incomplete?**
- Add synonyms to text prompt: `"plant . potted plant . planter"`
- Adjust `--dino_thresh` or `--sam_thresh`

**Object scale wrong?**
- Check depth alignment visualization in Module 07
- Verify reference view shows object clearly

**Background corrupted?**
- ROI may be too large; check `roi.pt` statistics
- Reduce `--roi_thresh` in Module 04

**Poor object quality?**
- Improve 2D reference in Module 06 (prompt engineering)
- Try different reference views

---

## Citation

Based on **GaussianEditor** methodology:
```
@article{gaussianeditor2023,
  title={GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting},
  author={Chen, Yiwen and Chen, Zilong and Zhang, Chi and Wang, Feng and Yang, Xiaofeng and Wang, Yikai and Cai, Zhongang and Yang, Lei and Liu, Huaping and Lin, Guosheng},
  journal={arXiv preprint arXiv:2311.14521},
  year={2023}
}
```

---

## License

MIT License - See LICENSE file
