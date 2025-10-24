# Instruct gs reconstruction

## Overview
**Instruct gs reconstruction** is a research prototype for **text-guided 3D scene editing** that couples **3D Gaussian Splatting** (via gsplat) with a **2D instruction-following diffusion model** (InstructPix2Pix). It supports:
- **Replace**: swap a specific object for another described in text.
- **Remove**: delete an object and plausibly inpaint the background.
- **Restyle**: change color/material/texture while mostly preserving geometry.

Edits follow an **iterative 2D ↔ 3D loop**: training views are re-rendered and edited in 2D each round; then the 3D Gaussian scene is re-optimized to match those edits with **multi-view consistency** and **preservation** losses to keep non-edited regions stable.

---

## Core Loop (Round-Based, Human-in-the-Loop)
Default **~2,500 3DGS updates per round**. Each round regenerates diffusion edits from the **latest** 3D scene and uses them as the **current training targets**.

**Per Round**
1) **Render current scene** at the training viewpoints (pre-edit renders for this round).  
2) **2D diffusion edit**: apply InstructPix2Pix to these **renders** using the text instruction and optional masks (one image at a time).  
   - The diffusion model edits the **rendered image** conditioned on the instruction (and mask). Original photos are *not* fed jointly to diffusion.  
3) **Swap training targets for this round**: the **freshly edited renders** (“diffusion images”) become the supervision for 3D optimization.  
4) **3D optimization**: train the Gaussian scene against these targets for **~2,500 iterations** with consistency and preservation regularizers.  
5) **Persist artifacts** (see below). The next round starts from the saved 3D snapshot and **repeats steps 1–4**, i.e., **re-render and re-edit** to reflect the updated 3D state.

### Optional: GS-only refinement (within a round)
- **Purpose:** tighten fit without another diffusion pass.  
- **Inputs:** reuse this round’s edited images (no new diffusion).  
- **Action:** run a short GS optimization on the same targets; save a new snapshot/metrics.  
- **Labeling:** this is part of the **current round** (not a new round).  
- **Next round:** always start by **re-rendering and re-editing** from the latest snapshot.

**When to use:** residuals in the RoI are close but not converged; changes are mostly appearance-level.  
**When to skip:** geometry still mismatched; CLIP/alignment or RoI metrics have plateaued; cross-view inconsistency persists.

---

## Saving Everything, Every Round (Progress Tracking)
Each round writes a complete, human-readable record to visualize progress and enable resumption:

- **Diffusion outputs (per view):** the edited images that serve as training targets **for that round**.  
- **Renders (per view):**
  - **Before-edit renders** (from the current 3D scene, same cameras as the edited views).
  - **Optional after-optimization renders** (to visualize the effect of this round’s training and, if used, GS-only refinement).
- **RoI diagnostics:** per-view masks, projected per-Gaussian edit weights, visibility maps.  
- **Metrics:** in-RoI photometric/perceptual, outside-RoI drift, optional instruction-alignment scores.  
- **Scene snapshot:** 3D parameters (and, ideally, optimizer state) captured at the end of the round.  
- **Provenance manifest:** prompt text, diffusion settings, random seeds, model versions, round index, iteration counts.

**Data reuse policy**
- The **edited images saved in round _k_** are used as the training targets **within round _k_**.  
- To begin **round _k+1_**, load the round _k_ snapshot, then **re-render and re-edit** to produce a **new** set of edited images for round _k+1_.  
- Optional short **GS-only refinements** reuse the same (round-k) edited targets and should be labeled as part of round _k_.

---

## Diffusion I/O Policy (Clarification)
- **Input per view:** the **current 3D render** of that view + **text instruction** (+ optional mask).  
- **Output per view:** the **edited image**, which becomes the training target **for that round**.  
- **Original photos:** useful for initialization, evaluation, and/or preservation losses; not concatenated with the render as diffusion input.

---

## Object-Scoped Editing
- **RoI definition:** prompt-guided or interactive 2D masks (points/boxes/polygons).  
- **Lift to 3D:** back-project masks to a **per-Gaussian edit weight** via visibility/coverage; edits are confined to this RoI.  
- **Modes:**
  - **Replace:** diffusion synthesizes the new object; 3D optimization **densifies/splits** Gaussians in high-gradient RoI regions to form structure.
  - **Remove:** diffusion inpaints background; 3D optimization **decays/prunes** object Gaussians and fills background where needed.
  - **Restyle:** primarily adjust **appearance parameters** (e.g., SH colors/opacity) with mild position/scale regularization.

---

## Losses & Regularizers (High-Level)
- **Photometric/perceptual in RoI:** robust reconstruction to edited targets (e.g., L1/Charbonnier + optional perceptual).  
- **Preservation outside RoI:** anchor non-edited regions to pre-edit renders/parameters; penalize unintended opacity growth.  
- **Cross-view consistency:** encourage coherent appearance of each Gaussian across views where visible.  
- **Instruction alignment (optional):** text-image similarity on RoI crops to keep edits faithful to the prompt.

---

## View Selection & Consistency Controls
- **Coverage-aware sampling:** prioritize views with strong RoI visibility; include some off-axis views for stability.  
- **Shared diffusion settings** across a batch to reduce inter-view variance.  
- **Soft mask edges** to mitigate boundary artifacts after back-projection.  
- **Residual-driven scheduling:** views with high RoI residual at the end of a round are prioritized in the next round.

---

## Datasets & Data Layout
- All datasets reside under **`datasets/`**.  
- **User chooses** the dataset and specific scene to edit (in addition to the text instruction).  
- **Inputs required:** multi-view images with **known camera poses** (e.g., COLMAP). Optional: sparse points for initialization.

**Recommended starting point**
- **Mip-NeRF 360:** real 360° captures with reliable poses; well-suited for evaluating multi-view consistency and large viewpoint changes.

**Other options**
- Small real captures via SfM (LLFF-style) for quick iteration.  
- Custom scenes prepared in a gsplat-compatible format (images + poses).

---

## Configuration Surface (Examples)
- Dataset/scene; round length (default **~2,500**); planned number of rounds.  
- Edit mode (replace/remove/restyle); instruction text; RoI/mask mode.  
- Diffusion settings (steps, guidance); fraction of views edited per round (all vs subset).  
- Consistency/preservation weights; densify/prune thresholds; logging cadence.

---

## Evaluation (Per Round & Final)
- **Photometric:** PSNR/SSIM/LPIPS to edited targets within RoI.  
- **Edit locality:** low change outside RoI (drift metric).  
- **Instruction faithfulness:** text-image alignment on RoI crops.  
- **3D consistency:** variance across views for the same surface/Gaussian footprint.

---

## Limitations, Tips, and the “GS-Only” Variant
- **Saving every view each round** is recommended for progress visualization and debugging; storage can be large—consider compression.  
- To **reduce compute**, you may intermittently insert **GS-only refinement** (no new diffusion, reusing the round’s edited targets) for a short step budget. Treat this as **part of the same round**.  
- For exact reproducibility across rounds, persist **random seeds, model versions, and optimizer state** (not just 3D parameters), or accept minor numerical drift.

---

## Roadmap
- Round-based loop with per-round dataset swapping and full artifact logging.  
- RoI tooling (interactive/prompted masks; 3D projection).  
- Mode-specific geometry/appearance policies (replace/remove/restyle).  
- Residual-driven view scheduling and instruction-alignment checks.  
- Ablations: round length, edited-view fraction, regularization strengths.

---

## Ethics & Data Use
- Respect dataset licenses and attributions.  
- Avoid harmful/misleading content; clearly disclose edits in disseminated results.

---

---

## Implementation Architecture

### **Project Structure**
The InstructGS implementation is organized into modular components for maintainability and extensibility:

```
instruct_gs/
├── __init__.py                   # Package initialization
├── main.py                       # Entry point with CLI
├── config/
│   ├── base_config.yaml          # Default configuration template
│   ├── config_manager.py         # YAML config loading/validation
│   └── schemas.py                # Configuration schemas
├── cli/
│   ├── interface.py              # Interactive CLI interface
│   └── commands.py               # Command definitions
├── core/
│   ├── trainer.py                # Main InstructGS trainer
│   ├── round_manager.py          # Round-based editing logic
│   └── state_manager.py          # Training state persistence
├── models/
│   ├── gaussian_model.py         # 3DGS model wrapper
│   ├── diffusion_model.py        # InstructPix2Pix wrapper
│   └── losses.py                 # Custom loss functions
├── data/
│   ├── dataset.py                # Extended dataset class
│   └── roi_utils.py              # Region-of-interest utilities
├── editing/
│   ├── edit_pipeline.py          # 2D editing orchestration
│   ├── mask_generator.py         # RoI mask generation
│   └── view_selector.py          # Coverage-aware view sampling
└── utils/
    ├── path_manager.py           # Unified path management
    ├── metrics.py                # Evaluation metrics
    └── visualization.py          # Progress tracking
```

### **Configuration System**
InstructGS uses YAML-based configuration for easy experimentation without code changes:

```yaml
# Example configuration (instruct_gs/config/base_config.yaml)
experiment:
  name: "instruct_gs_experiment"
  output_dir: "outputs"

editing:
  edit_prompt: "Turn it into a beautiful oil painting"
  edit_mode: "restyle"        # replace, remove, restyle
  cycle_steps: 2500           # 3DGS optimization steps per round
  max_rounds: 10              # Maximum editing rounds

diffusion:
  model_name: "timbrooks/instruct-pix2pix"
  steps: 20
  guidance_scale: 7.5

# ... and many more settings
```

### **Interactive CLI Interface**
The system provides a user-friendly interactive interface for experiment management:

```bash
# Start interactive mode
cd instruct_gs
python main.py --interactive

# Menu options:
# 1. 🚀 Setup New Experiment
# 2. 📂 Load Existing Experiment  
# 3. ▶️  Start Training
# 4. ⏯️  Resume Training
# 5. 📊 View Progress
# 6. ⚙️  Modify Configuration
# 7. 🔄 Reset Experiment
# 8. 💾 Export Results
# 9. ❓ Help
```

### **Experiment Organization**
Each experiment creates a structured output directory for organized results:

```
outputs/
└── my_experiment/
    ├── config.yaml              # Experiment configuration
    ├── training_state.yaml      # Current training state & progress
    ├── checkpoints/             # Model checkpoints per round
    │   ├── model_round_001.pt
    │   ├── model_round_002.pt
    │   └── model_latest.pt
    ├── rounds/                  # Round-specific artifacts
    │   ├── round_001/
    │   │   ├── renders/         # Before/after/mask images
    │   │   │   ├── pre_edit_view_001.png
    │   │   │   ├── edited_view_001.png
    │   │   │   └── post_opt_view_001.png
    │   │   └── artifacts/       # Round metadata
    │   └── round_002/
    ├── logs/                    # Training logs
    ├── metrics/                 # Quantitative metrics per round
    │   ├── round_001.yaml
    │   └── round_002.yaml
    └── artifacts/               # Final results & exports
```

### **Training State Management**
The system provides robust state management for long training runs:

- **Checkpointing**: Automatic model and optimizer state saving per round
- **Resume capability**: Restart training from any completed round
- **Progress tracking**: Metrics, timing, and status persistence
- **Reset functionality**: Clean slate restart while preserving data
- **Backup system**: Create/restore experiment snapshots

### **Key Features**

#### **Modular Design**
- **3DGS Backend**: Leverages gsplat's efficient CUDA implementation
- **Diffusion Integration**: Clean wrapper for InstructPix2Pix and future models
- **Pluggable Components**: Easy to swap loss functions, strategies, etc.

#### **Production Ready**
- **Error Handling**: Graceful recovery from interruptions
- **Memory Management**: Efficient handling of large scenes
- **Monitoring**: Real-time progress and metric tracking
- **Validation**: Configuration and data validation

#### **Research Friendly**
- **Reproducibility**: Full configuration and state persistence
- **Experimentation**: Easy parameter sweeps and ablations
- **Analysis**: Comprehensive metrics and artifact saving
- **Visualization**: Built-in progress and result visualization

---

## Getting Started

### **Installation**
```bash
# Clone repository
git clone https://github.com/your-repo/3DCV-3D-Scene-Edit-with-3DGS.git
cd 3DCV-3D-Scene-Edit-with-3DGS

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download example dataset
cd gsplat-src/examples
python datasets/download_dataset.py --dataset mipnerf360 --save-dir ../../datasets
```

### **Quick Start**
```bash
# Interactive setup (recommended)
cd instruct_gs
python main.py --interactive

# Follow the guided setup:
# 1. Choose "Setup New Experiment"
# 2. Enter experiment name: "my_first_edit"
# 3. Dataset directory: "datasets/360_v2/garden"
# 4. Edit instruction: "Turn it into a beautiful painting"
# 5. Start training!
```

### **Configuration Examples**

**Object Replacement:**
```yaml
editing:
  edit_prompt: "Replace the bicycle with a motorcycle"
  edit_mode: "replace"
  
roi:
  mask_type: "automatic"
  mask_prompt: "bicycle"
```

**Scene Restyling:**
```yaml
editing:
  edit_prompt: "Make it look like a Van Gogh painting"
  edit_mode: "restyle"
  
losses:
  photometric_weight: 1.0
  preservation_weight: 0.3  # Lower to allow more global changes
```

**Object Removal:**
```yaml
editing:
  edit_prompt: "Remove the person from the scene"
  edit_mode: "remove"
  
optimization:
  prune_opa: 0.001  # More aggressive pruning for removal
```

---

## Acknowledgments (Inspirations)
- **3D Gaussian Splatting** methods and training practices.  
- **Instruct-GS2GS** for the iterative 2D↔3D text-guided editing paradigm.  
- **InstructPix2Pix** for fast, instruction-following 2D edits.
