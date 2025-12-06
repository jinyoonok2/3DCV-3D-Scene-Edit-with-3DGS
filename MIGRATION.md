# Migration Guide: venv → Conda

This guide helps you transition from the old venv-based setup to the new conda-based setup.

## What Changed

### Old Setup (Removed)
- `setup-removal.sh` + `activate-removal.sh` → Phase 1 venv
- `setup-generation.sh` + `activate-generation.sh` → Phase 2 venv (never existed)
- Multiple virtual environments
- Pip-only dependency management

### New Setup
- **Single conda environment**: `3dgs-scene-edit`
- **One setup script**: `./setup.sh`
- **One activate script**: `source activate.sh`
- **Unified dependencies**: `configs/environment.yml`
- **No object generation**: Use external GaussianDreamer notebooks

## Migration Steps

### 1. Clean Up Old Environment (Optional)
```bash
# Remove old venv if it exists
rm -rf venv-removal/

# Remove old setup scripts (they're replaced)
# setup-removal.sh, activate-removal.sh are now obsolete
```

### 2. Install Conda (if needed)
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Optional: Install mamba for faster installation
conda install -n base -c conda-forge mamba
```

### 3. Run New Setup
```bash
# Setup conda environment (replaces ./setup-removal.sh)
./setup.sh

# For clean install, use reset flag
./setup.sh --reset
```

### 4. Activate Environment
```bash
# New way (replaces source activate-removal.sh)
source activate.sh

# Or directly
conda activate 3dgs-scene-edit

# Deactivate when done
conda deactivate
```

### 5. Update Your Workflow

**Before:**
```bash
source activate-removal.sh
python 01_train_gs_initial.py
```

**After:**
```bash
source activate.sh
python 01_train_gs_initial.py
```

## What Stays the Same

- All pipeline scripts (00-07) work identically
- Config structure unchanged (now in `configs/garden_config.yaml`)
- Dataset organization unchanged
- Output directory structure unchanged

## New Features

### External Object Generation
Object generation (old steps 06-08) is now external:

1. Use GaussianDreamer/GaussianDreamerPro notebooks separately
2. Save generated `.ply` file
3. Place it in your project directory
4. Run `python 06_place_object_at_roi.py`

**Example object path in configs/garden_config.yaml:**
```yaml
replacement:
  placement:
    object_gaussians: "GaussianDreamerResults/sugarfine_3Dgs5000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency03_gaussperface55.ply"
```

## Benefits of Conda

✅ **Faster setup**: Mamba can install in parallel
✅ **Better dependency resolution**: Conda handles C libraries
✅ **Reproducible**: `configs/environment.yml` pins all versions
✅ **Cross-platform**: Works on Linux, macOS, Windows
✅ **Simpler**: One environment instead of two phases

## Troubleshooting

### "conda command not found"
Install Miniconda or Anaconda first.

### "Environment already exists"
```bash
# Update existing environment
./setup.sh

# Or start fresh
./setup.sh --reset
```

### GPU/CUDA Issues
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check PyTorch version
python -c "import torch; print(torch.__version__)"
```

## For Vast.ai Users

The new setup works perfectly on Vast.ai:

```bash
# SSH into your instance
ssh root@<instance-ip>

# Clone and setup
git clone https://github.com/jinyoonok2/3DCV-3D-Scene-Edit-with-3DGS
cd 3DCV-3D-Scene-Edit-with-3DGS
./setup.sh

# Activate and run
source activate.sh
python 00_check_dataset.py
# ... rest of pipeline
```

## Support

If you encounter issues:
1. Check that conda is installed: `conda --version`
2. Try using mamba for faster setup: `conda install -n base mamba`
3. Use `--reset` flag: `./setup.sh --reset`
4. Check GPU is available: `nvidia-smi`
