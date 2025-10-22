# InstructGS: Text-to-3D Scene Editing with Gaussian Splatting

InstructGS is a novel approach for text-guided 3D scene editing that combines the power of 3D Gaussian Splatting with InstructPix2Pix for high-quality, view-consistent edits. The system uses an Iterative Dataset Update (IDU) methodology to ensure 3D consistency while maintaining photorealistic quality.

## 🎯 Overview

InstructGS enables you to edit 3D scenes using simple text prompts like:
- "Turn it into a painting"
- "Make it winter with snow"  
- "Set it on fire"
- "Turn it into a cartoon style"

The key innovation is the **Iterative Dataset Update (IDU)** cycle:

1. **3D Rendering**: Render the current 3D scene from all training viewpoints
2. **2D Editing**: Edit each rendered image using InstructPix2Pix with dual conditioning
3. **3D Training**: Train the 3D model to match the edited images
4. **Repeat**: Continue this cycle until convergence

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SplatfactoModel  │    │ InstructPix2Pix    │    │ Image Buffers    │
│   (3D Gaussians)   │◄──►│ (2D Editing)       │◄──►│ Original+Edited  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
         └────────────── IDU Controller ──────────────────┘
```

### Core Components

- **InstructGSModel**: Extends SplatfactoModel with image buffer management and IDU logic
- **InstructGSPipeline**: Coordinates training with IDU cycles
- **Image Buffers**: Stores original images and edited versions for dual conditioning
- **IDU Controller**: Manages the iterative editing cycles

## 📦 Installation

### Dependencies

```bash
# Install diffusers for InstructPix2Pix
pip install diffusers transformers accelerate

# Install additional requirements
pip install pillow torch torchvision
```

### Setup

1. Clone this repository
2. Ensure you have a working nerfstudio installation
3. The InstructGS module should be importable from your Python environment

## 🚀 Quick Start

### 1. Prepare Your Data

Use nerfstudio's data processing tools to prepare your dataset:

```bash
# Process images with COLMAP
ns-process-data images --data datasets/my_scene --output-dir datasets/my_scene_processed
```

### 2. Train a Base SplatfactoModel (Optional)

First, train a base 3D Gaussian Splatting model:

```bash
ns-train splatfacto --data datasets/my_scene_processed
```

### 3. Run InstructGS Editing

```bash
python instruct_gs/train_instruct_gs.py \
    --data datasets/my_scene_processed \
    --edit-prompt "Turn it into a painting" \
    --cycle-steps 2500 \
    --max-num-iterations 30000
```

## 📋 Usage Examples

### Basic Usage

```bash
# Turn a scene into a painting
python instruct_gs/train_instruct_gs.py \
    --data datasets/bicycle \
    --edit-prompt "Turn it into a painting"

# Winter scene with snow
python instruct_gs/train_instruct_gs.py \
    --data datasets/garden \
    --edit-prompt "Make it winter with snow" \
    --cycle-steps 2000

# Cartoon style
python instruct_gs/train_instruct_gs.py \
    --data datasets/room \
    --edit-prompt "Turn it into a cartoon style" \
    --cycle-steps 3000
```

### Using Pre-trained Models

```bash
# Load from existing SplatfactoModel
python instruct_gs/train_instruct_gs.py \
    --data datasets/bicycle \
    --load-dir outputs/bicycle/splatfacto/2024-01-01_120000 \
    --edit-prompt "Set it on fire"
```

### Advanced Configuration

```bash
python instruct_gs/train_instruct_gs.py \
    --data datasets/bicycle \
    --edit-prompt "Turn it into a watercolor painting" \
    --cycle-steps 2000 \
    --ip2p-guidance-scale 7.5 \
    --ip2p-image-guidance-scale 1.5 \
    --ip2p-device cuda:1 \
    --max-num-iterations 25000 \
    --steps-per-save 1000
```

## ⚙️ Configuration Options

### Core Parameters

- `--edit-prompt`: Text instruction for editing (e.g., "Turn it into a painting")
- `--cycle-steps`: Steps between IDU cycles (default: 2500)
- `--ip2p-guidance-scale`: Text guidance strength (default: 7.5)
- `--ip2p-image-guidance-scale`: Image conditioning strength (default: 1.5)
- `--ip2p-device`: GPU device for InstructPix2Pix (default: cuda:1)

### Training Parameters

- `--max-num-iterations`: Maximum training steps (default: 30000)
- `--steps-per-save`: Checkpoint frequency (default: 2000)
- `--steps-per-eval`: Evaluation frequency (default: 1000)

### Data Parameters

- `--data`: Path to processed dataset
- `--load-dir`: Path to pre-trained model checkpoint
- `--output-dir`: Output directory for results

## 🔧 Technical Details

### Iterative Dataset Update (IDU)

The IDU process consists of three phases:

1. **Internal 3D Rendering** ($\\mathbf{R}_{current}$):
   ```python
   rendered_images = model.get_outputs(camera) for camera in training_cameras
   ```

2. **Global 2D Editing**:
   ```python
   edited_image = ip2p_pipeline(
       prompt=edit_prompt,
       image=rendered_image,  # Current 3D render
       # Original image provides conditioning
   )
   ```

3. **Dataset Replacement**:
   ```python
   edited_buffer.update(edited_images)
   # Training uses edited images as ground truth
   ```

### Dual Conditioning

InstructPix2Pix receives two image inputs:
- **Rendered Image**: Current 3D scene render (image to edit)
- **Original Image**: Source photo for preserving details and lighting

This dual conditioning ensures:
- ✅ Semantic edits follow the text prompt
- ✅ Non-edited regions remain consistent
- ✅ Original lighting and details are preserved

### Memory Management

InstructGS uses separate GPU devices when possible:
- **Main GPU** (`cuda:0`): 3D Gaussian Splatting model
- **Secondary GPU** (`cuda:1`): InstructPix2Pix pipeline

This prevents OOM errors and allows for larger batch sizes.

## 📊 Expected Results

### Timing
- **IDU Cycle**: ~2-5 minutes (depends on dataset size)
- **Total Training**: 2-6 hours for 30k iterations
- **Memory Usage**: 8-16GB GPU memory (distributed across GPUs)

### Quality
- **3D Consistency**: High view consistency due to 3D training
- **Edit Quality**: Depends on InstructPix2Pix capabilities
- **Detail Preservation**: Good preservation of non-edited regions

## 🐛 Troubleshooting

### Common Issues

1. **OOM Errors**:
   ```bash
   # Use smaller image resolution
   --image-resolution 256 256
   
   # Use separate GPU for IP2P
   --ip2p-device cuda:1
   ```

2. **Poor Edit Quality**:
   ```bash
   # Adjust guidance scales
   --ip2p-guidance-scale 10.0
   --ip2p-image-guidance-scale 2.0
   
   # Try different prompts
   --edit-prompt "make it look like an oil painting"
   ```

3. **Slow Training**:
   ```bash
   # Increase cycle steps
   --cycle-steps 5000
   
   # Reduce image resolution
   --image-resolution 256 256
   ```

### Debug Mode

```bash
# Enable verbose logging
python instruct_gs/train_instruct_gs.py --data datasets/test --edit-prompt "test" --debug
```

## 📚 API Reference

### InstructGSModel

```python
from instruct_gs import InstructGSModel, InstructGSModelConfig

config = InstructGSModelConfig(
    edit_prompt="Turn it into a painting",
    cycle_steps=2500,
    ip2p_guidance_scale=7.5,
)

model = InstructGSModel(config)
```

### InstructGSPipeline

```python
from instruct_gs import InstructGSPipeline, InstructGSPipelineConfig

config = InstructGSPipelineConfig(
    model=InstructGSModelConfig(edit_prompt="Make it winter")
)

pipeline = InstructGSPipeline(config, device="cuda")
```

## 🔬 Research

This implementation is based on the InstructGS methodology for text-to-3D editing. Key research contributions:

1. **Iterative Dataset Update**: Novel training paradigm for 3D consistency
2. **Dual Conditioning**: Using both rendered and original images for better results  
3. **Gaussian Splatting Integration**: Real-time rendering capabilities

## 📄 License

This project is released under the same license as nerfstudio.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**Happy 3D editing!** 🎨✨