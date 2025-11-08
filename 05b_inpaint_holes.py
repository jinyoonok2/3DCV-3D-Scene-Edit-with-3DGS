#!/usr/bin/env python3#!/usr/bin/env python3

""""""

05b_inpaint_holes.py - Inpaint Holes with LaMa05b_inpaint_holes.py - Inpaint Holes with SDXL or LaMa



Goal: Use LaMa inpainting model to fill holes in rendered views.Goal: Use inpainting models to fill holes in rendered views.

      LaMa provides clean object removal without hallucination.

Models:

Approach:  - SDXL: Creative inpainting with text prompts (good for adding content)

  1. Load holed renders and masks from 05a  - LaMa: Object removal inpainting (good for clean removal, no hallucination)

  2. For each view, run LaMa inpainting to fill the hole

  3. Save inpainted target imagesApproach:

  1. Load holed renders and masks from 05a

Inputs:  2. For each view, run inpainting to fill the hole

  --holed_dir: Directory with holed renders (from 05a)  3. Save inpainted target images

  --output_dir: Output directory (default: sibling of holed_dir)

  --mask_blur: Mask blur radius (default: 20)Inputs:

  --mask_erode: Erode mask by N pixels (shrink hole)  --holed_dir: Directory with holed renders (from 05a)

  --mask_dilate: Dilate mask by N pixels (expand hole, default: 15)  --output_dir: Output directory (default: sibling of holed_dir)

  --model: Inpainting model to use (sdxl or lama, default: sdxl)

Outputs (saved in output_dir/05b_inpainted/):  --prompt: Inpainting prompt (SDXL only)

  - targets/train/*.png: Inpainted target images  --negative_prompt: Negative prompt (SDXL only)

  - manifest.json: Metadata  --strength: SDXL denoising strength (0-1)

  --guidance_scale: CFG scale (SDXL only, default: 7.5)

Note: For SDXL-based inpainting (ablation study), use 05b_inpaint_holes_sdxl.py  --num_steps: Number of denoising steps (SDXL only, default: 50)

"""

Outputs (saved in output_dir/05b_inpainted/):

import argparse  - targets/train/*.png: Inpainted target images

import sys  - manifest.json: Metadata

from datetime import datetime"""

from pathlib import Path

import argparse

import numpy as npimport json

import torchimport sys

from PIL import Imagefrom datetime import datetime

from tqdm import tqdmfrom pathlib import Path

from rich.console import Console

import numpy as np

from project_utils.config import ProjectConfigimport torch

from PIL import Image

console = Console()from tqdm import tqdm

from rich.console import Console



def parse_args():from project_utils.config import ProjectConfig

    parser = argparse.ArgumentParser(description="Inpaint holes with LaMa")

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")console = Console()

    parser.add_argument("--holed_dir", type=str, default=None, help="Directory from 05a (overrides config)")

    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")

    parser.add_argument("--mask_blur", type=int, default=20, help="Mask blur radius (default: 20)")def parse_args():

    parser.add_argument("--mask_erode", type=int, default=0, help="Erode mask by N pixels (shrink hole)")    parser = argparse.ArgumentParser(description="Inpaint holes with SDXL or LaMa")

    parser.add_argument("--mask_dilate", type=int, default=15, help="Dilate mask by N pixels (expand hole, default: 15)")    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")

    parser.add_argument("--device", type=str, default="cuda", help="Device")    parser.add_argument("--holed_dir", type=str, default=None, help="Directory from 05a (overrides config)")

    return parser.parse_args()    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")

    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "lama"], help="Inpainting model (sdxl or lama, default: sdxl)")

    parser.add_argument("--prompt", type=str, default=None, help="Prompt (SDXL only, overrides config)")

def load_lama_inpainting(device):    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt (SDXL only, overrides config)")

    """Load LaMa Inpainting model"""    parser.add_argument("--strength", type=float, default=None, help="Denoising strength (SDXL only, overrides config)")

    from simple_lama_inpainting import SimpleLama    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG scale (SDXL only, overrides config)")

        parser.add_argument("--num_steps", type=int, default=None, help="Denoising steps (SDXL only, overrides config)")

    console.print("[cyan]Loading LaMa Inpainting model...[/cyan]")    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")

    console.print("[dim](This may take a moment on first run - downloading ~200MB)[/dim]")    parser.add_argument("--mask_blur", type=int, default=8, help="Mask blur radius (default: 8)")

        parser.add_argument("--mask_erode", type=int, default=0, help="Erode mask by N pixels (shrink hole)")

    simple_lama = SimpleLama(device=device)    parser.add_argument("--mask_dilate", type=int, default=0, help="Dilate mask by N pixels (expand hole)")

    console.print("[green]✓ LaMa Inpainting loaded[/green]")    parser.add_argument("--device", type=str, default="cuda", help="Device")

    return simple_lama    return parser.parse_args()





def process_mask(mask, blur_radius=20, erode=0, dilate=15):def load_sdxl_inpainting(device):

    """Process mask with erosion/dilation and blurring"""    """Load SDXL Inpainting pipeline"""

    import cv2    from diffusers import StableDiffusionXLInpaintPipeline

        

    mask_np = np.array(mask)    console.print("[cyan]Loading SDXL Inpainting model...[/cyan]")

        console.print("[dim](This may take a few minutes on first run - downloading ~6GB)[/dim]")

    # Erode (shrink mask)    

    if erode > 0:    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(

        kernel = np.ones((erode, erode), np.uint8)        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",

        mask_np = cv2.erode(mask_np, kernel, iterations=1)        torch_dtype=torch.float16,

            variant="fp16",

    # Dilate (expand mask)    ).to(device)

    if dilate > 0:    

        kernel = np.ones((dilate, dilate), np.uint8)    pipe.enable_attention_slicing()

        mask_np = cv2.dilate(mask_np, kernel, iterations=1)    console.print("[green]✓ SDXL Inpainting loaded[/green]")

        return pipe

    # Blur edges

    if blur_radius > 0:

        mask_np = cv2.GaussianBlur(mask_np, (blur_radius*2+1, blur_radius*2+1), 0)def load_lama_inpainting(device):

        """Load LaMa Inpainting model"""

    return Image.fromarray(mask_np)    from simple_lama_inpainting import SimpleLama

    

    console.print("[cyan]Loading LaMa Inpainting model...[/cyan]")

def inpaint_view(lama_model, image_path, mask_path, mask_blur=20, mask_erode=0, mask_dilate=15):    console.print("[dim](This may take a moment on first run - downloading ~200MB)[/dim]")

    """Inpaint a single view with LaMa (pure texture continuation, no hallucination)"""    

    # Load images    simple_lama = SimpleLama(device=device)

    img_pil = Image.open(image_path).convert("RGB")    console.print("[green]✓ LaMa Inpainting loaded[/green]")

    mask_pil = Image.open(mask_path).convert("L")    return simple_lama

    

    # Process mask for inpainting (with dilation/blur)

    mask_pil_processed = process_mask(mask_pil, blur_radius=mask_blur, erode=mask_erode, dilate=mask_dilate)def process_mask(mask, blur_radius=8, erode=0, dilate=0):

        """Process mask with erosion/dilation and blurring"""

    # Run LaMa inpainting (simple API - no prompts, no seeds, no hallucination)    import cv2

    result = lama_model(img_pil, mask_pil_processed)    

        mask_np = np.array(mask)

    return result    

    # Erode (shrink mask)

    if erode > 0:

def main():        kernel = np.ones((erode, erode), np.uint8)

    args = parse_args()        mask_np = cv2.erode(mask_np, kernel, iterations=1)

        

    # Load config    # Dilate (expand mask)

    config = ProjectConfig(args.config)    if dilate > 0:

            kernel = np.ones((dilate, dilate), np.uint8)

    # Override config with command-line arguments        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    holed_dir = args.holed_dir if args.holed_dir else str(config.get_path('inpainting') / '05a_holed')    

    output_dir = args.output_dir if args.output_dir else str(config.get_path('inpainting') / '05b_inpainted')    # Blur edges

        if blur_radius > 0:

    device = torch.device(args.device)        mask_np = cv2.GaussianBlur(mask_np, (blur_radius*2+1, blur_radius*2+1), 0)

        

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")    return Image.fromarray(mask_np)

    console.print("[bold cyan]05b - Inpaint Holes with LaMa[/bold cyan]")

    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    def inpaint_view_sdxl(pipe, image_path, mask_path, prompt, negative_prompt, strength, guidance_scale, num_steps, seed, mask_blur=8, mask_erode=0, mask_dilate=0):

    # Setup paths    """Inpaint a single view with SDXL (with pre-fill of surrounding context color)"""

    holed_dir = Path(holed_dir)    import cv2

    renders_dir = holed_dir / "renders" / "train"    

    masks_dir = holed_dir / "masks" / "train"    # Load images

        img_pil = Image.open(image_path).convert("RGB")

    if not renders_dir.exists() or not masks_dir.exists():    mask_pil = Image.open(mask_path).convert("L")

        console.print(f"[red]ERROR: Could not find renders or masks in {holed_dir}[/red]")    

        console.print(f"[red]Make sure to run 05a first![/red]")    # Process mask for inpainting (with dilation/blur)

        sys.exit(1)    mask_pil_processed = process_mask(mask_pil, blur_radius=mask_blur, erode=mask_erode, dilate=mask_dilate)

        

    output_dir = Path(output_dir)    # --- PRE-FILL LOGIC: Fill hole with average surrounding color ---

    targets_dir = output_dir / "targets" / "train"    # This gives SDXL a strong hint: "the hole is already table-colored, just add texture"

    targets_dir.mkdir(parents=True, exist_ok=True)    img_np = np.array(img_pil)

        mask_np_orig = np.array(mask_pil)  # Use original mask for color sampling

    console.print(f"[cyan]Input directory:[/cyan] {holed_dir}")    

    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")    # Create inverse mask (everything EXCEPT the hole)

    console.print(f"[cyan]Model:[/cyan] LaMa (Large Mask Inpainting)")    # Dilate it slightly to avoid sampling edge pixels that might be artifacts

    console.print(f"[cyan]Mask processing:[/cyan] dilate={args.mask_dilate}, blur={args.mask_blur}")    kernel = np.ones((5, 5), np.uint8)

    console.print("")    inv_mask = cv2.dilate(255 - mask_np_orig, kernel, iterations=1)

        

    # Get image list    # Get average color from surrounding table pixels (not in hole)

    image_files = sorted(renders_dir.glob("*.png"))    avg_color_bgr = cv2.mean(img_np, mask=inv_mask)[:3]  # Returns BGR

    n_images = len(image_files)    avg_color_rgb = (int(avg_color_bgr[2]), int(avg_color_bgr[1]), int(avg_color_bgr[0]))

    console.print(f"[cyan]Found {n_images} images to inpaint[/cyan]\n")    

        # Create patch image filled with average color

    # Load LaMa model    patch_img = Image.new("RGB", img_pil.size, avg_color_rgb)

    lama_model = load_lama_inpainting(device)    

        # Composite: use patch where mask is white, original image elsewhere

    # Inpaint each view    img_prefilled = Image.composite(patch_img, img_pil, mask_pil)

    console.print("\n[yellow]Inpainting views...[/yellow]")    # --- END PRE-FILL LOGIC ---

        

    for img_path in tqdm(image_files, desc="Inpainting"):    # Run inpainting on pre-filled image

        idx = img_path.stem  # e.g., "00042"    generator = torch.Generator(device=pipe.device).manual_seed(seed)

        mask_path = masks_dir / f"{idx}.png"    result = pipe(

                prompt=prompt,

        if not mask_path.exists():        negative_prompt=negative_prompt,

            console.print(f"[yellow]Warning: No mask for {idx}, skipping[/yellow]")        image=img_prefilled,  # Use pre-filled image instead of original

            continue        mask_image=mask_pil_processed,

                strength=strength,

        # Inpaint with LaMa        num_inference_steps=num_steps,

        result = inpaint_view(        guidance_scale=guidance_scale,

            lama_model=lama_model,        generator=generator,

            image_path=img_path,    ).images[0]

            mask_path=mask_path,    

            mask_blur=args.mask_blur,    return result

            mask_erode=args.mask_erode,

            mask_dilate=args.mask_dilate,

        )def inpaint_view_lama(lama_model, image_path, mask_path, mask_blur=8, mask_erode=0, mask_dilate=0):

            """Inpaint a single view with LaMa (no prompts, pure texture continuation)"""

        # Save    # Load images

        result.save(targets_dir / f"{idx}.png")    img_pil = Image.open(image_path).convert("RGB")

        mask_pil = Image.open(mask_path).convert("L")

    console.print(f"[green]✓ Saved inpainted targets to {targets_dir}[/green]")    

        # Process mask for inpainting (with dilation/blur)

    # Save manifest    mask_pil_processed = process_mask(mask_pil, blur_radius=mask_blur, erode=mask_erode, dilate=mask_dilate)

    manifest = {    

        "module": "05b_inpaint_holes",    # Run LaMa inpainting (simple API - no prompts, no seeds)

        "timestamp": datetime.now().isoformat(),    result = lama_model(img_pil, mask_pil_processed)

        "config_file": args.config,    

        "inputs": {    return result

            "holed_dir": str(holed_dir),

        },

        "parameters": {def main():

            "model": "lama",    args = parse_args()

            "mask_blur": args.mask_blur,    

            "mask_erode": args.mask_erode,    # Load config

            "mask_dilate": args.mask_dilate,    config = ProjectConfig(args.config)

        },    

        "results": {    # Override config with command-line arguments

            "n_inpainted": n_images,    holed_dir = args.holed_dir if args.holed_dir else str(config.get_path('inpainting') / '05a_holed')

        },    output_dir = args.output_dir if args.output_dir else str(config.get_path('inpainting') / '05b_inpainted')

    }    

        prompt = args.prompt if args.prompt else config.config.get('inpainting', {}).get('prompt', 'wooden table surface')

    config.save_manifest("05b_inpaint_holes", manifest)    negative_prompt = args.negative_prompt if args.negative_prompt else config.config.get('inpainting', {}).get('negative_prompt', 'blurry, distorted')

    console.print(f"\n[green]✓ Saved manifest to {config.get_path('logs') / '05b_inpaint_holes_manifest.json'}[/green]")    strength = args.strength if args.strength is not None else config.config.get('inpainting', {}).get('strength', 0.99)

        guidance_scale = args.guidance_scale if args.guidance_scale is not None else config.config.get('inpainting', {}).get('guidance_scale', 7.5)

    console.print("\n[bold green]✓ Module 05b Complete![/bold green]")    num_steps = args.num_steps if args.num_steps is not None else config.config.get('inpainting', {}).get('num_steps', 50)

    console.print(f"[bold green]Next: Run 05c to optimize 3DGS to match targets[/bold green]\n")    seed = args.seed if args.seed is not None else config.config['dataset']['seed']

    

    device = torch.device(args.device)

if __name__ == "__main__":    

    main()    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")

    console.print(f"[bold cyan]05b - Inpaint Holes with {args.model.upper()}[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Setup paths
    holed_dir = Path(holed_dir)
    renders_dir = holed_dir / "renders" / "train"
    masks_dir = holed_dir / "masks" / "train"
    
    if not renders_dir.exists() or not masks_dir.exists():
        console.print(f"[red]ERROR: Could not find renders or masks in {holed_dir}[/red]")
        console.print(f"[red]Make sure to run 05a first![/red]")
        sys.exit(1)
    
    output_dir = Path(output_dir)
    targets_dir = output_dir / "targets" / "train"
    targets_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Input directory:[/cyan] {holed_dir}")
    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Model:[/cyan] {args.model.upper()}")
    if args.model == "sdxl":
        console.print(f"[cyan]Prompt:[/cyan] {prompt}")
        console.print(f"[cyan]Strength:[/cyan] {strength}")
    console.print("")
    
    # Get image list
    image_files = sorted(renders_dir.glob("*.png"))
    n_images = len(image_files)
    console.print(f"[cyan]Found {n_images} images to inpaint[/cyan]\n")
    
    # Load model
    if args.model == "sdxl":
        pipe = load_sdxl_inpainting(device)
    else:  # lama
        pipe = load_lama_inpainting(device)
    
    # Inpaint each view
    console.print("\n[yellow]Inpainting views...[/yellow]")
    
    for img_path in tqdm(image_files, desc="Inpainting"):
        idx = img_path.stem  # e.g., "00042"
        mask_path = masks_dir / f"{idx}.png"
        
        if not mask_path.exists():
            console.print(f"[yellow]Warning: No mask for {idx}, skipping[/yellow]")
            continue
        
        # Inpaint based on model choice
        if args.model == "sdxl":
            result = inpaint_view_sdxl(
                pipe=pipe,
                image_path=img_path,
                mask_path=mask_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                seed=seed,  # SAME seed for consistency across views
                mask_blur=args.mask_blur,
                mask_erode=args.mask_erode,
                mask_dilate=args.mask_dilate,
            )
        else:  # lama
            result = inpaint_view_lama(
                lama_model=pipe,
                image_path=img_path,
                mask_path=mask_path,
                mask_blur=args.mask_blur,
                mask_erode=args.mask_erode,
                mask_dilate=args.mask_dilate,
            )
        
        # Save
        result.save(targets_dir / f"{idx}.png")
    
    console.print(f"[green]✓ Saved inpainted targets to {targets_dir}[/green]")
    
    # Save manifest
    manifest = {
        "module": "05b_inpaint_holes",
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "inputs": {
            "holed_dir": str(holed_dir),
        },
        "parameters": {
            "model": args.model,
            "prompt": prompt if args.model == "sdxl" else None,
            "negative_prompt": negative_prompt if args.model == "sdxl" else None,
            "strength": strength if args.model == "sdxl" else None,
            "guidance_scale": guidance_scale if args.model == "sdxl" else None,
            "num_steps": num_steps if args.model == "sdxl" else None,
            "seed": seed if args.model == "sdxl" else None,
        },
        "results": {
            "n_inpainted": n_images,
        },
    }
    
    config.save_manifest("05b_inpaint_holes", manifest)
    console.print(f"\n[green]✓ Saved manifest to {config.get_path('logs') / '05b_inpaint_holes_manifest.json'}[/green]")
    
    console.print("\n[bold green]✓ Module 05b Complete![/bold green]")
    console.print(f"[bold green]Next: Run 05c to optimize 3DGS to match targets[/bold green]\n")


if __name__ == "__main__":
    main()
