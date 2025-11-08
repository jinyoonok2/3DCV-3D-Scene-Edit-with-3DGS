#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

05b_inpaint_holes.py - Inpaint Holes with LaMa""""""



Goal: Use LaMa inpainting model to fill holes in rendered views.05b_inpaint_holes.py - Inpaint Holes with LaMa05b_inpaint_holes.py - Inpaint Holes with SDXL or LaMa

      LaMa provides clean object removal without hallucination.



Approach:

  1. Load holed renders and masks from 05aGoal: Use LaMa inpainting model to fill holes in rendered views.Goal: Use inpainting models to fill holes in rendered views.

  2. For each view, run LaMa inpainting to fill the hole

  3. Save inpainted target images      LaMa provides clean object removal without hallucination.



Inputs:Models:

  --holed_dir: Directory with holed renders (from 05a)

  --output_dir: Output directory (default: sibling of holed_dir)Approach:  - SDXL: Creative inpainting with text prompts (good for adding content)

  --mask_blur: Mask blur radius (default: 8)

  --mask_erode: Erode mask by N pixels (shrink hole)  1. Load holed renders and masks from 05a  - LaMa: Object removal inpainting (good for clean removal, no hallucination)

  --mask_dilate: Dilate mask by N pixels (expand hole)

  2. For each view, run LaMa inpainting to fill the hole

Outputs (saved in output_dir/05b_inpainted/):

  - targets/train/*.png: Inpainted target images  3. Save inpainted target imagesApproach:

  - manifest.json: Metadata

  1. Load holed renders and masks from 05a

Note: For SDXL-based inpainting (ablation study), use 05b_inpaint_holes_sdxl.py

"""Inputs:  2. For each view, run inpainting to fill the hole



import argparse  --holed_dir: Directory with holed renders (from 05a)  3. Save inpainted target images

import json

import sys  --output_dir: Output directory (default: sibling of holed_dir)

from datetime import datetime

from pathlib import Path  --mask_blur: Mask blur radius (default: 20)Inputs:



import numpy as np  --mask_erode: Erode mask by N pixels (shrink hole)  --holed_dir: Directory with holed renders (from 05a)

import torch

from PIL import Image  --mask_dilate: Dilate mask by N pixels (expand hole, default: 15)  --output_dir: Output directory (default: sibling of holed_dir)

from tqdm import tqdm

from rich.console import Console  --model: Inpainting model to use (sdxl or lama, default: sdxl)



from project_utils.config import ProjectConfigOutputs (saved in output_dir/05b_inpainted/):  --prompt: Inpainting prompt (SDXL only)



console = Console()  - targets/train/*.png: Inpainted target images  --negative_prompt: Negative prompt (SDXL only)



  - manifest.json: Metadata  --strength: SDXL denoising strength (0-1)

def parse_args():

    parser = argparse.ArgumentParser(description="Inpaint holes with LaMa")  --guidance_scale: CFG scale (SDXL only, default: 7.5)

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")

    parser.add_argument("--holed_dir", type=str, default=None, help="Directory from 05a (overrides config)")Note: For SDXL-based inpainting (ablation study), use 05b_inpaint_holes_sdxl.py  --num_steps: Number of denoising steps (SDXL only, default: 50)

    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")

    parser.add_argument("--mask_blur", type=int, default=8, help="Mask blur radius (default: 8)")"""

    parser.add_argument("--mask_erode", type=int, default=0, help="Erode mask by N pixels (shrink hole)")

    parser.add_argument("--mask_dilate", type=int, default=0, help="Dilate mask by N pixels (expand hole)")Outputs (saved in output_dir/05b_inpainted/):

    parser.add_argument("--device", type=str, default="cuda", help="Device")

    return parser.parse_args()import argparse  - targets/train/*.png: Inpainted target images



import sys  - manifest.json: Metadata

def load_lama_inpainting(device):

    """Load LaMa Inpainting model"""from datetime import datetime"""

    from simple_lama_inpainting import SimpleLama

    from pathlib import Path

    console.print("[cyan]Loading LaMa Inpainting model...[/cyan]")

    console.print("[dim](This may take a moment on first run - downloading ~200MB)[/dim]")import argparse

    

    simple_lama = SimpleLama(device=device)import numpy as npimport json

    console.print("[green]✓ LaMa Inpainting loaded[/green]")

    return simple_lamaimport torchimport sys



from PIL import Imagefrom datetime import datetime

def process_mask(mask, blur_radius=8, erode=0, dilate=0):

    """Process mask with erosion/dilation and blurring"""from tqdm import tqdmfrom pathlib import Path

    import cv2

    from rich.console import Console

    mask_np = np.array(mask)

    import numpy as np

    # Erode (shrink mask)

    if erode > 0:from project_utils.config import ProjectConfigimport torch

        kernel = np.ones((erode, erode), np.uint8)

        mask_np = cv2.erode(mask_np, kernel, iterations=1)from PIL import Image

    

    # Dilate (expand mask)console = Console()from tqdm import tqdm

    if dilate > 0:

        kernel = np.ones((dilate, dilate), np.uint8)from rich.console import Console

        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    

    # Blur edges

    if blur_radius > 0:def parse_args():from project_utils.config import ProjectConfig

        mask_np = cv2.GaussianBlur(mask_np, (blur_radius*2+1, blur_radius*2+1), 0)

        parser = argparse.ArgumentParser(description="Inpaint holes with LaMa")

    return Image.fromarray(mask_np)

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")console = Console()



def inpaint_view_lama(lama_model, image_path, mask_path, mask_blur=8, mask_erode=0, mask_dilate=0):    parser.add_argument("--holed_dir", type=str, default=None, help="Directory from 05a (overrides config)")

    """Inpaint a single view with LaMa (no prompts, pure texture continuation)"""

    # Load images    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")

    img_pil = Image.open(image_path).convert("RGB")

    mask_pil = Image.open(mask_path).convert("L")    parser.add_argument("--mask_blur", type=int, default=20, help="Mask blur radius (default: 20)")def parse_args():

    

    # Process mask for inpainting (with dilation/blur)    parser.add_argument("--mask_erode", type=int, default=0, help="Erode mask by N pixels (shrink hole)")    parser = argparse.ArgumentParser(description="Inpaint holes with SDXL or LaMa")

    mask_pil_processed = process_mask(mask_pil, blur_radius=mask_blur, erode=mask_erode, dilate=mask_dilate)

        parser.add_argument("--mask_dilate", type=int, default=15, help="Dilate mask by N pixels (expand hole, default: 15)")    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")

    # Run LaMa inpainting (simple API - no prompts, no seeds)

    result = lama_model(img_pil, mask_pil_processed)    parser.add_argument("--device", type=str, default="cuda", help="Device")    parser.add_argument("--holed_dir", type=str, default=None, help="Directory from 05a (overrides config)")

    

    return result    return parser.parse_args()    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")



    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "lama"], help="Inpainting model (sdxl or lama, default: sdxl)")

def main():

    args = parse_args()    parser.add_argument("--prompt", type=str, default=None, help="Prompt (SDXL only, overrides config)")

    

    # Load configdef load_lama_inpainting(device):    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt (SDXL only, overrides config)")

    config = ProjectConfig(args.config)

        """Load LaMa Inpainting model"""    parser.add_argument("--strength", type=float, default=None, help="Denoising strength (SDXL only, overrides config)")

    # Override config with command-line arguments

    holed_dir = args.holed_dir if args.holed_dir else str(config.get_path('inpainting') / '05a_holed')    from simple_lama_inpainting import SimpleLama    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG scale (SDXL only, overrides config)")

    output_dir = args.output_dir if args.output_dir else str(config.get_path('inpainting') / '05b_inpainted')

            parser.add_argument("--num_steps", type=int, default=None, help="Denoising steps (SDXL only, overrides config)")

    device = torch.device(args.device)

        console.print("[cyan]Loading LaMa Inpainting model...[/cyan]")    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")

    console.print("[bold cyan]05b - Inpaint Holes with LaMa[/bold cyan]")    console.print("[dim](This may take a moment on first run - downloading ~200MB)[/dim]")    parser.add_argument("--mask_blur", type=int, default=8, help="Mask blur radius (default: 8)")

    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

            parser.add_argument("--mask_erode", type=int, default=0, help="Erode mask by N pixels (shrink hole)")

    # Setup paths

    holed_dir = Path(holed_dir)    simple_lama = SimpleLama(device=device)    parser.add_argument("--mask_dilate", type=int, default=0, help="Dilate mask by N pixels (expand hole)")

    renders_dir = holed_dir / "renders" / "train"

    masks_dir = holed_dir / "masks" / "train"    console.print("[green]✓ LaMa Inpainting loaded[/green]")    parser.add_argument("--device", type=str, default="cuda", help="Device")

    

    if not renders_dir.exists() or not masks_dir.exists():    return simple_lama    return parser.parse_args()

        console.print(f"[red]ERROR: Could not find renders or masks in {holed_dir}[/red]")

        console.print(f"[red]Make sure to run 05a first![/red]")

        sys.exit(1)

    

    output_dir = Path(output_dir)

    targets_dir = output_dir / "targets" / "train"def process_mask(mask, blur_radius=20, erode=0, dilate=15):def load_sdxl_inpainting(device):

    targets_dir.mkdir(parents=True, exist_ok=True)

        """Process mask with erosion/dilation and blurring"""    """Load SDXL Inpainting pipeline"""

    console.print(f"[cyan]Input directory:[/cyan] {holed_dir}")

    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")    import cv2    from diffusers import StableDiffusionXLInpaintPipeline

    console.print(f"[cyan]Model:[/cyan] LaMa")

    console.print("")        

    

    # Get image list    mask_np = np.array(mask)    console.print("[cyan]Loading SDXL Inpainting model...[/cyan]")

    image_files = sorted(renders_dir.glob("*.png"))

    n_images = len(image_files)        console.print("[dim](This may take a few minutes on first run - downloading ~6GB)[/dim]")

    console.print(f"[cyan]Found {n_images} images to inpaint[/cyan]\n")

        # Erode (shrink mask)    

    # Load LaMa model

    lama_model = load_lama_inpainting(device)    if erode > 0:    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(

    

    # Inpaint each view        kernel = np.ones((erode, erode), np.uint8)        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",

    console.print("\n[yellow]Inpainting views...[/yellow]")

            mask_np = cv2.erode(mask_np, kernel, iterations=1)        torch_dtype=torch.float16,

    for img_path in tqdm(image_files, desc="Inpainting"):

        idx = img_path.stem  # e.g., "00042"            variant="fp16",

        mask_path = masks_dir / f"{idx}.png"

            # Dilate (expand mask)    ).to(device)

        if not mask_path.exists():

            console.print(f"[yellow]Warning: No mask for {idx}, skipping[/yellow]")    if dilate > 0:    

            continue

                kernel = np.ones((dilate, dilate), np.uint8)    pipe.enable_attention_slicing()

        # Inpaint with LaMa

        result = inpaint_view_lama(        mask_np = cv2.dilate(mask_np, kernel, iterations=1)    console.print("[green]✓ SDXL Inpainting loaded[/green]")

            lama_model=lama_model,

            image_path=img_path,        return pipe

            mask_path=mask_path,

            mask_blur=args.mask_blur,    # Blur edges

            mask_erode=args.mask_erode,

            mask_dilate=args.mask_dilate,    if blur_radius > 0:

        )

                mask_np = cv2.GaussianBlur(mask_np, (blur_radius*2+1, blur_radius*2+1), 0)def load_lama_inpainting(device):

        # Save

        result.save(targets_dir / f"{idx}.png")        """Load LaMa Inpainting model"""

    

    console.print(f"[green]✓ Saved inpainted targets to {targets_dir}[/green]")    return Image.fromarray(mask_np)    from simple_lama_inpainting import SimpleLama

    

    # Save manifest    

    manifest = {

        "module": "05b_inpaint_holes",    console.print("[cyan]Loading LaMa Inpainting model...[/cyan]")

        "timestamp": datetime.now().isoformat(),

        "config_file": args.config,def inpaint_view(lama_model, image_path, mask_path, mask_blur=20, mask_erode=0, mask_dilate=15):    console.print("[dim](This may take a moment on first run - downloading ~200MB)[/dim]")

        "inputs": {

            "holed_dir": str(holed_dir),    """Inpaint a single view with LaMa (pure texture continuation, no hallucination)"""    

        },

        "parameters": {    # Load images    simple_lama = SimpleLama(device=device)

            "model": "lama",

            "mask_blur": args.mask_blur,    img_pil = Image.open(image_path).convert("RGB")    console.print("[green]✓ LaMa Inpainting loaded[/green]")

            "mask_erode": args.mask_erode,

            "mask_dilate": args.mask_dilate,    mask_pil = Image.open(mask_path).convert("L")    return simple_lama

        },

        "results": {    

            "n_inpainted": n_images,

        },    # Process mask for inpainting (with dilation/blur)

    }

        mask_pil_processed = process_mask(mask_pil, blur_radius=mask_blur, erode=mask_erode, dilate=mask_dilate)def process_mask(mask, blur_radius=8, erode=0, dilate=0):

    config.save_manifest("05b_inpaint_holes", manifest)

    console.print(f"\n[green]✓ Saved manifest to {config.get_path('logs') / '05b_inpaint_holes_manifest.json'}[/green]")        """Process mask with erosion/dilation and blurring"""

    

    console.print("\n[bold green]✓ Module 05b Complete![/bold green]")    # Run LaMa inpainting (simple API - no prompts, no seeds, no hallucination)    import cv2

    console.print(f"[bold green]Next: Run 05c to optimize 3DGS to match targets[/bold green]\n")

    result = lama_model(img_pil, mask_pil_processed)    



if __name__ == "__main__":        mask_np = np.array(mask)

    main()

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
