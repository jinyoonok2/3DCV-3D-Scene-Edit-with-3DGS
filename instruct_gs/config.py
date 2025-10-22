"""
InstructGS Configuration Integration with Nerfstudio

This module adds InstructGS method configuration to nerfstudio's method registry.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig  
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from .pipeline import InstructGSPipelineConfig
from .model import InstructGSModelConfig


def get_instruct_gs_config(
    edit_prompt: str = "Turn it into a painting",
    cycle_steps: int = 2500,
    ip2p_guidance_scale: float = 7.5,
    ip2p_image_guidance_scale: float = 1.5,
    ip2p_device: str = "cuda:1"
) -> TrainerConfig:
    """
    Get InstructGS training configuration.
    
    Args:
        edit_prompt: Text instruction for editing
        cycle_steps: Steps between IDU cycles
        ip2p_guidance_scale: Text guidance scale for IP2P
        ip2p_image_guidance_scale: Image guidance scale for IP2P
        ip2p_device: Device for IP2P model
        
    Returns:
        Complete trainer configuration for InstructGS
    """
    
    return TrainerConfig(
        method_name="instruct-gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=InstructGSPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=InstructGSModelConfig(
                # InstructGS specific parameters
                edit_prompt=edit_prompt,
                cycle_steps=cycle_steps,
                ip2p_guidance_scale=ip2p_guidance_scale,
                ip2p_image_guidance_scale=ip2p_image_guidance_scale,
                ip2p_device=ip2p_device,
                
                # SplatfactoModel parameters - optimized for editing
                warmup_length=500,
                refine_every=100,
                resolution_schedule=3000,
                background_color="random",
                num_downscales=2,
                cull_alpha_thresh=0.1,
                cull_scale_thresh=0.5,
                reset_alpha_every=30,
                densify_grad_thresh=0.0008,
                densify_size_thresh=0.01,
                n_split_samples=2,
                sh_degree_interval=1000,
                cull_screen_size=0.15,
                split_screen_size=0.05,
                stop_screen_size_at=4000,
                random_init=False,
                num_random=50000,
                random_scale=10.0,
                ssim_lambda=0.2,
                stop_split_at=15000,
                sh_degree=3,
                use_scale_regularization=False,
                max_gauss_ratio=10.0,
                output_depth_during_training=False,
                rasterize_mode="classic",
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )


# Pre-defined configurations for common use cases
INSTRUCT_GS_CONFIGS = {
    "instruct-gs": get_instruct_gs_config(),
    "instruct-gs-painting": get_instruct_gs_config(
        edit_prompt="Turn it into a painting",
        cycle_steps=2500,
    ),
    "instruct-gs-winter": get_instruct_gs_config(
        edit_prompt="Make it winter with snow",
        cycle_steps=2000,
    ),
    "instruct-gs-cartoon": get_instruct_gs_config(
        edit_prompt="Turn it into a cartoon style",
        cycle_steps=3000,
    ),
    "instruct-gs-fire": get_instruct_gs_config(
        edit_prompt="Set it on fire",
        cycle_steps=2500,
    ),
}