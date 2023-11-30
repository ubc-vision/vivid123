import os
from typing import List

import torch
from PIL import Image
import numpy as np
import imageio.v3 as imageio

from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers.pipelines import DiffusionPipeline
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import export_to_video

import pydantic_argparse

from models import CLIPCameraProjection
from pipelines import ViVid123Pipeline
from configs import ViVid123BaseSchema


def prepare_cam_pose_input(
    num_frames: int = 25,
    delta_elevation_start: float = 0.0,
    delta_elevation_end: float = 0.0,
    delta_azimuth_start: float = -45.0,
    delta_azimuth_end: float = 45.0,
    delta_radius_start: float = 0.0,
    delta_radius_end: float = 0.0,
):
    r"""
    The function to prepare the input to the vivid123 pipeline

    Args:
        delta_elevation_start (`float`, *optional*, defaults to 0.0):
            The starting relative elevation angle of the camera, in degree. Relative to the elevation of the reference image.
            The camera is facing towards the origin.
        delta_elevation_end (`float`, *optional*, defaults to 0.0):
            The ending relative elevation angle of the camera, in degree. Relative to the elevation of the reference image.
            The camera is facing towards the origin.
        delta_azimuth_start (`float`, *optional*, defaults to -45.0):
            The starting relative azimuth angle of the camera, in degree. Relative to the elevation of the reference image.
            The camera is facing towards the origin.
        delta_azimuth_end (`float`, *optional*, defaults to 45.0):
            The ending relative azimuth angle of the camera, in degree. Relative to the elevation of the reference image.
            The camera is facing towards the origin.
    
    Returns:

    """
    cam_elevation = np.radians(np.linspace(delta_elevation_start, delta_elevation_end, num_frames))[..., None]
    cam_azimuth = np.radians(np.linspace(delta_azimuth_start, delta_azimuth_end, num_frames))
    cam_azimuth_sin_cos = np.stack([np.sin(cam_azimuth), np.cos(cam_azimuth)], axis=-1)
    cam_radius = np.linspace(delta_radius_start, delta_radius_end, num_frames)[..., None]

    cam_pose_np = np.concatenate([cam_elevation, cam_azimuth_sin_cos, cam_radius], axis=-1)
    cam_pose_torch = torch.from_numpy(cam_pose_np)

    return cam_pose_torch


# refer to https://stackoverflow.com/a/33507138/6257375
def conver_rgba_to_rgb_white_bg(
    image_path: str,
    H: int = 256,
    W: int = 256,
):
    input_image = Image.open(image_path).convert("RGBA").resize((H, W), Image.Resampling.BICUBIC)
    background = Image.new("RGBA", input_image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, input_image)
    
    return alpha_composite


def prepare_fusion_schedule_linear(
    num_inference_steps: int = 50,
    video_linear_start_weight: float = 1.0,
    video_linear_end_weight: float = 0.5,
    video_start_step_percentage: float = 0.0,
    video_end_step_percentage: float = 1.0,
    zero123_linear_start_weight: float = 1.0,
    zero123_linear_end_weight: float = 1.0,
    zero123_start_step_percentage: float = 0.0,
    zero123_end_step_percentage: float = 1.0,
):
    """
    Prepare the fusion schedule of video diffusion and zero123 at all the denoising steps

    Args:
        video_linear_start_weight (`float`, *optional*, defaults to 1.0):
            The weight of the video diffusion at the start of the video. The weight is linearly increased from
            `video_linear_start_weight` to `video_linear_end_weight` during the video diffusion.
        video_linear_end_weight (`float`, *optional*, defaults to 0.5):
            The weight of the video diffusion at the end of the video. The weight is linearly increased from
            `video_linear_start_weight` to `video_linear_end_weight` during the video diffusion.
        video_start_step_percentage (`float`, *optional*, defaults to 0.0):
            The percentage of the total number of inference steps at which the video diffusion starts. The video
            diffusion is linearly increased from `video_linear_start_weight` to `video_linear_end_weight` between
            `video_start_step_percentage` and `video_end_step_percentage`.
        video_end_step_percentage (`float`, *optional*, defaults to 1.0):
            The percentage of the total number of inference steps at which the video diffusion ends. The video
            diffusion is linearly increased from `video_linear_start_weight` to `video_linear_end_weight` between
            `video_start_step_percentage` and `video_end_step_percentage`.
        zero123_linear_start_weight (`float`, *optional*, defaults to 1.0):
            The weight of the zero123 diffusion at the start of the video. The weight is linearly increased from
            `zero123_linear_start_weight` to `zero123_linear_end_weight` during the zero123 diffusion.
        zero123_linear_end_weight (`float`, *optional*, defaults to 1.0):
            The weight of the zero123 diffusion at the end of the video. The weight is linearly increased from
            `zero123_linear_start_weight` to `zero123_linear_end_weight` during the zero123 diffusion.
        zero123_start_step_percentage (`float`, *optional*, defaults to 0.0):
            The percentage of the total number of inference steps at which the zero123 diffusion starts. The
            zero123 diffusion is linearly increased from `zero123_linear_start_weight` to
            `zero123_linear_end_weight` between `zero123_start_step_percentage` and `zero123_end_step_percentage`.
        zero123_end_step_percentage (`float`, *optional*, defaults to 1.0):
            The percentage of the total number of inference steps at which the zero123 diffusion ends. The
            zero123 diffusion is linearly increased from `zero123_linear_start_weight` to
            `zero123_linear_end_weight` between `zero123_start_step_percentage` and `zero123_end_step_percentage`.
    
    Return:
        A tuple of two tensors, 
        video_schedule (`torch.Tensor`): The schedule of the video diffusion weighting, with shape `[num_inference_steps]`.
        zero123_schedule (`torch.Tensor`): The schedule of the zero123 diffusion weighting, with shape `[num_inference_steps]`.
    """

    assert (
        video_linear_start_weight >= 0.0 and video_linear_start_weight <= 1.0
    ), "video_linear_start_weight must be between 0.0 and 1.0"
    assert (
        video_linear_end_weight >= 0.0 and video_linear_end_weight <= 1.0
    ), "video_linear_end_weight must be between 0.0 and 1.0"
    assert (
        video_start_step_percentage >= 0.0 and video_start_step_percentage <= 1.0
    ), "video_start_step_percentage must be between 0.0 and 1.0"
    assert (
        video_end_step_percentage >= 0.0 and video_end_step_percentage <= 1.0
    ), "video_end_step_percentage must be between 0.0 and 1.0"
    assert (
        zero123_linear_start_weight >= 0.0 and zero123_linear_start_weight <= 1.0
    ), "zero123_linear_start_weight must be between 0.0 and 1.0"
    assert (
        zero123_linear_end_weight >= 0.0 and zero123_linear_end_weight <= 1.0
    ), "zero123_linear_end_weight must be between 0.0 and 1.0"
    assert (
        zero123_start_step_percentage >= 0.0 and zero123_start_step_percentage <= 1.0
    ), "zero123_start_step_percentage must be between 0.0 and 1.0"
    assert (
        zero123_end_step_percentage >= 0.0 and zero123_end_step_percentage <= 1.0
    ), "zero123_end_step_percentage must be between 0.0 and 1.0"

    video_schedule = torch.linspace(
        start=video_linear_start_weight,
        end=video_linear_end_weight,
        steps=int((video_end_step_percentage - video_start_step_percentage) * num_inference_steps),
    )
    zero123_schedule = torch.linspace(
        start=zero123_linear_start_weight,
        end=zero123_linear_end_weight,
        steps=int((zero123_end_step_percentage - zero123_start_step_percentage) * num_inference_steps),
    )
    if video_schedule.shape[0] < num_inference_steps:
        video_schedule = torch.cat(
            [
                video_linear_start_weight * torch.ones([video_start_step_percentage * num_inference_steps]),
                video_schedule,
                video_linear_end_weight
                * torch.ones([num_inference_steps - video_end_step_percentage * num_inference_steps]),
            ]
        )
    if zero123_schedule.shape[0] < num_inference_steps:
        zero123_schedule = torch.cat(
            [
                zero123_linear_start_weight * torch.ones([zero123_start_step_percentage * num_inference_steps]),
                zero123_schedule,
                zero123_linear_end_weight
                * torch.ones([num_inference_steps - zero123_end_step_percentage * num_inference_steps]),
            ]
        )
    
    return (video_schedule, zero123_schedule)


def save_videos_grid_zeroscope_nplist(video_frames: List[np.ndarray], path: str, n_rows=6, fps=8, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    f = len(video_frames)
    h, w, c = video_frames[0].shape
    #images = [(image).astype("uint8") for image in video_frames]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, video_frames, fps=fps)


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=ViVid123BaseSchema,
        prog="ViVid-1-to-3",
        description="Novel View Synthesis with ViVid-1-to-3",
        version="0.0.1",
        epilog="Program Epilog",
        add_help=True,
        exit_on_error=True,
    )
    cfg = parser.parse_typed_args()

    zero123_unet = UNet2DConditionModel.from_pretrained(cfg.zero123_model_id, subfolder="unet")
    zero123_cam_proj = CLIPCameraProjection.from_pretrained(cfg.zero123_model_id, subfolder="clip_camera_projection")
    zero123_img_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.zero123_model_id, subfolder="image_encoder")
    vivid123_pipe = ViVid123Pipeline.from_pretrained(
        cfg.video_model_id,
        # torch_dtype=torch.float16,
        novel_view_unet=zero123_unet,
        image_encoder=zero123_img_enc,
        cc_projection=zero123_cam_proj,
    )
    vivid123_pipe.to("cuda")

    input_img = conver_rgba_to_rgb_white_bg(cfg.input_image, H=cfg.height, W=cfg.width)

    cam_pose = prepare_cam_pose_input(
        num_frames=cfg.num_frames,
        delta_elevation_start=cfg.delta_elevation_start,
        delta_elevation_end=cfg.delta_elevation_end,
        delta_azimuth_start=cfg.delta_azimuth_start,
        delta_azimuth_end=cfg.delta_azimuth_end,
        delta_radius_start=cfg.delta_radius_start,
        delta_radius_end=cfg.delta_radius_end,
    )

    fusion_schedule = prepare_fusion_schedule_linear(
        num_inference_steps=cfg.num_inference_steps,
        video_linear_start_weight=cfg.video_linear_start_weight,
        video_linear_end_weight=cfg.video_linear_end_weight,
        video_start_step_percentage=cfg.video_start_step_percentage,
        video_end_step_percentage=cfg.video_end_step_percentage,
        zero123_linear_start_weight=cfg.zero123_linear_start_weight,
        zero123_linear_end_weight=cfg.zero123_linear_end_weight,
        zero123_start_step_percentage=cfg.zero123_start_step_percentage,
        zero123_end_step_percentage=cfg.zero123_end_step_percentage,
    )

    vid_output_frames = vivid123_pipe(
        image=input_img,
        cam_pose_torch=cam_pose,
        fusion_schedule=fusion_schedule,
        height=cfg.height,
        width=cfg.width,
        num_frames=cfg.num_frames,
        prompt=cfg.prompt,
        guidance_scale_video=cfg.guidance_scale_video,
        guidance_scale_zero123=cfg.guidance_scale_zero123,
        num_inference_steps=cfg.num_inference_steps,
    ).frames

    # save imgs
    os.makedirs(cfg.output_dir, exist_ok=True)
    bs = len(vid_output_frames)
    for i in range(bs):
        Image.fromarray(vid_output_frames[i]).save(f"{cfg.output_dir}/{i}.png")

    save_videos_grid_zeroscope_nplist(vid_output_frames, f"{cfg.output_dir}/base.mp4")
