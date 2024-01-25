from pydantic import BaseModel

class ViVid123BaseSchema(BaseModel):
    # Disable aliasing underscore to hyphen
    class Config:
        alias_generator = lambda string: string

    num_frames: int = 25
    delta_elevation_start: float = 0.0
    delta_elevation_end: float = 0.0
    delta_azimuth_start: float = -45.0
    delta_azimuth_end: float = 45.0
    delta_radius_start: float = 0.0
    delta_radius_end: float = 0.0
    height: int = 256
    width: int = 256
    # num_videos_per_image_prompt: int = 1  # Only support 1 for running on < 24G memory GPU
    num_inference_steps: int = 50
    guidance_scale_zero123: float = 3.0
    guidance_scale_video: float = 1.0
    eta: float = 1.0
    noise_identical_accross_frames: bool = False
    prompt: str = ""

    video_linear_start_weight: float = 1.0
    video_linear_end_weight: float = 0.5
    video_start_step_percentage: float = 0.0
    video_end_step_percentage: float = 1.0
    zero123_linear_start_weight: float = 1.0
    zero123_linear_end_weight: float = 1.0
    zero123_start_step_percentage: float = 0.0
    zero123_end_step_percentage: float = 1.0

    skip_refiner: bool = False
    refiner_strength: float = 0.3
    refiner_guidance_scale: float = 12.0

    name: str = "new_balance_used"
    input_image_path: str = "tmp/new_balance_used/012.png"
