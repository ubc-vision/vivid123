import os
import torch
from PIL import Image

from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection
from vivid123.models import CLIPCameraProjection
from vivid123.pipelines import Zero1to3StableDiffusionPipeline

from diffusers.utils import export_to_video

model_id = "bennyguo/zero123-xl-diffusers"

zero123_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", cache_dir="/scratch/.cache")
zero123_cam_proj = CLIPCameraProjection.from_pretrained(model_id, subfolder="clip_camera_projection", cache_dir="/scratch/.cache")
zero123_img_enc = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", cache_dir="/scratch/.cache")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", cache_dir="/scratch/.cache")
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", cache_dir="/scratch/.cache")
zero123_pipe = Zero1to3StableDiffusionPipeline(
    vae=vae,
    image_encoder=zero123_img_enc,
    unet=zero123_unet,
    scheduler=scheduler,
    cc_projection=zero123_cam_proj,
    requires_safety_checker=False,
    safety_checker=None,
    feature_extractor=None,
)

# zero123_pipe.enable_xformers_memory_efficient_attention()
# zero123_pipe.enable_vae_tiling()
# zero123_pipe.enable_attention_slicing()
zero123_pipe = zero123_pipe.to("cuda")

query_pose = [0, 45.0, 0.0]

# for single input
H, W = (256, 256)
num_images_per_prompt = 1

input_image = Image.open("data/Squirrel/img/012.png").convert("RGBA").resize((H, W), Image.BICUBIC)
background = Image.new("RGBA", input_image.size, (255, 255, 255))
alpha_composite = Image.alpha_composite(background, input_image)

input_images = [alpha_composite]
query_poses = [query_pose]

images = zero123_pipe(
    input_imgs=input_images,
    prompt_imgs=input_images,
    poses=query_poses,
    height=H,
    width=W,
    guidance_scale=3.0,
    num_inference_steps=50,
).images

# save imgs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
bs = len(input_images)
i = 0
for obj in range(bs):
    for idx in range(num_images_per_prompt):
        images[i].save(os.path.join(log_dir, f"obj{obj}_{idx}.jpg"))
        i += 1