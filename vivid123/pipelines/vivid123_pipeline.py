# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
import PIL

from diffusers.pipelines import TextToVideoSDPipeline
from diffusers.pipelines.text_to_video_synthesis import TextToVideoSDPipelineOutput
from diffusers.models import AutoencoderKL, UNet3DConditionModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
    BaseOutput,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from transformers import CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from ..models import CLIPCameraProjection
import kornia


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import TextToVideoSDPipeline
        >>> from diffusers.utils import export_to_video

        >>> pipe = TextToVideoSDPipeline.from_pretrained(
        ...     "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "Spiderman is surfing"
        >>> video_frames = pipe(prompt).frames
        >>> video_path = export_to_video(video_frames)
        >>> video_path
        ```
"""


def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # f h w c
    return images


class ViVid123Pipeline(TextToVideoSDPipeline):
    r"""
    Pipeline for text-to-video generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`UNet3DConditionModel`]):
            A [`UNet3DConditionModel`] to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        novel_view_unet: UNet2DConditionModel,
        image_encoder: CLIPVisionModelWithProjection,
        cc_projection: CLIPCameraProjection,
    ):
        super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

        self.register_modules(
            novel_view_unet=novel_view_unet,
            image_encoder=image_encoder,
            cc_projection=cc_projection,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=True,
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        num_inference_steps=50,
        fusion_schedule=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        
        if fusion_schedule is None:
            raise ValueError(
                "Fusion schedule is not provided."
            )
        
        if len(fusion_schedule[0]) != num_inference_steps or len(fusion_schedule[1]) != num_inference_steps:
            raise ValueError(
                "Fusion schedule length does not match the number of timesteps."
            )
        
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None, noise_identical_accross_frames=False
    ):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames if not noise_identical_accross_frames else 1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"User-prepared `latents` must have shape {shape}, when noise_identical_accross_frames={noise_identical_accross_frames} but got {latents.shape}."
                )
            latents = latents.to(device)

        if noise_identical_accross_frames:
            latents = latents.repeat(1, 1, num_frames, 1, 1)
        
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_img_latents(
        self, image, batch_size, dtype, device, generator=None, do_zero123_classifier_free_guidance=False
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            assert image.ndim == 4, "Image must have 4 dimensions"

            # Check image is in [-1, 1]
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        image = image.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.mode(generator[i]) for i in range(batch_size)  # sample
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.mode()

        # init_latents = self.vae.config.scaling_factor * init_latents  # todo in original zero123's inference gradio_new.py, model.encode_first_stage() is not scaled by scaling_factor
        if batch_size > init_latents.shape[0]:
            # init_latents = init_latents.repeat(batch_size // init_latents.shape[0], 1, 1, 1)
            num_images_per_prompt = batch_size // init_latents.shape[0]
            # duplicate image latents for each generation per prompt, using mps friendly method
            bs_embed, emb_c, emb_h, emb_w = init_latents.shape
            init_latents = init_latents.unsqueeze(1)
            init_latents = init_latents.repeat(1, num_images_per_prompt, 1, 1, 1)
            init_latents = init_latents.view(bs_embed * num_images_per_prompt, emb_c, emb_h, emb_w)

        # init_latents = torch.cat([init_latents]*2) if do_zero123_classifier_free_guidance else init_latents   # follow zero123
        init_latents = (
            torch.cat([torch.zeros_like(init_latents), init_latents])
            if do_zero123_classifier_free_guidance
            else init_latents
        )

        init_latents = init_latents.to(device=device, dtype=dtype)
        return init_latents

    def CLIP_preprocess(self, x):
        dtype = x.dtype
        # following openai's implementation
        # TODO HF OpenAI CLIP preprocessing issue https://github.com/huggingface/transformers/issues/22505#issuecomment-1650170741
        # follow openai preprocessing to keep exact same, input tensor [-1, 1], otherwise the preprocessing will be different, https://github.com/huggingface/transformers/pull/22608
        if isinstance(x, torch.Tensor):
            if x.min() < -1.0 or x.max() > 1.0:
                raise ValueError("Expected input tensor to have values in the range [-1, 1]")
        x = kornia.geometry.resize(
            x.to(torch.float32), (224, 224), interpolation="bicubic", align_corners=True, antialias=False
        ).to(dtype=dtype)
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(
            x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711])
        )
        return x

    # from stable_diffusion_image_variation
    def _encode_image(self, image, device, num_images_per_prompt, do_video_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            assert image.ndim == 4, "Image must have 4 dimensions"

            # Check image is in [-1, 1]
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        image = image.to(device=device, dtype=dtype)

        image = self.CLIP_preprocess(image)
        # if not isinstance(image, torch.Tensor):
        #     # 0-255
        #     print("Warning: image is processed by hf's preprocess, which is different from openai original's.")
        #     image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        image_embeddings = self.image_encoder(image).image_embeds.to(dtype=dtype)
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_video_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    def _encode_pose(self, pose, device, num_images_per_prompt, do_video_classifier_free_guidance):
        dtype = next(self.cc_projection.parameters()).dtype
        if isinstance(pose, torch.Tensor):
            pose_embeddings = pose.unsqueeze(1).to(device=device, dtype=dtype)
        else:
            if isinstance(pose[0], list):
                pose = torch.Tensor(pose)
            else:
                pose = torch.Tensor([pose])
            x, y, z = pose[:, 0].unsqueeze(1), pose[:, 1].unsqueeze(1), pose[:, 2].unsqueeze(1)
            pose_embeddings = (
                torch.cat([torch.deg2rad(x), torch.sin(torch.deg2rad(y)), torch.cos(torch.deg2rad(y)), z], dim=-1)
                .unsqueeze(1)
                .to(device=device, dtype=dtype)
            )  # B, 1, 4
        # duplicate pose embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = pose_embeddings.shape
        pose_embeddings = pose_embeddings.repeat(1, num_images_per_prompt, 1)
        pose_embeddings = pose_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if do_video_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(pose_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            pose_embeddings = torch.cat([negative_prompt_embeds, pose_embeddings])
        return pose_embeddings

    def _encode_image_with_pose(self, image, pose, device, num_images_per_prompt, do_video_classifier_free_guidance):
        img_prompt_embeds = self._encode_image(image, device, num_images_per_prompt, False)
        pose_prompt_embeds = self._encode_pose(pose, device, num_images_per_prompt, False)
        prompt_embeds = torch.cat([img_prompt_embeds, pose_prompt_embeds], dim=-1)
        prompt_embeds = self.cc_projection(prompt_embeds)
        # prompt_embeds = img_prompt_embeds
        # follow 0123, add negative prompt, after projection
        if do_video_classifier_free_guidance:
            negative_prompt = torch.zeros_like(prompt_embeds)
            prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale_video: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        # vivid123 params below
        image: Optional[
            Union[
                torch.FloatTensor,
                PIL.Image.Image,
                np.ndarray,
                List[torch.FloatTensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ]
        ] = None,
        cam_pose_torch: Optional[torch.FloatTensor] = None,
        fusion_schedule: Optional[tuple[float]] = None,
        ddim_eta_0123: float = 1.0,
        guidance_scale_zero123: float = 3.0,
        noise_identical_accross_frames: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between `torch.FloatTensor` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            guidance_scale_zero123 (`float`, *optional*, defaults to 3.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            cam_pose_torch: (`torch.FloatTensor`, *optional*):
                Camera pose in torch tensor, shape (4,). The elements mean (el, sin(az), cos(az), radius)
            fusion_schedule (`tuple[float]`, *optional*):
                Fusion schedule for video diffusion and zero123. The first element is the schedule for video diffusion, and the
                second element is the schedule for zero123. The length of each schedule should be the same as the number
                of timesteps.
            ddim_eta_0123 (`float`, *optional*, defaults to 1.0):
                The eta value for the 0123 diffusion steps. Only applies to the [`~schedulers.DDIMScheduler`], and is
                ignored in other schedulers.
        
        Example:
        

        Returns:
            [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_videos_per_image_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            num_inference_steps,
            fusion_schedule
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_video_classifier_free_guidance = guidance_scale_video > 1.0
        do_zero123_classifier_free_guidance = guidance_scale_zero123 > 1.0

        # 3.1 Encode input prompt for video diffusion
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            # by diffusers v0.23.1, the naming of diffusers.pipelines.TextToVideoSDPipeline is still "num_images_per_prompt",
            # where it should be "num_videos_per_prompt"
            num_images_per_prompt=num_videos_per_image_prompt,
            do_classifier_free_guidance=do_video_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_video_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3.2 Encode input image for zero123
        zero123_cond_images = [image for _ in range(num_frames)]
        zero123_embeds = self._encode_image_with_pose(
            zero123_cond_images,
            cam_pose_torch,
            device,
            num_videos_per_image_prompt,
            do_zero123_classifier_free_guidance,
        )  # (2xF) x 1 x 768

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_image_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            noise_identical_accross_frames,
        )

        # 6. Prepare Zero123 image latents
        img_latents = self.prepare_img_latents(
            zero123_cond_images,
            batch_size=num_frames,
            dtype=zero123_embeds.dtype,
            device=device,
            generator=generator,
            do_zero123_classifier_free_guidance=True,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_video_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual with video diffusion
                noise_pred_video = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform classifier-free guidance for video diffusion
                if do_video_classifier_free_guidance:
                    noise_pred_video_uncond, noise_pred_video_text = noise_pred_video.chunk(2)
                    noise_pred_video = noise_pred_video_uncond + guidance_scale_video * (
                        noise_pred_video_text - noise_pred_video_uncond
                    )
                # if do_video_classifier_free_guidance and guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred_video = rescale_noise_cfg(
                #         noise_pred_video, noise_pred_video_text, guidance_rescale=guidance_rescale
                #     )

                # zero123 denoising
                latent_model_input_zero123 = torch.cat([latents] * 2) if do_zero123_classifier_free_guidance else latents
                augmented_latent_model_input_zero123 = torch.cat(
                    [rearrange(latent_model_input_zero123, "B C F H W -> (B F) C H W"), img_latents],
                    dim=1,
                ).to(self.novel_view_unet.dtype)
                noise_pred_zero123 = self.novel_view_unet(
                    augmented_latent_model_input_zero123,
                    t,
                    encoder_hidden_states=zero123_embeds,
                    return_dict=True,
                ).sample
                noise_pred_zero123 = rearrange(noise_pred_zero123, "(B F) C H W -> B C F H W", F=num_frames)

                if do_zero123_classifier_free_guidance:
                    noise_pred_zero123_uncond, noise_pred_zero123_text = noise_pred_zero123.chunk(2)
                    noise_pred_zero123 = noise_pred_zero123_uncond + guidance_scale_zero123 * (
                        noise_pred_zero123_text - noise_pred_zero123_uncond
                    )

                # fusing video diffusion with zero123
                noise_pred = fusion_schedule[0][i] * noise_pred_video + fusion_schedule[1][i] * noise_pred_zero123

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            return TextToVideoSDPipelineOutput(frames=latents)

        video_tensor = self.decode_latents(latents)

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor)

        # Offload last model to CPU
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return TextToVideoSDPipelineOutput(frames=video)
