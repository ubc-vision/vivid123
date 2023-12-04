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
import inspect
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
import torch
import torchvision.transforms.functional as TF
from diffusers.configuration_utils import ConfigMixin, FrozenDict, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging
from diffusers.utils.torch_utils import randn_tensor
from packaging import version
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CLIPCameraProjection(ModelMixin, ConfigMixin):
    """
    A Projection layer for CLIP embedding and camera embedding.
    Parameters:
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the model input `clip_embed`
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected `hidden_states`. The actual length of the used `hidden_states` is `num_embeddings +
            additional_embeddings`.
    """

    @register_to_config
    def __init__(self, embedding_dim: int = 768, additional_embeddings: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.additional_embeddings = additional_embeddings

        self.input_dim = self.embedding_dim + self.additional_embeddings
        self.output_dim = self.embedding_dim

        self.proj = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(
        self,
        embedding: torch.FloatTensor,
    ):
        """
        The [`PriorTransformer`] forward method.
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, input_dim)`):
                The currently input embeddings.
        Returns:
            The output embedding projection (`torch.FloatTensor` of shape `(batch_size, output_dim)`).
        """
        proj_embedding = self.proj(embedding)
        return proj_embedding