#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from typing import Any

import torch

from lerobot_policy_vjepa2.configuration_vjepa2 import VJEPA2PolicyConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


class VJEPA2ImagePreprocessorStep:
    """Applies V-JEPA2's official preprocessor to all image observations.

    V-JEPA2 was pretrained on video data, so its hub preprocessor expects input
    as (B, T, C, H, W) — a batch of frame sequences. LeRobot feeds single images
    per camera as (B, C, H, W). This step:

        1. Unsqueezes a fake temporal dim T=1 to satisfy the preprocessor's
           expected input shape.
        2. Runs the official vjepa2_preprocessor (resize to 224x224 + ImageNet
           mean/std normalization).
        3. Squeezes the temporal dim back out so the backbone receives (B, C, H, W)
           as it would for single-frame inference.

    This runs AFTER the standard NormalizerProcessorStep. The standard normalizer
    handles state/action via dataset_stats. This step then overwrites image tensors
    with V-JEPA2's expected preprocessing, which is fixed from pretraining — not
    learned from your dataset.

    When vjepa2_pretrained=False (training from scratch), this is a no-op.
    """

    def __init__(self, pretrained: bool):
        self.pretrained = pretrained
        if self.pretrained:
            # This loads Facebook's official preprocessor via torch.hub.
            # Internally it does: resize -> normalize with ImageNet stats.
            # Expects (B, T, C, H, W) video tensor input.
            self.vjepa2_preprocessor = torch.hub.load(
                "facebookresearch/vjepa2", "vjepa2_preprocessor"
            )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.pretrained:
            return data

        for key in list(data.keys()):
            if not key.startswith("observation.image"):
                continue

            img = data[key]  # (B, C, H, W)

            # The hub preprocessor expects (B, T, C, H, W).
            # We have single frames, so inject T=1.
            img = img.unsqueeze(1)  # (B, 1, C, H, W)

            # Run official preprocessor: resize + ImageNet normalization.
            img = self.vjepa2_preprocessor(img)  # (B, 1, C, 224, 224)

            # Squeeze T back out — the backbone will see single frames.
            img = img.squeeze(1)  # (B, C, 224, 224)

            data[key] = img

        return data


def make_vjepa2_pre_post_processors(
    config: VJEPA2PolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Constructs pre- and post-processor pipelines for the V-JEPA2 policy.

    Pre-processing pipeline (in order):
        1. Rename observations — passthrough by default, extend rename_map if needed.
        2. Add batch dimension — wraps single-sample dicts into batch dim=1.
        3. Move to device (GPU).
        4. Normalize state/action features using dataset statistics.
        5. V-JEPA2 image preprocessing — handles the video-format mismatch
           (unsqueeze T, run official preprocessor, squeeze T back) and applies
           the backbone's expected resize + ImageNet normalization. No-op when
           training from scratch.

    Post-processing pipeline (in order):
        1. Unnormalize action outputs back to the original dataset scale.
        2. Move outputs back to CPU for environment interaction.

    Args:
        config: The VJEPA2PolicyConfig instance. Controls device placement,
            feature definitions, normalization mappings, and whether pretrained
            weights are active (which gates image preprocessing).
        dataset_stats: Per-feature statistics (mean, std, min, max) from the
            training dataset. Used by the normalizer/unnormalizer steps.
            Defaults to None.

    Returns:
        A tuple of (pre_processor_pipeline, post_processor_pipeline).
    """
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
        # Must be last in the input pipeline. The hub preprocessor overwrites
        # image tensors with V-JEPA2's expected format. State/action normalization
        # from the step above is already applied and untouched by this.
        VJEPA2ImagePreprocessorStep(pretrained=config.vjepa2_pretrained),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )