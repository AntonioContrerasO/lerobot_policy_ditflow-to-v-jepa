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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("vjepa2")
@dataclass
class VJEPA2PolicyConfig(PreTrainedConfig):
    """Configuration class for the V-JEPA2 policy.

    Uses a pretrained V-JEPA2 ViT backbone for visual feature extraction,
    with an action head that predicts chunked action sequences.

    Supports:
        - Full backbone retraining (freeze_backbone=False)
        - Partial fine-tuning via freeze_backbone_layers (e.g. freeze first N layers)
        - Frozen backbone, only train the action head and optional state projection

    Notes on inputs and outputs:
        - At least one key starting with "observation.image" is required.
        - May optionally include "observation.state" for proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        # --- Input / Output structure ---
        n_obs_steps: Number of observation steps to feed to the policy.
            Currently only 1 is supported.
        chunk_size: Number of future action steps the model predicts at once.
        n_action_steps: Number of predicted action steps actually executed
            per policy invocation. Must be <= chunk_size.

        # --- V-JEPA2 Backbone ---
        vjepa2_model_name: Which V-JEPA2 model to load via torch.hub.
            Options: "vjepa2_vit_large", "vjepa2_vit_huge", "vjepa2_vit_giant".
        vjepa2_pretrained: If True, load pretrained weights from Facebook's hub.
            Set to False to train the ViT from scratch (random init).
        freeze_backbone: If True, the entire V-JEPA2 backbone is frozen.
        freeze_backbone_layers: Number of transformer layers (from the bottom)
            to freeze. Only used when freeze_backbone=False. Set to 0 to
            train all layers. E.g. if the ViT has 24 layers and you set this
            to 20, only the top 4 layers + action head are trained.

        # --- Action Head ---
        # The action head sits on top of the pooled V-JEPA2 features.
        # It uses an adaptive LayerNorm (adaLN) conditioned on the robot
        # proprioceptive state to modulate the features before projecting
        # to actions.
        action_head_hidden_dim: Hidden dimension of the intermediate MLP
            layer in the action head. Set to 0 to use a single linear layer
            (no intermediate layer).
        action_head_activation: Activation function in the action head MLP.
            Options: "relu", "gelu", "silu".
        use_ada_ln: If True, use adaptive LayerNorm (adaLN) modulation in
            the action head, conditioned on robot state. Requires
            "observation.state" to be present. If False, uses standard
            LayerNorm — no conditioning.

        # --- Temporal Ensembling (inference) ---
        temporal_ensemble_coeff: Exponential weighting coefficient for
            temporal ensembling at inference. None disables ensembling.
            When enabled, n_action_steps must be 1.

        # --- Training ---
        dropout: Dropout rate applied in the action head.
        optimizer_lr: Learning rate for the action head (and unfrozen backbone layers).
        optimizer_weight_decay: Weight decay for AdamW.
        optimizer_lr_backbone: Separate (usually lower) LR for unfrozen
            backbone layers. Only used when freeze_backbone=False.
    """

    # --- Input / Output structure ---
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- V-JEPA2 Backbone ---
    vjepa2_model_name: str = "vjepa2_vit_large"
    vjepa2_pretrained: bool = True
    freeze_backbone: bool = False
    freeze_backbone_layers: int = 0  # 0 = train all unfrozen layers

    # --- Action Head ---
    action_head_hidden_dim: int = 1024  # 0 = single linear, no MLP
    action_head_activation: str = "silu"
    use_ada_ln: bool = True  # adaLN conditioned on robot state

    # --- Temporal Ensembling ---
    temporal_ensemble_coeff: float | None = None

    # --- Training ---
    dropout: float = 0.1
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        # --- Validate backbone model name ---
        _VALID_VJEPA2_MODELS = {
            "vjepa2_vit_large",
            "vjepa2_vit_huge",
            "vjepa2_vit_giant",
        }
        if self.vjepa2_model_name not in _VALID_VJEPA2_MODELS:
            raise ValueError(
                f"`vjepa2_model_name` must be one of {_VALID_VJEPA2_MODELS}. "
                f"Got '{self.vjepa2_model_name}'."
            )

        # --- Validate action head activation ---
        _VALID_ACTIVATIONS = {"relu", "gelu", "silu"}
        if self.action_head_activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"`action_head_activation` must be one of {_VALID_ACTIVATIONS}. "
                f"Got '{self.action_head_activation}'."
            )

        # --- Chunk / action step logic (same contract as ACT) ---
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) cannot exceed "
                f"`chunk_size` ({self.chunk_size})."
            )

        # --- Temporal ensemble requires step-by-step inference ---
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. "
                "The policy must be queried every step to form the ensemble."
            )

        # --- Only 1 obs step supported for now ---
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not supported yet. Got `n_obs_steps={self.n_obs_steps}`."
            )

        # --- adaLN requires proprioceptive state ---
        if self.use_ada_ln and not self.robot_state_feature:
            raise ValueError(
                "`use_ada_ln=True` requires 'observation.state' in the inputs "
                "(needed as the conditioning signal for adaLN modulation). "
                "Either add a state input or set `use_ada_ln=False`."
            )

        # --- freeze_backbone_layers only meaningful when not fully frozen ---
        if self.freeze_backbone and self.freeze_backbone_layers > 0:
            raise ValueError(
                "`freeze_backbone_layers` is ignored when `freeze_backbone=True`. "
                "Either set `freeze_backbone=False` or set `freeze_backbone_layers=0`."
            )

    # --- Optimizer / scheduler presets ---
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    # --- Feature validation (called by the policy __init__) ---
    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError(
                "V-JEPA2 policy requires at least one image input "
                "(key starting with 'observation.image')."
            )

    # --- Delta indices (same semantics as ACT) ---
    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None