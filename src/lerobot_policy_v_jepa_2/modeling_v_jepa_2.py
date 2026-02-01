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

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Any

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_v_jepa_2.configuration_v_jepa_2 import VJEPA2PolicyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_activation_fn(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Choose from {list(activations.keys())}.")
    return activations[name]()


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Temporal Ensembler
# ---------------------------------------------------------------------------

class VJEPA2TemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int):
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(
            -temporal_ensemble_coeff * torch.arange(chunk_size)
        )
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)

        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[
                self.ensembled_actions_count - 1
            ]
            self.ensembled_actions += (
                actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            )
            self.ensembled_actions /= self.ensemble_weights_cumsum[
                self.ensembled_actions_count
            ]
            self.ensembled_actions_count = torch.clamp(
                self.ensembled_actions_count + 1, max=self.chunk_size
            )
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions[:, -1:]], dim=1
            )
            self.ensembled_actions_count = torch.cat(
                [
                    self.ensembled_actions_count,
                    torch.ones_like(self.ensembled_actions_count[-1:]),
                ]
            )

        action = self.ensembled_actions[:, 0]
        self.ensembled_actions = self.ensembled_actions[:, 1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return action


# ---------------------------------------------------------------------------
# Action Head
# ---------------------------------------------------------------------------

class VJEPA2ActionHead(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int,
        activation: str,
        use_ada_ln: bool,
        dropout: float,
    ):
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        if use_ada_ln:
            self.ada_ln = nn.LayerNorm(visual_dim, elementwise_affine=False, eps=1e-6)
            self.ada_ln_modulation = nn.Sequential(
                nn.Linear(state_dim, visual_dim),
                nn.SiLU(),
                nn.Linear(visual_dim, 2 * visual_dim),
            )
        else:
            self.ada_ln = nn.LayerNorm(visual_dim)

        self.has_hidden = hidden_dim > 0
        if self.has_hidden:
            self.hidden_linear = nn.Linear(visual_dim, hidden_dim)
            self.hidden_activation = _get_activation_fn(activation)
            self.hidden_dropout = nn.Dropout(dropout)
            proj_input_dim = hidden_dim
        else:
            proj_input_dim = visual_dim

        self.action_proj = nn.Linear(proj_input_dim, chunk_size * action_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        robot_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = visual_features

        if self.use_ada_ln:
            x_normed = self.ada_ln(x)
            shift, scale = self.ada_ln_modulation(robot_state).chunk(2, dim=-1)
            x = _modulate(x_normed, shift, scale)
        else:
            x = self.ada_ln(x)

        if self.has_hidden:
            x = self.hidden_linear(x)
            x = self.hidden_activation(x)
            x = self.hidden_dropout(x)

        x = self.action_proj(x)
        x = x.view(-1, self.chunk_size, self.action_dim)
        return x


# ---------------------------------------------------------------------------
# V-JEPA2 Policy
# ---------------------------------------------------------------------------

class VJEPA2Policy(PreTrainedPolicy):
    config_class = VJEPA2PolicyConfig
    name = "vjepa2"

    def __init__(
        self,
        config: VJEPA2PolicyConfig,
        dataset_stats: dict[str, Any] | None = None,
    ):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config

        # --- Backbone ---
        if config.vjepa2_pretrained:
            self.backbone = torch.hub.load(
                "facebookresearch/vjepa2", config.vjepa2_model_name
            )
        else:
            self.backbone = torch.hub.load(
                "facebookresearch/vjepa2",
                config.vjepa2_model_name,
                pretrained=False,
            )

        self._backbone_dim = self._probe_backbone_dim()
        self._apply_backbone_freezing()

        # --- Action head ---
        action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else 0
        n_cameras = len(config.image_features)

        self.action_head = VJEPA2ActionHead(
            visual_dim=self._backbone_dim * n_cameras,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=config.chunk_size,
            hidden_dim=config.action_head_hidden_dim,
            activation=config.action_head_activation,
            use_ada_ln=config.use_ada_ln,
            dropout=config.dropout,
        )

        # --- Inference state ---
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = VJEPA2TemporalEnsembler(
                temporal_ensemble_coeff=config.temporal_ensemble_coeff,
                chunk_size=config.chunk_size,
            )

        self.reset()

    # ------------------------------------------------------------------
    # Backbone utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _probe_backbone_dim(self) -> int:
        dummy = torch.randn(1, 3, 224, 224, device=next(self.backbone.parameters()).device)
        out = self.backbone(dummy)

        if isinstance(out, dict):
            for key in ("x", "features", "output", "last_hidden_state"):
                if key in out:
                    out = out[key]
                    break
            else:
                raise RuntimeError(
                    f"Backbone returned a dict with no expected keys. Got: {list(out.keys())}"
                )

        return out.shape[-1]

    def _apply_backbone_freezing(self):
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        if self.config.freeze_backbone_layers == 0:
            return

        # Freeze all, then unfreeze top layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        blocks = self.backbone.blocks
        total_layers = len(blocks)
        n_freeze = self.config.freeze_backbone_layers

        if n_freeze >= total_layers:
            raise ValueError(
                f"freeze_backbone_layers ({n_freeze}) must be < total layers ({total_layers}). "
                f"Use freeze_backbone=True to freeze everything."
            )

        for block in blocks[n_freeze:]:
            for param in block.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def get_optim_params(self) -> list[dict]:
        if self.config.freeze_backbone:
            return [{"params": self.action_head.parameters()}]

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.action_head.parameters())

        param_groups = []
        if len(backbone_params) > 0:
            param_groups.append({
                "params": backbone_params,
                "lr": self.config.optimizer_lr_backbone,
            })
        param_groups.append({"params": head_params})

        return param_groups

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self._predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self._predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def _predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        visual_features = self._encode_images(batch)
        robot_state = batch.get(OBS_STATE, None)
        return self.action_head(visual_features, robot_state)

    def _encode_images(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image_list = batch[OBS_IMAGES]

        pooled = []
        for img in image_list:
            out = self.backbone(img)

            if isinstance(out, dict):
                for key in ("x", "features", "output", "last_hidden_state"):
                    if key in out:
                        out = out[key]
                        break

            # (B, N, D) -> (B, D)
            pooled.append(out.mean(dim=1))

        # (B, n_cameras * D)
        return torch.cat(pooled, dim=-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        visual_features = self._encode_images(batch)
        robot_state = batch.get(OBS_STATE, None)
        actions_hat = self.action_head(visual_features, robot_state)

        actions_target = batch[ACTION]

        if "action_is_pad" in batch:
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            mse_loss = (
                F.mse_loss(actions_hat, actions_target, reduction="none") * mask
            ).sum() / mask.sum()
        else:
            mse_loss = F.mse_loss(actions_hat, actions_target)

        loss_dict = {"mse_loss": mse_loss.item()}

        return mse_loss, loss_dict