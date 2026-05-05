from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from alphagen.config import OPERATORS
from alphagen.rl.policy import PositionalEncoding
from alphagen.rl.env.wrapper import OFFSET_OP
from new.env import HEAD_NAMES


class MultiHeadTransformerFeatures(BaseFeaturesExtractor):
    """Shared Transformer encoder for expression tokens; the final obs slot is head_id."""

    def __init__(
        self,
        observation_space: gym.Space,
        n_encoder_layers: int = 2,
        d_model: int = 128,
        n_head: int = 4,
        d_ffn: int = 256,
        dropout: float = 0.1,
        device: Union[str, th.device] = "cpu",
    ):
        super().__init__(observation_space, d_model)
        assert isinstance(observation_space, gym.spaces.Box)
        token_high = int(np.asarray(observation_space.high)[0])
        self._n_token_ids = token_high + 1
        self._beg_id = self._n_token_ids
        self._token_emb = nn.Embedding(self._n_token_ids + 1, d_model, padding_idx=0)
        self._pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=lambda x: F.leaky_relu(x),
            batch_first=True,
            device=device,
        )
        self._transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model, eps=1e-5, device=device),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        tokens = obs[:, :-1].long()
        bs = tokens.shape[0]
        beg = th.full((bs, 1), self._beg_id, dtype=th.long, device=tokens.device)
        tokens = th.cat((beg, tokens), dim=1)
        pad_mask = tokens == 0
        x = self._pos_enc(self._token_emb(tokens))
        h = self._transformer(x, src_key_padding_mask=pad_mask)
        return h.mean(dim=1)


class MultiHeadMaskablePolicy(MaskableActorCriticPolicy):
    """Maskable PPO policy with separate actor/value heads selected by obs head_id."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = MultiHeadTransformerFeatures,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        simple_bias: float = 0.4,
        ts_bias: float = 0.5,
        temperatures: Optional[Dict[str, float]] = None,
    ):
        self.head_names = tuple(HEAD_NAMES)
        self.simple_bias = float(simple_bias)
        self.ts_bias = float(ts_bias)
        self.temperatures_cfg = temperatures or {"base": 1.0, "simple": 0.9, "ts": 1.0, "explore": 1.3}
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch or [],
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        action_dim = int(self.action_space.n)
        self.action_nets = nn.ModuleList([nn.Linear(latent_dim_pi, action_dim) for _ in self.head_names])
        self.value_nets = nn.ModuleList([nn.Linear(latent_dim_vf, 1) for _ in self.head_names])
        self.register_buffer("head_bias", self._build_head_bias(action_dim))
        self.register_buffer("head_temperature", self._build_head_temperature())

        # Keep attributes expected by SB3 serialization/introspection.
        self.action_net = self.action_nets[0]
        self.value_net = self.value_nets[0]

        if self.ortho_init:
            module_gains = {self.features_extractor: np.sqrt(2), self.mlp_extractor: np.sqrt(2)}
            for module in self.action_nets:
                module_gains[module] = 0.01
            for module in self.value_nets:
                module_gains[module] = 1
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build_head_temperature(self) -> th.Tensor:
        return th.tensor(
            [float(self.temperatures_cfg.get(name, 1.0)) for name in self.head_names],
            dtype=th.float32,
        )

    def _build_head_bias(self, action_dim: int) -> th.Tensor:
        bias = th.zeros((len(self.head_names), action_dim), dtype=th.float32)
        complex_ops = {"Corr", "Cov", "Div", "Log", "Std", "Var", "Mad"}
        ts_ops = {"Ref", "Delta", "Mean", "Std", "Corr", "Cov", "WMA", "EMA", "Sum"}
        for idx, op in enumerate(OPERATORS):
            action_idx = OFFSET_OP + idx - 1
            if action_idx < 0 or action_idx >= action_dim:
                continue
            name = op.__name__
            if name in complex_ops:
                bias[self.head_names.index("simple"), action_idx] -= self.simple_bias
            if name in ts_ops:
                bias[self.head_names.index("ts"), action_idx] += self.ts_bias
        return bias

    def _head_ids(self, obs: th.Tensor) -> th.Tensor:
        return obs[:, -1].long().clamp(0, len(self.head_names) - 1)

    def _select_by_head(self, modules: nn.ModuleList, latent: th.Tensor, head_ids: th.Tensor) -> th.Tensor:
        outputs = th.stack([module(latent) for module in modules], dim=1)
        gather_idx = head_ids.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
        return outputs.gather(1, gather_idx).squeeze(1)

    def _dist_from_obs_latent(self, obs: th.Tensor, latent_pi: th.Tensor) -> MaskableDistribution:
        head_ids = self._head_ids(obs)
        logits = self._select_by_head(self.action_nets, latent_pi, head_ids)
        logits = logits + self.head_bias[head_ids]
        logits = logits / self.head_temperature[head_ids].unsqueeze(1).clamp_min(1e-6)
        return self.action_dist.proba_distribution(action_logits=logits)

    def _values_from_obs_latent(self, obs: th.Tensor, latent_vf: th.Tensor) -> th.Tensor:
        return self._select_by_head(self.value_nets, latent_vf, self._head_ids(obs))

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self._values_from_obs_latent(obs, latent_vf)
        distribution = self._dist_from_obs_latent(obs, latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._dist_from_obs_latent(obs, latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return self._values_from_obs_latent(obs, latent_vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        features = self.extract_features(obs)
        if not self.share_features_extractor:
            features = features[0]
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._dist_from_obs_latent(obs, latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if not self.share_features_extractor:
            features = features[1]
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self._values_from_obs_latent(obs, latent_vf)
