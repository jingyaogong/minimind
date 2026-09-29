"""Inference-only MiniMind implementation for vLLM 0.17.x.

Parameter names match the exported Transformers checkpoint. Gated Delta Net
recurrent state is allocated and scheduled by vLLM's hybrid Mamba cache.
"""

from __future__ import annotations

import math
from itertools import islice
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class MiniMindRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normalized = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * normalized).to(x.dtype)


def _precompute_rope(config) -> tuple[torch.Tensor, torch.Tensor]:
    dim = config.head_dim
    base = config.rope_theta
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    scaling = getattr(config, "rope_scaling", None)
    attention_factor = 1.0
    if scaling is not None:
        original = scaling.get("original_max_position_embeddings", 2048)
        factor = scaling.get("factor", 16)
        beta_fast = scaling.get("beta_fast", 32.0)
        beta_slow = scaling.get("beta_slow", 1.0)

        def correction_dim(beta: float) -> float:
            return (dim * math.log(original / (beta * 2 * math.pi))) / (2 * math.log(base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim // 2 - 1)
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low)
            / max(high - low, 0.001),
            0,
            1,
        )
        inv_freq = inv_freq * (1 - ramp + ramp / factor)
        attention_factor = scaling.get("attention_factor", 1.0)

    positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
    phase = torch.outer(positions, inv_freq)
    phase = torch.cat((phase, phase), dim=-1)
    return phase.cos() * attention_factor, phase.sin() * attention_factor


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    pos = positions.reshape(-1).long()
    cos_pos = cos.index_select(0, pos).unsqueeze(1)
    sin_pos = sin.index_select(0, pos).unsqueeze(1)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos_pos + rotated * sin_pos).to(x.dtype)


class MiniMindFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniMindFullAttention(nn.Module):
    def __init__(self, config, cache_config: CacheConfig, prefix: str):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = MiniMindRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = MiniMindRMSNorm(self.head_dim, config.rms_norm_eps)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x, positions, cos, sin):
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_kv_heads * self.head_dim)
        q = _apply_rope(self.q_norm(q), positions, cos, sin)
        k = _apply_rope(self.k_norm(k), positions, cos, sin)
        y = self.attn(q.flatten(1), k.flatten(1), v)
        return self.o_proj(y)


class MiniMindGatedDeltaNet(MambaBase):
    """MiniMind GDN layer using vLLM-managed per-sequence recurrent state."""

    def __init__(self, config, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.model_config: ModelConfig = vllm_config.model_config
        self.cache_config: CacheConfig = vllm_config.cache_config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.beta_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.decay_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.decay_bias = nn.Parameter(torch.full((self.num_heads,), float(config.linear_decay_bias)))
        self.q_norm = MiniMindRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = MiniMindRMSNorm(self.head_dim, config.rms_norm_eps)
        self.out_norm = MiniMindRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.kv_cache: tuple[torch.Tensor, ...] = ()

    @property
    def mamba_type(self):
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        # MiniMind has no short convolution state. A zero-sized conv entry keeps
        # the GDN backend's standard two-state layout; the second is [H, D, D].
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, self.num_heads, self.num_heads, self.head_dim, self.head_dim, 1, 0
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def forward(self, x, positions, cos, sin):
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        q = _apply_rope(self.q_norm(q), positions, cos, sin)
        k = _apply_rope(self.k_norm(k), positions, cos, sin)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        q = F.normalize(q.float(), p=2, dim=-1).to(x.dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(x.dtype)
        beta = torch.sigmoid(self.beta_proj(x))
        log_decay = -F.softplus(
            self.decay_proj(x).float() + self.decay_bias.float()
        ).to(x.dtype)

        forward_context = get_forward_context()
        raw_metadata = forward_context.attn_metadata
        if raw_metadata is None:
            # The engine runs a metadata-free profile pass before cache binding.
            return x.new_zeros((n, self.num_heads * self.head_dim))
        metadata = raw_metadata[self.prefix] if isinstance(raw_metadata, dict) else raw_metadata
        if not isinstance(metadata, GDNAttentionMetadata):
            raise TypeError(f"Expected GDN metadata for {self.prefix}, got {type(metadata)!r}")
        if metadata.spec_sequence_masks is not None:
            raise RuntimeError("Disable speculative decoding for the MiniMind adapter")
        if not self.kv_cache:
            raise RuntimeError(f"The vLLM state cache was not bound for {self.prefix}")

        # vLLM may expose state tensors directly or grouped by virtual engine.
        cache = self.kv_cache
        if not isinstance(cache[0], torch.Tensor):
            cache = cache[forward_context.virtual_engine]

        actual = metadata.num_actual_tokens
        q, k, v = (t[:actual].unsqueeze(0) for t in (q, k, v))
        beta = beta[:actual].unsqueeze(0)
        log_decay = log_decay[:actual].unsqueeze(0)
        ssm_state = cache[1]

        if metadata.num_prefills > 0:
            state_indices = metadata.non_spec_state_indices_tensor
            initial_state = ssm_state[state_indices].contiguous()
            initial_state[~metadata.has_initial_state, ...] = 0
            recurrent, final_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=log_decay,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=metadata.non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=False,
            )
            ssm_state[state_indices] = final_state.to(ssm_state.dtype)
        elif metadata.num_decodes > 0:
            state_indices = metadata.non_spec_state_indices_tensor
            recurrent, _ = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=log_decay,
                beta=beta,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=metadata.non_spec_query_start_loc[: metadata.num_decodes + 1],
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            recurrent = q.new_zeros(q.shape)

        recurrent = self.out_norm(recurrent.squeeze(0)).reshape(actual, -1)
        output = self.o_proj(recurrent)
        if actual < n:
            output = F.pad(output, (0, 0, 0, n - actual))
        return output


class MiniMindDecoderLayer(nn.Module):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str, layer_type: str):
        super().__init__()
        if layer_type == "linear":
            self.self_attn = MiniMindGatedDeltaNet(
                config, vllm_config, prefix=f"{prefix}.self_attn"
            )
        elif layer_type == "full":
            self.self_attn = MiniMindFullAttention(
                config, vllm_config.cache_config, prefix=f"{prefix}.self_attn"
            )
        else:
            raise ValueError(f"Unsupported MiniMind layer type {layer_type!r}")
        self.input_layernorm = MiniMindRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MiniMindRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MiniMindFeedForward(config)

    def forward(self, hidden_states, positions, cos, sin):
        attn_out = self.self_attn(self.input_layernorm(hidden_states), positions, cos, sin)
        hidden_states = hidden_states + attn_out
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class MiniMindModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        def make_layer(layer_prefix: str):
            index = extract_layer_index(layer_prefix)
            return MiniMindDecoderLayer(
                config, vllm_config, layer_prefix, config.layer_types[index]
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            make_layer,
            prefix=maybe_prefix(prefix, "layers"),
        )
        self.norm = (
            MiniMindRMSNorm(config.hidden_size, config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )
        cos, sin = _precompute_rope(config)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None):
        pp = get_pp_group()
        if pp.is_first_rank:
            if input_ids is None:
                raise ValueError("input_ids is required on the first pipeline rank")
            hidden_states = self.embed_input_ids(input_ids)
        else:
            if intermediate_tensors is None:
                raise ValueError("intermediate_tensors is required on non-first ranks")
            hidden_states = intermediate_tensors["hidden_states"]

        cos = self.freqs_cos.to(hidden_states.device)
        sin = self.freqs_sin.to(hidden_states.device)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states, positions, cos, sin)
        if not pp.is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return self.norm(hidden_states)


class MiniMindForCausalLM(nn.Module, HasInnerState, SupportsPP, IsHybrid):
    has_inner_state = True
    is_hybrid = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = vllm_config.model_config.hf_text_config
        self.model = MiniMindModel(vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            raise NotImplementedError("This text-only MiniMind adapter does not accept inputs_embeds")
        return self.model(input_ids, positions, intermediate_tensors)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name == "lm_head.weight":
                name = "model.embed_tokens.weight"
            param = params.get(name)
            if param is None:
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor)
            loaded.add(name)

        missing = sorted(set(params) - loaded)
        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise ValueError(f"Checkpoint misses {len(missing)} parameters: {preview}{suffix}")
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            config.num_attention_heads,
            config.num_attention_heads,
            config.head_dim,
            config.head_dim,
            1,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
