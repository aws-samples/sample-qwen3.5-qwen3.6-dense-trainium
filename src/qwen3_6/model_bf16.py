# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5/3.6 dense hybrid (DeltaNet + GQA) model — BF16 implementation.

Serves the whole dense family; dims and the layer split come from the HF
config. Example (Qwen3.5-27B): 64 layers as [3 DeltaNet + 1 GQA] x 16.

What this file implements:
- Hand-written RMSNorm and partial RoPE, plus full-attention (GQA) layers on
  NF.qkv_proj / NF.flash_attention / NF.o_proj, for both prefill and decode
  (decode attention is written out — see `forward_decode`).
- Dense SwiGLU MLP with NF.mlp on all layers (TP-sharded intermediate).
- DeltaNet linear-attention layers, wrapping PR #152's fused NKI kernel
  verbatim (kept in `nki_kernels/deltanet_fused.py`):
    - prefill via the chunked NKI kernel;
    - decode via a single-step PyTorch recurrent update;
    - recurrent/conv state carried in side-channel buffers on the layer
      (`recurrent_state_buffer`, `conv_state_buffer`), not the paged KV
      cache — the KV spec reports dummy 1x1 heads for DeltaNet layers.
- DecoderLayer dispatch by `config.layer_types[layer_idx]`.

Parallelism:
- GQA, MLP, and DeltaNet layers are all TP-sharded; DeltaNet shards by value
  head. See `Qwen3_6DeltaNetAttention` for the strategy and the three
  divisibility guards its `__init__` raises.

This file follows `vllm_neuron.model.qwen3_vl.model_bf16` for its GQA,
model-backbone, cache-interface and weight-loading structure — including the
vocab-sharded embedding, sequence parallelism, and lm_head, with tie state and
tensor prefix auto-detected in `load_weights`;
`vllm_neuron.model.llama3.model` for the dense MLP; and PR #152's
GatedDeltaNet (HF `Qwen3_5GatedDeltaNet`) for the DeltaNet pipeline.

vllm-neuron runner contracts (each note below explains a contract this file
has to satisfy; see the README's Compatibility Matrix for the pinned stack):
- Official Qwen3.5/3.6 HF configs carry `mrope_section` in
  `rope_parameters`, so vLLM marks the model `uses_mrope=True`. The
  runner then requires the `SupportsMRoPE` protocol at request init and
  passes a `rotary_position_ids` kwarg to forward(). We implement the
  protocol text-only (all three M-RoPE rows equal the 1-D position) and
  ignore `rotary_position_ids` in forward — this model's partial RoPE is
  computed from `positions` directly, which is identical for pure text.
- forward() accepts `is_token_ids` and `**kwargs` so the runner's
  prompt-embeds and async-spec kwargs don't TypeError (both unused).
- vLLM core registers this arch as multimodal; serve TEXT-ONLY with
  `--limit-mm-per-prompt '{"image": 0, "video": 0}'` so the runner skips
  the encoder-cache / vision-kwargs path (this model has no vision tower).
"""

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    last_dim_padding_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from .config import Qwen3_6Config

logger = logging.getLogger(__name__)


# ============================================================================
# Custom weight loaders for gated Qwen3.5/Qwen3.6 q_proj layouts
# ----------------------------------------------------------------------------
# On dense Qwen3.5/Qwen3.6 variants with attn_output_gate=True, HF
# safetensors store q_proj as [2*Q*head_dim, hidden], INTERLEAVED PER HEAD:
#     [h0_q(head_dim), h0_gate(head_dim), h1_q, h1_gate, ...]
# Thus head h's Q rows are [h*2*head_dim : h*2*head_dim + head_dim], and
# its gate rows are the following head_dim rows. For the 27B configuration,
# q_proj.weight has shape (12288, 5120) instead of the ungated
# (6144, 5120).
#
# We can't use the stock fused_qkv_weight_loader: it takes one contiguous
# q_size-row block per rank, which assumes the Q slice's leading dim is
# n_q*head_dim. With Q and gate interleaved per head that block straddles
# gate rows. So we write a thin wrapper that builds a custom
# SafetensorsWeightLoader.
# ============================================================================


def _spliced_q_kv_loader(kv_size_full, num_shards, num_kv_replicas,
                         head_dim, num_heads):
    """Loader for `qkv_proj_weight`: extracts the Q sub-slice of each head
    from q_proj (which interleaves Q and gate PER HEAD), plus K + V.

    On gated dense Qwen3.5/Qwen3.6 variants, q_proj on disk has shape
    (num_heads * head_dim * 2, hidden), laid out per-head INTERLEAVED:
        [h0_q(head_dim), h0_gate(head_dim), h1_q, h1_gate, ...]
    (HF: `q_proj(x).view(*input_shape, -1, head_dim*2)` then chunk(2, dim=-1).)

    So Q for head h is rows [h*2*head_dim : h*2*head_dim + head_dim].
    We gather the Q rows for this rank's head slice, then concat K + V.
    Returns storage-transposed (hidden, qkv_per_rank).
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    assert num_heads % num_shards == 0
    heads_per_rank = num_heads // num_shards
    kv_per_rank = kv_size_full // max(1, (num_shards // max(1, num_kv_replicas)))

    def transform(slices, rank):
        assert len(slices) == 3, "expected (Q, K, V) slices"
        q_slice, k_slice, v_slice = slices
        q_rank = rank % num_shards
        kv_rank = q_rank // max(1, num_kv_replicas)

        # Gather Q rows for this rank's heads (skip the gate half of each head).
        first_head = q_rank * heads_per_rank
        q_rows = []
        for h in range(first_head, first_head + heads_per_rank):
            base = h * 2 * head_dim          # start of this head's [q|gate] block
            q_rows.append(q_slice[base : base + head_dim, :])  # the Q half only
        q_t = torch.cat(q_rows, dim=0)       # (heads_per_rank*head_dim, hidden)

        kv_start = kv_rank * kv_per_rank
        kv_end = kv_start + kv_per_rank
        k_t = k_slice[kv_start:kv_end, :]
        v_t = v_slice[kv_start:kv_end, :]

        cat = torch.cat([q_t, k_t, v_t], dim=0)
        return cat.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _spliced_q_gate_loader(num_shards, head_dim, num_heads):
    """Loader for `attn_gate_weight`: extracts the GATE sub-slice of each head.

    Gate for head h is rows [h*2*head_dim + head_dim : (h+1)*2*head_dim].
    Returns storage-transposed (hidden, q_per_rank).
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    assert num_heads % num_shards == 0
    heads_per_rank = num_heads // num_shards

    def transform(slices, rank):
        assert len(slices) == 1, "expected single q_proj slice"
        q_slice = slices[0]
        q_rank = rank % num_shards
        first_head = q_rank * heads_per_rank
        g_rows = []
        for h in range(first_head, first_head + heads_per_rank):
            base = h * 2 * head_dim
            g_rows.append(q_slice[base + head_dim : base + 2 * head_dim, :])  # gate half
        g_t = torch.cat(g_rows, dim=0)
        return g_t.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _deltanet_qkv_block_loader(key_dim, key_dim_local, value_dim_local, num_shards):
    """Loader for DeltaNet's fused `in_proj_qkv_weight` AND `conv1d_weight`.

    Unlike GQA's `_spliced_q_kv_loader` (three source tensors), this receives
    a SINGLE fused checkpoint tensor whose dim 0 is laid out as
    [Q(key_dim) | K(key_dim) | V(value_dim)]. Head-sharding by value head
    means rank r takes its key-head rows out of the Q and K blocks and its
    value-head rows out of the V block, concatenated back in [q|k|v] order:

        Q rows: [r*key_dim_local   : (r+1)*key_dim_local]
        K rows: key_dim   + [r*key_dim_local   : (r+1)*key_dim_local]
        V rows: 2*key_dim + [r*value_dim_local : (r+1)*value_dim_local]

    conv1d_weight [conv_dim, 1, K] is depthwise over the same concatenated
    [q|k|v] channels, so the identical dim-0 split applies (trailing dims
    pass through untouched).

    NOT storage-transposed: DeltaNet weights are [out, in] and used via
    plain F.linear(x, W).
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    def transform(slices, rank):
        assert len(slices) == 1, "expected single fused [Q|K|V] tensor"
        src = slices[0]
        ndim = len(src.get_shape())
        r = rank % num_shards

        def rows(start, end):
            sl = [slice(None)] * ndim
            sl[0] = slice(start, end)
            return src[tuple(sl)]

        q = rows(r * key_dim_local, (r + 1) * key_dim_local)
        k = rows(key_dim + r * key_dim_local, key_dim + (r + 1) * key_dim_local)
        v = rows(
            2 * key_dim + r * value_dim_local,
            2 * key_dim + (r + 1) * value_dim_local,
        )
        return torch.cat([q, k, v], dim=0)

    return SafetensorsWeightLoader(transform=transform)


# ============================================================================
# Section 1: RMSNorm
# Unpadded RMSNorm using the Qwen3.5/3.6 `(1 + weight)` convention.
# ============================================================================


class Qwen3_6RMSNorm(nn.Module):
    """RMS Normalization with the Qwen3.5/3.6 `(1 + weight)` convention.

    Weights are stored centered on ZERO and applied as
    `output * (1.0 + weight)`, not the standard `output * weight` (matches HF
    `modeling_qwen3_5.Qwen3_5RMSNorm`).

    Used for input_layernorm, post_attention_layernorm, q_norm, k_norm, and
    the final backbone `model.norm`. The gated DeltaNet output norm uses a
    DIFFERENT convention (plain `weight *`, HF `Qwen3_5RMSNormGated`) and is
    handled inside the DeltaNet block — do NOT route it through this class.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        # Stored centered at 0 (the checkpoint weights are ~0, not ~1).
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        return x.to(input_dtype)


# ============================================================================
# Section 2: Rotary Position Embedding (Partial RoPE)
# Only the first `partial_rotary_factor * head_dim` dims are rotated
# (this family: 25%, so 64 of head_dim 256).
# ============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NeoX-style half-and-half RoPE rotation (matches HF qwen3_5 reference)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE only to the first `rotary_dim` dimensions of head_dim.

    Matches HF `transformers.models.qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb`:
    standard NeoX rotate_half (NOT GPT-J interleaved pairing). The
    `mrope_interleaved` config flag affects only how the cos/sin
    *frequencies* are assembled across the 3 mRoPE sections — for
    text-only input all three sections share the sequence position, so
    the frequencies reduce to standard RoPE and the rotation is plain
    rotate_half.

    `cos`/`sin` come in as [T, rotary_dim/2]; we double them to
    [T, rotary_dim] via cat (HF builds them at full rotary_dim width
    directly, equivalent result).

    Args:
        q, k: [..., head_dim]
        cos, sin: [T, rotary_dim/2]
        rotary_dim: number of leading dims of head_dim to rotate
        interleaved: kept for signature stability; ignored (always NeoX).
    """
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_full = torch.cat((cos, cos), dim=-1).unsqueeze(0)  # [1, T, rotary_dim]
    sin_full = torch.cat((sin, sin), dim=-1).unsqueeze(0)  # [1, T, rotary_dim]

    q_rot = (q_rot * cos_full) + (rotate_half(q_rot) * sin_full)
    k_rot = (k_rot * cos_full) + (rotate_half(k_rot) * sin_full)

    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class Qwen3_6RotaryEmbedding(nn.Module):
    """Partial RoPE: produces (cos, sin) for the first `rotary_dim` dims.

    rotary_dim = head_dim * partial_rotary_factor.
    """

    def __init__(self, config: Qwen3_6Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        # partial RoPE
        self.rotary_dim = int(round(config.head_dim * config.partial_rotary_factor))
        if self.rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be even; got {self.rotary_dim} "
                f"(head_dim={self.head_dim}, factor={config.partial_rotary_factor})"
            )

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device="cpu")
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :].float()  # [1, rot/2]
        position_ids_expanded = position_ids[:, None].float()  # [T, 1]
        freqs = position_ids_expanded @ inv_freq_expanded  # [T, rot/2]
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


# ============================================================================
# Section 3: Full-Attention (GQA) Layer — the config-selected full-attention
# layers (16 of 64 on the 27B). Uses the public qwen3_vl GQA/TP structure,
# adapted for:
#   - Partial RoPE (only first `rotary_dim` of head_dim)
#   - Optional sigmoid attention-output gate
# ============================================================================


class Qwen3_6GQAAttention(nn.Module):
    """GQA attention with TP head sharding and partial RoPE.

    Uses qwen3_vl-style TP GQA structure with the Qwen3.5/3.6 partial-RoPE
    and optional attention-output-gate behavior.
    """

    def __init__(self, config: Qwen3_6Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim ** -0.5
        # rotary applied only to part of head_dim
        self.rotary_dim = int(round(config.head_dim * config.partial_rotary_factor))
        # Optional sigmoid gate on attention output.
        # On gated dense Qwen3.5/Qwen3.6 variants, q_proj.weight stores
        # Q and gate rows interleaved inside each head's 2*head_dim block:
        # [h0_q, h0_gate, h1_q, h1_gate, ...]. It does not append all gate
        # rows as a global second half. We honour `config.attn_output_gate`
        # here; the custom loaders in _setup_weight_loaders gather each
        # local head's Q sub-slice into qkv_proj_weight and its following
        # gate sub-slice into attn_gate_weight.
        self.attn_output_gate = bool(getattr(config, "attn_output_gate", False))

        # TP group setup
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # Head sharding (replicate KV when fewer than world_size)
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // self.world_size
        )
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        # QKV / O weights
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_in = (self.num_attention_heads * self.head_dim) // self.world_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_in, self.hidden_size, dtype=self.dtype)
        )

        # Per-head Q/K layernorm (Qwen3 family)
        self.q_layernorm = Qwen3_6RMSNorm(self.head_dim, self.rms_norm_eps, self.dtype)
        self.k_layernorm = Qwen3_6RMSNorm(self.head_dim, self.rms_norm_eps, self.dtype)

        # gate projection on attention output
        # PR #152: g = sigmoid(linear(hidden)); attn_out = g * attn_out
        # Gate is full Q width: one value per (head, head_dim) element.
        if self.attn_output_gate:
            self.attn_gate_weight = nn.Parameter(
                torch.empty(self.hidden_size, q_size, dtype=self.dtype)
            )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.k_cache = None
        self.v_cache = None

        # FP8 KV cache scales. Buffers, not Parameters: the checkpoint ships
        # no `k_scale`/`v_scale` keys, and Parameters would trip strict weight
        # loading. Scale 32 keeps typical BF16 K/V (~[-3, 3]) inside FP8 e4m3
        # range without saturating; a runtime-calibrated per-layer scale is
        # the follow-up. The float copies avoid Dynamo graph breaks
        # (tensor.item() is host-side). This path is unverified end-to-end —
        # see docs/known-boundaries.md §2.
        self.register_buffer(
            "k_scale",
            torch.tensor(32.0, dtype=torch.float32, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "v_scale",
            torch.tensor(32.0, dtype=torch.float32, device="cpu"),
            persistent=False,
        )
        self.k_scale_float = 32.0
        self.v_scale_float = 32.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        if self.attn_output_gate:
            # Qwen3.6 spliced layout: q_proj on disk is
            # (num_heads * head_dim * 2, hidden) interleaved PER HEAD as
            # [h0_q, h0_gate, h1_q, h1_gate, ...].
            kv_size_full = self.num_key_value_heads * self.head_dim
            set_weight_loader(
                self.qkv_proj_weight,
                _spliced_q_kv_loader(
                    kv_size_full=kv_size_full,
                    num_shards=self.world_size,
                    num_kv_replicas=self.num_kv_replicas,
                    head_dim=self.head_dim,
                    num_heads=self.num_attention_heads,
                ),
            )
        else:
            set_weight_loader(
                self.qkv_proj_weight,
                fused_qkv_weight_loader(
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    shard_dim=1,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                    num_kv_replicas=self.num_kv_replicas,
                ),
            )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=(self.num_attention_heads * self.head_dim)
                // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        if self.attn_output_gate:
            # Spliced gate: load the gate sub-slice of each head from q_proj.
            set_weight_loader(
                self.attn_gate_weight,
                _spliced_q_gate_loader(
                    num_shards=self.world_size,
                    head_dim=self.head_dim,
                    num_heads=self.num_attention_heads,
                ),
            )

    # ── Forward dispatch ────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ── Prefill ─────────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # 1. Fused QKV
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

        # 2. QK-norm, then partial RoPE
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        cos, sin = position_embeddings
        # partial RoPE (only first `rotary_dim` of head_dim)
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        # 3. Update KV cache (per-rank)
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        # FP8 KV write — quantize when the cache is FP8, otherwise plain cast.
        k_raw = k.reshape(-1, self.head_dim)
        v_raw = v.reshape(-1, self.head_dim)
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_flat = (k_raw * self.k_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.k_cache.dtype)
            v_flat = (v_raw * self.v_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.v_cache.dtype)
        else:
            k_flat = k_raw.to(self.k_cache.dtype)
            v_flat = v_raw.to(self.v_cache.dtype)

        head_indices = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_idx_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        pos_idx_put = position_indices.repeat(self.num_key_value_heads_per_rank)

        self.k_cache.index_put_((block_idx_put, head_indices, pos_idx_put), k_flat)
        self.v_cache.index_put_((block_idx_put, head_indices, pos_idx_put), v_flat)

        # 4. Flash attention (no sinks, no SWA)
        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.repeat_interleave(self.num_key_value_groups, dim=0)

        attn_output = NF.flash_attention(
            q.transpose(1, 2),  # [Nh, Dh, T]
            k.transpose(1, 2),  # [Nh, Dh, T]
            v,                  # [Nh, T, Dh]
            scale=self.scaling,
            tp_q=False,
            tp_out=True,
        )

        # sigmoid output gate (verified against HF qwen3_5 reference:
        # the attention output gate is sigmoid; output_gate_type="swish"
        # refers to the DeltaNet gated norm, not this gate)
        if self.attn_output_gate:
            # gate = sigmoid(hidden @ W_gate)  → shape [tokens, q_size_per_rank]
            gate = torch.sigmoid(hidden_states @ self.attn_gate_weight)
            # Reshape to match attn_output layout from NF.flash_attention: [Nh, Dh, T]
            gate = gate.view(tokens, self.num_attention_heads_per_rank, self.head_dim)
            # (tokens, Nh, Dh) → (Nh, Dh, tokens) to broadcast with attn_output
            attn_output = attn_output * gate.permute(1, 2, 0)

        # 5. O-proj + reduce-scatter
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight)
        attn_output = attn_output.squeeze(0)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode ──────────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode path — NF.qkv_proj -> RMSNorm -> partial RoPE -> read prior
        KV via block_table -> masked split-K/split-V attention -> NF.o_proj.

        Two vllm_neuron kernels are unusable here, which is why the attention
        itself is written out rather than delegated:
        - `NF.attention_decode` takes a mask, but this family's 256-wide head
          trips the NKI tensor-engine limit underneath it ("Tensor engine
          transpose requires shape <= [128, 128]").
        - `NF.flash_attention` takes no mask tensor (only a `causal_mask`
          bool), and its built-in
          single-stream causal triangle is wrong when several requests share
          the gathered K/V tensor.

        Everything else stays inside NF (storage layout, TP sharding).
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        S_ctx = max_blocks_per_seq * block_size
        Nh = self.num_attention_heads_per_rank
        Nkh = self.num_key_value_heads_per_rank
        Dh = self.head_dim

        hidden_states = hidden_states.to(self.dtype)

        # 1) Fused QKV via NF (handles storage layout + TP sharding)
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, Nh, Dh).transpose(0, 1)   # [Nh, T, Dh]
        k = k.view(tokens, Nkh, Dh).transpose(0, 1)  # [Nkh, T, Dh]
        v = v.view(tokens, Nkh, Dh).transpose(0, 1)  # [Nkh, T, Dh]

        # 2) Per-head RMSNorm
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        # 3) Partial RoPE (first rotary_dim entries only)
        cos, sin = position_embeddings
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        # 4) Write new K/V into the paged cache.
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]

        k_new_flat = k.reshape(-1, Dh)  # [Nkh*T, Dh]
        v_new_flat = v.reshape(-1, Dh)
        head_indices = torch.arange(
            Nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_idx_put = block_indices.repeat(Nkh)
        pos_idx_put = position_indices.repeat(Nkh)
        # FP8 KV write — quantize on FP8 cache, plain cast otherwise.
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_q = (k_new_flat * self.k_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.k_cache.dtype)
            v_q = (v_new_flat * self.v_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.v_cache.dtype)
        else:
            k_q = k_new_flat.to(self.k_cache.dtype)
            v_q = v_new_flat.to(self.v_cache.dtype)
        self.k_cache.index_put_((block_idx_put, head_indices, pos_idx_put), k_q)
        self.v_cache.index_put_((block_idx_put, head_indices, pos_idx_put), v_q)

        # 5) Gather full prior K/V context for each request via block_table.
        # k_cache is [num_blocks, Nkh, block_size, Dh]
        # block_table is [B, max_blocks_per_seq] of block ids
        # The batch axis stays separate: flattening to a single [B*S_ctx]
        # stream would let one request attend into another's KV.
        # On an FP8 cache, K's dequant scale is folded into the softmax
        # scaling factor (the Llama3 trick) instead of materializing a
        # dequantized K_full, since
        #   (Q @ (K_raw/k_scale).T) * s  ==  (Q @ K_raw.T) * (s / k_scale)
        # V is dequantized explicitly — folding it would mean baking the
        # scale into o_proj_weight.
        K_raw = self.k_cache[block_table]   # [B, MB, Nkh, BS, Dh]
        V_raw = self.v_cache[block_table]
        K_raw = K_raw.permute(0, 2, 1, 3, 4).reshape(B, Nkh, S_ctx, Dh)
        V_raw = V_raw.permute(0, 2, 1, 3, 4).reshape(B, Nkh, S_ctx, Dh)
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Just cast to bf16; do NOT divide by scale here.
            K_full = K_raw.to(self.dtype)
            V_full = V_raw.to(self.dtype) / self.v_scale  # V dequant still explicit
            # Compute the effective scale: scaling / k_scale folded into one float.
            effective_scale = self.scaling / self.k_scale_float
        else:
            K_full = K_raw.to(self.dtype)
            V_full = V_raw.to(self.dtype)
            effective_scale = self.scaling

        # 6) Repeat KV for GQA grouping
        if self.num_key_value_groups > 1:
            K_full = K_full.repeat_interleave(self.num_key_value_groups, dim=1)
            V_full = V_full.repeat_interleave(self.num_key_value_groups, dim=1)
        # K_full, V_full now [B, Nh, S_ctx, Dh]

        # 7) Masked attention with split-K / split-V, written out because
        # neither NF decode kernel fits head_dim=256 (see the docstring).
        # Each 256-wide head is split into two 128-wide halves:
        #   scores = (Q_lo @ K_lo^T + Q_hi @ K_hi^T) * scale
        #   weights = softmax(scores + mask_bias)
        #   out    = cat([weights @ V_lo, weights @ V_hi], dim=-1)
        # which is the same math as the unsplit form, but every matmul has a
        # 128 reduction dim — inside the tensor engine's transpose limit, so
        # the compiler emits PSUM-accumulating nc_matmul calls and fuses the
        # whole chain into one NEFF. Pure PyTorch; torch.compile lowers it.

        # 7a) Reshape Q from [Nh, B*S_decode, Dh] -> [B, Nh, S_decode, Dh]
        q_b = q.transpose(0, 1).reshape(B, S_decode, Nh, Dh).transpose(1, 2)

        # 7b) Per-request causal mask.
        positions_b = positions.view(B, S_decode)              # [B, S_decode]
        k_idx = torch.arange(S_ctx, device=positions.device)   # [S_ctx]
        mask = (k_idx[None, None, None, :] <= positions_b[:, None, :, None])

        # 7c) Split-K: Q_lo @ K_lo^T + Q_hi @ K_hi^T (see the note above).
        Dh_half = Dh // 2  # = 128 for head_dim=256
        q_lo = q_b[..., :Dh_half]                 # [B, Nh, S_decode, 128]
        q_hi = q_b[..., Dh_half:]                 # [B, Nh, S_decode, 128]
        k_lo = K_full[..., :Dh_half]              # [B, Nh, S_ctx, 128]
        k_hi = K_full[..., Dh_half:]              # [B, Nh, S_ctx, 128]

        scores = (
            torch.matmul(q_lo, k_lo.transpose(-2, -1))
            + torch.matmul(q_hi, k_hi.transpose(-2, -1))
        )                                          # [B, Nh, S_decode, S_ctx]
        scores = scores * effective_scale

        # Apply mask via additive bias.
        neg_bias = (~mask).to(scores.dtype) * -1e4
        scores = scores + neg_bias
        attn_weights = torch.softmax(scores, dim=-1)

        # 7d) Split-V matmul: weights @ V_lo  cat  weights @ V_hi
        v_lo = V_full[..., :Dh_half]              # [B, Nh, S_ctx, 128]
        v_hi = V_full[..., Dh_half:]              # [B, Nh, S_ctx, 128]
        attn_output_lo = torch.matmul(attn_weights, v_lo)   # [B, Nh, S_decode, 128]
        attn_output_hi = torch.matmul(attn_weights, v_hi)   # [B, Nh, S_decode, 128]
        attn_output = torch.cat([attn_output_lo, attn_output_hi], dim=-1)
        # attn_output: [B, Nh, S_decode, Dh]

        # Re-flatten to [Nh, tokens, Dh].
        attn_output = attn_output.transpose(0, 1).reshape(Nh, tokens, Dh)

        # 8) Optional sigmoid attention-output gate
        if self.attn_output_gate:
            gate = torch.sigmoid(hidden_states @ self.attn_gate_weight)
            gate = gate.view(tokens, Nh, Dh)
            attn_output = attn_output * gate.transpose(0, 1)

        # 9) O-projection via NF.
        # NF.o_proj expects [B, N, D, S] (batch, heads, head_dim, seq_len), but
        # attn_output is [Nh, tokens, Dh] = [N, S, D]: swap the last two dims
        # and add the batch axis. The transpose is load-bearing at batch > 1 —
        # at batch=1 an incorrect reshape here is harmless because S=1.
        attn_output = attn_output.transpose(1, 2).unsqueeze(0)   # [1, Nh, Dh, tokens]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight)  # → [1, tokens, hidden]
        attn_output = attn_output.squeeze(0)                       # [tokens, hidden]

        # 10) All-reduce across TP (NOT reduce_scatter — decode tokens may
        # not be divisible by world_size, e.g. batch=1 with TP=8).
        if self.world_size > 1:
            attn_output = self.tp_group.all_reduce(attn_output)

        return attn_output.contiguous()


# ============================================================================
# Section 4: DeltaNet (linear attention) Layer
# Wraps the validated PR #152 fused NKI kernel with a vllm_neuron-style
# nn.Module. The kernel itself is in `nki_kernels/deltanet_fused.py`
# (verbatim from PR #152 — never edit; fix wrappers instead).
# ============================================================================


def _deltanet_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization along `dim`. Q and K are l2-normed before the kernel.

    Matches the HF qwen3_5 reference `l2norm`: eps added INSIDE the rsqrt
    (`x * rsqrt(sum(x^2) + eps)`), not as a floor on the norm.
    """
    return x * torch.rsqrt(x.pow(2).sum(dim=dim, keepdim=True) + eps)


class Qwen3_6DeltaNetAttention(nn.Module):
    """GatedDeltaNet linear-attention layer.

    Wraps `qwen3_6.nki_kernels.call_deltanet_fused` (PR #152).
    Pipeline mirrors `NeuronGatedDeltaNet.forward` from PR #152:

        in_proj_qkv  → split into q, k, v
        in_proj_z    → output gate input
        in_proj_a    → softplus(a + dt_bias) → scale by -exp(A_log)  (g)
        in_proj_b    → sigmoid                         (beta)
        causal conv1d (kernel=4) on concat(q, k, v)    (mixed_post_conv)
        silu(...)
        reshape to (B, H_v, S, head_dim) — q and k expanded from H_k via repeat
        l2norm q and k; scale q by 1/sqrt(k_dim)
        pad S to multiple of 128
        per-(B*H) NKI fused chunked DeltaNet kernel call
        gather outputs
        RMSNorm over head_v_dim
        z gate: out = out * silu(z)
        out_proj  → hidden_size

    State lives in side-channel buffers on the layer
    (`recurrent_state_buffer`, `conv_state_buffer`) rather than the paged KV
    cache, addressed by request-stable slot rows — rationale and the bug it
    fixes are in docs/implementation-notes.md §3. Writes use an out-of-place
    `index_copy` then `.data.copy_()` — the `.data` form is what the aliasing
    override recognizes (§2), so do not "simplify" it — which touches only
    this step's rows while keeping the whole-buffer graph dependency the
    compiled graphs need.

    TP: head-sharded by VALUE head, including the recurrent and conv state.
    Heads are independent through the recurrence, per-head RMSNorm and z
    gate; only `out_proj` mixes them, so it is column-sharded and the partial
    contributions are summed across ranks (reduce_scatter on prefill,
    all_reduce on decode — as in the GQA/MLP layers).

    The forwards use ONLY the `*_local` dims; the full dims exist for the
    weight-loader slice offsets, so a stray full-dim reference in a forward
    is a shape error rather than silent corruption.
    """

    def __init__(self, config: Qwen3_6Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        self.num_v_heads = config.deltanet_num_v_heads      # 48
        self.num_k_heads = config.deltanet_num_k_heads      # 16
        self.head_k_dim = config.deltanet_k_head_dim        # 128
        self.head_v_dim = config.deltanet_v_head_dim        # 128
        self.conv_kernel_size = config.deltanet_conv_kernel_size  # 4

        # Full (checkpoint) dims. `key_dim` supplies the weight-loader slice
        # offsets; `value_dim`/`conv_dim` are kept for reference only.
        self.key_dim = self.head_k_dim * self.num_k_heads       # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads     # 6144
        self.conv_dim = self.key_dim * 2 + self.value_dim       # 10240

        # Head sharding by value head. Each rank must own whole key-head
        # groups so the k->v repeat_interleave stays rank-local.
        ws = self.world_size
        if self.num_v_heads % ws != 0 or self.num_k_heads % ws != 0:
            raise ValueError(
                f"DeltaNet head sharding requires num_v_heads "
                f"({self.num_v_heads}) and num_k_heads ({self.num_k_heads}) "
                f"to be divisible by TP world_size ({ws})"
            )
        self.v_heads_local = self.num_v_heads // ws
        self.k_heads_local = self.num_k_heads // ws
        if self.v_heads_local % self.k_heads_local != 0:
            raise ValueError(
                f"DeltaNet head sharding requires v_heads per rank "
                f"({self.v_heads_local}) to be a multiple of k_heads per "
                f"rank ({self.k_heads_local}) so each rank keeps whole "
                f"key-head groups (world_size={ws})"
            )
        self.kv_repeat = self.v_heads_local // self.k_heads_local
        self.key_dim_local = self.head_k_dim * self.k_heads_local
        self.value_dim_local = self.head_v_dim * self.v_heads_local
        self.conv_dim_local = self.key_dim_local * 2 + self.value_dim_local

        # PR #152 sanity: kernel assumes head_k_dim == head_v_dim == 128
        if self.head_k_dim != 128 or self.head_v_dim != 128:
            raise NotImplementedError(
                f"PR #152 fused DeltaNet kernel hardcodes head_dim=128. "
                f"Got head_k_dim={self.head_k_dim}, head_v_dim={self.head_v_dim}."
            )

        # Input projections — row-sharded by head (this rank's slice)
        self.in_proj_qkv_weight = nn.Parameter(
            torch.empty(self.conv_dim_local, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_z_weight = nn.Parameter(
            torch.empty(self.value_dim_local, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_a_weight = nn.Parameter(
            torch.empty(self.v_heads_local, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_b_weight = nn.Parameter(
            torch.empty(self.v_heads_local, self.hidden_size, dtype=self.dtype)
        )

        # Causal conv1d on concat(q, k, v) — depthwise, kernel=4
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim_local, 1, self.conv_kernel_size, dtype=self.dtype)
        )

        # Decay parameters
        self.A_log = nn.Parameter(torch.zeros(self.v_heads_local, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.ones(self.v_heads_local, dtype=torch.float32))

        # Output: per-head RMSNorm over head_v_dim (replicated — it acts
        # within a head), then column-sharded linear back to hidden.
        self.norm_weight = nn.Parameter(torch.ones(self.head_v_dim, dtype=self.dtype))
        self.out_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.value_dim_local, dtype=self.dtype)
        )

        # Recurrent/conv state buffers — the DeltaNet equivalent of a KV cache
        # (docs/implementation-notes.md §3). Buffers, not Parameters, so the
        # vllm_neuron loader (which walks named_parameters()) never sees them;
        # `persistent=False` additionally keeps them out of state_dict. This
        # state is zero-initialized at runtime, not loaded.
        #
        # IMPORTANT: explicit `device="cpu"` overrides the
        # `with torch.device("meta")` context that vllm_neuron uses to
        # build the model skeleton. Without it, these buffers land on
        # `meta` and `model.to(device)` later fails with "Cannot copy out
        # of meta tensor; no data!".
        nc = config.neuron_config
        # vllm-neuron exposes the decode batch-size cap via
        # `num_seqs_buckets` (a list); that is the path taken on this stack.
        # A scalar `max_batch_size` attribute is honored first if some config
        # flavor supplies one, but `NeuronConfig` does not define it.
        if nc is None:
            max_batch = 1
        elif hasattr(nc, "max_batch_size") and nc.max_batch_size is not None:
            max_batch = int(nc.max_batch_size)
        elif hasattr(nc, "num_seqs_buckets") and nc.num_seqs_buckets:
            max_batch = int(max(nc.num_seqs_buckets))
        else:
            # Nothing usable on the config — assume batch 1.
            max_batch = int(getattr(nc, "max_batch_size", 1) or 1)
        # KEEP THIS BUFFER FP32. It is the only state carried token-to-token
        # in decode; storing it bf16 re-quantizes every step, which compounds
        # and collapses generation (the HF qwen3_5 reference is fp32 too).
        #
        # Rows are indexed by request-stable slot, never by batch position.
        # The extra row (+1) is the scratch slot for padded decode rows and DP
        # dummy batches, and is never read back into a real request.
        self.state_scratch_slot = max_batch
        self.register_buffer(
            "recurrent_state_buffer",
            torch.zeros(
                max_batch + 1, self.v_heads_local, self.head_k_dim, self.head_v_dim,
                dtype=torch.float32, device="cpu",
            ),
            persistent=False,
        )
        self.register_buffer(
            "conv_state_buffer",
            torch.zeros(
                max_batch + 1, self.conv_dim_local, self.conv_kernel_size - 1,
                dtype=self.dtype, device="cpu",
            ),
            persistent=False,
        )

        # Pre-built per-layer DeltaNet masks. Stored as buffers so they
        # move with the module to the Neuron device, and so Dynamo can
        # see them as constants (no global-dict guard failure).
        # device="cpu" for the same meta-tensor reason as the state buffers.
        chunk = 128
        self.register_buffer(
            "deltanet_lower_mask",
            torch.tril(torch.ones(chunk, chunk, dtype=torch.float32, device="cpu"), diagonal=-1),
            persistent=False,
        )
        self.register_buffer(
            "deltanet_identity_mat",
            torch.eye(chunk, dtype=torch.float32, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "deltanet_lower_mask_diag",
            torch.tril(torch.ones(chunk, chunk, dtype=torch.float32, device="cpu"), diagonal=0),
            persistent=False,
        )

        # Dummy KV cache attrs to satisfy bind_kv_cache contract. We never
        # read these in the DeltaNet forward — the real state lives in the
        # buffers above.
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        # All DeltaNet weights are stored [out, in] and used via plain
        # F.linear(x, W) — NOT storage-transposed (unlike the GQA weights,
        # which are [in, out] for NF.qkv_proj). Row shards slice dim 0;
        # out_proj [hidden, value_dim] column-shards its INPUT dim (dim 1).
        qkv_block_loader = _deltanet_qkv_block_loader(
            key_dim=self.key_dim,
            key_dim_local=self.key_dim_local,
            value_dim_local=self.value_dim_local,
            num_shards=self.world_size,
        )
        set_weight_loader(self.in_proj_qkv_weight, qkv_block_loader)
        set_weight_loader(self.conv1d_weight, qkv_block_loader)
        set_weight_loader(
            self.in_proj_z_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.value_dim_local,
                num_shards=self.world_size,
            ),
        )
        v_head_row_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.v_heads_local,
            num_shards=self.world_size,
        )
        set_weight_loader(self.in_proj_a_weight, v_head_row_loader)
        set_weight_loader(self.in_proj_b_weight, v_head_row_loader)
        set_weight_loader(self.A_log, v_head_row_loader)
        set_weight_loader(self.dt_bias, v_head_row_loader)
        set_weight_loader(
            self.out_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=self.value_dim_local,
                num_shards=self.world_size,
            ),
        )
        # norm_weight is per-head-dim ([head_v_dim]) and replicated — no loader.

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Forward pass — dispatches between prefill (CTE, fused NKI kernel)
        and decode (TKG, single-step recurrent update in PyTorch).

        Both paths run this rank's value-head slice (see the class
        docstring's TP strategy note) and reduce the partial out_proj
        contributions across ranks at the end.

        `state_slots` (from attn_metadata) maps each sequence in this step to
        its request-stable row in the state buffers. Prefill steps carry one
        sequence → one slot; decode steps carry one slot per batch row, with
        padded rows pointing at the scratch slot.
        """
        state_slots = self._get_state_slots(attn_metadata, hidden_states)
        is_prefill = self._is_prefill(attn_metadata)
        if is_prefill:
            return self._forward_prefill(hidden_states, state_slots)
        return self._forward_decode(hidden_states, state_slots)

    def _get_state_slots(self, attn_metadata, hidden_states) -> torch.Tensor:
        """Resolve this step's request→buffer-row mapping.

        Falls back to identity (row i = batch position i) when the runner
        does not provide `state_slots` — correct for single-request use and
        for unit tests that call the layer directly.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        if attn_metadata is not None and layer_name in attn_metadata:
            slots = attn_metadata[layer_name].get("state_slots")
            if slots is not None:
                return slots
        if self._is_prefill(attn_metadata):
            n = 1  # prefill handles a single sequence
        else:
            n = hidden_states.shape[0]
        return torch.arange(n, dtype=torch.long, device=hidden_states.device)

    # ── Prefill (CTE) ────────────────────────────────────────────────────

    def _forward_prefill(
        self, hidden_states: torch.Tensor, state_slots: torch.Tensor
    ) -> torch.Tensor:
        """CTE prefill via fused NKI kernel.

        `state_slots` is a length-1 tensor: the buffer row owned by the
        single sequence being prefilled.
        """
        # SP: hidden states arrive sharded [T/ws, H] during prefill (the
        # embedding reduce-scatters the sequence). The DeltaNet recurrence is
        # sequential over the WHOLE sequence, so each rank must see all tokens
        # — a per-rank contiguous shard would start the recurrence with zero
        # state and drop cross-shard history. Gather the full sequence, run
        # the recurrence on this rank's head slice, then reduce_scatter the
        # partial hidden output (sum across ranks + back to SP layout).
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # vllm_neuron passes [tokens, hidden]; treat as [B=1, S=tokens, hidden]
        # and follow PR #152's [B, S, H] convention through the rest of the layer.
        tokens, hidden = hidden_states.shape
        batch_size = 1
        seq_len = tokens
        x = hidden_states.view(batch_size, seq_len, hidden).to(self.dtype)

        # 1. Project: QKV (fused), Z, A, B
        qkv = torch.nn.functional.linear(x, self.in_proj_qkv_weight)
        z = torch.nn.functional.linear(x, self.in_proj_z_weight)
        a = torch.nn.functional.linear(x, self.in_proj_a_weight)
        b = torch.nn.functional.linear(x, self.in_proj_b_weight)

        q_raw = qkv[..., : self.key_dim_local]
        k_raw = qkv[..., self.key_dim_local : self.key_dim_local * 2]
        v_raw = qkv[..., self.key_dim_local * 2 :]

        # 2. Causal Conv1d on concat(q, k, v), then SiLU
        mixed = torch.cat([q_raw, k_raw, v_raw], dim=-1)  # [B, S, conv_dim_local]
        mixed = mixed.transpose(1, 2)                      # [B, conv_dim_local, S]
        # Depthwise conv1d via grouped F.conv1d
        conv_out = torch.nn.functional.conv1d(
            mixed,
            self.conv1d_weight,                           # [conv_dim_local, 1, K]
            bias=None,
            stride=1,
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim_local,
        )[:, :, :seq_len]
        mixed_post_conv = torch.nn.functional.silu(conv_out).transpose(1, 2)
        # [B, S, conv_dim_local]

        # Split back to q, k, v
        q = mixed_post_conv[..., : self.key_dim_local]
        k = mixed_post_conv[..., self.key_dim_local : self.key_dim_local * 2]
        v = mixed_post_conv[..., self.key_dim_local * 2 :]

        # 3. Reshape to head layout
        q = q.reshape(batch_size, seq_len, self.k_heads_local, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.k_heads_local, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.v_heads_local, self.head_v_dim)

        # 4. Compute decay g and write-gate beta
        # g = -exp(A_log) * softplus(a + dt_bias)        [B, S, H_v] in fp32
        # beta = sigmoid(b)                              [B, S, H_v]
        a_f32 = a.float()
        g = -self.A_log.exp() * torch.nn.functional.softplus(a_f32 + self.dt_bias)
        beta = b.sigmoid().to(self.dtype)

        # 5. Expand the key-side heads (q and k) to this rank's V heads
        #    (e.g. 4 -> 12 local = repeat 3x).
        # The guard in __init__ ensures each rank owns whole key-head groups,
        # so the repeat stays rank-local.
        if self.kv_repeat > 1:
            q = (
                q.unsqueeze(3)
                .expand(-1, -1, -1, self.kv_repeat, -1)
                .reshape(batch_size, seq_len, self.v_heads_local, self.head_k_dim)
            )
            k = (
                k.unsqueeze(3)
                .expand(-1, -1, -1, self.kv_repeat, -1)
                .reshape(batch_size, seq_len, self.v_heads_local, self.head_k_dim)
            )

        # 6. Transpose to (B, H, S, dim), float32 for kernel
        q = q.transpose(1, 2).contiguous().float()
        k = k.transpose(1, 2).contiguous().float()
        v = v.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        # 7. l2norm Q and K, scale Q by 1/sqrt(k_dim)
        q = _deltanet_l2norm(q, dim=-1)
        k = _deltanet_l2norm(k, dim=-1)
        scale = 1.0 / (self.head_k_dim ** 0.5)
        q = q * scale

        # 8. Pad S to multiple of 128
        chunk = 128
        pad = (chunk - seq_len % chunk) % chunk
        if pad > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
            g = torch.nn.functional.pad(g, (0, pad))
            beta = torch.nn.functional.pad(beta, (0, pad))
        total_seq = seq_len + pad

        # 9. Flatten (B, H) -> per-(b,h) kernel calls.
        BH = batch_size * self.v_heads_local
        q_flat = q.reshape(BH, total_seq, self.head_k_dim).contiguous()
        k_flat = k.reshape(BH, total_seq, self.head_k_dim).contiguous()
        v_flat = v.reshape(BH, total_seq, self.head_v_dim).contiguous()
        g_flat = g.reshape(BH, total_seq).unsqueeze(-1).contiguous()
        beta_flat = beta.reshape(BH, total_seq).unsqueeze(-1).contiguous()

        # 10. Kernel masks (registered buffers — moved with the module).
        from .nki_kernels import call_deltanet_fused
        lower_mask = self.deltanet_lower_mask
        identity_mat = self.deltanet_identity_mat
        lower_mask_diag = self.deltanet_lower_mask_diag

        # 11. Per-(b,h) kernel calls
        outputs = []
        states = []
        for bh in range(BH):
            out_bh, state_bh = call_deltanet_fused(
                q_flat[bh],
                k_flat[bh],
                v_flat[bh],
                g_flat[bh],
                beta_flat[bh],
                lower_mask,
                identity_mat,
                lower_mask_diag,
            )
            outputs.append(out_bh)
            states.append(state_bh)

        # 12. Reassemble: [B, H_v_local, S+pad, head_v_dim] → [B, S, value_dim_local]
        output = torch.stack(outputs, dim=0)
        output = output.reshape(batch_size, self.v_heads_local, total_seq, self.head_v_dim)
        output = output[:, :, :seq_len]                              # drop padding
        # [B, H, S, D] -> [B, S, H, D] -> [B, S, value_dim_local]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.value_dim_local)

        # 13. Stash recurrent state so the decode path can pick up from here.
        # Write ONLY this request's slot row. The old PR #152 pattern
        # (`final_state + buffer * 0` then copy_ the whole buffer) broadcast
        # the [1, ...] prefill state across every row, clobbering other live
        # requests' states under continuous batching (one prefill per step).
        # Out-of-place index_copy keeps the buffer-dependency the `+ buf * 0`
        # trick provided — the result depends on the whole old buffer — while
        # replacing only the target row.
        final_state = torch.stack(states, dim=0)
        final_state = final_state.reshape(batch_size, self.v_heads_local, self.head_k_dim, self.head_v_dim)
        # Keep fp32 — the recurrent_state_buffer is fp32 (see __init__).
        # state_slots may be padded (e.g. DP batch padding); prefill carries
        # exactly one sequence, so use the first entry only.
        prefill_slot = state_slots[:batch_size]
        new_rec_state = self.recurrent_state_buffer.index_copy(
            0, prefill_slot, final_state.float()
        )
        self.recurrent_state_buffer.data.copy_(new_rec_state)

        # Stash conv state from last 3 mixed pre-conv tokens
        if seq_len >= self.conv_kernel_size - 1:
            new_conv_state = mixed[:, :, -(self.conv_kernel_size - 1):].contiguous()
        else:
            # Pad if seq is shorter than (kernel - 1)
            new_conv_state = torch.nn.functional.pad(
                mixed, (self.conv_kernel_size - 1 - seq_len, 0)
            )[:, :, -(self.conv_kernel_size - 1):]
        new_conv_full = self.conv_state_buffer.index_copy(
            0, prefill_slot, new_conv_state.to(self.dtype)
        )
        self.conv_state_buffer.data.copy_(new_conv_full)

        # 14. RMSNorm over head_v_dim, then z gate (silu), then out_proj
        # output: [B, S, value_dim_local] -> [B, S, v_heads_local, head_v_dim]
        out_h = output.reshape(batch_size, seq_len, self.v_heads_local, self.head_v_dim)
        # Per-head RMSNorm: variance over head_v_dim
        x_f32 = out_h.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        out_h = (x_f32 * torch.rsqrt(variance + self.rms_norm_eps)).to(self.dtype)
        out_h = out_h * self.norm_weight  # broadcasts on last dim

        # z gate: [B, S, value_dim_local] silu (computed in fp32, matching HF
        # `F.silu(gate.float())`), then cast back.
        z_gate = torch.nn.functional.silu(z.float())
        out_flat = out_h.reshape(batch_size, seq_len, self.value_dim_local)
        gated = (out_flat.float() * z_gate).to(self.dtype)

        # Output projection: column-sharded, so this is a PARTIAL hidden
        # contribution from this rank's heads.
        result = torch.nn.functional.linear(gated, self.out_proj_weight)

        # vllm_neuron's residual layout is [tokens, hidden]
        result = result.view(seq_len, hidden)

        # Sum the partial contributions across ranks and return to the SP
        # (sequence-sharded) layout the residual add expects — same pattern
        # as the GQA prefill and MLP prefill paths.
        if self.world_size > 1:
            result = self.tp_group.reduce_scatter(result, dim=0)

        return result.contiguous()

    def _is_prefill(self, attn_metadata) -> bool:
        if attn_metadata is None:
            return True  # default to prefill for unit testing
        layer_name = f"layers.{self.layer_idx}.self_attn"
        if layer_name not in attn_metadata:
            return True
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len > decode_token_threshold

    # ── Decode (TKG) ─────────────────────────────────────────────────────

    def _forward_decode(
        self, hidden_states: torch.Tensor, state_slots: torch.Tensor
    ) -> torch.Tensor:
        """TKG single-step recurrent update.

        Mirrors PR #152's `_recurrent_step`. No NKI kernel — just PyTorch
        elementwise ops plus contractions over the
        [B, H_v, head_k_dim, head_v_dim] state. Reads/writes
        `recurrent_state_buffer` and `conv_state_buffer`, addressed by
        `state_slots` (one request-stable buffer row per batch row; padded
        rows point at the scratch slot).

        Math (per-(b, h)):
            new_state = state * exp(g_t)
            kv_mem    = sum(new_state * k_t, axis=-2)
            delta     = (v_t - kv_mem) * beta_t
            new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            output    = sum(new_state * q_t, axis=-2)
        """
        tokens, hidden = hidden_states.shape
        # Decode path expects exactly one token per sequence; vllm_neuron
        # batches B sequences into [B*1, hidden] = [B, hidden].
        batch_size = tokens
        seq_len = 1
        x = hidden_states.view(batch_size, seq_len, hidden).to(self.dtype)

        # 1. Project (same as prefill but for one token)
        qkv = torch.nn.functional.linear(x, self.in_proj_qkv_weight)
        z = torch.nn.functional.linear(x, self.in_proj_z_weight)
        a = torch.nn.functional.linear(x, self.in_proj_a_weight)
        b = torch.nn.functional.linear(x, self.in_proj_b_weight)

        q_raw = qkv[..., : self.key_dim_local]
        k_raw = qkv[..., self.key_dim_local : self.key_dim_local * 2]
        v_raw = qkv[..., self.key_dim_local * 2 :]

        # 2. Causal Conv1d using stored state
        # PR #152 pattern: conv_state holds last (kernel-1) tokens; new input
        # is concatenated, then a per-channel weighted sum over the 4-tap window
        # gives one output token per channel.
        mixed_now = torch.cat([q_raw, k_raw, v_raw], dim=-1)  # [B, 1, conv_dim_local]
        mixed_now = mixed_now.transpose(1, 2)                  # [B, conv_dim_local, 1]

        # Gather each batch row's conv state from its request-stable slot.
        conv_state = self.conv_state_buffer.index_select(0, state_slots)
        # [B, conv_dim_local, 3]
        conv_input = torch.cat([conv_state, mixed_now], dim=-1)  # [B, conv_dim_local, 4]

        # Depthwise: weight [conv_dim_local, 1, 4] → [conv_dim_local, 4]
        w = self.conv1d_weight.squeeze(1)
        # Sum over kernel taps: out = sum_k w[:, k] * conv_input[:, :, k]
        conv_out = (w.unsqueeze(0) * conv_input).sum(dim=-1, keepdim=True)
        # [B, conv_dim_local, 1]

        mixed_post_conv = torch.nn.functional.silu(conv_out).transpose(1, 2)
        # [B, 1, conv_dim_local]

        # New conv state: shift left, append latest pre-conv mixed. Scatter
        # back to the slot rows (out-of-place index_copy keeps the
        # buffer-dependency; padded rows write the scratch slot only).
        new_conv_state = torch.cat([conv_state[:, :, 1:], mixed_now], dim=-1)
        new_conv_full = self.conv_state_buffer.index_copy(
            0, state_slots, new_conv_state.to(self.dtype)
        )
        self.conv_state_buffer.data.copy_(new_conv_full)

        # Split q/k/v
        q = mixed_post_conv[..., : self.key_dim_local]
        k = mixed_post_conv[..., self.key_dim_local : self.key_dim_local * 2]
        v = mixed_post_conv[..., self.key_dim_local * 2 :]

        # 3. Reshape to head layout
        q = q.reshape(batch_size, seq_len, self.k_heads_local, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.k_heads_local, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.v_heads_local, self.head_v_dim)

        # 4. Decay g + write-gate beta
        a_f32 = a.float()
        g = -self.A_log.exp() * torch.nn.functional.softplus(a_f32 + self.dt_bias)
        beta = b.sigmoid().to(torch.float32)

        # 5. Expand K heads to this rank's V heads (rank-local repeat)
        if self.kv_repeat > 1:
            q = (
                q.unsqueeze(3)
                .expand(-1, -1, -1, self.kv_repeat, -1)
                .reshape(batch_size, seq_len, self.v_heads_local, self.head_k_dim)
            )
            k = (
                k.unsqueeze(3)
                .expand(-1, -1, -1, self.kv_repeat, -1)
                .reshape(batch_size, seq_len, self.v_heads_local, self.head_k_dim)
            )

        # 6. Transpose to (B, H, S=1, dim), float32 for the recurrence
        q = q.transpose(1, 2).contiguous().float()
        k = k.transpose(1, 2).contiguous().float()
        v = v.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        # 7. l2norm + scale
        q = _deltanet_l2norm(q, dim=-1)
        k = _deltanet_l2norm(k, dim=-1)
        scale = 1.0 / (self.head_k_dim ** 0.5)
        q = q * scale

        # 8. Pull each batch row's recurrent state from its slot
        recurrent_state = self.recurrent_state_buffer.index_select(
            0, state_slots
        ).float()
        # Shape: [B, H_local, head_k_dim, head_v_dim]

        # 9. Single-step recurrent update (PR #152 _recurrent_step)
        q_t = q[:, :, 0]                # [B, H, head_k_dim]
        k_t = k[:, :, 0]                # [B, H, head_k_dim]
        v_t = v[:, :, 0]                # [B, H, head_v_dim]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        # [B, H, 1, 1]
        beta_t = beta[:, :, 0].unsqueeze(-1)
        # [B, H, 1]

        new_state = recurrent_state * g_t
        # kv_mem[b, h, v_dim] = sum_k new_state[b, h, k_dim, v_dim] * k_t[b, h, k_dim]
        kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        # outer product update: new_state += k_t ⊗ delta
        new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # output = sum_k new_state * q_t  →  [B, H, head_v_dim]
        out_one = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)
        # Add a singleton seq dim → [B, H, 1, head_v_dim]
        out_h = out_one.unsqueeze(2)

        # 10. Write back state, scattered to the slot rows.
        # fp32 buffer — rationale in __init__.
        new_rec_full = self.recurrent_state_buffer.index_copy(
            0, state_slots, new_state.float()
        )
        self.recurrent_state_buffer.data.copy_(new_rec_full)

        # 11. Reshape and finish: RMSNorm, z gate, out_proj
        # out_h: [B, H, 1, head_v_dim] → [B, 1, H, head_v_dim]
        out_h = out_h.transpose(1, 2).contiguous()
        # Per-head RMSNorm
        x_f32 = out_h.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        out_h = (x_f32 * torch.rsqrt(variance + self.rms_norm_eps)).to(self.dtype)
        out_h = out_h * self.norm_weight

        # Flatten to [B, 1, value_dim_local] then z gate + out_proj.
        # silu in fp32 to match HF `F.silu(gate.float())`.
        out_flat = out_h.reshape(batch_size, seq_len, self.value_dim_local)
        z_gate = torch.nn.functional.silu(z.float())
        gated = (out_flat.float() * z_gate).to(self.dtype)
        # Column-sharded out_proj — partial hidden contribution per rank.
        result = torch.nn.functional.linear(gated, self.out_proj_weight)

        # Back to vllm_neuron's [tokens, hidden]
        result = result.view(tokens, hidden)

        # Sum partial contributions across ranks. all_reduce, NOT
        # reduce_scatter — decode token counts may not divide by world_size
        # (mirrors the GQA decode path).
        if self.world_size > 1:
            result = self.tp_group.all_reduce(result)

        return result


# ============================================================================
# Section 5: Dense SwiGLU MLP — used by every layer (all 64 on the 27B).
# ============================================================================


class Qwen3_6MLP(nn.Module):
    """Dense SwiGLU MLP with NF.mlp.

    SiLU activation, no bias; dims come from the config (e.g. Qwen3.6-27B:
    hidden 5120, intermediate 17408). Pattern mirrors
    `vllm_neuron.model.llama3.LlamaMLP` minus the optional MLP-DP supergroup,
    which is intentionally not implemented here (add it if MLP
    data-parallelism is ever needed).

    Parallelism:
        - gate/up_proj weights: [hidden, intermediate / TP] (column-parallel)
        - down_proj weight: [intermediate / TP, hidden] (row-parallel)
        - Prefill: all-gather → NF.mlp → reduce-scatter
        - Decode: NF.mlp → all-reduce
    """

    def __init__(self, config: Qwen3_6Config) -> None:
        super().__init__()

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) must be "
                f"divisible by TP world_size ({self.world_size})"
            )
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        self.dtype = config.torch_dtype

        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size_per_rank, dtype=self.dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size_per_rank, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, self.hidden_size, dtype=self.dtype)
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor | None,
    ) -> torch.Tensor:
        is_prefill = not is_decode

        # SP: all-gather to full sequence before MLP during prefill
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.to(self.dtype)
        # SwiGLU: down(silu(gate(x)) * up(x))
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            self.tp_group.all_reduce(output)

        return output


# ============================================================================
# Section 6: Decoder Layer (dispatches by layer_types[layer_idx])
# Uses the public qwen3_vl decoder-block structure, with hybrid
# layer-type dispatch selecting GQA or DeltaNet attention.
# ============================================================================


class Qwen3_6DecoderLayer(nn.Module):
    """One transformer decoder block with attn-type dispatch.

    layer_types[layer_idx] decides whether self_attn is
    a full-attention GQA layer or a linear-attention DeltaNet layer.
    """

    def __init__(self, config: Qwen3_6Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.input_layernorm = Qwen3_6RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        # dispatch on layer type
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_6GQAAttention(config, layer_idx=layer_idx)
        elif self.layer_type == "linear_attention":
            self.self_attn = Qwen3_6DeltaNetAttention(config, layer_idx=layer_idx)
        else:
            raise ValueError(
                f"Unknown layer_type {self.layer_type!r} at layer {layer_idx}"
            )

        self.post_attention_layernorm = Qwen3_6RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.mlp = Qwen3_6MLP(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            positions=positions,
            is_decode=is_decode,
            rank=rank,
        )
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Section 7: Model Backbone
# Uses the public qwen3_vl text-backbone structure. DeltaNet layers do
# not use the paged KV cache: get_kv_spec reports dummy 1x1 KV heads for them,
# while their real state lives in the layers' side-channel buffers.
# ============================================================================


class Qwen3_6Model(nn.Module):
    """Qwen3.5/3.6 dense transformer backbone."""

    def __init__(self, config: Qwen3_6Config) -> None:
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            Qwen3_6DecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        )

        self.norm = Qwen3_6RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3_6RotaryEmbedding(config)

        # Embedding sharding
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.hidden_size,
            ),
        )
        set_weight_loader(
            self.norm.weight, last_dim_padding_weight_loader(config.hidden_size)
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use any GQA layer's metadata to detect prefill (DeltaNet layers
        # share the same flag because the dispatch is global).
        first_full_idx = next(
            i for i, lt in enumerate(self.config.layer_types) if lt == "full_attention"
        )
        meta_key = f"layers.{first_full_idx}.self_attn"
        max_query_len = attn_metadata[meta_key]["max_query_len"]
        decode_token_threshold = attn_metadata[meta_key]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )
        position_embeddings = self.rotary_emb(
            positions, dtype=hidden_states.dtype
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                rank=rank,
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


# ============================================================================
# Section 8: ForCausalLM (LM head + sampler)
# Wires backbone + column-parallel lm_head (tied or untied) + an optional
# on-device sampler. The weight-mapping
# table lives in `weight_loaders_bf16.build_weight_mappings`.
# ============================================================================


class Qwen3_6ForConditionalGeneration(nn.Module):
    """Qwen3.6 model for conditional generation.

    Follows the public `Qwen3VLForConditionalGeneration` structural
    pattern: column-parallel lm_head plus on-device sampler. `load_weights`
    detects the checkpoint layout (tensor prefix, lm_head tie state).
    """

    def __init__(self, config: Qwen3_6Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_6Model(config)

        from vllm.distributed.parallel_state import get_tp_group as _get_tp_group
        import vllm_neuron.nn as _neuron_nn
        from vllm_neuron.nn.sampler import Sampler as _Sampler
        from vllm_neuron.utils.weight_loader import (
            sharding_weight_loader_with_padding as _padding_loader,
            set_weight_loader as _set_loader,
        )

        self.tp_group = _get_tp_group()
        self.world_size = self.tp_group.world_size

        # Sampling config (on-device sampling when configured)
        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config is not None
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and getattr(config.neuron_config, "debug_logits_dir", None) is not None
        )
        max_logprobs = (
            getattr(config.neuron_config, "max_logprobs", 0)
            if config.neuron_config is not None else 0
        )
        self._gather_logits = (max_logprobs != 0) or debug_logits_enabled

        # Column-parallel LM head — vocab dim is sharded across TP ranks
        self.lm_head = _neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )

        if self.on_device_sampling_config is not None:
            self.sampler = _Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

        # Shard lm_head on the vocab dim (same spec for tied and untied
        # checkpoints — only the source tensor differs).
        _set_loader(
            self.lm_head.weight,
            _padding_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.hidden_size,
            ),
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> "Qwen3_6ForConditionalGeneration":
        cfg = Qwen3_6Config.from_configs(hf_config, neuron_config)
        return cls(cfg)

    def get_weight_mappings(
        self,
        hf_prefix: str = "model",
        tie_word_embeddings: bool | None = None,
    ) -> dict[str, list[str]]:
        """Return the {flat_name: [hf_keys]} map for `SafetensorsCheckpoint`.

        `load_weights` calls this after detecting the checkpoint's prefix and
        tie state, then hands the result to `SafetensorsCheckpoint`; see
        `weight_loaders_bf16` for the layouts the two arguments select.
        """
        from .weight_loaders_bf16 import build_weight_mappings
        return build_weight_mappings(
            self.config,
            hf_prefix=hf_prefix,
            tie_word_embeddings=tie_word_embeddings,
        )

    @staticmethod
    def _detect_hf_prefix(checkpoint) -> str:
        """Detect the checkpoint's text-decoder prefix.

        Dense text-only checkpoints store the decoder under plain ``model.``;
        the multimodal HF wrapper nests it under ``model.language_model.``.
        We probe for the embedding tensor under each candidate.
        """
        keys = checkpoint._tensor_name_to_file
        if "model.language_model.embed_tokens.weight" in keys:
            return "model.language_model"
        return "model"

    @staticmethod
    def _detect_tied_embeddings(checkpoint, config) -> bool:
        """Decide whether lm_head is tied to the embedding weight.

        Untied checkpoints ship a top-level ``lm_head.weight`` tensor; tied
        ones don't. Prefer the checkpoint's ground truth (presence of the
        tensor) and fall back to the config flag when the index is empty.
        """
        keys = checkpoint._tensor_name_to_file
        if keys:
            return "lm_head.weight" not in keys
        return bool(getattr(config, "tie_word_embeddings", False))

    def get_kv_spec(self):
        """Return the KV cache spec for this hybrid model.

        Follows `Qwen3VLForConditionalGeneration.get_kv_spec`'s public
        interface, with hybrid handling:
        - Full-attention (GQA) layers: real KV cache spec
        - DeltaNet layers: dummy 1x1 KV spec (PR #152 pattern). The
          real recurrent state lives in side-channel buffers on the layer
          (`recurrent_state_buffer`, `conv_state_buffer`).
        """
        from vllm_neuron.model.kv_cache import KVSpec, LayerSpec

        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            attn = layer.self_attn

            if layer.layer_type == "full_attention":
                # Real KV cache for GQA layers
                layers.append(
                    LayerSpec(
                        name=layer_name,
                        num_kv_heads=attn.num_key_value_heads_per_rank,
                        head_size=attn.head_dim,
                        dtype=attn.dtype,
                        sliding_window_size=None,
                        chunk_size=None,
                    )
                )
            else:
                # DeltaNet: dummy KV with 1 head, head_size=1 to keep the
                # cache manager happy. The actual recurrent state lives
                # in `attn.recurrent_state_buffer` + `attn.conv_state_buffer`.
                layers.append(
                    LayerSpec(
                        name=layer_name,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=self.config.torch_dtype,
                        sliding_window_size=None,
                        chunk_size=None,
                    )
                )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        """Bind external KV cache tensors to each attention layer.

        Follows the public vllm-neuron
        `Qwen3VLForConditionalGeneration.bind_kv_cache` interface. For
        DeltaNet layers the bound K/V tensors are dummies
        (1 head × 1 head_size); their real state lives in the side-channel
        buffers.
        """
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            attn = layer.self_attn
            attn.k_cache = kv_caches[layer_name][0]
            attn.v_cache = kv_caches[layer_name][1]

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        """Load HF safetensors → our flat-named parameters.

        Follows the public vllm-neuron
        `Qwen3VLForConditionalGeneration.load_weights` interface. The mapping
        covers every HF key in the checkpoint; weight loaders (attached to each
        parameter via `set_weight_loader`) handle TP sharding.
        """
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tp_rank = self.model.tp_group.rank_in_group
        tp_size = self.model.world_size

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)

        # Index the tensor names first: the prefix and tie state are detected
        # from the checkpoint itself, before the mapping table is built.
        checkpoint._ensure_indexed()
        hf_prefix = self._detect_hf_prefix(checkpoint)
        tie_word_embeddings = self._detect_tied_embeddings(checkpoint, self.config)
        logger.info(
            "Qwen3_6 checkpoint layout: hf_prefix=%r tie_word_embeddings=%s",
            hf_prefix,
            tie_word_embeddings,
        )

        # Build the flat mappings. SafetensorsCheckpoint expects values
        # to be a single string when length-1, list when fused. We get
        # lists from build_weight_mappings; flatten singletons.
        raw = self.get_weight_mappings(
            hf_prefix=hf_prefix, tie_word_embeddings=tie_word_embeddings
        )
        mappings: dict[str, object] = {}
        for k, v in raw.items():
            mappings[k] = v[0] if len(v) == 1 else v

        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        ).state_dict

        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list,
    ) -> tuple[torch.Tensor, int]:
        """Text-only M-RoPE positions (SupportsMRoPE protocol).

        The runner requires this protocol whenever the HF config marks
        the model `uses_mrope` (Qwen3.5/3.6 configs carry `mrope_section`),
        and raises TypeError at request init without it. This model serves
        text only, where every M-RoPE row (t/h/w) equals the 1-D position,
        so return `arange` replicated to [3, seq_len] with delta 0.
        `mm_features` must be empty — vision input would need real grid
        positions and a vision tower this model doesn't have.
        """
        if mm_features:
            raise ValueError(
                "Qwen3_6ForConditionalGeneration is text-only; got "
                f"{len(mm_features)} multimodal feature(s). Serve with "
                "--limit-mm-per-prompt '{\"image\": 0, \"video\": 0}'."
            )
        seq_len = len(input_tokens)
        positions = (
            torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).expand(3, -1)
        )
        return positions, 0

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        rotary_position_ids: torch.Tensor | None = None,
        **kwargs,  # runner may inject async-spec / future kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError("Input Embedding as Inputs is Not Supported Yet.")
        # `rotary_position_ids` unused: for text-only serving M-RoPE reduces
        # to `positions` (see the module docstring).

        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        # Sampling slice + LM head
        hidden_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        hidden_for_logits = hidden_for_logits.to(self.config.torch_dtype)

        logits = self.lm_head(hidden_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        gathered_logits = None
        if self._gather_logits:
            if self.tp_group is not None:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler
            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits
