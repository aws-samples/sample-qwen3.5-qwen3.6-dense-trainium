# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 / Qwen3.6 dense (hybrid GatedDeltaNet + GQA) configuration.

The dataclass field values below are only defaults (they match Qwen3.5-27B);
`from_configs()` overrides them from the target model's HF `config.json` at
load time, so smaller/tied variants and different head counts need no code
changes.

Architecture (shared across the family; example values are Qwen3.5-27B):
- N layers total, split DeltaNet (linear-attn) + GQA (full-attn)
  27B: 64 layers = 48 DeltaNet + 16 GQA
- Layer pattern: [3 DeltaNet + 1 GQA], repeated  (from `layer_types`)
- SwiGLU MLP        27B: hidden 5120 / intermediate 17408
- GQA               27B: 24 Q heads, 4 KV heads, head_dim 256
- DeltaNet          27B: 48 value heads, 16 key heads, k_dim=v_dim=128
- Conv1d kernel 4 (state stores last 3 pre-conv QKV tokens)
- RoPE: partial (`partial_rotary_factor` of head_dim rotated), large theta
- Vocab is 248,320 family-wide; `tie_word_embeddings` varies — the 27B is
  untied (lm_head is its own tensor), smaller dense variants are typically
  tied.

Config layout: the per-layer values live in the top-level dict for dense
text-only checkpoints, and inside the `text_config` block for the
multimodal HF wrapper. `from_configs()` merges both so either layout works.

Fields consumed by this model implementation (read per-model from HF config):
  vocab_size, hidden_size, num_hidden_layers, intermediate_size,
  num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps,
  tie_word_embeddings, torch_dtype/dtype, layer_types, partial_rotary_factor,
  attn_output_gate, rope_theta (top-level or `rope_parameters`), and the
  DeltaNet dims (the `linear_*` keys, plus short aliases).

Informational fields, not consumed by this repository's model implementation:
  `max_position_embeddings`, `full_attention_interval` and `eos_token_id` are
  carried over from the HF `text_config`; `rope_scaling`, `output_gate_type`,
  `pad_token_id` and `bos_token_id` are placeholders that published
  Qwen3.5/3.6 `text_config` blocks do not ship.
The original HF config remains available to vLLM/tokenizer code; within this
model, `layer_types` determines the attention pattern and the attention and
DeltaNet gate operations are implemented directly.

CAVEAT — behavior-driving fields that fall back to the 27B-shaped default ONLY
when the HF config omits them: `attn_output_gate` (True) and
`partial_rotary_factor` (0.25). Real Qwen3.5/3.6 checkpoints ship these, so the
defaults are inert in practice; a hand-stripped config that drops them could
mis-build (e.g. a gate the checkpoint lacks). Prefer serving with the model's
real `config.json`.

"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3_6Config:
    """Configuration for the Qwen3.5/3.6 dense hybrid (DeltaNet + GQA) family.

    See the module docstring for the defaults' provenance and for which
    fields are read per-model vs. which fall back to a default.
    """

    # Core dims (Qwen3.5-27B defaults; from_configs() overrides from HF config)
    vocab_size: int = 248320
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    intermediate_size: int = 17408
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144  # informational — not consumed here
    torch_dtype: torch.dtype = torch.bfloat16
    # 27B ships untied embeddings — lm_head is a separate tensor.
    tie_word_embeddings: bool = False

    # Attention (GQA, full-attn layers)
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    rope_theta: float = 10000000.0
    # Informational — not consumed here; RoPE uses rope_theta and
    # partial_rotary_factor directly.
    rope_scaling: dict | None = None
    # Partial RoPE: rotate only `partial_rotary_factor * head_dim` dims.
    # Qwen3.5/3.6 family: 25% of head_dim (e.g. 64 for head_dim=256).
    # CAVEAT: only defaults to 0.25 if the HF config omits it — real
    # checkpoints ship this value, so per-model behavior comes from there.
    partial_rotary_factor: float = 0.25
    # CAVEAT: defaults to True only if the HF config omits `attn_output_gate`.
    # A dense variant that has no attn gate must set this False in its config;
    # otherwise the loader would build/expect a gate the checkpoint lacks.
    attn_output_gate: bool = True
    # Informational — not consumed here; the gate ops are encoded directly
    # ("swish" is silu, which is what the gated DeltaNet norm applies).
    output_gate_type: str = "swish"

    # DeltaNet (linear-attn layers)
    deltanet_num_v_heads: int = 48
    deltanet_num_k_heads: int = 16
    deltanet_k_head_dim: int = 128
    deltanet_v_head_dim: int = 128
    deltanet_conv_kernel_size: int = 4

    # Layer pattern — list of "linear_attention" / "full_attention" of length
    # num_hidden_layers. Loaded from HF config.json `text_config.layer_types`.
    layer_types: list[str] = field(default_factory=list)
    # Informational — not consumed here; dispatch uses layer_types directly.
    full_attention_interval: int = 4

    # Informational — not consumed here. `eos_token_id` matches the published
    # 27B text_config; bos/pad are from its generation_config.json.
    pad_token_id: int | None = None
    bos_token_id: int | None = 248044
    eos_token_id: int | None = 248044

    neuron_config: NeuronConfig | None = None

    # Validation
    def __post_init__(self) -> None:
        # Sanity: layer_types length must match num_hidden_layers when provided
        if self.layer_types and len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}"
            )
        # Sanity: only the two expected layer types
        bad = [lt for lt in self.layer_types
               if lt not in ("linear_attention", "full_attention")]
        if bad:
            raise ValueError(
                f"Unexpected layer_types entries: {set(bad)}. "
                "Expected only 'linear_attention' and 'full_attention'."
            )

    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for lt in self.layer_types if lt == "full_attention")

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(1 for lt in self.layer_types if lt == "linear_attention")

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig,
    ) -> "Qwen3_6Config":
        """Build from a HF config (path, PretrainedConfig, or dict).

        For HF Qwen3_5ForConditionalGeneration (used by both 3.5 and 3.6),
        the per-layer values live in `text_config`. We pull from there if
        present, then fall back to the top-level dict.
        """
        # Normalize input to a dict
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            # Strip None quantization_config (HF round-trip quirk)
            qc = getattr(hf_config, "quantization_config", "MISSING")
            if qc is None:
                delattr(hf_config, "quantization_config")
                config_dict = hf_config.to_dict()
                hf_config.quantization_config = None
            else:
                config_dict = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            config_dict = hf_config
        else:
            raise TypeError(
                f"Unsupported hf_config type: {type(hf_config).__name__}"
            )

        # Prefer text_config block when present (multimodal HF wrapper)
        text_cfg = config_dict.get("text_config", config_dict)
        merged = {**config_dict, **text_cfg}

        # Pull RoPE from rope_parameters if present (Qwen3.5/3.6 use it).
        # Both published checkpoints nest `partial_rotary_factor` in here too,
        # so read it from the same place rather than falling back to the
        # dataclass default (which happens to agree at 0.25 for this family —
        # but would silently mis-build a sibling that differs).
        rope_params = merged.get("rope_parameters") or {}
        for key in ("rope_theta", "partial_rotary_factor"):
            if key in rope_params:
                merged[key] = rope_params[key]

        # Map HF DeltaNet field names to ours. Published Qwen3.5/3.6
        # `text_config` blocks (all four published models) use `linear_*` keys;
        # the shorter aliases below come from the PR #152 contrib config and
        # are accepted for compatibility.
        mapping = {
            # Short aliases (PR #152 contrib config)
            "num_v_heads": "deltanet_num_v_heads",
            "num_k_heads": "deltanet_num_k_heads",
            "v_head_dim": "deltanet_v_head_dim",
            "k_head_dim": "deltanet_k_head_dim",
            "conv_kernel_size": "deltanet_conv_kernel_size",
            # 27B-style names
            "linear_num_value_heads": "deltanet_num_v_heads",
            "linear_num_key_heads": "deltanet_num_k_heads",
            "linear_value_head_dim": "deltanet_v_head_dim",
            "linear_key_head_dim": "deltanet_k_head_dim",
            "linear_conv_kernel_dim": "deltanet_conv_kernel_size",
        }
        for src, dst in mapping.items():
            if src in merged and dst not in merged:
                merged[dst] = merged[src]

        # Coerce torch_dtype string -> torch.dtype.
        # 27B's text_config uses "dtype" (not "torch_dtype"); accept either.
        dtype_val = merged.get("torch_dtype") or merged.get("dtype")
        if isinstance(dtype_val, str):
            merged["torch_dtype"] = getattr(torch, dtype_val)

        # Filter to known fields
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in field_names}
        filtered["neuron_config"] = neuron_config

        return cls(**filtered)
