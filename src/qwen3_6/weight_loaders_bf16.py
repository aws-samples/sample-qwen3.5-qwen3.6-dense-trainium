# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-to-parameter name mapping for the Qwen3.5/3.6 dense BF16 format.

Provides `build_weight_mappings(config, ...)` which returns a dict mapping
flat parameter names (the names our `Qwen3_6ForConditionalGeneration`
exposes) to lists of HuggingFace safetensors keys.

vllm_neuron's `SafetensorsCheckpoint` consumes this mapping plus the
per-parameter weight loader (set via `set_weight_loader` in
`model_bf16.py`) to materialize on-device tensors.

HuggingFace checkpoint layout (example dims are Qwen3.6-27B; layer count
and which layers are GQA vs DeltaNet come from `config.layer_types`). The
`{HF}` prefix is `model` for a text-only export of the decoder and
`model.language_model` for the native multimodal checkpoint — see below.

  Embeddings / norm / lm_head:
    {HF}.embed_tokens.weight        [vocab, hidden]
    {HF}.norm.weight                [hidden]
    lm_head.weight                  [vocab, hidden]  (untied variants only;
                                     tied variants reuse embed_tokens)

  Per layer L:
    Per-layer norms (pre-attention and pre-MLP):
      {HF}.layers.{L}.input_layernorm.weight              [hidden]
      {HF}.layers.{L}.post_attention_layernorm.weight     [hidden]

    MLP (all layers):
      {HF}.layers.{L}.mlp.gate_proj.weight   [intermediate, hidden]
      {HF}.layers.{L}.mlp.up_proj.weight     [intermediate, hidden]
      {HF}.layers.{L}.mlp.down_proj.weight   [hidden, intermediate]

    GQA full-attention layers (the "full_attention" entries in layer_types):
      {HF}.layers.{L}.self_attn.q_proj.weight        [2*Q*head_dim, hidden]
                                                    (Q and gate interleaved
                                                     per head; see below)
      {HF}.layers.{L}.self_attn.k_proj.weight        [KV*head_dim, hidden]
      {HF}.layers.{L}.self_attn.v_proj.weight        [KV*head_dim, hidden]
      {HF}.layers.{L}.self_attn.o_proj.weight        [hidden, Q*head_dim]
      {HF}.layers.{L}.self_attn.q_norm.weight        [head_dim]
      {HF}.layers.{L}.self_attn.k_norm.weight        [head_dim]
      (attn output gate, when attn_output_gate=True, is spliced into
       q_proj's rows rather than a separate tensor — see model_bf16.py)

    DeltaNet linear-attention layers (the "linear_attention" entries):
      {HF}.layers.{L}.linear_attn.in_proj_qkv.weight  [conv_dim, hidden]   (conv_dim = key_dim*2 + value_dim)
      {HF}.layers.{L}.linear_attn.in_proj_z.weight    [value_dim, hidden]
      {HF}.layers.{L}.linear_attn.in_proj_a.weight    [num_v_heads, hidden]
      {HF}.layers.{L}.linear_attn.in_proj_b.weight    [num_v_heads, hidden]
      {HF}.layers.{L}.linear_attn.conv1d.weight       [conv_dim, 1, kernel]   (no conv1d.bias — bias=False)
      {HF}.layers.{L}.linear_attn.dt_bias             [num_v_heads]
      {HF}.layers.{L}.linear_attn.A_log               [num_v_heads]
      {HF}.layers.{L}.linear_attn.norm.weight         [head_v_dim]
      {HF}.layers.{L}.linear_attn.out_proj.weight     [hidden, value_dim]

      State buffers (`recurrent_state_buffer`, `conv_state_buffer`)
      are runtime state, NOT loaded from the checkpoint — they live as
      zero-init buffers on the layer.

Layout notes handled by `build_weight_mappings`:
- Q+K+V are fused into one `qkv_proj_weight` tensor (NF.qkv_proj expects
  that layout), while the checkpoint keeps them as separate q/k/v tensors.
- lm_head: untied variants (tie_word_embeddings=False, e.g. the 27B) map
  from their own top-level `lm_head.weight`; tied variants (typically the
  smaller dense models) alias lm_head to the embedding key.
"""

from __future__ import annotations

from .config import Qwen3_6Config


def build_weight_mappings(
    config: Qwen3_6Config,
    hf_prefix: str = "model",
    tie_word_embeddings: bool | None = None,
) -> dict[str, list[str]]:
    """Return a `{flat_param_name: [hf_safetensors_keys]}` mapping.

    Keys in the returned dict are dot-paths into our
    `Qwen3_6ForConditionalGeneration` module tree. Values are lists
    because vllm_neuron's loaders accept either a single source key
    (length-1 list) or fused multiple sources (e.g. q/k/v → qkv).

    Args:
        config: the resolved Qwen3.5/3.6 config.
        hf_prefix: dot-path prefix for the text decoder tensors in the
            checkpoint. Two layouts exist:
              - a text-only export of the decoder stores weights under plain
                ``model.``;
              - the native multimodal checkpoint (the whole Qwen3.5/3.6
                family is multimodal) nests the text decoder under
                ``model.language_model.``.
            `load_weights` detects which one the checkpoint uses and passes
            it here; the default assumes the plain decoder layout.
        tie_word_embeddings: whether lm_head shares the embedding weight.
            When None, falls back to ``config.tie_word_embeddings``. Tied
            checkpoints ship no ``lm_head.weight`` tensor, so lm_head is
            loaded from the embedding key instead.
    """
    mappings: dict[str, list[str]] = {}

    if tie_word_embeddings is None:
        tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))

    HF = hf_prefix

    # Backbone
    mappings["model.embed_tokens.weight"] = [f"{HF}.embed_tokens.weight"]
    mappings["model.norm.weight"] = [f"{HF}.norm.weight"]

    # LM head. Untied models (e.g. the 27B) ship lm_head as its own
    # top-level `lm_head.weight` tensor. Tied models (typically the smaller
    # dense variants) share the embedding weight and have no `lm_head.weight`
    # key, so we alias lm_head to the embedding tensor. Either way lm_head's
    # weight loader shards the [vocab, hidden] tensor identically.
    if tie_word_embeddings:
        mappings["lm_head.weight"] = [f"{HF}.embed_tokens.weight"]
    else:
        mappings["lm_head.weight"] = ["lm_head.weight"]

    if not config.layer_types:
        raise ValueError("config.layer_types is empty; can't build mappings")
    if len(config.layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"layer_types length ({len(config.layer_types)}) != "
            f"num_hidden_layers ({config.num_hidden_layers})"
        )

    for L, lt in enumerate(config.layer_types):
        layer = f"model.layers.{L}"
        prefix = f"{HF}.layers.{L}"  # HF source key prefix

        # Pre-norms (always present)
        mappings[f"{layer}.input_layernorm.weight"] = [f"{prefix}.input_layernorm.weight"]
        mappings[f"{layer}.post_attention_layernorm.weight"] = [
            f"{prefix}.post_attention_layernorm.weight"
        ]

        # MLP (always present, identical across layer types)
        mappings[f"{layer}.mlp.gate_proj_weight"] = [f"{prefix}.mlp.gate_proj.weight"]
        mappings[f"{layer}.mlp.up_proj_weight"] = [f"{prefix}.mlp.up_proj.weight"]
        mappings[f"{layer}.mlp.down_proj_weight"] = [f"{prefix}.mlp.down_proj.weight"]

        if lt == "full_attention":
            # GQA: fuse Q + K + V into one tensor. On gated variants
            # (attn_output_gate=True), q_proj.weight interleaves Q and gate
            # within each head's 2*head_dim block:
            # [h0_q, h0_gate, h1_q, h1_gate, ...]. The custom
            # _spliced_q_kv_loader gathers each head's Q sub-slice; the same
            # q_proj.weight is also referenced by attn_gate_weight, whose
            # loader gathers the following gate sub-slice from each head.
            mappings[f"{layer}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{layer}.self_attn.o_proj_weight"] = [
                f"{prefix}.self_attn.o_proj.weight"
            ]
            # Per-head Q/K layernorm
            mappings[f"{layer}.self_attn.q_layernorm.weight"] = [
                f"{prefix}.self_attn.q_norm.weight"
            ]
            mappings[f"{layer}.self_attn.k_layernorm.weight"] = [
                f"{prefix}.self_attn.k_norm.weight"
            ]
            # Attention output gate (the sub-slice following Q inside each
            # head's interleaved 2*head_dim block). Mapped only when
            # config.attn_output_gate is True.
            if getattr(config, "attn_output_gate", False):
                mappings[f"{layer}.self_attn.attn_gate_weight"] = [
                    f"{prefix}.self_attn.q_proj.weight"
                ]

        elif lt == "linear_attention":
            la = f"{prefix}.linear_attn"
            mappings[f"{layer}.self_attn.in_proj_qkv_weight"] = [f"{la}.in_proj_qkv.weight"]
            mappings[f"{layer}.self_attn.in_proj_z_weight"] = [f"{la}.in_proj_z.weight"]
            mappings[f"{layer}.self_attn.in_proj_a_weight"] = [f"{la}.in_proj_a.weight"]
            mappings[f"{layer}.self_attn.in_proj_b_weight"] = [f"{la}.in_proj_b.weight"]
            mappings[f"{layer}.self_attn.conv1d_weight"] = [f"{la}.conv1d.weight"]
            mappings[f"{layer}.self_attn.A_log"] = [f"{la}.A_log"]
            mappings[f"{layer}.self_attn.dt_bias"] = [f"{la}.dt_bias"]
            mappings[f"{layer}.self_attn.norm_weight"] = [f"{la}.norm.weight"]
            mappings[f"{layer}.self_attn.out_proj_weight"] = [f"{la}.out_proj.weight"]
            # State buffers are NOT loaded from checkpoint — zero-init at runtime.

        else:
            raise ValueError(
                f"Unknown layer type {lt!r} at layer {L}; "
                "expected 'full_attention' or 'linear_attention'."
            )

    return mappings
