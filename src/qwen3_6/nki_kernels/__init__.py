# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""NKI kernels for Qwen3.5/Qwen3.6 models

PR #152's kernel is exposed at **two integration levels**:

- `deltanet_fused_chunked_fwd_jit`: the NKI kernel object returned by
  `nki.jit()`. It is the input to the vllm-neuron wrapper below; model code
  does not invoke it as a torch HOP directly.

- `call_deltanet_fused`: a vllm-neuron-friendly wrapper. Internally
  uses `libtorch_neuronx_lite.nki.nki_hop.wrap_nki` so the kernel call is a
  proper torch HOP that survives `torch.compile` graph extraction.
  Use this from inside `Qwen3_6DeltaNetAttention.forward()`.

  NOTE: the NKI-HOP and fx-pass machinery ships in the standalone
  `libtorch-neuronx-lite` distribution rather than in `vllm_neuron` itself —
  the same package that holds the aliasing pass this repo overrides
  (`overrides/README.md`).

The PR #152 source (`deltanet_fused.py`) is kept close to verbatim: we
removed the `@nki.jit` decorator so this module can construct the NKI
`Kernel` wrapper explicitly before vllm-neuron turns it into a torch HOP.
We also added bf16 I/O so q/k/v and the output follow the input dtype while
all math stays fp32. The supported bf16-in / fp32-compute / bf16-out path
(with fp32 recurrent state) matches the HF Qwen3.5 gated-delta-rule op; see
the header note in `deltanet_fused.py` for details. The current model prefill
caller passes fp32 q/k/v after preprocessing, so the bf16 path is supported
but not currently exercised by this repository.
"""

import nki

from .deltanet_fused import deltanet_fused_chunked_fwd as _kernel_fn

# Construct the nki.framework.kernel.Kernel wrapper at import time; no kernel
# compilation occurs here. `wrap_nki` registers its underlying function as a
# torch HOP for later model tracing and lowering.
deltanet_fused_chunked_fwd_jit = nki.jit()(_kernel_fn)


def call_deltanet_fused(
    query, key, value, g_in, beta_in, lower_mask, identity, lower_mask_diag,
    *,
    grid: int = 2,
):
    """Wrap the kernel via vllm_neuron's torch HOP and invoke.

    The kernel supports bf16 and fp32 q/k/v inputs, preserves ``query``'s
    dtype for the output, performs its internal math in fp32, and returns
    ``final_state`` in fp32. This repository's prefill caller currently passes
    fp32: query and key have already been L2-normalized, query has also been
    scaled by ``1 / sqrt(head_k_dim)``, and value has undergone the DeltaNet
    projection/conv preprocessing. The supported bf16 path is not currently
    exercised here.

    Args:
        query: (S, 128) bf16|fp32 — preprocessed query; in the current caller,
            L2-normalized, scaled, and converted to fp32.
        key: (S, 128) bf16|fp32 — preprocessed key; in the current caller,
            L2-normalized and converted to fp32.
        value: (S, 128) bf16|fp32 — preprocessed value; converted to fp32 by
            the current caller.
        g_in: (S, 1) float32 — RAW per-token log-decay
        beta_in: (S, 1) float32 — sigmoid(b)
        lower_mask, identity, lower_mask_diag: (128, 128) float32 constants
        grid: NKI grid size to launch under (default 2; cumsum uses 2 too).

    Returns:
        (output (S, 128) — same dtype as `query`, final_state (128, 128) float32)
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    wrapped = wrap_nki(deltanet_fused_chunked_fwd_jit)
    # The [grid](**kwargs) syntax is the vllm_neuron convention. We pass
    # positionally because the source PR #152 kernel takes positional args.
    return wrapped[grid](
        query, key, value, g_in, beta_in, lower_mask, identity, lower_mask_diag,
    )


__all__ = [
    "deltanet_fused_chunked_fwd_jit",
    "call_deltanet_fused",
]
