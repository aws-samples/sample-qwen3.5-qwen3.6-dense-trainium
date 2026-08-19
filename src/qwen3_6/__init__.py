# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 / Qwen3.6 dense (hybrid GatedDeltaNet + GQA) for vllm_neuron.

Serves the Qwen3.5 and Qwen3.6 dense models on Trainium in both inference
modes — `vllm serve` (online) and `LLM.generate()` (offline). ONE package
serves the whole dense family.

All models in this family share one architecture — hybrid linear-attn +
full-attn in a [3 lin + 1 full] block pattern, with partial RoPE — and differ
only in scale plus a couple of checkpoint-layout flags. Everything
model-specific is read at load time, never hardcoded: dimensions and head
counts come from the HF config (`config.py`), while the tensor prefix and
tie state are detected from the checkpoint index
(`weight_loaders_bf16.build_weight_mappings`).

The HF arch class name is `Qwen3_5ForConditionalGeneration` for both 3.5
and 3.6, so a single registry slot covers them.

Package name note: `qwen3_6` serves 3.5 and 3.6 alike — they are the same
architecture family, and the HF class name is shared.

See ../../README.md for validation status and architecture notes.
"""

from .config import Qwen3_6Config
from .factory import Qwen3_6ForConditionalGeneration

# Register into the vLLM / vllm_neuron model registries at import time.
# This overlay package is not part of the installed vllm_neuron tree, so
# it must inject itself; importing the package (e.g. via the bundled
# sitecustomize.py on PYTHONPATH) makes the registration effective in
# every Python process, including vLLM worker subprocesses.
try:
    from .register import register as _register, install_post_plugin_hook

    _register()
    install_post_plugin_hook()
except Exception:  # pragma: no cover — never block import on registration
    import logging

    logging.getLogger(__name__).warning(
        "qwen3_6: auto-registration failed; run `python -m qwen3_6.register` "
        "manually before serving",
        exc_info=True,
    )

__all__ = [
    "Qwen3_6Config",
    "Qwen3_6ForConditionalGeneration",
]
