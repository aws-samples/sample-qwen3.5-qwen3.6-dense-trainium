# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Factory for the Qwen3.5/3.6 dense hybrid model, selected by platform/config.

Validates the config and hands off to the BF16 implementation, which reads
the per-model dimensions, tie state, and checkpoint layout from the HF config
at load time. Same platform/config factory pattern as the public vllm-neuron
`qwen3_vl` implementation.

Only BF16 is supported today; mxfp4 quantization is rejected at validation
time (see `_validate_config`).
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_6ForConditionalGeneration(nn.Module):
    """Factory that validates config and selects the Qwen3.5/3.6 dense impl."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: object | None = None,
    ) -> nn.Module:
        # The runner calls multimodal-registered archs with
        # (hf_config=..., text_neuron_config=..., vision_neuron_config=...).
        # This model is text-only, so accept that convention, take the text
        # config from either parameter, and ignore the vision config.
        if neuron_config is None:
            neuron_config = text_neuron_config
        del vision_neuron_config
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        # Only a BF16 implementation exists today.
        from .model_bf16 import Qwen3_6ForConditionalGeneration as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None

        if quantization == "mxfp4":
            raise ValueError(
                "quantization='mxfp4' is not yet supported for Qwen3.5/3.6. "
                "Please use quantization='bf16' or leave unset."
            )
