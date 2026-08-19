# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register the Qwen3.5/3.6 dense family in vllm_neuron without forking it.

Run before `vllm serve` (e.g. via `python -m qwen3_6.register` in the
container, or automatically via this package's __init__ / the bundled
sitecustomize.py).

After this call, both vLLM's and vllm_neuron's model registries dispatch
the HF architecture class "Qwen3_5ForConditionalGeneration" — which every
Qwen3.5 AND Qwen3.6 dense checkpoint ships under (Alibaba kept the class
name across the version bump) — to this package's implementation.
"""

import importlib
import logging

logger = logging.getLogger(__name__)


_PATCHED = False
_HF_ARCH = "Qwen3_5ForConditionalGeneration"


def register() -> None:
    """Idempotently inject the Qwen3.5/3.6 factory into both registries."""
    global _PATCHED
    if _PATCHED:
        logger.debug("Qwen3.5/3.6 dense already registered, skipping")
        return

    from . import Qwen3_6ForConditionalGeneration

    # 1. vllm_neuron's own registry (its internal code paths, plus this
    #    module's __main__ self-check). If the installed plugin ever ships this
    #    arch itself, replace the slot rather than skipping, so THIS package's
    #    implementation — with TP-sharded DeltaNet and the state-slot
    #    contract — is the one that serves.
    registry = importlib.import_module("vllm_neuron.model.registry")
    original_get_models = registry.get_models

    def patched_get_models() -> list[tuple[str, type]]:
        models = list(original_get_models())
        models = [m for m in models if m[0] != _HF_ARCH]
        models.append((_HF_ARCH, Qwen3_6ForConditionalGeneration))
        logger.info("Registered %s -> qwen3_6 package in vllm_neuron", _HF_ARCH)
        return models

    registry.get_models = patched_get_models

    # 2. vLLM's `_ModelRegistry` — the one `neuron_model_runner.load_model`
    #    actually queries via `ModelRegistry.resolve_model_cls`. vLLM ships
    #    its own GPU implementation for this arch, lazily registered among the
    #    MULTIMODAL models and with no `from_configs` — which is also why the
    #    code below has to force `is_text_generation_model=True`. We must
    #    FORCE-REPLACE the slot, not skip when it exists.
    try:
        vllm_registry_mod = importlib.import_module(
            "vllm.model_executor.models.registry"
        )
        ModelRegistry = vllm_registry_mod.ModelRegistry  # the singleton
        _RegisteredModel = vllm_registry_mod._RegisteredModel  # eager wrapper

        slot = _RegisteredModel.from_model_cls(Qwen3_6ForConditionalGeneration)

        # vLLM's introspection on our factory class doesn't detect that it
        # is a text-generation model; force the flag. Both dataclasses are
        # frozen, so rebuild with dataclasses.replace.
        try:
            import dataclasses

            new_interfaces = dataclasses.replace(
                slot.interfaces, is_text_generation_model=True
            )
            slot = dataclasses.replace(slot, interfaces=new_interfaces)
        except Exception as exc:
            logger.warning(
                "Could not patch _ModelInfo.is_text_generation_model: %r", exc
            )

        ModelRegistry.models[_HF_ARCH] = slot
        logger.info(
            "Force-registered %s -> qwen3_6 package in vllm.ModelRegistry",
            _HF_ARCH,
        )
    except Exception as exc:
        logger.warning("Could not patch vllm.ModelRegistry: %r", exc)

    _PATCHED = True


def install_post_plugin_hook() -> None:
    """Re-apply the registry patch after vLLM's plugin loader runs.

    Plugin loading can (re)import `vllm.model_executor.models.registry` in a
    process where our patch has not run yet, leaving the arch pointing at
    vLLM's own default slot. Wrapping `load_general_plugins`
    re-applies the patch after plugins finish, in every process that loads
    them (including vLLM worker subprocesses).

    Idempotent — safe to call multiple times.
    """
    try:
        import vllm.plugins as _plugins_mod
    except Exception as exc:
        logger.warning("Could not import vllm.plugins to install hook: %r", exc)
        return

    if getattr(_plugins_mod.load_general_plugins, "_qwen36_wrapped", False):
        return  # already wrapped

    _orig_loader = _plugins_mod.load_general_plugins

    def _wrapped_loader(*args, **kwargs):
        result = _orig_loader(*args, **kwargs)
        global _PATCHED
        _PATCHED = False
        try:
            register()
        except Exception as exc:
            logger.warning("post-plugin re-register failed: %r", exc)
        return result

    _wrapped_loader._qwen36_wrapped = True  # type: ignore[attr-defined]
    _plugins_mod.load_general_plugins = _wrapped_loader
    logger.info("Installed post-plugin re-register hook on load_general_plugins")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register()

    from vllm_neuron.model.registry import get_models

    names = [name for name, _ in get_models()]
    print("Registered model architectures:")
    for n in names:
        marker = " <-- this package" if n == _HF_ARCH else ""
        print(f"  - {n}{marker}")
