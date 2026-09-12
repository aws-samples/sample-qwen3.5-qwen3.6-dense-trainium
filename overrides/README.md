# overrides/ — core file replacements (REQUIRED)

Unlike a pure overlay package, this contribution needs changes in TWO core
files that a PYTHONPATH overlay cannot deliver. This directory ships **full
replacement copies** of those files. Both are required for correct serving;
skipping them produces the failure modes described below (silently — outputs
degrade, nothing crashes).

## Layout — one directory per target package

The two files live in **different installed packages**: the runner is part of
`vllm_neuron`, while the fx-pass / NKI-HOP compilation machinery ships in the
standalone `libtorch-neuronx-lite` distribution. Each top-level directory here
is named for its target package, and the tree below it mirrors that package's
own layout:

```
overrides/
├── apply.sh                                        # installs both + clears cache
├── vllm_neuron/vllm/worker/neuron_model_runner.py  # DeltaNet state-slot lifecycle
└── libtorch_neuronx_lite/fx_passes/aliasing_pass.py  # .data.copy_() graph aliasing
```

## Base version pin

These replacements are based on:

- **Installed packages:** vllm-neuron `0.24.0.1.1.0` with vllm `0.24.0`, and
  libtorch-neuronx-lite base version `2.11.0.1.0.1284` (the validated wheel
  reports `2.11.0.1.0.1284+f49d8626`; the suffix is local build metadata).
- **Public source pin:** tag
  [`v0.24.0.1.1.0`](https://github.com/vllm-project/vllm-neuron/tree/v0.24.0.1.1.0)
  on branch `release-0.24.0.1.1.0`.

The installed `vllm_neuron/vllm/worker/neuron_model_runner.py` is
**byte-identical** to that tag, so the runner replacement is that public file
plus purely additive changes — no upstream line removed. Its features are
therefore all intact: async and EAGLE3 speculative decode, data-parallel
coordination, fp8-packed KV, SWA and decode-context bucket trimming, decode
context parallelism, and the disaggregated encoder / EPD paths. The aliasing
pass likewise keeps its NKI write-alias serialization.

`libtorch_neuronx_lite` is not published in the vllm-neuron repo, so its
replacement is diffed against the copy shipped in the wheel above.

Both replacements are validated end-to-end on device on this 0.24 stack (see
the top-level README's validation section). On a different version, do NOT
blind-copy — diff the summaries below against your version's files and
re-apply the changes manually (they are small and well-isolated).

## How to apply

```bash
./overrides/apply.sh
```

The script first verifies both installed package base versions (accepting PEP
440 local build metadata after `+`), all target paths, and the cache location. It stages both replacements and backups before
mutation, restores the originals if replacement or cache cleanup fails, and
preserves the validated cache root while deleting only its contents. The
standard cache locations are accepted automatically; a custom absolute cache
root requires `QWEN36_ALLOW_CUSTOM_CACHE_DELETE=1` after you verify the path.
Missing targets are never created because they indicate that the active
environment does not match the pinned stack.

To inspect or reproduce the installation manually, read `apply.sh`; keeping the
version checks, full preflight, rollback, and cache-path guard intact is
strongly recommended.

## What each file changes and why

### 1. `vllm_neuron/vllm/worker/neuron_model_runner.py` — DeltaNet state-slot lifecycle

**Fixes: output crossing between sequences under continuous batching.**
Root cause and the reasoning behind the design: `docs/implementation-notes.md`
§3. In short, the runner owns a request→slot mapping — mirroring how it
already owns KV block tables — and threads it to the model each step, so
DeltaNet's side-channel state buffers are addressed by a request-stable row
instead of the batch position.

Delta summary (all additions, no behavior change for non-hybrid models —
the feature self-disables when the model has no DeltaNet state buffers):

- `__init__`: `_state_slot_capacity` / `_state_slot_by_req` bookkeeping.
- `load_model`: `_resolve_state_slot_capacity()` — detects the
  `state_scratch_slot` attribute on model modules (only the DeltaNet layer
  has one) before torch.compile wrapping.
- New helpers by `_reset_state`: `_acquire_state_slot` (first-free
  allocation; loud RuntimeError on exhaustion), `_release_state_slot`,
  `_release_state_slots_for` (per-step release policy),
  `_build_state_slots_tensor` (per-step `[padded_num_reqs]` tensor: batch
  position → buffer row; padded rows → scratch slot).
- `execute_model`: calls `_release_state_slots_for(scheduler_output)`
  immediately before `_update_states`. Slots are released for
  `finished_req_ids` AND `preempted_req_ids`; merely-unscheduled requests keep
  theirs. **`_update_states` itself is left byte-identical to upstream**, per
  the NEURON MAINTAINER NOTE on that method (it is a verbatim copy of
  `GPUModelRunner._update_states` and is meant to be replaced wholesale on
  rebase), so the release policy lives at the single call site instead.
- `_reset_state` clears the map after warmup.
- `state_slots` attached by both attention-metadata builders:
  `_build_attention_metadata` (real steps) and
  `_build_warmup_attention_metadata` (identity `arange` — keeps compiled
  graph shapes matching runtime). `execute_dummy_batch` is an overwrite site,
  not a third builder: after `_build_decode_synthetic_inputs` creates the
  metadata, it replaces every `state_slots` tensor with all-scratch slots so DP
  dummy runs never touch live state.

### 2. `libtorch_neuronx_lite/fx_passes/aliasing_pass.py` — `.data.copy_()` graph aliasing

**Fixes: gibberish output after the first decoded tokens** (decode state
silently not persisting across steps). Full analysis:
`docs/implementation-notes.md` §2. In short, `tensor.data` traces to
`_get_data_attr`, which the pass did not treat as an alias, so DeltaNet's
`buffer.data.copy_(new_state)` never mapped back to the buffer placeholder
and emitted no `io_map` entry.

Delta summary:

- New predicate `_is_data_attr_op(node)` — recognizes `_get_data_attr` as
  an aliasing op.
- `_is_aliasing_op` extended to include it (one line).

That is the entire delta — the file is otherwise byte-identical to the shipped
2.11.0.1.0.1284 source, so it diffs cleanly against it.

To confirm the fix is live, look for a populated `io_map` in the trace log —
two entries per hybrid layer:

```
FX Pass metadata: {... 'aliasing_output_rewrite': {'io_map': {2: 17, 3: 18, ...}}}
```

An empty `io_map` means the override is not installed (or the compile cache
still holds a graph built without it).
