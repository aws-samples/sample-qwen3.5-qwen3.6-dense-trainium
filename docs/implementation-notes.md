# Implementation Notes

Three things in this package are non-obvious and generalize to any
linear-attention / hybrid model on the vLLM-Neuron stack. Read this before
porting a sibling model or debugging output quality.

## 1. DeltaNet TP head-sharding

DeltaNet heads are fully independent through the recurrence, the per-head
RMSNorm, and the z-gate; only `out_proj` mixes them. So the layer shards
cleanly by value head:

- Row-shard the input projections, decay params, and recurrent/conv state by
  value head; column-shard `out_proj`; replicate the tiny per-head-dim norm.
- Reduce the partial `out_proj` outputs across ranks — `reduce_scatter` on
  prefill, `all_reduce` on decode — exactly like the GQA and MLP layers.
- The fused `[Q|K|V]` input projection and the depthwise `conv1d` must be
  sliced with the **same** per-block offsets (Q and K by key head, V by value
  head), so each rank owns whole key-head groups.

Two guards at layer init (clear `ValueError`s, never a silent mis-shard):
`num_v_heads % TP == 0` and `num_k_heads % TP == 0`, and per-rank v-heads a
multiple of per-rank k-heads.

Pitfall: DeltaNet weights are `[out, in]` and used via plain `F.linear` — do
**not** copy the GQA loaders' `is_storage_transposed=True`. A transposed load
produces plausible-looking garbage rather than a crash.

Sharding replaces (does not supplement) an earlier replicated implementation.
Correctness was verified against that replicated path as an oracle: an offline
weight-reconstruction check on the loader slice offsets (`torch.equal`), and a
sharded-vs-full forward-equivalence check in fp32 over prefill plus multi-step
decode (max abs err 2.05e-08).

Note that on-device output is NOT bit-identical across TP degrees, and should
not be expected to be — see the README's Validation Summary for the measured
behavior.

## 2. Silent state-drop in the compiled graph (the aliasing fix)

DeltaNet persists recurrent/conv state between the separately-compiled prefill
and decode graphs via `buffer.data.copy_(new_state)`. `tensor.data` traces to
`_get_data_attr`, which the fx aliasing pass did not recognize as an alias — so
the mutation never mapped back to the buffer placeholder and the state update
was **dropped between graphs**. Generation ran fine for a few tokens, then
collapsed into loops/gibberish.

Fix: one predicate in
`overrides/libtorch_neuronx_lite/fx_passes/aliasing_pass.py` that recognizes
`_get_data_attr` as an aliasing op. (That pass ships in the standalone
`libtorch-neuronx-lite` distribution, not in `vllm_neuron` — see
`overrides/README.md`.) Also load-bearing: keep the recurrent state
**fp32** — re-quantizing to bf16 every decode step compounds and collapses
generation.

You can confirm the fix is active in a trace log: the
`aliasing_output_rewrite` pass reports a populated `io_map` (two entries per
hybrid layer). An empty `io_map` means the state writes were dropped.

## 3. Output crossing under continuous batching (the state-slot fix)

DeltaNet carries recurrent/conv state in per-layer side-channel buffers, not
the paged KV cache. The original buffers were indexed by **batch position**,
which is correct under static batching but wrong under vLLM's continuous
batching in two ways:

- The prefill write `final_state + buffer * 0` broadcasts one sequence's state
  over **every** row (the Neuron scheduler prefills one sequence per step), so
  the last prefill clobbers all other requests' state.
- Batch position stops being a stable request identity once requests finish and
  the batch condenses/swaps rows.

Result: prompts answered with other prompts' content.

Fix — **request-stable state slots**, mirroring the paged-KV idiom. The runner
owns a `req_id → slot` map (allocate at first prefill, free on finish/preempt)
and threads a per-step `state_slots` tensor through `attn_metadata`; the model
gathers/scatters state by slot, with a scratch row for padded batch positions.
Buffers are sized `[max_batch + 1, ...]`; the extra row is the scratch slot
that padded decode rows and dummy batches write to and is never read back.
Writes stay `.data.copy_()` so the aliasing fix (#2) keeps covering them.

Model side: `src/qwen3_6/model_bf16.py` (`Qwen3_6DeltaNetAttention`).
Runner side: `overrides/vllm_neuron/vllm/worker/neuron_model_runner.py`.

The runner-side release policy hangs off the single `_update_states` call site
in `execute_model` rather than off edits inside `_update_states`, which upstream
marks as a verbatim copy of `GPUModelRunner._update_states` — keeping that
method untouched means it can be replaced wholesale on the next rebase.

Verified with an offline state-isolation oracle (prefill clobber, slot-following
decode, row swap, scratch isolation, slot reuse) and an on-device churn test
(more prompts than `max_num_seqs`, staggered `max_tokens`) where every batched
output was token-exact vs its single-prompt reference.
