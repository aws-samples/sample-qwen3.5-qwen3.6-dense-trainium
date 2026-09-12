# Qwen3.5 / Qwen3.6 Dense Models on AWS Trainium2 (vLLM-Neuron)

Run Alibaba's **Qwen3.5 and Qwen3.6 dense models** — the hybrid
GatedDeltaNet + GQA family — on AWS Trainium2 through the PyTorch-native
vLLM-Neuron plugin, in **both inference modes**:

- **Online** — the OpenAI-compatible server via `vllm serve`
- **Offline** — in-process batch inference via `LLM.generate()`

Both modes share the same model package, the same required overrides, and the
same continuous-batching state machinery, so a fix verified in one mode holds
in the other. **One package serves the whole dense family** — every
model-specific dimension is read from the HF config at load time, and all four
published dense models (Qwen3.5-4B/9B/27B and Qwen3.6-27B) are validated on
device.

Inference-focused: this package covers online serving and offline batch
inference (no training).

## Model Information

- **HuggingFace IDs:** [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) ·
  [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) ·
  [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) ·
  [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B)
  (all four validated on device; the smaller 0.8B / 2B siblings are expected
  to work by construction but were not run)
- **Model type:** hybrid decoder — linear-attention (GatedDeltaNet) +
  full-attention (GQA) in a `[3 × DeltaNet + 1 × GQA]` repeating block;
  multimodal-capable checkpoints served text-only
- **Shared architecture:** partial RoPE (25% of head_dim 256, NeoX
  rotate_half), per-head QK-norm, attention output gate spliced into q_proj,
  `(1 + weight)` RMSNorm convention, SwiGLU MLP, DeltaNet head_dim 128
- **Per-model dims (read from HF config):**

  | Model | Layers (DN+GQA) | Hidden | Q/KV heads | DN v/k heads | lm_head |
  |---|---|---|---|---|---|
  | Qwen3.5-4B | 24 + 8 | 2560 | 16 / 4 | 32 / 16 | tied |
  | Qwen3.5-9B | 24 + 8 | 4096 | 16 / 4 | 32 / 16 | untied |
  | Qwen3.5-27B | 48 + 16 | 5120 | 24 / 4 | 48 / 16 | untied |
  | Qwen3.6-27B | 48 + 16 | 5120 | 24 / 4 | 48 / 16 | untied |

- **HF arch class:** `Qwen3_5ForConditionalGeneration` for both 3.5 and 3.6
  (the class name is shared across the version bump), so one registry slot
  serves the family
- **License:** Except for files explicitly marked `Apache-2.0`, this
  repository's code and documentation are licensed under MIT-0. Bundled or
  adapted files marked `Apache-2.0`, and the separately downloaded model
  weights, retain Apache License 2.0. See `LICENSE`,
  `LICENSE-APACHE-2.0`, `NOTICE`, and `THIRD-PARTY-LICENSES`.

## Validation Summary

Validated on vllm-neuron `0.24.0.1.1.0` with vllm `0.24.0` (see Compatibility
Matrix): trn2.3xlarge (LNC=2), greedy sampling, both online and offline paths.

Verified on device:

| Model | TP | Offline `generate()` | Churn test (token-exact) | Online serve + concurrent API |
|---|---|---|---|---|
| Qwen3.5-4B | 1 | PASS | PASS | — |
| Qwen3.5-4B | 4 | PASS | PASS | PASS |
| Qwen3.5-9B | 4 | PASS | PASS | PASS |
| Qwen3.5-27B | 4 | PASS | PASS | PASS |
| Qwen3.6-27B | 4 | PASS | PASS | PASS |

- **TP sharding correctness:** CPU oracle suite green — weight-reconstruction
  exact (`torch.equal`, ws=1/2/4), sharded-vs-full forward equivalence max err
  2.05e-08 (fp32, prefill + multi-step decode), divisibility guard raises.
- **Continuous batching:** the churn test (6 prompts > `max_num_seqs`=4,
  staggered `max_tokens`, mid-run batch condensation and slot reuse) is
  token-exact vs single-prompt references on every row of the table above.
  Four simultaneous HTTP completions each stay on their own topic.
- **Aliasing fix live:** the `aliasing_output_rewrite` pass reports a populated
  `io_map` — two entries per layer, so 64 on the 4B/9B and 128 on either 27B —
  meaning the DeltaNet state writes map back to their buffer placeholders.
- **27B serving (the sharding goal):** both 27B models fit the KV budget at
  TP=4 on trn2.3xlarge — 12.72 GiB HBM used / 11.28 GiB free per rank after
  weight load, 429,056 KV tokens. The 9B leaves far more room (4.24 GiB used /
  19.76 GiB free, 858,144 KV tokens). Replicating the DeltaNet weights instead
  of head-sharding them does not fit at 27B.
- **Config-driven equivalence, empirically:** Qwen3.6-27B reused the
  Qwen3.5-27B NEFF cache entry outright — identical dimensions produce an
  identical compiled graph, and only the weights differ.
- **Output is not bit-identical across TP degrees**, and should not be expected
  to be. Comparing the 4B's greedy output at TP=1 vs TP=4 on four prompts: two
  matched byte-for-byte, and two diverged mid-continuation. Both divergences
  occurred where the next token was a near-tie between interchangeable
  candidates (the position-4 entry of a list of European capitals; a degenerate
  `100%` loop), while the prompts with sharp distributions — integer counting
  and a `fibonacci` body — were identical. TP changes the collective reduction
  order, so bf16 partial sums land differently and a near-tied argmax can flip.
  Token-exactness is therefore asserted *within* a fixed TP (batched vs
  single-prompt, which is what the churn test checks), not across TP.

Not exercised on device: TP=1 for anything but the 4B (the larger models are
run at TP=4, the configuration the head-sharding work targets) and the smaller
0.8B / 2B family members. `docs/known-boundaries.md` lists the rest of the
untested envelope.

Throughput / TTFT benchmarking was out of scope (correctness-first); see
`docs/known-boundaries.md` §5.

## Usage

### Prerequisites

```bash
# 1. Weights (never commit these)
hf download Qwen/Qwen3.5-4B --local-dir "$HOME/models/Qwen3.5-4B"
# ... or Qwen/Qwen3.5-9B / Qwen/Qwen3.5-27B / Qwen/Qwen3.6-27B

# 2. Activate the pinned vllm-neuron environment (see Compatibility).
#    On an AWS Neuron image that provides the validated environment:
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
#    If your image uses a different path, activate an environment containing
#    the exact versions in the Compatibility Matrix instead. To create one,
#    follow the installation instructions for the pinned vLLM-Neuron release:
#    https://github.com/vllm-project/vllm-neuron/tree/v0.24.0.1.1.0
#    QWEN36_DISABLE_AUTO_REGISTER=1 is a debugging-only escape hatch; do not
#    set it for online or offline inference because workers need registration.

# 3. REQUIRED (both online and offline): apply the two core-file overrides
#    (state-slot lifecycle + aliasing fix). Without them, batched inference
#    crosses outputs and decode degrades into gibberish — silently.
#    The script installs both files (they live in DIFFERENT installed
#    packages) and clears the NEFF compile cache.
./overrides/apply.sh
```

See `overrides/README.md` for what each change does and the base version pin.

**Instances without EFA** (including trn2.3xlarge, the instance this repo is
validated on): the plugin resolves the EFA interface from a per-instance-family
BDF table and raises at engine startup when the device isn't there. All three
entry points call `configure_efa_affinity()` (`src/neuron_host_env.py`), which
sets `NEURON_SKIP_EFA_AFFINITY=1` only when the host genuinely has no
InfiniBand/EFA device — EFA affinity is a CPU-placement optimization, not a
correctness requirement. Set the variable yourself to override the detection.

### Online — serve (OpenAI-compatible API)

```bash
# 4B, TP=4, 4K context (defaults)
./src/serve.sh

# 27B
MODEL="$HOME/models/Qwen3.5-27B" TP=4 ./src/serve.sh

# quick-iteration smoke config (small buckets = fast compile)
MAX_LEN=256 BUCKET=256 MAX_NUM_SEQS=4 ./src/serve.sh
```

The launcher derives the whole coordinated vllm-neuron configuration from
those few env vars — buckets, sequence caps, the bounded `vision_neuron_config`
that official checkpoints force (they ship a `vision_config` even at zero
`--limit-mm-per-prompt`), and explicitly disabled chunked prefill / prefix
caching. `src/_serve_main.py`'s docstring documents every variable and the
exact argv it builds.

The one invariant to remember: prefill is single-shot, so the final `BUCKET`
entry must equal `MAX_LEN` (`BUCKET=128,256` with `MAX_LEN=256` is fine). The
launcher enforces this rather than letting DeltaNet see a split prompt.

`serve.sh` puts `src/` on PYTHONPATH; the bundled `sitecustomize.py`
auto-registers the architecture in every Python process (including vLLM worker
subprocesses).

#### Secure access to the vLLM API

A vLLM HTTP endpoint can serve requests from any client that can reach its
listening address. Never expose an unauthenticated vLLM port directly to the
public internet. A request URL containing `localhost` only identifies where
that particular client connects; it does not prove that the server itself is
bound only to the loopback interface. Verify both the vLLM bind address and the
EC2 network controls.

Prefer **Option 2 or Option 3**, and combine controls where practical for
defense in depth.

1. **AWS Security Group filtering — demonstration only; do not enable as-is.**
   For a controlled customer demonstration, an inbound rule can allow TCP port
   8000 only from explicitly trusted sources. A rule that permits
   `0.0.0.0/0` or `::/0` is not acceptable. Any Security Group or
   infrastructure-as-code lines that expose this port must remain commented
   out by default, for example:

   ```text
   # DEMO ONLY — leave commented until the controls below are in place.
   # Inbound TCP 8000 from <TRUSTED_SOURCE_IP>/32
   ```

   Uncomment the equivalent rule only after applying appropriate controls,
   such as replacing the placeholder with narrowly restricted source IP
   addresses, running the instance in a private VPC with no direct public
   route, and/or adding authentication. Remove temporary demo access when it
   is no longer needed. Security Group exposure alone is not the recommended
   way to protect a public production endpoint.

2. **vLLM API-key authentication — secure approach for authorized clients.**
   The bundled launcher maps `VLLM_API_KEY` to vLLM's `--api-key` argument.
   Supply the key at runtime rather than adding raw launcher arguments or
   placing it in a script. For an interactive local launch:

   ```bash
   printf "vLLM API key: "
   IFS= read -r -s VLLM_API_KEY && printf '\n'
   export VLLM_API_KEY
   ./src/serve.sh
   ```

   Clients must then send the key with each inference request:

   ```bash
   -H "Authorization: Bearer ${VLLM_API_KEY}"
   ```

   Store the key in a managed secret store or protected environment variable;
   do not commit it to this repository or place it directly in scripts. Rotate
   it when access changes, use TLS for any network connection so the key is not
   sent in cleartext, and continue to restrict network reachability.

3. **Bind vLLM to localhost — secure approach for local testing.**
   The launcher defaults to the loopback interface. The explicit equivalent
   is:

   ```bash
   HOST=127.0.0.1 ./src/serve.sh
   ```

   Do not set `HOST=0.0.0.0` for this mode. Run the client on the same instance,
   or use a separately secured and authenticated tunnel if remote access is
   required. The local `curl` example below assumes that the server has been
   bound to `127.0.0.1`; add the `Authorization` header shown above if API-key
   authentication is also enabled.

Query the running server:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${HOME}/models/Qwen3.5-4B\",\"prompt\":\"The capital of France is\",\"max_tokens\":20,\"temperature\":0}"
```

### Offline — batch inference (`LLM.generate()`)

No server needed — construct an `LLM` in-process and call `generate()`. The
bundled example submits 4 prompts in one continuous batch (exercising the same
state-slot machinery as online serving):

```bash
PYTHONPATH=src python src/run_offline_example.py \
    --model-checkpoint "$HOME/models/Qwen3.5-4B" --tp 4
```

Import `qwen3_6` before constructing the `LLM` so the architecture is
registered, and keep the `overrides/` files in place (same requirement as
online). See `src/run_offline_example.py` for a minimal template.

## The key fixes

Three failure modes were found and fixed during this enablement; all three
generalize to any linear-attention / hybrid model on this stack. Full detail in
`docs/implementation-notes.md`.

1. **DeltaNet TP head-sharding** — shard the linear-attention layers by value
   head (weights, compute, and recurrent state) instead of replicating them.
   This is what makes the 27B fit the KV budget at TP=4.
2. **Aliasing fix** — DeltaNet's `buffer.data.copy_()` state writes were
   silently dropped by the compiled-graph aliasing pass, degrading generation
   into gibberish. Fixed in
   `overrides/libtorch_neuronx_lite/fx_passes/aliasing_pass.py` (and keeping
   the recurrent state fp32).
3. **Request-stable state slots** — under continuous batching, batch-position
   state indexing crossed outputs between sequences. Fixed with a request→slot
   map threaded through `attn_metadata`.

## Compatibility Matrix

| Instance | TP | Models | Status |
|---|---|---|---|
| trn2.3xlarge (LNC=2) | 4 | 4B / 9B / 27B / Qwen3.6-27B | **VALIDATED** (correctness + batching, offline + online) |
| trn2.3xlarge (LNC=2) | 1 | 4B | VALIDATED (regression config) |
| larger trn2 / higher TP | divides head counts | any dense | expected by construction; not run |

TP constraints (validated at model init with clear errors): TP must divide the
DeltaNet v- and k-head counts (32/16 on 4B/9B, 48/16 on 27B). It must also
divide the GQA Q-head count (16 on 4B/9B, 24 on 27B), which is required by the
sharding math. TP=4 divides all of them.

Target stack:

| Component | Version (as validated) |
|---|---|
| vllm-neuron plugin | package `0.24.0.1.1.0`; public source pin tag [`v0.24.0.1.1.0`](https://github.com/vllm-project/vllm-neuron/tree/v0.24.0.1.1.0) on `release-0.24.0.1.1.0` |
| vllm | `0.24.0` |
| libtorch-neuronx-lite | base `2.11.0.1.0.1284`; validated wheel `2.11.0.1.0.1284+f49d8626` (ships the fx passes / NKI HOP) |
| transformers | `5.15.0` (the plugin requires `>=5.5.1,<6`) |
| torch / torch-xla | `2.11.0` / `2.11.0` |
| neuronx-cc | `2.27.5334.0` |
| nki | `0.6.0` |
| Neuron driver (dkms) / runtime-lib / collectives | `2.30.2.0` / `2.34.10.0` / `2.34.10.0` |
| Instance / logical-neuroncore-config | trn2.3xlarge, LNC=2 (4 cores, 96 GB) |
| Python | `3.12.3` |

The installed `neuron_model_runner.py` is **byte-identical** to the public
`v0.24.0.1.1.0` tag, so the runner override is that file plus additive changes.

### Stack gotchas worth knowing

Two plugin behaviors are easy to re-trip when changing the launcher config,
and both fail at startup rather than quietly.

1. **`vision_neuron_config` must state its buckets explicitly.**
   `NeuronPlatform._resolve_vision_auto_config` writes its derived
   `num_vision_tokens_buckets` back into `additional_config`, and
   `check_and_update_config` then runs again in the EngineCore process on the
   already-mutated copy — where it rejects the very combination it just created
   ("Cannot set both num_vision_tokens_buckets and max_vision_seq_len"). So the
   launcher and both Python entry points pass
   `num_vision_tokens_buckets: [block_size]` rather than `max_vision_seq_len`.
   That is the same value the auto path derives, but it takes the idempotent
   early-return branch. Setting `max_vision_seq_len` instead fails at startup.
2. **EFA affinity must be skipped on hosts without EFA** — see Prerequisites.

## Testing

```bash
# 1. CPU oracle suite — no device needed, runs in ~1 min. Covers the
#    divisibility guard, weight-loader slice offsets, sharded-vs-full forward
#    equivalence (ws=2/4), and batched state isolation.
python test/test_deltanet_tp_sharding.py

# 2. On-device batched-consistency (churn) test — pass/fail exit code.
#    Run per model; every published dense checkpoint is validated at TP=4.
for M in Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B Qwen3.6-27B; do
  PYTHONPATH=src python test/batched_consistency_test.py --model "$HOME/models/$M" --tp 4
done

# 3. Concurrent-request check against a running server (start it first with
#    MODEL="$HOME/models/Qwen3.5-4B" TP=4 MAX_LEN=256 BUCKET=256 ./src/serve.sh).
./test/concurrent_api_test.sh 8000
```

Expect ~10 min of cold compile for the first 4B smoke run and ~15 min for a
27B; subsequent runs at the same config hit the NEFF cache and start in
2–3 minutes. Qwen3.6-27B reuses the Qwen3.5-27B cache entry, so whichever you
compile second starts warm.

## Known Boundaries

Full detail in `docs/known-boundaries.md`. Highlights: chunked prefill with
DeltaNet is not implemented (keep `MAX_LEN` ≤ the largest prefill bucket); FP8
KV cache is present for GQA layers but unverified end-to-end; preemption slot
release is implemented but never exercised live; and performance is
correctness-first, with the per-`(batch, head)` prefill kernel loop as the
obvious optimization target.

## Repo Map

| Path | Contents |
|---|---|
| `src/qwen3_6/` | the model package (family-wide) |
| `src/serve.sh`, `src/_serve_main.py` | online: env-var-driven server launcher |
| `src/run_offline_example.py` | offline: `LLM.generate()` batch-inference example |
| `src/sitecustomize.py` | auto-registers the architecture in every Python process |
| `src/neuron_host_env.py` | host-env guards (EFA affinity on EFA-less instances) |
| `overrides/` | **required** core-file replacements + `apply.sh` + rationale |
| `docs/` | implementation notes (the three fixes) + known boundaries |
| `test/` | CPU oracle suite + on-device consistency tests |
| `LICENSE`, `LICENSE-APACHE-2.0`, `NOTICE`, `THIRD-PARTY-LICENSES` | MIT-0 project license + Apache-2.0 text and third-party attribution |
