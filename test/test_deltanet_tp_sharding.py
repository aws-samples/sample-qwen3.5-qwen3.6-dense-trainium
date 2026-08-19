# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline oracle tests for DeltaNet TP head sharding (no Neuron device).

Validates, in pure PyTorch on CPU:
  A. The divisibility guard raises a clear ValueError (never mis-shards).
  B. Weight-loader slice offsets: per-rank shards loaded through the REAL
     attached loaders reconstruct the full weights exactly.
  C. Forward equivalence: the REAL `_forward_prefill` / `_forward_decode`
     run on full (ws=1) vs sharded (ws=2, 4) modules — with fake TP
     collectives that expose the partial outputs — must agree to fp32
     tolerance when the partials are summed. Multi-step decode after
     prefill exercises state carry through the shrunk local buffers.
  D. Batched-generation state isolation (see docs/implementation-notes.md,
     "Output crossing under continuous batching — the state-slot fix"):
     sequential prefills must not clobber other requests' state rows;
     batched decode must follow request-stable slots (including after a
     batch-position swap); scratch-slot writes must stay isolated; a freed
     slot reused by a new request must not disturb survivors.

The NKI kernel is monkeypatched with a faithful pure-torch sequential
recurrence (same math as `_forward_decode`'s single step) — the kernel
itself is untouched by the sharding change, and this keeps the oracle
runnable without a device. Follow it with the on-device check in
`test/batched_consistency_test.py`.

Usage:  python test/test_deltanet_tp_sharding.py   (from the repo root;
        requires the vllm-neuron plugin installed — CPU only, no device)
"""

import sys
import types

import torch

# Make the overlay package importable when run from the repo root or test/.
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import qwen3_6.model_bf16 as m
import qwen3_6.nki_kernels as nk
from vllm_neuron.utils.weight_loader import get_weight_loader

torch.manual_seed(0)

# ── Small test config (head dims must be 128 per the kernel guard) ─────────
HIDDEN = 256
NUM_V_HEADS = 8
NUM_K_HEADS = 4
HEAD_DIM = 128
KERNEL = 4
KEY_DIM = NUM_K_HEADS * HEAD_DIM      # 512
VALUE_DIM = NUM_V_HEADS * HEAD_DIM    # 1024
CONV_DIM = 2 * KEY_DIM + VALUE_DIM    # 2048
DTYPE = torch.float32                 # fp32 so the oracle tolerance is tight


def make_config(num_v=NUM_V_HEADS, num_k=NUM_K_HEADS):
    return types.SimpleNamespace(
        torch_dtype=DTYPE,
        rms_norm_eps=1e-6,
        hidden_size=HIDDEN,
        deltanet_num_v_heads=num_v,
        deltanet_num_k_heads=num_k,
        deltanet_k_head_dim=HEAD_DIM,
        deltanet_v_head_dim=HEAD_DIM,
        deltanet_conv_kernel_size=KERNEL,
        neuron_config=None,
    )


class FakeTPGroup:
    """Fake TP group: runs each rank's module in-process and records the
    partial tensors that the real collectives would sum across ranks."""

    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank_in_group = rank
        self.recorded = []  # partials passed to reduce_scatter/all_reduce

    def all_gather(self, x, dim=0):
        # Tests always feed the FULL sequence, so gather is identity.
        return x

    def reduce_scatter(self, x, dim=0):
        self.recorded.append(x.clone())
        chunk = x.shape[dim] // self.world_size
        return x.narrow(dim, self.rank_in_group * chunk, chunk)

    def all_reduce(self, x):
        self.recorded.append(x.clone())
        return x


def pure_torch_deltanet_kernel(query, key, value, g_in, beta_in,
                               lower_mask, identity, lower_mask_diag):
    """Sequential gated-delta-rule recurrence — same math as the NKI kernel
    and as `_forward_decode`'s single step, applied token by token."""
    S = query.shape[0]
    q = query.float()
    k = key.float()
    v = value.float()
    g = g_in.float().squeeze(-1)
    beta = beta_in.float().squeeze(-1)
    state = torch.zeros(HEAD_DIM, HEAD_DIM, dtype=torch.float32)
    out = torch.zeros(S, HEAD_DIM, dtype=torch.float32)
    for t in range(S):
        state = state * torch.exp(g[t])
        kv_mem = (state * k[t].unsqueeze(-1)).sum(dim=-2)
        delta = (v[t] - kv_mem) * beta[t]
        state = state + k[t].unsqueeze(-1) * delta.unsqueeze(-2)
        out[t] = (state * q[t].unsqueeze(-1)).sum(dim=-2)
    return out.to(query.dtype), state


def make_full_weights():
    w = {
        "in_proj_qkv_weight": torch.randn(CONV_DIM, HIDDEN, dtype=DTYPE) * 0.02,
        "in_proj_z_weight": torch.randn(VALUE_DIM, HIDDEN, dtype=DTYPE) * 0.02,
        "in_proj_a_weight": torch.randn(NUM_V_HEADS, HIDDEN, dtype=DTYPE) * 0.02,
        "in_proj_b_weight": torch.randn(NUM_V_HEADS, HIDDEN, dtype=DTYPE) * 0.02,
        "conv1d_weight": torch.randn(CONV_DIM, 1, KERNEL, dtype=DTYPE) * 0.2,
        "A_log": torch.randn(NUM_V_HEADS, dtype=torch.float32) * 0.1,
        "dt_bias": torch.randn(NUM_V_HEADS, dtype=torch.float32) * 0.1,
        "norm_weight": torch.randn(HEAD_DIM, dtype=DTYPE) * 0.1 + 1.0,
        "out_proj_weight": torch.randn(HIDDEN, VALUE_DIM, dtype=DTYPE) * 0.02,
    }
    return w


class TensorShim:
    """Duck-types safetensors' PySafeSlice over a plain tensor."""

    def __init__(self, t):
        self._t = t

    def get_shape(self):
        return list(self._t.shape)

    def __getitem__(self, idx):
        return self._t[idx]


def build_module(config, world_size, rank, full_weights):
    group = FakeTPGroup(world_size, rank)
    orig = m.get_tp_group
    m.get_tp_group = lambda: group
    try:
        layer = m.Qwen3_6DeltaNetAttention(config, layer_idx=0)
    finally:
        m.get_tp_group = orig
    for name, param in layer.named_parameters():
        loader = get_weight_loader(param)
        shard = loader.load([TensorShim(full_weights[name])], rank)
        assert shard.shape == param.shape, (
            f"{name}: loader produced {tuple(shard.shape)}, "
            f"param is {tuple(param.shape)} (ws={world_size}, rank={rank})"
        )
        param.data.copy_(shard.to(param.dtype))
    return layer, group


def qkv_block_slices(ws, rank):
    """Expected [Q|K|V] row indices for `rank` of `ws` (the sharding map)."""
    kd_loc, vd_loc = KEY_DIM // ws, VALUE_DIM // ws
    rows = (
        list(range(rank * kd_loc, (rank + 1) * kd_loc))
        + list(range(KEY_DIM + rank * kd_loc, KEY_DIM + (rank + 1) * kd_loc))
        + list(range(2 * KEY_DIM + rank * vd_loc, 2 * KEY_DIM + (rank + 1) * vd_loc))
    )
    return torch.tensor(rows)


# ── A. Divisibility guard ───────────────────────────────────────────────────

def test_guard():
    # v=8, k=4 not divisible by ws=3
    try:
        build_module(make_config(), 3, 0, make_full_weights())
    except ValueError as e:
        assert "divisible" in str(e), e
    else:
        raise AssertionError("ws=3 with 8/4 heads should raise ValueError")
    # v=12, k=8 divisible by ws=4 but v_loc=3 not a multiple of k_loc=2
    try:
        cfg = make_config(num_v=12, num_k=8)
        group = FakeTPGroup(4, 0)
        orig = m.get_tp_group
        m.get_tp_group = lambda: group
        try:
            m.Qwen3_6DeltaNetAttention(cfg, layer_idx=0)
        finally:
            m.get_tp_group = orig
    except ValueError as e:
        assert "key-head groups" in str(e), e
    else:
        raise AssertionError("v_loc %% k_loc != 0 should raise ValueError")
    print("A. divisibility guard: PASS")


# ── B. Weight-loader slice offsets ─────────────────────────────────────────

def test_loaders():
    full = make_full_weights()
    for ws in (1, 2, 4):
        for rank in range(ws):
            layer, _ = build_module(make_config(), ws, rank, full)
            rows = qkv_block_slices(ws, rank)
            v_loc = NUM_V_HEADS // ws
            vd_loc = VALUE_DIM // ws
            checks = {
                "in_proj_qkv_weight": full["in_proj_qkv_weight"][rows],
                "conv1d_weight": full["conv1d_weight"][rows],
                "in_proj_z_weight": full["in_proj_z_weight"][rank * vd_loc:(rank + 1) * vd_loc],
                "in_proj_a_weight": full["in_proj_a_weight"][rank * v_loc:(rank + 1) * v_loc],
                "in_proj_b_weight": full["in_proj_b_weight"][rank * v_loc:(rank + 1) * v_loc],
                "A_log": full["A_log"][rank * v_loc:(rank + 1) * v_loc],
                "dt_bias": full["dt_bias"][rank * v_loc:(rank + 1) * v_loc],
                "norm_weight": full["norm_weight"],
                "out_proj_weight": full["out_proj_weight"][:, rank * vd_loc:(rank + 1) * vd_loc],
            }
            for name, expected in checks.items():
                got = dict(layer.named_parameters())[name].data
                assert torch.equal(got, expected.to(got.dtype)), (
                    f"{name} mis-sharded at ws={ws} rank={rank}"
                )
    print("B. weight-loader slice offsets: PASS")


# ── C. Forward equivalence (prefill + multi-step decode state carry) ───────

def _slots(*rows):
    return torch.tensor(rows, dtype=torch.long)


def test_forward_equivalence():
    nk.call_deltanet_fused = pure_torch_deltanet_kernel

    full = make_full_weights()
    seq_len = 24
    n_decode = 4
    prefill_hidden = torch.randn(seq_len, HIDDEN, dtype=DTYPE) * 0.5
    decode_hiddens = [torch.randn(1, HIDDEN, dtype=DTYPE) * 0.5 for _ in range(n_decode)]

    # Reference: ws=1 (no collectives fire)
    ref_layer, _ = build_module(make_config(), 1, 0, full)
    ref_prefill = ref_layer._forward_prefill(prefill_hidden.clone(), _slots(0))
    # Snapshot post-prefill state BEFORE decode mutates the buffers.
    ref_rec_state = ref_layer.recurrent_state_buffer.clone()
    ref_conv_state = ref_layer.conv_state_buffer.clone()
    ref_decode = [
        ref_layer._forward_decode(h.clone(), _slots(0)) for h in decode_hiddens
    ]

    for ws in (2, 4):
        layers, groups = zip(*(build_module(make_config(), ws, r, full) for r in range(ws)))

        # Prefill: every rank sees the full sequence (fake all_gather is
        # identity); sum the partials each rank handed to reduce_scatter.
        for layer in layers:
            layer._forward_prefill(prefill_hidden.clone(), _slots(0))
        prefill_sum = sum(g.recorded.pop() for g in groups)
        err = (prefill_sum - ref_prefill).abs().max().item()
        assert err < 1e-3, f"prefill mismatch at ws={ws}: max abs err {err}"

        # Recurrent state: concat of local head slices == full state.
        state_cat = torch.cat([l.recurrent_state_buffer for l in layers], dim=1)
        serr = (state_cat - ref_rec_state).abs().max().item()
        assert serr < 1e-4, f"recurrent state mismatch at ws={ws}: {serr}"

        # Conv state: per-rank buffer == full buffer at that rank's rows.
        for r, layer in enumerate(layers):
            expected = ref_conv_state[:, qkv_block_slices(ws, r)]
            cerr = (layer.conv_state_buffer - expected).abs().max().item()
            assert cerr < 1e-5, f"conv state mismatch at ws={ws} rank={r}: {cerr}"

        # Multi-step decode after prefill: state carry through local buffers.
        for step, h in enumerate(decode_hiddens):
            for layer in layers:
                layer._forward_decode(h.clone(), _slots(0))
            decode_sum = sum(g.recorded.pop() for g in groups)
            derr = (decode_sum - ref_decode[step]).abs().max().item()
            assert derr < 1e-3, (
                f"decode step {step} mismatch at ws={ws}: max abs err {derr}"
            )
        print(f"C. forward equivalence at ws={ws}: PASS "
              f"(prefill err {err:.2e}, state err {serr:.2e})")


# ── D. Batched-generation state isolation (the state-crossing fix) ─────────

def test_state_isolation():
    """Repro of the state-crossing bug (docs/implementation-notes.md,
    "Output crossing under continuous batching"), now expected green.

    Scenario: prefill A (slot 0), prefill B (slot 1) — B's prefill must NOT
    clobber A's state row — then batched decodes addressed by slots must
    match each request's single-request reference, including with swapped
    batch positions and after slot reuse.
    """
    nk.call_deltanet_fused = pure_torch_deltanet_kernel
    import types
    cfg = make_config()
    cfg.neuron_config = types.SimpleNamespace(max_batch_size=2)

    full = make_full_weights()
    promptA = torch.randn(16, HIDDEN, dtype=DTYPE) * 0.5
    promptB = torch.randn(16, HIDDEN, dtype=DTYPE) * 0.5
    promptC = torch.randn(16, HIDDEN, dtype=DTYPE) * 0.5
    dec = torch.randn(1, HIDDEN, dtype=DTYPE) * 0.5

    def single_ref(prompt, n_dec=2):
        layer, _ = build_module(cfg, 1, 0, full)
        layer._forward_prefill(prompt.clone(), _slots(0))
        return [layer._forward_decode(dec.clone(), _slots(0)) for _ in range(n_dec)]

    refA, refB, refC = single_ref(promptA), single_ref(promptB), single_ref(promptC)

    # Server-like sequence on ONE layer: prefill A -> slot 0, prefill B -> slot 1.
    layer, _ = build_module(cfg, 1, 0, full)
    layer._forward_prefill(promptA.clone(), _slots(0))
    stateA = layer.recurrent_state_buffer[0].clone()
    layer._forward_prefill(promptB.clone(), _slots(1))
    assert torch.equal(layer.recurrent_state_buffer[0], stateA), (
        "B's prefill clobbered A's state row (broadcast bug regressed)"
    )

    # Batched decode step 1: batch order [A, B] -> slots [0, 1].
    out = layer._forward_decode(torch.cat([dec, dec], 0), _slots(0, 1))
    assert torch.allclose(out[0], refA[0][0], atol=1e-4), "row A != A's reference"
    assert torch.allclose(out[1], refB[0][0], atol=1e-4), "row B != B's reference"

    # Batched decode step 2 with SWAPPED batch positions (InputBatch
    # condense/swap): batch order [B, A] -> slots [1, 0].
    out = layer._forward_decode(torch.cat([dec, dec], 0), _slots(1, 0))
    assert torch.allclose(out[0], refB[1][0], atol=1e-4), "swapped row B wrong"
    assert torch.allclose(out[1], refA[1][0], atol=1e-4), "swapped row A wrong"

    # Scratch-slot isolation: a padded decode row pointing at scratch (=2)
    # must not disturb live rows. (Fresh layer state: A after 2 decode steps.)
    stateA2 = layer.recurrent_state_buffer[0].clone()
    layer._forward_decode(torch.cat([dec, dec], 0), _slots(2, 1))  # row0 = pad
    assert torch.equal(layer.recurrent_state_buffer[0], stateA2), (
        "scratch-slot write leaked into a live slot"
    )

    # Slot reuse: A finishes, slot 0 freed, new request C prefills into slot
    # 0. B (slot 1) must be undisturbed and both decode correctly.
    stateB = layer.recurrent_state_buffer[1].clone()
    layer._forward_prefill(promptC.clone(), _slots(0))
    assert torch.equal(layer.recurrent_state_buffer[1], stateB), (
        "C's prefill into reused slot 0 disturbed B's slot 1"
    )
    out = layer._forward_decode(torch.cat([dec, dec], 0), _slots(0, 1))
    assert torch.allclose(out[0], refC[0][0], atol=1e-4), "reused slot C wrong"

    print("D. batched state isolation: PASS")


if __name__ == "__main__":
    test_guard()
    test_loaders()
    test_forward_equivalence()
    test_state_isolation()
    print("ALL ORACLE CHECKS PASS")
    sys.exit(0)
