# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-environment guards that must run before the Neuron worker starts.

Not model code — these adjust process env vars for properties of the *host*
that vllm-neuron cannot infer. Called from `_serve_main.py`,
`run_offline_example.py`, and `test/batched_consistency_test.py`.
"""

import logging
import os

logger = logging.getLogger(__name__)

_IB_CLASS_DIR = "/sys/class/infiniband"


def host_has_efa() -> bool:
    """True if this host exposes at least one InfiniBand/EFA device."""
    try:
        return any(os.scandir(_IB_CLASS_DIR))
    except FileNotFoundError:
        return False


def configure_efa_affinity() -> None:
    """Skip EFA affinity on hosts without EFA.

    `NeuronWorker._set_efa_affinity` resolves the EFA interface from a
    per-instance-FAMILY BDF table, so on a small single-device instance
    (trn2.3xlarge — the instance this repo is validated on) it looks up a BDF
    that belongs to the 48xlarge layout, finds no `infiniband` sysfs entry, and
    raises during engine startup:

        RuntimeError: No EFA device found at
        /sys/bus/pci/devices/0000:c9:00.0/infiniband ...

    EFA affinity is a CPU-placement performance optimization, not a
    correctness requirement, and the plugin ships `NEURON_SKIP_EFA_AFFINITY`
    for exactly this case. We set it only when the host genuinely has no EFA,
    so EFA-equipped instances keep the optimization.

    An explicit `NEURON_SKIP_EFA_AFFINITY` in the environment always wins.
    """
    if "NEURON_SKIP_EFA_AFFINITY" in os.environ:
        return
    if host_has_efa():
        return
    os.environ["NEURON_SKIP_EFA_AFFINITY"] = "1"
    logger.info(
        "No EFA device under %s — setting NEURON_SKIP_EFA_AFFINITY=1 "
        "(performance optimization only; set it explicitly to override).",
        _IB_CLASS_DIR,
    )
