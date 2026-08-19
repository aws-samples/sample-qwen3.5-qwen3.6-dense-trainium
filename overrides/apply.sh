#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the REQUIRED core-file overrides into the active environment and
# clear the NEFF compile cache.
#
#   ./overrides/apply.sh
#
# The two files live in DIFFERENT installed packages (the runner in
# vllm_neuron, the fx-pass / NKI-HOP machinery in libtorch-neuronx-lite), so
# each top-level directory under overrides/ is named for its target package
# and the tree below it mirrors that package's layout.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Importing either package emits plugin/log chatter on stdout, so tag the
# lines we care about and filter on the tag.
roots=$(python - <<'PY' | sed -n 's/^@ROOT //p'
import os
import libtorch_neuronx_lite
import vllm_neuron
print("@ROOT vllm_neuron", os.path.dirname(vllm_neuron.__file__))
print("@ROOT libtorch_neuronx_lite", os.path.dirname(libtorch_neuronx_lite.__file__))
PY
)

while read -r pkg root; do
    src_dir="${HERE}/${pkg}"
    [ -d "$src_dir" ] || continue
    while IFS= read -r -d '' src; do
        rel="${src#"${src_dir}/"}"
        dst="${root}/${rel}"
        if [ ! -f "$dst" ]; then
            echo "ERROR: target does not exist: $dst" >&2
            echo "       The installed version may not match this branch's pin." >&2
            exit 1
        fi
        cp "$src" "$dst"
        echo "installed ${pkg}/${rel}"
    done < <(find "$src_dir" -name '*.py' -print0)
done <<< "$roots"

# Clear the NEFF compile cache after ANY change to these files. 0.24 resolves
# the path via libtorch_neuronx_lite (NEURON_LIBTORCH_CACHE_ROOT, then
# VLLM_CACHE_ROOT, then ~/.cache/neuron_libtorch, falling back to
# /tmp/neuron_compile_wdir_$USER on NFS homes) — ask the library rather than
# guessing.
cache_dir=$(python -c \
    "import libtorch_neuronx_lite.envs as e; print('@CACHE', e.get_neuron_compile_cache_dir())" \
    | sed -n 's/^@CACHE //p')
rm -rf "$cache_dir"
echo "cleared compile cache: $cache_dir"
