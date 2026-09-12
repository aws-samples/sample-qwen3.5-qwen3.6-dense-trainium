#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
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
EXPECTED_VLLM_NEURON_VERSION="0.24.0.1.1.0"
EXPECTED_LIBTORCH_NEURONX_LITE_VERSION="2.11.0.1.0.1284"

# Importing either package emits plugin/log chatter on stdout, so tag the
# lines we care about and filter on the tag. Include distribution versions so
# a different stack fails before either installed file is changed.
roots="$(python - <<'PY' | sed -n 's/^@ROOT //p'
import importlib.metadata
import os

import libtorch_neuronx_lite
import vllm_neuron

print(
    "@ROOT vllm_neuron",
    importlib.metadata.version("vllm-neuron"),
    os.path.dirname(vllm_neuron.__file__),
)
print(
    "@ROOT libtorch_neuronx_lite",
    importlib.metadata.version("libtorch-neuronx-lite"),
    os.path.dirname(libtorch_neuronx_lite.__file__),
)
PY
)"

sources=()
targets=()
while read -r pkg version root; do
    [[ -n "${pkg}" && -n "${version}" && -n "${root}" ]] || continue

    case "${pkg}" in
        vllm_neuron)
            expected_version="${EXPECTED_VLLM_NEURON_VERSION}"
            ;;
        libtorch_neuronx_lite)
            expected_version="${EXPECTED_LIBTORCH_NEURONX_LITE_VERSION}"
            ;;
        *)
            echo "ERROR: unexpected package record: ${pkg}" >&2
            exit 1
            ;;
    esac
    # PEP 440 local versions append build metadata after `+` (for example,
    # 2.11.0.1.0.1284+f49d8626). The source-compatible base version must
    # match exactly; the wheel's local build identifier may vary.
    base_version="${version%%+*}"
    if [[ "${base_version}" != "${expected_version}" ]]; then
        echo "ERROR: ${pkg} version ${version} does not match base ${expected_version}" >&2
        exit 1
    fi

    src_dir="${HERE}/${pkg}"
    [[ -d "${src_dir}" ]] || {
        echo "ERROR: override directory does not exist: ${src_dir}" >&2
        exit 1
    }

    while IFS= read -r -d '' src; do
        rel="${src#"${src_dir}/"}"
        dst="${root}/${rel}"
        if [[ ! -f "${dst}" ]]; then
            echo "ERROR: target does not exist: ${dst}" >&2
            echo "       The installed version may not match this branch's pin." >&2
            exit 1
        fi
        sources+=("${src}")
        targets+=("${dst}")
    done < <(find "${src_dir}" -name '*.py' -print0)
done <<< "${roots}"

if ((${#sources[@]} != 2)); then
    echo "ERROR: expected exactly 2 override files, found ${#sources[@]}" >&2
    exit 1
fi

# Resolve and canonicalize the cache path before changing either installed
# file. Standard locations are accepted automatically; custom cache roots
# require an explicit opt-in after broad system paths are rejected.
cache_dir="$(python - <<'PY' | sed -n 's/^@CACHE //p'
import os
import libtorch_neuronx_lite.envs as envs

print("@CACHE", os.path.realpath(os.path.expanduser(envs.get_neuron_compile_cache_dir())))
PY
)"
home_cache="$(python -c 'import os; print(os.path.realpath(os.path.expanduser("~/.cache/neuron_libtorch")))')"
tmp_cache_root="/tmp/neuron_compile_wdir_$(id -un)"

case "${cache_dir}" in
    ""|/|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var|"${HOME}")
        echo "ERROR: refusing to clear unsafe cache path: ${cache_dir:-<empty>}" >&2
        exit 1
        ;;
    "${home_cache}"|"${home_cache}/"*|"${tmp_cache_root}"|"${tmp_cache_root}/"*)
        ;;
    /*)
        if [[ "${QWEN36_ALLOW_CUSTOM_CACHE_DELETE:-0}" != "1" ]]; then
            echo "ERROR: refusing custom cache path without explicit opt-in: ${cache_dir}" >&2
            echo "       Verify it, then set QWEN36_ALLOW_CUSTOM_CACHE_DELETE=1." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: refusing non-absolute cache path: ${cache_dir}" >&2
        exit 1
        ;;
esac
if [[ -e "${cache_dir}" && ! -d "${cache_dir}" ]]; then
    echo "ERROR: cache path exists but is not a directory: ${cache_dir}" >&2
    exit 1
fi

# Stage both replacements and backups before changing either target. The EXIT
# trap restores both originals if a later replacement or cache clear fails.
tmp_files=()
backup_files=()
rollback_required=0
cleanup_or_rollback() {
    exit_code=$?
    rollback_failed=0
    if ((rollback_required)); then
        for i in "${!backup_files[@]}"; do
            if [[ -f "${backup_files[$i]}" ]]; then
                if mv -f -- "${backup_files[$i]}" "${targets[$i]}"; then
                    echo "restored ${targets[$i]}" >&2
                else
                    echo "ERROR: could not restore ${targets[$i]}" >&2
                    echo "       backup retained at ${backup_files[$i]}" >&2
                    rollback_failed=1
                fi
            fi
        done
    fi
    if ((${#tmp_files[@]})); then
        rm -f -- "${tmp_files[@]}"
    fi
    if ((rollback_required == 0 || rollback_failed == 0)); then
        if ((${#backup_files[@]})); then
            rm -f -- "${backup_files[@]}"
        fi
    fi
    if ((rollback_failed)); then
        exit_code=1
    fi
    trap - EXIT
    exit "${exit_code}"
}
trap cleanup_or_rollback EXIT

for i in "${!sources[@]}"; do
    dst="${targets[$i]}"
    tmp="${dst}.qwen36-tmp.$$"
    backup="${dst}.qwen36-backup.$$"
    tmp_files+=("${tmp}")
    backup_files+=("${backup}")
    cp "${sources[$i]}" "${tmp}"
    chmod --reference="${dst}" "${tmp}"
    cp -p "${dst}" "${backup}"
done

rollback_required=1
for i in "${!targets[@]}"; do
    mv -f -- "${tmp_files[$i]}" "${targets[$i]}"
    echo "installed ${targets[$i]}"
done

# Clear cache contents, but preserve the validated cache root itself.
if [[ -d "${cache_dir}" ]]; then
    find "${cache_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
fi
rollback_required=0
echo "cleared compile cache: ${cache_dir}"
