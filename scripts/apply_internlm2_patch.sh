#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! -f "$1/modeling_internlm2.py" ]]; then
    echo "Usage: bash scripts/apply_internlm2_patch.sh /absolute/path/to/InternVL2-8B" >&2
    exit 2
fi

model_dir=$(cd -- "$1" && pwd -P)
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
patch_file="$script_dir/../patches/internlm2-generation-last-token.patch"

if git -C "$model_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "InternLM2 generation patch already applied: $model_dir"
elif git -C "$model_dir" apply --check "$patch_file"; then
    git -C "$model_dir" apply "$patch_file"
    echo "Applied InternLM2 generation patch: $model_dir"
else
    echo "Model source does not match this patch; inspect it before applying." >&2
    exit 1
fi
