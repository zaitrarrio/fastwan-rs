#!/usr/bin/env bash
# Time DMD steps at a real geometry. LAYERS=30 is the full DiT; 4 is a look.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib.sh
source "$root/scripts/lib.sh"
cd "$root"
bin="$(fw_bin "$root")"
fw_need_bin "$bin"

args=(
  bench
  --layers "${LAYERS:-4}"
  --width "${WIDTH:-832}"
  --height "${HEIGHT:-480}"
  --frames "${FRAMES:-81}"
  --backend "${BACKEND:-auto}"
  --weights "${WEIGHTS:-native}"
  --attention "${ATTENTION:-auto}"
)
if [[ -n "${VSA_SPARSITY:-}" ]]; then
  args+=(--vsa-sparsity "$VSA_SPARSITY")
fi
if [[ "${DECODE:-0}" == "1" ]]; then
  args+=(--decode)
fi
exec "$bin" "${args[@]}" "$@"
