#!/usr/bin/env bash
# Memory plan. BACKEND=cuda counts 2-byte activations; anything else counts 4.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib.sh
source "$root/scripts/lib.sh"
cd "$root"
bin="$(fw_bin "$root")"
fw_need_bin "$bin"

args=(
  plan
  --vram-gib "${VRAM_GIB:-12}"
  --backend "${BACKEND:-auto}"
  --width "${WIDTH:-832}"
  --height "${HEIGHT:-480}"
  --frames "${FRAMES:-81}"
)
if [[ -n "${MAX_BINDING_MIB:-}" ]]; then
  args+=(--max-binding-mib "$MAX_BINDING_MIB")
fi
exec "$bin" "${args[@]}" "$@"
