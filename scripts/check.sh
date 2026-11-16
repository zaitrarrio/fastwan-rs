#!/usr/bin/env bash
# Every kernel and the tiny model against the CPU reference.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib.sh
source "$root/scripts/lib.sh"
cd "$root"
bin="$(fw_bin "$root")"
fw_need_bin "$bin"
exec "$bin" check --backend "${BACKEND:-auto}" "$@"
