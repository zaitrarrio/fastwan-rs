#!/usr/bin/env bash
# Release build by default. FEATURES=cuda writes target/feat-cuda so it does
# not replace target/release/fastwan. PROFILE=debug is an unoptimized build.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib.sh
source "$root/scripts/lib.sh"
cd "$root"

if [[ -n "${FEATURES:-}" && -z "${CARGO_TARGET_DIR:-}" ]]; then
  CARGO_TARGET_DIR="$(fw_target_dir "$root")"
  export CARGO_TARGET_DIR
fi

args=(build)
case "${PROFILE:-release}" in
  release) args+=(--release) ;;
  debug | dev) ;;
  *) args+=(--profile "$PROFILE") ;;
esac
if [[ -n "${FEATURES:-}" ]]; then
  args+=(--features "$FEATURES")
fi

echo "cargo ${args[*]}"
cargo "${args[@]}" "$@"
bin="$(fw_built_bin "$root")"
if [[ ! -x "$bin" ]]; then
  echo "build did not produce $bin" >&2
  exit 1
fi
echo "binary: $bin"
