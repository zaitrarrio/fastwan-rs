#!/usr/bin/env bash
# Local prompt-to-clip page. SYNTHETIC=1 skips weights. SERVE_GALLERY limits
# the page to prompts already encoded into that directory.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib.sh
source "$root/scripts/lib.sh"
cd "$root"
bin="$(fw_bin "$root")"
fw_need_bin "$bin"

model_dir="${MODEL_DIR:-fastwan}"
args=(
  serve
  --addr "${ADDR:-127.0.0.1:7860}"
  --backend "${BACKEND:-auto}"
  --weights "${WEIGHTS:-native}"
  --attention "${ATTENTION:-auto}"
  --tae-chunk "${TAE_CHUNK:-2}"
)
if [[ -n "${VSA_SPARSITY:-}" ]]; then
  args+=(--vsa-sparsity "$VSA_SPARSITY")
fi
if [[ -n "${MAX_PIXELS:-}" ]]; then
  args+=(--max-pixels "$MAX_PIXELS")
fi
if [[ -n "${MAX_FRAMES:-}" ]]; then
  args+=(--max-frames "$MAX_FRAMES")
fi

if [[ "${SYNTHETIC:-0}" == "1" ]]; then
  args+=(--synthetic)
else
  if [[ ! -d "$model_dir" ]]; then
    echo "model dir not found: $model_dir" >&2
    echo "run: task weights    or    task serve:synthetic" >&2
    exit 1
  fi
  args+=(--model-dir "$model_dir")
  if [[ -n "${TAE:-}" ]]; then
    args+=(--tae "$TAE")
  fi
  if [[ -n "${TRANSFORMER:-}" ]]; then
    args+=(--transformer "$TRANSFORMER")
  fi
  if [[ -n "${SERVE_GALLERY:-}" ]]; then
    if [[ ! -f "$SERVE_GALLERY/gallery.json" ]]; then
      echo "gallery not found: $SERVE_GALLERY/gallery.json" >&2
      echo "run: task embed:gallery" >&2
      exit 1
    fi
    args+=(--gallery "$SERVE_GALLERY")
  fi
fi
if [[ -n "${ALLOW_ORIGIN:-}" ]]; then
  args+=(--allow-origin "$ALLOW_ORIGIN")
fi

echo "studio: http://${ADDR:-127.0.0.1:7860}/"
exec "$bin" "${args[@]}" "$@"
