#!/usr/bin/env bash
# Encode prompts with the native UMT5 into a gallery directory.
# One load of the text encoder covers every prompt (PROMPT plus PROMPT_FILE).
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=../lib.sh
source "$root/scripts/lib.sh"
cd "$root"

model_dir="${MODEL_DIR:-fastwan}"
gallery="${GALLERY:-gallery}"
bin="$(fw_bin "$root")"
fw_need_bin "$bin"

if [[ ! -f "$model_dir/tokenizer/tokenizer.json" || ! -d "$model_dir/text_encoder" ]]; then
  echo "text encoder not found under $model_dir" >&2
  echo "run: task weights" >&2
  exit 1
fi

args=(encode --model-dir "$model_dir" --out-dir "$gallery")
if [[ -n "${PROMPT:-}" ]]; then
  args+=(--prompt="$PROMPT")
fi
if [[ -n "${PROMPT_FILE:-}" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "prompt file not found: $PROMPT_FILE" >&2
    exit 1
  fi
  args+=(--prompt-file="$PROMPT_FILE")
fi
if [[ -z "${PROMPT:-}" && -z "${PROMPT_FILE:-}" && $# -eq 0 ]]; then
  echo "set PROMPT= or PROMPT_FILE= (one prompt per line; blank and # lines are skipped)" >&2
  exit 1
fi

echo "encoding into $gallery (native UMT5 under $model_dir; about 12 GB of RAM)"
exec "$bin" "${args[@]}" "$@"
