#!/usr/bin/env bash
# Quantize the DiT. QUANT is native, int8, or int4[-hqq][-gN][-ffn].
# A file written here is loaded with TRANSFORMER= on clip and serve.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib.sh
source "$root/scripts/lib.sh"
cd "$root"
bin="$(fw_bin "$root")"
fw_need_bin "$bin"

model_dir="${MODEL_DIR:-fastwan}"
quant="${QUANT:-int4-hqq-g64}"
transformer="${TRANSFORMER:-$model_dir/transformer}"
out_dir="${OUT_DIR:-out}"
if [[ ! -e "$transformer" ]]; then
  echo "transformer not found: $transformer" >&2
  echo "run: task weights" >&2
  exit 1
fi
if [[ -n "${QUANT_OUT:-}" ]]; then
  out="$QUANT_OUT"
else
  out="$out_dir/fastwan-${quant}.safetensors"
fi
mkdir -p "$(dirname "$out")"
exec "$bin" quantize --transformer "$transformer" --weights "$quant" --out "$out" "$@"
