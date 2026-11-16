#!/usr/bin/env bash
# FastWan (transformer, UMT5, tokenizer) and TAEHV into MODEL_DIR.
# Layout matches `huggingface-cli download … --local-dir fastwan` in the README.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

model_dir="${MODEL_DIR:-fastwan}"
dit_repo="${DIT_REPO:-FastVideo/FastWan2.1-T2V-1.3B-Diffusers}"
tae_repo="${TAE_REPO:-madebyollin/taehv}"
tae_file="${TAE_FILE:-taew2_1.safetensors}"

if command -v huggingface-cli >/dev/null 2>&1; then
  tool=(huggingface-cli)
elif command -v hf >/dev/null 2>&1; then
  tool=(hf)
else
  echo "huggingface-cli not found. Install it with: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "downloading $dit_repo into $model_dir"
"${tool[@]}" download "$dit_repo" \
  --include "transformer/*" "text_encoder/*" "tokenizer/*" \
  --local-dir "$model_dir"

echo "downloading $tae_repo $tae_file into $model_dir"
"${tool[@]}" download "$tae_repo" "$tae_file" --local-dir "$model_dir"

if [[ ! -d "$model_dir/transformer" || ! -d "$model_dir/text_encoder" || ! -f "$model_dir/tokenizer/tokenizer.json" ]]; then
  echo "download finished but $model_dir is missing transformer/, text_encoder/, or tokenizer/tokenizer.json" >&2
  exit 1
fi
if [[ ! -f "$model_dir/$tae_file" ]]; then
  echo "TAEHV file missing: $model_dir/$tae_file" >&2
  exit 1
fi
echo "weights: $model_dir"
