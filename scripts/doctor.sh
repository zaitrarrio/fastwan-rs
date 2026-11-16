#!/usr/bin/env bash
# Toolchain and local-tree check. Missing weights are reported, not fatal:
# `task setup` runs this before `task weights`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=lib.sh
source "$root/scripts/lib.sh"
cd "$root"

missing=0
model_dir="${MODEL_DIR:-fastwan}"
gallery="${GALLERY:-gallery}"

have() {
  local cmd="$1" hint="$2" line
  if command -v "$cmd" >/dev/null 2>&1; then
    line="$("$cmd" --version 2>&1 || true)"
    printf 'ok        %s\n' "${line%%$'\n'*}"
  else
    printf 'missing   %s — %s\n' "$cmd" "$hint"
    missing=1
  fi
}

optional_cmd() {
  local cmd="$1" hint="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    printf 'ok        %s\n' "$cmd"
  else
    printf 'optional  %s — %s\n' "$cmd" "$hint"
  fi
}

printf 'toolchain\n'
have rustc "install rustup: https://rustup.rs"
have cargo "install rustup: https://rustup.rs"
have python3 "needed for gallery lookup and web/build.py"

printf '\noptional\n'
if command -v ffmpeg >/dev/null 2>&1; then
  line="$(ffmpeg -version 2>&1 || true)"
  printf 'ok        %s\n' "${line%%$'\n'*}"
else
  printf 'optional  ffmpeg — task clip still writes .y4m; mp4 conversion is skipped\n'
fi
if command -v huggingface-cli >/dev/null 2>&1 || command -v hf >/dev/null 2>&1; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    printf 'ok        huggingface-cli\n'
  else
    printf 'ok        hf\n'
  fi
else
  printf 'optional  huggingface-cli — task weights (pip install -U huggingface_hub)\n'
fi
if command -v rustup >/dev/null 2>&1; then
  if rustup target list --installed 2>/dev/null | grep -qx 'wasm32-unknown-unknown'; then
    printf 'ok        wasm32-unknown-unknown\n'
  else
    printf 'optional  wasm32-unknown-unknown — rustup target add wasm32-unknown-unknown\n'
  fi
else
  printf 'optional  rustup — wasm target for task web:wasm\n'
fi
if command -v wasm-bindgen >/dev/null 2>&1; then
  line="$(wasm-bindgen --version 2>&1 || true)"
  line="${line%%$'\n'*}"
  case "$line" in
    *0.2.127*) printf 'ok        %s\n' "$line" ;;
    *) printf 'optional  %s — task web:wasm wants 0.2.127\n' "$line" ;;
  esac
else
  printf 'optional  wasm-bindgen 0.2.127 — task web:wasm\n'
fi
optional_cmd task "https://taskfile.dev"

printf '\nweights\n'
weight_path() {
  local path="$1" hint="$2"
  if [[ -e "$path" ]]; then
    printf 'ok        %s\n' "$path"
  else
    printf 'missing   %s — %s\n' "$path" "$hint"
  fi
}
weight_path "$model_dir/transformer" "task weights"
weight_path "$model_dir/text_encoder" "task weights"
weight_path "$model_dir/tokenizer/tokenizer.json" "task weights"
weight_path "$model_dir/${TAE_FILE:-taew2_1.safetensors}" "task weights"
if [[ -f "$gallery/gallery.json" ]]; then
  printf 'ok        %s/gallery.json\n' "$gallery"
else
  printf 'missing   %s/gallery.json — task embed or task embed:gallery\n' "$gallery"
fi

printf '\nbinary\n'
bin="$(fw_bin "$root")"
if [[ -x "$bin" ]]; then
  printf 'ok        %s\n' "$bin"
else
  printf 'missing   %s — task build\n' "$bin"
fi

printf '\n'
if [[ "$missing" -ne 0 ]]; then
  echo "doctor: install the missing toolchain pieces, then re-run task doctor" >&2
  exit 1
fi
if [[ ! -x "$bin" ]]; then
  echo "next: task build"
elif [[ ! -d "$model_dir/transformer" || ! -f "$model_dir/${TAE_FILE:-taew2_1.safetensors}" ]]; then
  echo "next: task weights"
else
  echo "next: task clip"
fi
