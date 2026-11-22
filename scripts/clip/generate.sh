#!/usr/bin/env bash
# Generate one clip. ENCODE=auto writes a gallery embedding when PROMPT is not
# already there; ENCODE=always re-encodes; ENCODE=never requires EMBEDS or a hit.
# The FastWan checkpoint is not causal: CAUSAL=1 needs TRANSFORMER set to a
# Self-Forcing checkpoint from scripts/convert_self_forcing.py.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=../lib.sh
source "$root/scripts/lib.sh"
cd "$root"

model_dir="${MODEL_DIR:-fastwan}"
gallery="${GALLERY:-gallery}"
bin="$(fw_bin "$root")"
fw_need_bin "$bin"

prompt="${PROMPT-a fox running through snow}"
export PROMPT="$prompt"
width="${WIDTH:-832}"
height="${HEIGHT:-480}"
frames="${FRAMES:-81}"
seed="${SEED:-7}"
backend="${BACKEND:-auto}"
weights="${WEIGHTS:-native}"
attention="${ATTENTION:-auto}"
out_dir="${OUT_DIR:-out}"
encode_mode="${ENCODE:-auto}"
tae_file="${TAE_FILE:-taew2_1.safetensors}"
transformer="${TRANSFORMER:-$model_dir/transformer}"
tae="${TAE:-$model_dir/$tae_file}"

if [[ ! -e "$transformer" ]]; then
  echo "transformer not found: $transformer" >&2
  echo "run: task weights" >&2
  exit 1
fi
if [[ ! -f "$tae" ]]; then
  echo "TAEHV not found: $tae" >&2
  echo "run: task weights" >&2
  exit 1
fi
case "$encode_mode" in
  auto | always | never) ;;
  *)
    echo "ENCODE must be auto, always, or never (got $encode_mode)" >&2
    exit 1
    ;;
esac
case "${MP4:-auto}" in
  0 | false | no | auto | 1 | true | yes) ;;
  *)
    echo "MP4 must be auto, 1, or 0 (got ${MP4})" >&2
    exit 1
    ;;
esac
if [[ "${CAUSAL:-0}" == "1" && -z "${TRANSFORMER:-}" ]]; then
  echo "CAUSAL=1 needs TRANSFORMER= set to a Self-Forcing checkpoint" >&2
  echo "convert with scripts/convert_self_forcing.py; the FastWan checkpoint is not causal" >&2
  exit 1
fi
if [[ -n "${OUT:-}" && "$OUT" != *.y4m ]]; then
  echo "OUT must end in .y4m (got $OUT)" >&2
  exit 1
fi

resolve() {
  python3 "$root/scripts/embed/resolve-embed.py" "$gallery" "$prompt"
}

embeds=""
if [[ -n "${EMBEDS:-}" ]]; then
  embeds="$EMBEDS"
  if [[ -d "$embeds" ]]; then
    echo "EMBEDS is a directory: $embeds" >&2
    echo "pass a safetensors file, or leave EMBEDS empty and set PROMPT" >&2
    exit 1
  fi
  if [[ ! -f "$embeds" ]]; then
    echo "embeddings not found: $embeds" >&2
    exit 1
  fi
else
  if [[ -z "$prompt" ]]; then
    echo "set PROMPT= or EMBEDS=" >&2
    exit 1
  fi
  status=0
  if [[ "$encode_mode" != "always" ]]; then
    set +e
    embeds="$(resolve)"
    status=$?
    set -e
  else
    status=1
  fi
  if [[ "$status" -eq 2 ]]; then
    exit 2
  fi
  if [[ "$status" -ne 0 ]]; then
    if [[ "$encode_mode" == "never" ]]; then
      echo "no embedding for this prompt in $gallery" >&2
      echo "run: task embed    or    task clip" >&2
      exit 1
    fi
    bash "$root/scripts/embed/encode.sh"
    set +e
    embeds="$(resolve)"
    status=$?
    set -e
    if [[ "$status" -ne 0 || -z "$embeds" ]]; then
      echo "encoded the prompt but it is not listed in $gallery/gallery.json" >&2
      exit 1
    fi
  else
    echo "embedding exists: $embeds"
  fi
fi

if [[ -n "${OUT:-}" ]]; then
  out="$OUT"
else
  if [[ -n "$prompt" ]]; then
    slug="$(python3 -c 'import re,sys; s=re.sub(r"[^a-z0-9]+","-",sys.argv[1].lower()).strip("-")[:48]; print(s or "clip")' "$prompt")"
  else
    slug="$(basename "$embeds" .safetensors)"
  fi
  out="$out_dir/${slug}-${width}x${height}x${frames}-s${seed}.y4m"
fi
mkdir -p "$(dirname "$out")"

echo "clip: ${width}x${height}x${frames} seed ${seed} backend ${backend} attention ${attention} weights ${weights}"
echo "embeds: $embeds"
echo "y4m: $out"

args=(
  generate
  --transformer "$transformer"
  --tae "$tae"
  --embeds "$embeds"
  --width "$width"
  --height "$height"
  --frames "$frames"
  --seed "$seed"
  --backend "$backend"
  --weights "$weights"
  --attention "$attention"
  --runs "${RUNS:-1}"
  --tae-chunk "${TAE_CHUNK:-2}"
  --out "$out"
)
if [[ -n "${VSA_SPARSITY:-}" ]]; then
  args+=(--vsa-sparsity "$VSA_SPARSITY")
fi
if [[ -n "${FFN_CHUNK:-}" ]]; then
  args+=(--ffn-chunk "$FFN_CHUNK")
fi
if [[ -n "${UPSCALE:-}" ]]; then
  args+=(--upscale "$UPSCALE")
fi
if [[ -n "${UPSCALE_X:-}" ]]; then
  args+=(--upscale-x "$UPSCALE_X")
fi
if [[ "${CAUSAL:-0}" == "1" ]]; then
  args+=(--causal --cache-frames "${CACHE_FRAMES:-12}" --sink-frames "${SINK_FRAMES:-3}")
fi
if [[ -n "${MEMORY:-}" ]]; then
  args+=(--memory "$MEMORY")
fi

"$bin" "${args[@]}" "$@"

mp4="${out%.y4m}.mp4"
case "${MP4:-auto}" in
  0 | false | no) ;;
  auto | 1 | true | yes)
    if command -v ffmpeg >/dev/null 2>&1; then
      ffmpeg -y -hide_banner -i "$out" "$mp4"
      echo "mp4: $mp4"
    elif [[ "${MP4:-auto}" == "auto" ]]; then
      echo "ffmpeg not installed; left $out"
    else
      echo "ffmpeg not installed (needed because MP4=${MP4})" >&2
      exit 1
    fi
    ;;
esac
