#!/usr/bin/env bash
# One prompt → one safetensors file via scripts/encode_prompt.py (torch).
# This does not update the gallery. `task embed` is the native gallery path.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

prompt="${PROMPT:-a fox running through snow}"
out_dir="${OUT_DIR:-out}"
if [[ -z "$prompt" ]]; then
  echo "set PROMPT=" >&2
  exit 1
fi

if [[ -n "${EMBEDS:-}" ]]; then
  out="$EMBEDS"
else
  slug="$(python3 -c 'import re,sys; s=re.sub(r"[^a-z0-9]+","-",sys.argv[1].lower()).strip("-")[:48]; print(s or "prompt")' "$prompt")"
  out="$out_dir/$slug.safetensors"
fi
mkdir -p "$(dirname "$out")"
python3 "$root/scripts/encode_prompt.py" "$prompt" "$out"
echo "embeds: $out"
