# Shared path helpers for the Taskfile scripts. Source this file; do not execute it.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source scripts/lib.sh" >&2
  exit 1
fi

# Cargo target directory for this build. A non-empty FEATURES set gets its
# own directory so `task build:cuda` does not replace the WebGPU binary.
fw_target_dir() {
  local root="$1" safe
  if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    printf '%s\n' "$CARGO_TARGET_DIR"
    return 0
  fi
  if [[ -n "${FEATURES:-}" ]]; then
    safe="$(printf '%s' "$FEATURES" | tr -c 'A-Za-z0-9' '-')"
    safe="${safe#-}"
    safe="${safe%-}"
    printf '%s\n' "$root/target/feat-$safe"
    return 0
  fi
  printf '%s\n' "$root/target"
}

fw_profile_dir() {
  case "${PROFILE:-release}" in
    release) printf 'release\n' ;;
    debug | dev) printf 'debug\n' ;;
    *) printf '%s\n' "$PROFILE" ;;
  esac
}

# Path cargo will write. Ignores BIN.
fw_built_bin() {
  printf '%s/%s/fastwan\n' "$(fw_target_dir "$1")" "$(fw_profile_dir)"
}

# Binary to run. BIN overrides the path just built.
fw_bin() {
  if [[ -n "${BIN:-}" ]]; then
    printf '%s\n' "$BIN"
    return 0
  fi
  fw_built_bin "$1"
}

fw_need_bin() {
  local bin="$1"
  if [[ -x "$bin" ]]; then
    return 0
  fi
  echo "fastwan binary not found: $bin" >&2
  if [[ -n "${FEATURES:-}" ]]; then
    echo "run: task build FEATURES=${FEATURES}" >&2
  elif [[ "${PROFILE:-release}" == "debug" || "${PROFILE:-release}" == "dev" ]]; then
    echo "run: task build:debug" >&2
  else
    echo "run: task build" >&2
  fi
  return 1
}
