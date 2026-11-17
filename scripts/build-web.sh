#!/usr/bin/env bash
# Build the in-browser engine (crates/fastwan-web) into web/pkg for Prompt
# Studio's "In this browser" mode. Needs the wasm32 target and wasm-bindgen
# CLI matching the crate: rustup target add wasm32-unknown-unknown;
# cargo install wasm-bindgen-cli --version 0.2.127 --locked.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown -p fastwan-web
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/fastwan_web.wasm
ls -la web/pkg
