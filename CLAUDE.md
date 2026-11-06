# CLAUDE.md: fastwan-rs

State and invariants only; the reasoning is in `docs/design.md`.

- One op vocabulary (`fastwan_core::Backend`), three implementations. A new op
  means: CPU reference, WGSL kernel, CUDA kernel, and a case in
  `fastwan-model/src/check.rs`, all in the same change.
- The CPU backend is ground truth. GPU code is trusted only after
  `fastwan check --backend <b>` passes on that hardware. Both pass on an RTX
  4090: 480p step CUDA 1.35 s (VSA) / 2.48 s (dense); WebGPU 12.9 s (VSA),
  clip 42.7 s (`docs/runs/2026-11-03-vast-4090-webgpu-perf`). Neither has run
  on AMD. WebGPU speed varies by host Vulkan driver (152 vs 83 s/step on
  570.86 vs 595.99 with the old kernels): record the driver with every number.
- TAEHV decode on CUDA uses `fw_conv2d_mma` (tensor-core implicit GEMM,
  sm_80+; `FASTWAN_CUDA_SCALAR_CONV=1` for the old kernel): 1.55 → 0.56 s on
  an RTX PRO 6000 (325 W cap), clip 4.45 s; fastvideo-rs on that card class:
  denoise 1.92 s / decode 0.26 s vs our 3.80 / 0.56
  (`docs/runs/2026-11-06-vast-pro6000`).
- Attention: VSA (sparsity 0.8) is the default whenever every block has
  `to_gate_compress` (FastWan does); `--attention dense` to compare. The VSA
  CPU path is fastvideo-rs's host oracle; keep GPU kernels checked against it.
- Rentals: `scripts/vast/` (no SSH; R2 command channel). Arm a destroy timer
  at rent time, `timeout` every command, run the decisive work first.
- WebGPU refuses software adapters for bench/generate/serve
  (`FASTWAN_ALLOW_SOFTWARE=1` overrides): a missing driver dependency makes
  wgpu fall back to llvmpipe silently.
- WGSL kernels and their CUDA twins share structure (`fw_<op>` in
  `kernels.cu`). Change both. `fw_` because `norm` collides with a CUDA builtin.
- Shader change → `python3 web/build.py` (the test page inlines the shaders;
  `web_page_embeds_current_shaders` fails otherwise).
- WGSL: write per-thread register blocks as named vec4 accumulators, never
  arrays indexed in loops. NVIDIA's Vulkan compiler kept those arrays in local
  memory: unrolling made the GEMM 5.75× and flash attention 7.8× faster.
  `FASTWAN_PROFILE=1` gives per-kernel GPU time; check it after kernel work.
- WebGPU f16 math (`linear3h`, `flash_h`, `FASTWAN_WGPU_F16=1`) is opt-in: correct
  (PSNR 43.8 dB vs f32 on a real clip) but 8.5% slower on an RTX 4090, where
  f16 runs at the f32 rate (`docs/runs/2026-11-03-vast-4090-f16`). Unmeasured
  on AMD / Intel / Apple, where it should pay; measure before defaulting it on.
- `--weights int8` (`fastwan_core::quant::Q8`: symmetric int8, one f32 scale
  per 128 input columns) applies to the block linears only. The CPU backend
  dequantizes and is the reference the kernels are checked against. On an RTX
  4090 it cuts peak VRAM by 1.3 GB (CUDA) / 1.5 GB (WebGPU) at equal speed,
  and moves the clip less than switching backends does: 34.6 dB vs a 33.5 dB
  bf16-vs-f32 baseline (`docs/runs/2026-11-03-vast-4090-int8`). Off by default.
- 4-bit (`--weights int4[-hqq][-gN][-ffn]`, `quant::Q4`): real kernels on all
  three backends. Every variant changes the sample for a seed (fox 21–23 dB vs
  int8's 34.6) without visible artifacts. Quantized checkpoints
  (`fastwan quantize`, format in `export.rs`, stored in R2 under
  `fv-weights/weights/fastwan21-1.3b-quant/`): int4-hqq-g64-ffn 1.18 GB,
  int4-hqq-g64 0.88 GB vs ~2.9 GB f16; CUDA peak 3.4 / 3.1 GB, WebGPU 5.4 /
  5.1 GB (`docs/runs/2026-11-03-vast-4090-q4-store`). A stored file decides
  its own format. `WeightQuant::for_key` is the one rule for which linears get
  which format; check gate / weight presence with `has_linear`, never
  `contains("….weight")` (a quantized file stores `.weight.q4`).
- WGSL: zero loop-local accumulators explicitly. Dawn (Chrome) on SwiftShader
  carried a `var` declared in a loop body across iterations, which broke
  multi-chunk flash attention while naga/wgpu passed. Run the bench page in
  Chrome as well as `cargo test` after shader changes.
- WGSL auto layouts drop unused bindings, so every declared binding must be
  statically used (see the guard lines at the end of `conv.wgsl` and
  `elementwise.wgsl`).
- Weight keys are the Diffusers / taehv positional names, checked against
  `weights::wan_manifest` / `tae_manifest`. A missing key is an error, never a
  default.
- No speed claim without a measurement on named hardware.
- The Wan DiT and UMT5 are tested against transformers / diffusers fixtures
  (`scripts/reference/make_fixtures.py`, tiny random weights, checked in). A
  change to either model must keep `hf_parity` green; regenerate fixtures
  only when the reference library changes, and say which version.
- `fastwan encode` + `fastwan serve --gallery <dir>`: precomputed prompt
  embeddings (f16, real tokens, `gallery.rs`), engine without the text
  encoder; only gallery prompts are accepted (400 otherwise). Gallery vs live
  encoder: 39.4 dB on a clip (`docs/runs/2026-11-04-vast-4090-gallery`).
- `fastwan serve` binds 127.0.0.1 and sends CORS headers only for the one
  `--allow-origin` site; `/api/generate` also rejects non-JSON bodies and
  foreign `Origin`s (a cross-site form POST skips the preflight). Keep both.
  `web/demo.html` is compiled into the binary (`include_str!`); rebuild after
  editing it.
- Site: https://fastwan.pages.dev (Cloudflare Pages project `fastwan`, account
  "Maximal"), built by `web/build.py` into `web/site/` and deployed with
  `scripts/deploy-pages.sh`. Classic Pages, not Workers: no `--force` needed now.
- Tests: `cargo test --workspace` (needs a Vulkan driver for the WebGPU suite:
  `mesa-vulkan-drivers`); NVRTC kernel compile needs libnvrtc on
  `LD_LIBRARY_PATH` (`pip install nvidia-cuda-nvrtc-cu12`).
