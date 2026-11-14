# CLAUDE.md: fastwan-rs

State and invariants only; the reasoning is in `docs/design.md`.

- One op vocabulary (`fastwan_core::Backend`), three implementations. A new op
  means: CPU reference, WGSL kernel, CUDA kernel, and a case in
  `fastwan-model/src/check.rs`, all in the same change.
- The CPU backend is ground truth. GPU code is trusted only after
  `fastwan check --backend <b>` passes on that hardware. Both pass on an RTX
  4090: 480p step CUDA 1.35 s (VSA) / 2.48 s (dense); WebGPU 12.9 s (VSA),
  clip 42.7 s (`docs/runs/2026-11-03-vast-4090-webgpu-perf`; 6.67 s/step
  with the 2026-11-14 kernels). Neither has run
  on AMD. WebGPU speed varies by host Vulkan driver (152 vs 83 s/step on
  570.86 vs 595.99 with the old kernels): record the driver with every number.
- CUDA speed on an RTX PRO 6000 Server Edition (600 W), 832×480×81 VSA:
  step **0.54 s**, clip 2.21 s (denoise 1.61 s, decode 0.43 s); fastvideo-rs
  on that card class: denoise 1.92 s, decode 0.26 s
  (`docs/runs/2026-11-06-vast-pro6000-opt`). Kernels: tensor-core implicit
  GEMM conv (`fw_conv2d_mma`), radix-select VSA top-k, cuBLASLt bias/GELU
  epilogues (`FASTWAN_CUDA_NO_LT=1` to disable), cp.async-pipelined
  attention with `ldmatrix.trans` V. `FASTWAN_PROFILE=1` gives per-op times
  on CUDA. A 325 W-capped card is ~1.4× slower: record the power limit.
- The decode is pipelined: `TaeHv::launch_chunk` for chunk i+1 is queued
  before chunk i is converted on the host (`Chunk::rgb8`, parallel). Keep
  host work off the GPU's critical path; `--tae-chunk 2` beats 4.
- The MMA attention kernels are not memory-bound: skipping all K/V loads saves
  12% (VSA), skipping the MMAs 37%; the rest is softmax, barriers and latency
  (`docs/runs/2026-11-07-attention-probe-and-decode-pool`). `ldmatrix` K and
  `ex2.approx` were 2% slower; runtime probe branches cost 1.7%: keep
  diagnostics out of the hot kernel. TAEHV reuses buffers across chunks
  (`DecodeState::pool`): 583 → 40 allocations per decode.
- `FASTWAN_CUDA_ATTN_VARIANT` 0|1|2 selects the MMA attention kernels; v2
  (swizzled tiles, 3 blocks/SM) is the default on sm_120+ only: 2% faster
  step there (`docs/runs/2026-11-07-attention-variants`). Check v2 on another
  architecture before defaulting it there. `examples/ptx.rs` + ptxas gives
  registers / spills / smem without a GPU.
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
- WebGPU kernel defaults (2026-11-14, RTX 4090): `linear4` (128×128 GEMM, 8×8
  per thread, vec4 shared reads; bit-identical to linear3) + `flash3` (4
  threads per softmax row, vec4 panels). Chrome 10.9 → 5.75 s/step at
  832×480, native 10.7 → 6.67 s (`docs/runs/2026-11-14-vast-4090-webgpu-kernels`).
  `FASTWAN_WGPU_LINEAR` / `FASTWAN_WGPU_FLASH` natively, `?kernels=L,F` in the
  page, select others for A/B. Measure kernel changes in Chrome too: Dawn and
  naga compile the same WGSL very differently (flash2 was 1.7× slower
  natively, faster in Chrome).
- WGSL: never index a vector dynamically (`v[j]` in a loop), not even small
  vec4s: NVIDIA's Vulkan compiler moves them to local memory (flash2 → flash3:
  119 → 47 ms by unrolling only that). Never store one component of a shared
  vec4 (may compile to read-modify-write and race): store whole vec4s.
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
- Site: https://fastwan.dev (also https://fastwan.pages.dev; Cloudflare Pages project `fastwan`, account
  "Maximal"), built by `web/build.py` into `web/site/` and deployed with
  `scripts/deploy-pages.sh`. Classic Pages, not Workers: no `--force` needed now.
- In-browser mode (`crates/fastwan-web`, wasm via `scripts/build-web.sh` →
  `web/pkg`, wasm-bindgen pinned `=0.2.127` to match wgpu 30): weights stream
  from the public bucket `fastwan-public/v1/` (`manifest.json`, served at
  `PUBLIC_WEIGHTS` in `web/demo.html` = `https://weights.fastwan.dev/v1`, the bucket's custom domain; fv-weights stays
  private; CORS is an origin allowlist, add new site domains to it), one block per Range request, so DiT files must be block-ordered
  (`export::write` keeps the given order; `quantize_dit` sorts head first, then
  `blocks.N`). Blocking `download` panics on wasm: use `download_async`; no
  `Instant` on wasm. In the browser wgpu's default error handler panics
  inside a JS callback, which left generate's promise pending forever (the
  "stuck after step 3" report): `Wgpu` records uncaptured errors and device
  loss (`fault()`), the web driver uses `try_download_async` and refuses a
  size whose largest buffer exceeds the binding cap before denoising, and the
  page fails the call on a wasm `RuntimeError`. Tiny-model parity vs the CPU (SwiftShader Chrome): f4/f16
  clips and the upscaler 99.9% byte-exact, max 1 level.
- Upscaler (`upscale.rs`, Real-ESRGAN `realesr-animevideov3`, BSD-3): convs on
  the backend with `ConvEpilogue::Prelu`, pixel-shuffle + nearest skip in
  `pack_rgba8` (below). Real weights match a numpy SRVGGNetCompact 100%
  (`scripts/upscale/parity.py`). `generate --upscale`, Prompt Studio checkbox.
  4090 CUDA: 81 frames 480×272 → 1920×1088 in 0.87 s, of which 285 ms is GPU.
- `pack_rgba8` (+ `alloc_words` / `upload_words` / `download_rgba8`): TAEHV's
  and the upscaler's last step on the device, frames out as packed RGBA. A
  packed word is not a valid f32 in general: WGSL writes it through a u32
  view, CUDA through an f32-flagged buffer (activations are bf16), and the
  upscaler's residual is the exact 8-bit input (bf16 lost a level). Native
  win is small (decode 0.36 → 0.33 s, upscale 0.99 → 0.87 s, output
  identical); the browser loses its per-pixel loops
  (`docs/runs/2026-11-13-vast-4090-gpu-pack-browser`).
- In the browser on an RTX 4090 (Chrome, real GPU): 832×480×81 in 18.9 s
  (5.75 s/step; 35.7 s before the 2026-11-14 kernels), 480×272×81 in 5.2 s. Upscaled frames stay on
  the GPU and are drawn on a WebGPU canvas over the 2D one
  (`fastwan_wgpu::CanvasView`, wasm only): the low-res clip plays at once,
  first 1080p frame 0.5 s after the decode, all 81 in 1.4–1.7 s (was 2.4–2.6
  s with readback; `docs/runs/2026-11-13-vast-4090-browser-upscale-gpu`). In
  a browser dropping a buffer destroys it: drop the upscaler's intermediates
  only after a fence. Upscale is offered for 480×272 only. Playwright's headless
  *shell* has no `navigator.gpu`: use the full Chromium build. Vast hosts
  differ in whether they expose Vulkan; `fastwan check --backend wgpu` must
  name NVIDIA, not llvmpipe. `fastwan serve` marks its page
  (`fastwan-served` meta); only that page hides browser mode.
- Tests: `cargo test --workspace` (needs a Vulkan driver for the WebGPU suite:
  `mesa-vulkan-drivers`); NVRTC kernel compile needs libnvrtc on
  `LD_LIBRARY_PATH` (`pip install nvidia-cuda-nvrtc-cu12`).
