# 2026-11-17: Chrome long-video stalls, f16 kernels, 4-bit Self-Forcing (RTX 4090)

Vast instance 61118463 (RTX 4090, Japan, driver 615.71.09, 450 W). Full
output in `run.log`. Code: K/V-cache ops (`kv_store` / `attention_kv`, one
dispatch per run instead of ~3,600 copies per block, cache read in place,
optional 16-bit storage), pipelined `generateLong` (block b queued before
block b-1's frames are awaited; latent never leaves the GPU), f16 kernels
`linear4h` / `flash3h`.

## Correctness on the GPU

`fastwan check --backend wgpu`: 62/62 in f32 (K/V cache rel <= 7.4e-7) and
62/62 in f16 mode (GEMM 7.2e-4, VSA model grid 2.4e-3, f16-cache rollout
1.7e-4).

## Chrome long-video stalls: not fixed

Headless Chromium 159 (Vulkan), Prompt Studio long video, 480×272, 20 s
(27 blocks), seed 7. Per-block seconds:

| arm | total | blocks |
|---|---|---|
| f32 | 293.9 s | 1.3–2.2, except 8.1 35.5 8.9 / 24.0 35.5 / 18.8 35.5 / 29.1 40.5 / 29.2 |
| `?halfkv=1` | 355.7 s | same pattern, 23.7–42.0 |
| `?f16=1` | 215.4 s | same pattern, slow blocks 14.0–26.0 (24.8 recurring) |

Native wgpu on the same GPU: 321 frames in 48.2 s, flat.

Ruled out now: the per-head copy flood (gone: one dispatch per run), the
latent readback (gone), the GPU idling while the page waits (the next block
is queued first). The slow blocks shrink with faster kernels (35.5 → 24.8 s
with f16), so during them the GPU itself appears to run ~20× slower under
Chrome, rather than results arriving late. Next: GPU timestamps inside
Chrome (`--enable-dawn-features=allow_unsafe_apis`) or a Chrome trace, and a
desktop Chrome with a display (every run so far is headless Linux).

## f16 kernels: faster on NVIDIA too now

Prompt Studio "Benchmark f32 vs f16" in Chrome:

| kernel | f32 | f16 | speed-up |
|---|---|---|---|
| GEMM 4096×1536→8960 (FFN up, GELU) | 4.20 ms | 3.80 ms | 1.11× |
| GEMM 4096×8960→1536 (FFN down) | 5.10 ms | 4.10 ms | 1.24× |
| attention 12×4096×4096 d128 | 7.30 ms | 4.50 ms | 1.62× |
| attention over 16-bit cache 1530×6120 | 5.50 ms | 5.80 ms | 0.95× |

Native, 832×480×81 VSA, 30 layers, GPU time of the top kernels per denoise:
f32 17.7 s → f16 16.5 s (flash3 44.5 → 42.1 ms/call, FFN GEMMs 31.9/30.8 →
29.6/27.9 ms). The old f16 kernels (linear3h, flash_h) were 8.5% slower on
this card class. AMD / Apple / Intel (2:1 packed f16) unmeasured: run the
benchmark there. Still off by default until a real-clip PSNR check and those
numbers.

## 4-bit Self-Forcing (phones)

`self-forcing-1.3b-f4.safetensors` (int4-hqq-g64, 837 MB, public). CUDA,
832×480, 60 s, seed 7: f16 90.1 s, f4 91.9 s. f4 is a different sample of the
prompt (as 4-bit weights are for FastWan) and noticeably less stable: the
camera wanders into close-ups, the astronaut's look changes, sky patches and
light streaks (`astro-sf4-cuda.jpg` vs `astro-sf16-cuda.jpg`). Usable on
phones, where it is the only variant that fits.
