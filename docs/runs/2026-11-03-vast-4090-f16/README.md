# 2026-11-03: WebGPU f16 math on an RTX 4090

Vast instance 59004159 (offer 57627089): RTX 4090, Macao, driver 595.112.07,
450 W power limit, $0.39/h, ~15 minutes. `FASTWAN_WGPU_F16=0|1` on one host.

## Correctness

All 34 checks pass in f16 mode (tolerance 1e-2): GEMM 7.2e-4, attention
6.5e-4 … 2.0e-3, VSA 1.4e-3 … 1.9e-3; the rest unchanged.

Same prompt and seed, real checkpoint, 832×480×81: f16 vs f32 output
**PSNR 43.8 dB** average (Y 40.0, U 48.6, V 49.8; worst frame 39.6 dB).
Visually the same clip (`wgpu-f16-fox-832x480.mp4`).

## Speed on NVIDIA: none (as the hardware predicts)

| | f32 (default) | f16 | |
|---|---|---|---|
| GEMM 8960×1536 (FFN up) | 64.7 ms | 65.5 ms | +1% |
| flash (VSA fine + cross) | 70.9 ms | 84.7 ms | +19% |
| 4-layer bench | 4.32 s | 4.69 s | +8.5% |
| real clip (2nd run) | **35.3 s** | 37.9 s | +7% |

GeForce Ada runs non-tensor f16 at the f32 rate (82.6 TFLOPS either way), so
packed f16 FMAs buy nothing and the f32→f16 conversions on every tile load
cost a little. **f16 stays off by default.** It is for adapters with 2:1
packed f16 (AMD RDNA, Intel Arc, Apple): none were rentable on Vast to
measure, so no speed claim is made for them.

This 450 W host is also the fastest WebGPU host so far: 10.8 s per 480p VSA
step and a 35.3 s clip in f32 (France, 300 W cap: 12.9 s / 42.7 s).
