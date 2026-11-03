# 2026-11-03: the WebGPU performance disparity, and the fix (RTX 4090)

Vast instance 58998916 (offer 58886860): RTX 4090, France, driver 595.99.02,
300 W power limit, SM clock 2.1 GHz throughout, $0.402/h, ~35 minutes.
Profiled with `FASTWAN_PROFILE=1` (GPU timestamp queries per dispatch).

## The disparity: hosts, not the code path

The same WebGPU VSA 480p step measured 65 s (scaled from a 4-layer bench,
Netherlands) and 152 s (full real-weights run, Michigan). On one host:

| run (France, same binary) | per layer | per 480p step |
|---|---|---|
| `bench`, 4 layers, unprofiled | 2.77 s | 83 s |
| `bench`, 4 layers, profiled | 2.77 s | 83 s |
| `bench`, 30 layers, profiled | 2.76 s | 83 s |
| `generate`, real weights, 30 layers | 2.76 s | 83 s |

Every kernel matched to < 0.1 ms per call across those runs; clocks, power
and temperature were steady. So the earlier spread was the hosts:

| host | driver | WebGPU VSA per 480p step (old kernels) |
|---|---|---|
| Netherlands | 595.99.02 | ≈ 65 s |
| France | 595.99.02 (300 W cap) | 83 s |
| Michigan | **570.86.15** | 152 s |

CUDA barely moved across hosts (1.35–1.39 s/step); the WebGPU path is
sensitive to the Vulkan driver's shader compiler, which is consistent with
what the profile showed next.

## The cause of the slowness: register blocks in local memory

The profile put GEMMs at ~2 TFLOP/s for both 64×64 and 128×128 tiles
(`linear2.wgsl`, 128×128, was 38% *slower*), and the flash kernel at
~1.3 TFLOP/s. Both kept per-thread accumulators in arrays indexed inside
loops; the driver kept them in local memory. Writing the blocks as named
vec4 accumulators:

| kernel | before | after | |
|---|---|---|---|
| GEMM 1536→8960 (FFN up + GELU) | 446 ms | 77.5 ms | **5.75×**, ≈ 11.6 TFLOP/s |
| `flash` (VSA fine + cross attention) | 662 ms | 84.9 ms | **7.8×** |
| TAEHV conv (decode of 81 frames) | 9.7 s | 3.5 s | **2.8×** |

## End to end (France, real checkpoint, VSA, 832×480×81, warm)

| | denoise | decode | clip |
|---|---|---|---|
| before | 249 s (83 s/step) | 9.9 s | 260 s |
| **after** | **38.8 s (12.9 s/step)** | **3.5 s** | **42.7 s** |
| dense attention, after | ≈ 72 s (24 s/step, 4-layer bench scaled) | | |

All 33 conformance checks pass on the GPU after each change, and the 21 of
the bench page pass in Chrome. Clip: `wgpu-fox-832x480.mp4`.

Where WebGPU time goes now (4-layer profile): flash 39%, GEMMs 58%, all else
3%. CUDA does the same clip in 6.9 s; WebGPU is at ~6×. Next levers: f16
math with `shader-f16`, and subgroup / cooperative-matrix ops where adapters
have them. The 570-driver host was not re-measured with the new kernels.
