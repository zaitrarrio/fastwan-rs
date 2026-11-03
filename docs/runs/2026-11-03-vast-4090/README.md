# 2026-11-03: first real-GPU run (Vast.ai RTX 4090)

Vast instance 58974620, offer 51255313: RTX 4090 24 GB, driver 580.119.02,
32 vCPU, 108 GB RAM, California. Image `nvidia/cuda:12.8.1-runtime-ubuntu24.04`.
$0.53/h with disk, about an hour, **$1.82 total** (Vast credit 31.61 → 29.79).
Weights: FastWan2.1-T2V-1.3B-Diffusers (`fv-weights` R2 tree 25e7ed7f…) and
`taew2_1`, 27 GB pulled in 101 s. Binaries built from 23648d3 (bin-2) and its
parent (bin-1). Driven through R2 with `scripts/vast/`.

## Conformance on the GPU

| backend | result |
|---|---|
| CUDA (sm_89) | all 25 checks pass first run, rel 1.7e-3 … 7.6e-3 (bf16); after the tensor-core attention kernel, all 27 pass (d=128 attention 3.7e-3 … 6.1e-3) |
| WebGPU, NVIDIA Vulkan | all checks pass, once libx11/libxext/libegl/libgl were installed; before that the NVIDIA ICD failed to load and wgpu silently used llvmpipe |

## Speed, 832×480×81 (32,760 tokens), 3 DMD steps

| | per step | denoise | TAEHV decode | clip |
|---|---|---|---|---|
| CUDA, scalar attention (bin-1) | 137 s (4.56 s/layer) | 412 s | 1.92 s | — |
| **CUDA, tensor-core attention (bin-2)** | **2.9 s** (96.5 ms/layer) | **8.7 s** | **1.95 s** | **10.8 s** |
| WebGPU on the 4090 | > 20 min (bench timed out before step 0 ended) | — | — | — |

The tensor-core kernel made a step **47× faster**. Real-weight runs (2 runs
each, same seed, warm numbers):

| clip | UMT5 (CPU, fp32) | load | denoise | decode | total |
|---|---|---|---|---|---|
| fox, 480×272×33 | 5.6 s, 33 tokens | 4.0 s | 0.46 s (0.15 s/step) | 0.25 s | 0.77 s |
| fox, 832×480×81 | 5.7 s, 33 tokens | 3.1 s | 8.75 s | 1.94 s | 10.8 s |
| market, 832×480×81 | 5.8 s, 34 tokens | 4.0 s | 8.77 s | 1.93 s | 10.8 s |

Clips: `fox-832x480.mp4`, `market-832x480.mp4`, `fox-480x272.mp4` (x264 crf 18).
Both 480p clips are coherent and on-prompt. The market prompt's "handheld
camera" was rendered as a camera in someone's hand.

For scale, fastvideo-rs's gold run of the same checkpoint (RTX PRO 6000
Blackwell, VSA on, `lab/gold/fastwan21-1.3b.toml`) denoises in 1.92 s. Dense
attention on a 4090 at 8.7 s is in that neighbourhood.

## What it says

- Attention is the whole game: 137 → 2.9 s/step from that one kernel.
  The DiT's remaining 97 ms/layer is now worth profiling (cuBLAS GEMMs plus
  separate bias/GELU passes, norms, head splits).
- WebGPU needs the same treatment before 480p is usable: its scalar attention
  is the bottleneck there too (subgroup / cooperative-matrix attention, or VSA,
  or a lower resolution).
- UMT5 on the CPU costs 5.6 s per prompt with fp32 weights; bf16 weights would
  halve the bytes it reads.
