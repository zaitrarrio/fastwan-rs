# 2026-11-06: DiT step 0.88 → 0.54 s on an RTX PRO 6000 (600 W)

Vast instance 59457529 (offer 51056495): RTX PRO 6000 Blackwell **Server
Edition**, Bulgaria, driver 595.71.05, 600 W, $1.72/h, ~15 min. The
previous build (7741ba4, "old") and this one on the same box; 832×480×81,
VSA, 30 layers. Raw output: `output.txt`.

## Changes

1. **VSA coarse + top-k** (`fw_vsa_coarse`): warp-per-tile float4 scores,
   radix select with 8-bit digits (4 histogram passes instead of 32
   single-bit passes), ballot/prefix-sum compaction instead of thread 0
   writing the selection serially.
2. **Bias / GELU in the GEMM** (cuBLASLt `BIAS` / `GELU_BIAS` epilogues,
   plans cached per shape; `FASTWAN_CUDA_NO_LT=1` keeps cuBLAS +
   `fw_bias_act`).
3. **Attention body** (dense and VSA fine): K/V chunks by 16-byte `cp.async`
   (invalid keys zero-filled), pipelined so chunk c+1's K loads during
   softmax / P·V and its V during the next Q·Kᵀ; V fragments by
   `ldmatrix.trans` instead of a transposed scalar copy.

## Correctness

`fastwan check --backend cuda`: 50/50 (attention d16–d128 with partial key
tiles, VSA on three grids, the tiny DiT end to end in native / int8 / int4).
Real clip, new vs old build: 35.7 dB PSNR. The bias is now added in f32
before the one bf16 rounding (it was rounded, added, rounded again), and
attention sums in a different order; the 3-step sampler turns that into a
slightly different sample, within the 33.5 dB bf16-vs-f32 backend gap.

## Speed (same card, unprofiled)

| | old | new | |
|---|---|---|---|
| VSA step (bench) | 0.876 s | **0.538 s** | 1.63× |
| dense step (bench) | 1.66 s (earlier box) | 0.99 s | |
| real clip: denoise | 2.54 s | **1.62 s** | |
| real clip: decode | 0.64 s | 0.69 s | |
| real clip: total | 3.29 s | **2.41 s** | 1.37× |
| without the cuBLASLt epilogue | | 0.594 s/step | the epilogue is worth 56 ms |

Profiled per step (new): VSA fine 217 ms (was 393), coarse + top-k 18.5 ms
(was 111), GEMMs with fused bias 233 ms at 369 TFLOP/s, separate bias pass
gone (was 70).

## Against fastvideo-rs (RTX PRO 6000 Blackwell)

| | fastvideo-rs | fastwan-rs |
|---|---|---|
| denoise, 3 steps, VSA | 1.92 s | **1.62 s** |
| TAEHV decode | **0.26 s** | 0.69 s (0.28 s of it on the GPU) |
| peak VRAM | 21.5–27.1 GiB | 3.5–5.2 GiB |

The denoise is now faster than fastvideo-rs's on this card class; the
decode is not, and its remaining cost is host-side (per-chunk allocation,
f32 download and conversion), not GPU work.
