# 2026-11-06: profile of a DiT step on an RTX PRO 6000 (600 W)

Vast instance 59416487 (offer 55977091): **RTX PRO 6000 Blackwell Server
Edition**, Utah, driver 580.173.02, **600 W limit** (drew 485 W avg / 540 W
max, SM 2329 MHz, 45 °C while busy), $1.54/h, ~10 min. `fastwan bench
--backend cuda --layers 30` (synthetic weights at the real geometry,
832×480×81, 32,760 tokens); `FASTWAN_PROFILE=1` adds the per-op table
(each op synchronised, ~3% slower than unprofiled). Raw output:
`bench-output.txt`.

## Unprofiled

VSA step **0.88 s** (denoise 2.65 s), vs 1.27 s on the 325 W-capped WS card
(`../2026-11-06-vast-pro6000`): that run was power-bound. fastvideo-rs on
this card class: 0.64 s/step (1.92 s). Dense attention: 1.66 s/step.

## Where a VSA step goes (per step, from the 3-step profile)

| op | ms / step | share | note |
|---|---|---|---|
| VSA fine attention (mma) | 393 | 43% | 13.1 ms/layer for ~1.3 TFLOP: ~100 TFLOP/s |
| VSA coarse + top-k | 111 | 12% | ~2.4 GFLOP of work: pure overhead |
| GEMMs (cuBLAS) | 234 | 26% | **332 TFLOP/s**: already near the card |
| bias / GELU pass after each GEMM | 70 | 8% | a separate kernel: fusable into the GEMM epilogue |
| norm, qk-norm + RoPE, gate, pool | 57 | 6% | elementwise |
| cross attention (mma) | 22 | 2.5% | |

The one 76 ms text-projection GEMM is a one-time cuBLAS warm-up in
`prepare_prompt`, not a per-step cost.

Dense attention for comparison: 43 ms per self-attention call, ~150
TFLOP/s. So the VSA fine kernel runs at about two thirds of our own dense
kernel's rate, and both are well short of FlashAttention-class kernels.

## Decode

GPU time 288 ms (98% `conv2d (mma)`, 385 calls), against a wall time of
0.86–0.96 s in `bench` (0.56 s in `generate` on the other card): the rest is
on the host (per-chunk allocation, the f32 download and conversion).
fastvideo-rs: 0.26 s.

## What would close the gap to fastvideo-rs (0.64 s/step)

| change | saves / step (est.) |
|---|---|
| top-k on the GPU in parallel (coarse is tiny) | ~100 ms |
| bias + GELU in the cuBLASLt epilogue | ~65 ms |
| VSA fine kernel at dense-kernel efficiency (cp.async double buffer, larger tiles) | ~130–200 ms |

Together ≈ 0.52–0.58 s/step: at or under fastvideo-rs, on estimates only.
Decode: convert to 8-bit RGB on the GPU and download that, pinned and
overlapped with the next chunk, to bring the wall time toward the 0.29 s of
GPU work.
