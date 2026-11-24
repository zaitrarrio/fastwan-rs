# 2026-11-24: the FP8 option (`--weights fp8`) on an RTX 4090

Vast instance 62398895 (RTX 4090, Pennsylvania, driver 595.112.07, 450 W).
Full output in `run.log`. Seed 7, 832×480. The decision to build it came
from `docs/runs/2026-11-24-vast-4090-int8-fp8-eval` (FP8 GEMMs 1.8–2.1×,
GEMMs 42% of a clip step, projected ~19%).

What it is: W8A8. Block-linear weights E4M3 with one scale per output row;
at every linear each input row (token) is quantized to E4M3 at its own scale
(`quant::fq8_rows`); on sm_89+ cuBLASLt multiplies the codes (f32
accumulate, bf16 out) and `fw_fp8_epilogue` applies both scales, the bias
and GELU. Without FP8 tensor cores (or `FASTWAN_CUDA_NO_FP8=1`) the same
rounded values are multiplied in bf16. The CPU reference and WebGPU do the
same arithmetic in f32 (WebGPU: 0 – 8e-8 vs the CPU).

## Checks

CUDA 82 / 82 on the FP8 path and on the fallback; the FP8 linears sit at
rel 1.4–2.0e-2 against the CPU (tolerance 3e-2): CUDA rounds the inputs to
bf16 before quantizing, which moves some row scales and codes.

## Speed, memory, quality

| run | native | fp8 | int8 (for scale) |
|---|---|---|---|
| clip denoise (3 steps, warm) | 3.42 s | **3.23 s (−5.6%)** | 3.42 s |
| clip peak GPU memory | 5.05 GB | 4.00 GB | 3.74 GB |
| clip PSNR vs native, astronaut | | 21.6 dB | 29.9 dB |
| clip PSNR vs native, fox | | 27.4 dB | |
| long video 165 frames | 15.67 s | **14.45 s (−7.8%)** | 16.2 s (earlier run) |
| long peak GPU memory | 8.29 GB | 6.97 GB | 7.04 GB |
| long PSNR vs native | | 21.4 dB | 21.2 dB |

Same-weights CUDA vs WebGPU is 30.5 dB (clip) / 20.9 dB (long). FP8 keeps
the clips' scene and composition with no artifacts (the planets' shading and
horizon details change: `clip-astro-*-strip.png`); the long video is another
sample of the prompt, as any change gives (one astronaut instead of two).

## Why ~6–8% and not ~19%

Profile of a clip step (`FASTWAN_PROFILE=1`, which syncs per op, so absolute
times are inflated): the GEMMs went from 1577 ms to 835 ms, as the micro-
benchmark said, but the two passes FP8 adds cost ~500 ms: the epilogue
(scales, bias, GELU over the bf16 output, 310 ms) and the per-token quantize
(189 ms). Both are memory-bound passes over activations the bf16 path never
makes (there cuBLASLt fuses bias + GELU). Fusing them needs a per-tensor
activation scale (cuBLASLt on Ada takes a per-channel alpha vector and the
bias / GELU epilogue, but no per-token scale), which risks quality: not done.

Verdict: kept as an opt-in. It is slightly faster than native and int8 on
sm_89+, uses about as little memory as int8, and changes the clip more than
int8 does.
