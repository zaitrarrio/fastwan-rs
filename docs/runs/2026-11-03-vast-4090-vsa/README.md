# 2026-11-03: VSA and the new attention kernels on an RTX 4090

Vast instance 58988269 (offer 45460723): RTX 4090, driver 595.99.02, 126 GB
RAM, the Netherlands, ~12 minutes of use. Binary from 8e0b97b. Weights as in
`../2026-11-03-vast-4090` (27 GB in 245 s). Driven with `scripts/vast/`.

## Conformance on the GPU

All 31 checks pass on both backends: CUDA (bf16, rel ≤ 7.6e-3) and WebGPU on
NVIDIA Vulkan (f32, rel ≤ 1.6e-4), including the three VSA cases (partial
tiles, d 16/64/128) and the tiny head_dim-32 VSA model end to end. First
hardware run of the VSA kernels and the WGSL flash kernel.

## Speed, 832×480×81 (32,760 tokens, 624 tiles, VSA top-k 125), 3 DMD steps

| backend | attention | per step | per layer | denoise |
|---|---|---|---|---|
| CUDA | dense (tensor-core flash) | 2.48 s | 82.7 ms | 7.46 s |
| **CUDA** | **VSA 0.8** | **1.35 s** | **45.1 ms** | **4.07 s** |
| WebGPU | dense, new flash kernel | 18.1 s / 4 layers | 4.5 s | ≈ 407 s (30 layers, scaled) |
| WebGPU | VSA 0.8 | 8.7 s / 4 layers | 2.16 s | ≈ 197 s (scaled) |

- VSA makes the CUDA denoise **1.83× faster** than dense.
- The new WGSL flash kernel takes WebGPU dense from **> 20 min per step** (the
  old kernel never finished one) to **136 s**; VSA halves that again.
- From dense − VSA: WebGPU spends ≈ 2.9 s per layer in dense attention and
  ≈ 1.6 s in everything else (GEMMs: ≈ 2.7 TFLOP per layer, so ~1.7 TFLOP/s).
  Both are a few % of a 4090's f32 rate. The WGSL GEMM and flash kernels are
  the next target: wider register tiles, vec4 loads, f16 math where the
  adapter has `shader-f16`.

## Real clips (CUDA, seed 7, 2 runs each, warm numbers)

| clip | UMT5 (CPU) | denoise | TAEHV decode | total |
|---|---|---|---|---|
| fox, VSA | 5.6 s | 4.07 s | 2.62 s | **6.85 s** |
| market, VSA | 5.5 s | 4.07 s | 2.63 s | **6.87 s** |
| fox, dense | 5.7 s | 7.46 s | 2.53 s | 10.17 s |

TAEHV decoded in 1.9 s on the first rental and 2.5 s here: host variance, same
binary path.

`vsa-fox` and `dense-fox` share subject, pose and motion but are different
samples (background, trunk layout), as fastvideo-rs measured for this
checkpoint (LPIPS ≈ 0.55 between its VSA and dense arms). `dense-fox`
reproduces the first rental's dense clip on a different host.

## Against fastvideo-rs (same checkpoint, both bf16, VSA 0.8)

fastvideo-rs denoises in 1.96 s on an RTX PRO 6000 Blackwell; fastwan-rs in
4.07 s on an RTX 4090. Different GPUs, so the 2.1× is hardware and kernels
together; still unmeasured on one card.

Cost: the Vast credit read $29.79 before this rental and $24.92 after. This
run's instance time and download should be well under $1; check Vast billing
for the split with the first rental's final charges.
