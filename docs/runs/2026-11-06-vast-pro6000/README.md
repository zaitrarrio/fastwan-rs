# 2026-11-06: RTX PRO 6000 Blackwell, tensor-core decode

Vast instance 59414999 (offer 57473068): **RTX PRO 6000 Blackwell
Workstation Edition**, Tennessee, driver 615.71.09, sm_120, **power limit
325 W** (the WS edition ships at 600 W; this host caps it), $1.58/h, ~25 min.
Fox prompt, seed 7, VSA, 832×480×81 unless noted; second of two runs.

## Correctness

`fastwan check`: 50/50 on CUDA (conv cases incl. 18 K tiles, N tails,
upsample + concat) and 50/50 on WebGPU (Vulkan). The tensor-core conv's clip
matches the scalar conv's at 58.5 dB PSNR (bf16 rounding).

## The decode fix (`fw_conv2d_mma`)

| | decode | clip |
|---|---|---|
| scalar conv (`FASTWAN_CUDA_SCALAR_CONV=1`) | 1.55 s | 5.42 s |
| tensor-core conv | **0.56 s** | **4.45 s** |

## CUDA on this card

| config | denoise (3 steps) | decode | clip | peak VRAM |
|---|---|---|---|---|
| native (bf16) | 3.80 s (1.27 s/step) | 0.56 s | **4.45 s** | 5180 MiB |
| int8 file | 3.83 s | 0.56 s | 4.47 s | 3868 MiB |
| int4-hqq-g64-ffn file | 3.84 s | 0.56 s | 4.48 s | 3548 MiB |
| dense attention | 7.19 s | 0.59 s | 7.87 s | 5148 MiB |
| 832×480×33 | 1.22 s | 0.25 s | 1.52 s | 4220 MiB |
| 480×272×33 | 0.29 s | 0.09 s | **0.41 s** | 3740 MiB |

WebGPU (Vulkan, same card): 10.8 s/step, decode 1.7 s, clip 34.4 s, peak
7169 MiB; int4-ffn 11.6 s/step, 5375 MiB.

## Against fastvideo-rs (`lab/gold/fastwan21-1.3b.toml`, RTX PRO 6000 Blackwell)

| | fastvideo-rs | fastwan-rs (this run) |
|---|---|---|
| denoise, 3 steps, VSA | 1.92 s (MXFP8: 2.26 s gen) | 3.80 s |
| TAEHV decode | 0.26 s | 0.56 s |
| peak VRAM | 21.5–27.1 GiB | 3.5–5.2 GiB |

Same model, same card family, but the power limit of the fastvideo-rs run is
not recorded, and this one was capped at 325 W: the gap is ≤ 2× and an
unknown part of it is power. Denoise barely moved from an RTX 4090 (1.35 →
1.27 s/step), which says the DiT step is not compute-bound on our kernels:
the next lever is profiling it (attention and the non-GEMM ops), not the
decoder.
