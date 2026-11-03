# 2026-11-03: stored quantized checkpoints, real int4 kernels (RTX 4090)

Vast instance 59015923 (offer 58785629): RTX 4090, British Columbia, $0.368/h,
~20 minutes. Driver 615.71.09 (Vulkan). Fox prompt, seed 7, 832×480×81, VSA.
This host ran ~12% slower than earlier 4090s (1.5 s/step CUDA native vs
1.35), so compare speeds within this run only.

## The files (`fastwan quantize`, format in `crates/fastwan-core/src/export.rs`)

Stored in R2, add-only, next to the source checkpoint:
`fv-weights/weights/fastwan21-1.3b-quant/25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef/`

| file | size | block linears |
|---|---|---|
| source `transformer/` (f32) | 5.96 GB | f32 (≈ 2.9 GB as f16) |
| `fastwan-1.3b-int8.safetensors` | 1.56 GB | int8, g128 |
| `fastwan-1.3b-int4-hqq-g64-ffn.safetensors` | **1.18 GB** | FFN 4-bit HQQ g64, attention + gate int8 |
| `fastwan-1.3b-int4-hqq-g64.safetensors` | **0.88 GB** | all 4-bit HQQ g64 |

Non-block tensors are f16 in every file. Export takes 22–40 s with a warm
page cache (540 s cold on a 4-core / 15 GB box).

## Correctness

`fastwan check`: 48/48 on CUDA (int4 GEMMs ≤ 6.4e-3, bf16) and on WebGPU
(≤ 4e-7 against the CPU's dequantized reference).

Stored vs load-time quantization (`--weights int4-hqq-g64-ffn` on the f32
source), CUDA: 38.2 dB. The block weights are bit-identical (unit test); the
difference is the f16 storage of the other tensors, amplified by the 3-step
sampler, and is far smaller than quantization itself (23.1 dB vs native).

## Memory, speed, quality

| | CUDA peak | CUDA step | WebGPU peak | WebGPU step | PSNR vs native (CUDA / WebGPU) |
|---|---|---|---|---|---|
| native | 5022 MiB | 1.55 s | 7182 MiB | 11.5 s | — |
| int8 file | 3710 MiB | 1.53 s | — | — | 32.7 / — |
| int4-hqq-g64-ffn file | **3390 MiB** | 1.54 s | **5389 MiB** | 11.8 s | 23.1 / 23.0 |
| int4-hqq-g64 file | **3134 MiB** | 1.53 s | **5133 MiB** | 11.8 s | 21.1 / 21.0 |

Load from a stored file: 3.4–6.5 s, vs 16.4 s quantizing at load.

4-bit costs ~2% on WebGPU (11.8 vs 11.5 s per step: the nibble unpack in
the GEMM's tile load); CUDA is unchanged (one dequant into a bf16 scratch per
GEMM). PSNR is the same on both backends, as it should be: the same codes.
The 4-bit clips are other samples of the prompt, not degraded ones (see
`docs/runs/2026-11-03-vast-4090-int4`); `wgpu-int4-hqq-g64-ffn-fox.mp4` is
the WebGPU clip from the stored file.

## What this does not do yet

The browser does not run the model: the web pages talk to `fastwan serve`.
An in-browser run needs a wasm build of the engine, a reader for this file
(safetensors is a JSON header plus raw buffers), and a text encoder story
(UMT5-XXL is 11 GB; precomputed or server-side embeddings).
