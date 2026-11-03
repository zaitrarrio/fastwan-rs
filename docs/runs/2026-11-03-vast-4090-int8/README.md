# 2026-11-03: int8 weight-only DiT on an RTX 4090

Vast instance 59007763 (offer 41504588): RTX 4090, Texas, driver 580.126.09,
450 W, $0.376/h, about 10 minutes. Real FastWan checkpoint, 832×480×81, VSA,
seed 7, the fox prompt; `--weights native` vs `--weights int8`.

## Correctness

`fastwan check` passes all 41 checks on both backends, including the int8
GEMMs (grouped and per-row, odd K) and the tiny VSA DiT in int8: WebGPU
≤ 3.4e-7 against the CPU dequantized reference, CUDA ≤ 5.7e-3 (bf16).

Weight error, from `cargo run --release -p fastwan-core --example q8_error`
on the checkpoint: every block matrix keeps 41.7–44.4 dB SNR (43.6 dB
overall, worst `blocks.29.attn1.to_v`). No outlier matrix.

## Memory and speed

| | native | int8 | |
|---|---|---|---|
| CUDA peak VRAM | 5022 MiB | **3710 MiB** | −1.3 GB |
| CUDA step / clip (2nd run) | 1.34 s / 6.67 s | 1.35 s / 6.71 s | same |
| WebGPU peak VRAM | 7182 MiB | **5645 MiB** | −1.5 GB |
| WebGPU step / clip (2nd run) | 10.37 s / 35.49 s | 10.49 s / 35.69 s | +1% |
| load | 6.3 s / 10.5 s | 13.5 s / 14.7 s | quantized on the CPU at load |

The block weights go from 2.93 GB to 1.51 GB. On CUDA the int8 weight is
expanded into a shared bf16 scratch per GEMM (one matrix, ≤ 27 MB) and the
cuBLAS GEMM is unchanged; on WebGPU the GEMM reads the codes directly. Neither
gets faster: at 32k tokens these GEMMs are compute bound.

## Output

PSNR of the int8 clip against the native clip on the same backend, next to
the gap that already exists between the two native backends:

| pair | Y | average |
|---|---|---|
| CUDA int8 vs CUDA bf16 | 30.6 | 34.6 dB |
| WebGPU int8 vs WebGPU native | 30.9 | 34.9 dB |
| **baseline:** WebGPU native (f32) vs CUDA native (bf16) | 29.5 | 33.5 dB |
| WebGPU int8 vs CUDA int8 | 28.5 | 32.6 dB |

int8 moves the clip less than switching backends does. A 3-step sampler
amplifies any perturbation into a slightly different sample rather than
noise: `cuda-native-vs-int8.png` (frames 0 / 40 / 80, bf16 top, int8 bottom)
shows the same scene, fox and composition, with fine texture (grass, fur)
re-rolled. `cuda-int8-fox-832x480.mp4` is the int8 clip.

## Verdict

Opt-in, worth it where memory is the limit: `--weights int8` takes 1.3–1.5 GB
off the peak at no speed cost. It does not shrink the download yet (weights
are quantized at load); an int8 safetensors export would.
