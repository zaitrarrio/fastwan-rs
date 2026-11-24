# 2026-11-24: int8 for the browser, FP8 for CUDA (RTX 4090)

Vast instance 62387940 (RTX 4090, Michigan, driver 580.65.06, 450 W, 24 GB).
Full output in `run.log`. Seed 7, gallery prompt 07 (astronaut). Nothing
public changed: the int8 files and a test manifest were served from the
rental.

## int8 weights (`fastwan quantize --weights int8`)

Block linears int8 (one f32 scale per 128 inputs), the rest f16, as the
existing int8 path.

| | f16 | int8 |
|---|---|---|
| clip DiT file | 2.98 GB | **1.56 GB** |
| long-video (Self-Forcing) file | 2.84 GB | **1.49 GB** |

| run | f16 | int8 | f16 vs int8 |
|---|---|---|---|
| CUDA clip 832×480×81 | 4.04 s, 5.05 GB peak | 4.08 s, 3.74 GB | 30.9 dB |
| native WebGPU clip 832×480×81 | 25.6 s, 8.99 GB | 24.6 s, 7.71 GB | 29.3 dB |
| CUDA long 832×480, 165 frames | 15.7 s, 8.29 GB | 16.2 s, 7.04 GB | 21.2 dB |
| native WebGPU long 480×272 | 28.4 s, 6.58 GB | 27.7 s, 5.23 GB | 18.0 dB |
| Chrome clip 832×480×81 | 19.3 s, 7.54 GB | 22.1 s, 6.41 GB | |
| Chrome long 480×272, 321 frames | 37.2 s (1.3 s/block), 6.34 GB | 42.1 s (1.5 s/block), 5.23 GB | |

Peak GPU memory is `nvidia-smi` over the whole device. Chrome load from the
rental's own disk: 38–40 s f16, 19–20 s int8 (the real gain is the download:
half the bytes).

For scale, the same f16 weights on CUDA vs WebGPU differ by 30.5 dB (clip)
and 20.9 dB (long). int8 moves the result as much as switching backend does:
the clip strips are near-identical, and the long video keeps the scene, the
two astronauts and the moons, with small changes in detail (the backpack) —
the same kind of difference as CUDA vs WebGPU (`*-strip.png`, frames 0, 40, …).
Long videos score lower for any change because differences compound block
by block.

Cost: Chrome is 13–14% slower with int8 (dequantizing in the GEMM costs more
there than natively, where it is free).

## FP8 (cuBLASLt, `crates/fastwan-cuda/examples/fp8_bench.rs`)

E4M3 inputs, bf16 out, f32 accumulate, vs the bf16 GEMMs the backend uses:

| linear | rows | bf16 ms | FP8 ms | speed-up | activation quantize |
|---|---|---|---|---|---|
| attn q/k/v/o 1536→1536 | 1530 | 0.059 | 0.031 | 1.88× | 0.006 ms |
| ffn up 1536→8960 | 4680 | 0.833 | 0.420 | 1.98× | 0.014 ms |
| ffn down 8960→1536 | 4680 | 0.823 | 0.413 | 1.99× | 0.139 ms |
| ffn up | 32760 | 5.452 | 2.605 | 2.09× | 0.162 ms |
| ffn down | 32760 | 5.152 | 2.606 | 1.98× | 0.948 ms |

Fast accumulation changes nothing. Profile of a CUDA clip step (832×480×81,
VSA): GEMMs are **42%** (FFN up 13.4%, down 12.9%, q/k/v/o 9.1%, fused QKV
6.6%), VSA attention 38%, convs (decode) 7.8%. Halving the GEMMs and paying
the quantize passes projects the denoise from 3.48 s to about 2.8 s (**~19%**)
on this card; not measured end to end.
