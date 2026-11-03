# 2026-11-03: 4-bit weights, quality evaluation (RTX 4090, CUDA)

Vast instance 59010637 (offer 30801833): RTX 4090, Minnesota, driver
580.126.09, $0.376/h, ~15 minutes. Real FastWan checkpoint, 832×480×81, VSA,
seed 7, two prompts (the snow fox; a rainy neon night market).

4-bit here is **simulated** (`--weights int4-sim…`): weights are quantized,
expanded back and stored at native precision. Every backend computes
`Σ x·(q·s)` with `q·s` in f32, so the pictures are what int4 kernels would
make; the memory saving is computed, not measured. Speed is unchanged by
construction (1.35 s per step for every row).

## Weight error (`examples/quant_error`, all 330 block matrices)

| format | SNR | worst matrix | block weights |
|---|---|---|---|
| int8 g128 | 43.6 dB | 41.7 | 1.51 GB |
| int4 g128 | 19.9 | 18.5 | 0.78 GB |
| int4 g128 HQQ | 20.1 | 18.7 | 0.78 GB |
| int4 g64 | 20.8 | 19.9 | 0.82 GB |
| int4 g64 HQQ | 21.2 | 20.2 | 0.82 GB |
| int4 g32 | 22.1 | 21.5 | 0.91 GB |
| int4 g32 HQQ | 22.5 | 21.9 | 0.91 GB |

HQQ (ported from the paper / Pruna's quantizer) buys only 0.2–0.4 dB over
min/max rounding on these weights, which have no heavy outliers.

## Clips: PSNR against the native clip (average / Y)

| format | fox | market | est. peak VRAM |
|---|---|---|---|
| int8 | 34.6 / 30.6 | 27.1 / 23.0 | 3.7 GB (measured) |
| int4 g128 | 19.7 / 15.8 | 17.4 / 13.6 | ≈ 3.0 GB |
| int4 g64 | 21.0 / 17.1 | 17.8 / 14.0 | ≈ 3.0 GB |
| int4 g64 HQQ | 21.0 / 17.1 | 17.4 / 13.6 | ≈ 3.0 GB |
| int4 g32 HQQ | 22.2 / 18.4 | 18.7 / 15.5 | ≈ 3.1 GB |
| int4 g64 HQQ, FFN only (attention int8) | **23.1 / 19.2** | 18.4 / 14.8 | ≈ 3.3 GB |

For scale: the two native backends (bf16 CUDA, f32 WebGPU) differ by 33.5 dB.

`fox-grid.png` / `market-grid.png`: frames 0 / 40 / 80, rows in the table's
order with native on top. int8 reproduces the native sample (the market's
lower int8 PSNR is fine detail in rain and crowds). **Every 4-bit variant
produces a different sample:** the fox's snow line, grass and pose shift; the
market loses its foreground figure and rearranges the stalls and crowd. None
show 4-bit artifacts at this size (banding, colour shifts, broken anatomy);
they read as other seeds of the same prompt. Clips:
`*-int4-sim-hqq-g64-ffn.mp4`.

## Verdict

4-bit does not preserve the model's output the way int8 does: it changes
*which* video you get for a seed, not (visibly) how good it is. That makes
PSNR the wrong final judge; settling "as good" needs a seed sweep with a
perceptual / prompt-alignment metric. Against that, the gain is small: the
DiT's int8 weights are 1.5 GB of a 3.7 GB peak, so 4-bit saves ~0.4–0.7 GB
more. Not worth int4 kernels for VRAM; the case for it is download size in
the browser (2.9 GB → ~0.8 GB of block weights). If built, the candidate is
g64 HQQ on the FFN with int8 attention (best PSNR on the fox, ~1.05 GB).
