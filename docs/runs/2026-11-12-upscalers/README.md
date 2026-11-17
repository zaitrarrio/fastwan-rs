# 2026-11-12: upscaling 480×272 clips (RTX 4090)

Vast instance 60379959: RTX 4090, Spain, driver 615.78.08, 450 W. Two
prompts (fox, night market), seed 7, 81 frames, CUDA, default kernels.
Scripts: `scripts/upscale/upscale.py` (models, timing, flicker score) and
`grid.py` (crop grid). Every checkpoint is loaded tensors-only (spandrel's
restricted unpickler; `torch.load(weights_only=True)` for RealBasicVSR).
All clips are in R2 under `fastwan-runs/vast-20261112-1630/up/`; this
directory keeps the grids, results and the two fox clips to compare.

## Generation (same box)

| | 480×272×81 | 832×480×81 |
|---|---|---|
| clip | **1.05 s** (0.27 s/step, decode 0.15 s) | 3.81 s (1.09 s/step, decode 0.41 s) |

## Upscalers on the 480×272 clip (81 frames, fp16)

| model | out | time for 81 frames | fps | flicker fox / market |
|---|---|---|---|---|
| bicubic ×4 | 1920×1088 | 0.009 s | — | 0.62 / 0.66 |
| Real-ESRGAN `realesr-animevideov3` (compact, 2.5 MB) | 1920×1088 | **0.22 s** | **362** | **1.11 / 1.43** |
| Real-ESRGAN `realesr-general-x4v3` (compact, 4.9 MB) | 1920×1088 | 0.41 s | 195 | 1.17 / 1.87 |
| Real-ESRGAN `RealESRGAN_x4plus` (RRDB, 67 MB) | 1920×1088 | 7.34 s | 11 | 1.38 / 1.88 |
| RealBasicVSR ×4 (video, 210 MB) | 1920×1088 | 2.6–3.0 s | 27–31 | 2.66 / 2.75 |
| bicubic ×2 | 960×544 | 0.002 s | — | 1.62 / 1.71 |
| Real-ESRGAN `RealESRGAN_x2plus` | 960×544 | 2.30 s | 35 | 2.21 / 2.46 |

Flicker = mean absolute frame-to-frame change of a 5×5 high-pass of luma,
in 8-bit levels; compare only within one output size. It also rises with
plain sharpness, so it is a guide, not a verdict.

## What the frames show (`*/crops_f40.png`, frame 40, same LR region)

- **animevideov3 / general-x4v3**: clean, sharp edges (eyes, whiskers,
  stall signs) with a slightly painterly finish; animevideov3 is the
  steadiest of the learned models and the fastest.
- **x4plus**: the most natural texture, softer, 30× slower than
  animevideov3.
- **RealBasicVSR**: its cleaning stage over-processes generated video into
  an oil-paint look (posterised fur, smeared signs), and it flickers the
  most here despite being temporal. Not suited to this content.
- **x2plus**: barely distinguishable from bicubic ×2 on these clips.

The native 832×480 clip (`fox/native_f40.png`) is a different sample of
the prompt (other framing), and it still shows real fur and whisker detail
that no upscaler of the 480×272 clip invents.

## Verdict

480×272 + `realesr-animevideov3` ×4 = **1.27 s for a 1920×1088 clip** on a
4090, against 3.81 s for native 832×480. It is the fastest and steadiest
learned option, 2.5 MB of weights (phone-friendly). Native 832×480 still
has more genuine detail; upscaling makes 480×272 look sharp, not richer.
