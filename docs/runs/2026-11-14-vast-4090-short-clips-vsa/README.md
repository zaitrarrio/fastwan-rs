# 2026-11-14: short clips lose their last frames under VSA (RTX 4090)

Report: on an iPhone, a 1 s clip (480×272×17) came out with its last frames
"eroded". Reproduced on Vast instance 60695491 (RTX 4090, Spain, driver
615.71.09) with the f4 browser weights, fox, seed 7, CUDA and WebGPU.

| clip | attention | last frames |
|---|---|---|
| 480×272×17 | VSA (the old auto) | dissolve into blocky noise from ~frame 11 |
| 480×272×17 | dense | clean |
| 480×272×21, 33, 49, 65 | VSA | the last ~4–8 frames break up the same way |
| 480×272×33, 49 | dense | clean |
| 832×480×17, 49 | VSA | same failure |
| 480×272×81, 832×480×81 | VSA | clean |
| 480×272×81 | dense | clean |

- Not the decoder: TAEHV chunk 1 vs 2 gives identical frames (99 dB).
- Not the phone or WebGPU: CUDA shows it, and WebGPU on the 4090 too.
- Not partial VSA tiles as such: 81 frames (21 latent = 5×4 + 1) also ends in
  a one-frame tile and is clean, as do 65 (17 = 4×4 + 1) and 33 (9 = 2×4 + 1)
  which are not.

VSA is only well behaved at the length FastWan was distilled at (81 frames).
Whether upstream FastVideo shows the same at other lengths is not checked
here. Fix: `Attention` left to the default (`--attention auto`, the engine,
the browser) uses VSA only from 21 latent frames (`VSA_MIN_LATENT_FRAMES`)
and dense below. Cost on CUDA at those sizes: 49 frames 0.18 vs 0.16 s/step,
17 and 33 frames equal.

Images: `cuda-17-vsa-all-frames.jpg` (all 17 frames, VSA), and the last 12
frames of each case in `last12-dense-vs-vsa-*.jpg`.
