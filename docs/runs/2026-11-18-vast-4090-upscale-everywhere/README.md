# 2026-11-18: ×2 / ×4 upscaling everywhere (RTX 4090)

Two Vast rentals: 61268316 (Netherlands, driver 595.99.02) and 61275108
(Japan, driver 615.71.09), both RTX 4090 at 450 W. Full output in `run.log`.

Code: ×2 = the ×4 `realesr-animevideov3` network then a 2×2 box filter in
`pack_rgba8` (`PackShape::down`); `unpack_rgba8` turns packed RGBA frames
into the upscaler's input on the device (`Upscaler::launch_words`), so
decoded frames go in without a host pass. CLI `--upscale-x 2|4`, and
`--causal --upscale` upscales each block as it is decoded. Browser: frames
are upscaled as they play into a window of at most 768 MB (96 MB on phones)
from the playhead forward; playback waits for each upscaled frame. `fastwan
serve --upscaler` takes `upscale: 2|4` per request.

## Checks

WebGPU 66/66. CUDA: all upscale / pack / unpack / word-copy cases pass; the
three VSA cases fail (rel 0.28-0.29) because random inputs rounded to bf16
flip top-k picks: a check-input problem, fixed on the unmerged branch
`claude/modest-hypatia-qt2bff` (`e72ff08`), not a kernel bug. Run 1 found
CUDA `copy` refusing packed-word buffers (every CUDA upscale panicked):
fixed, with a check case.

## CLI, CUDA (run 2)

| job | upscale | total |
|---|---|---|
| clip 832×480×81 ×4 → 3328×1920 | 2.99 s | |
| clip 832×480×81 ×2 → 1664×960 | 1.39 s | |
| long 832×480, 165 frames, none (run 1) | | 15.3 s (10.8 fps) |
| long, ×2 | 0.33 s per 12-frame block | 20.0 s (8.2 fps) |
| long, ×4 | 0.58 s per block | 31.8 s (5.2 fps; writing 3.2 GB of y4m) |

Native WebGPU, long 480×272 ×2 (run 1): 34.7 s for 165 frames vs ~29 s
without (the per-block "denoise" line is queue time on wgpu, so the upscale
column there includes waiting for it). Chrome runs the same kernels faster.

## Chrome (headless, real GPU)

| clip / video | upscale ms/frame | playback | kept on GPU |
|---|---|---|---|
| clip 832×480 ×4 (3328×1920) | 49-53 | waits (window 29 of 81 frames) | 29 |
| clip 832×480 ×2 (1664×960) | 43-49 | 14.5 fps, 5 waits in 10 s | 81 of 81 |
| clip 480×272 ×4 (1920×1088) | 19 | | 81 of 81 |
| long 480×272 ×2 | 16 | 14.5 fps | 321 of 321 |
| long 480×272 ×4 | 22-26 | 14.9 fps, 1 wait in 10 s | 91 |
| long 640×352 ×2 | 26 | 14.6 fps | 217 |

Long-video generation is unchanged with upscaling on (1.3 s per block,
37 s for 20 s of video). Playback is measured as frames drawn
(`?updebug=1` counters); run 1's script measured the playhead, which wraps on
an 81-frame clip, and reported clips at "6.3 fps" when they drew 14.5. The
frame on screen was checked by reading it back from the GPU: the upscaled
canvas is visible and its box-filtered image is within 3.3-4.8 levels (mean)
of the low-res frame. Headless screenshots do not capture the WebGPU canvas
(×2 and ×4 screenshots were byte-identical), so none are kept here.

At 832×480 ×4 a 768 MB window holds 29 frames, so a looping clip is
re-upscaled each pass at ~20 frames/s: playback waits there (not measured
with the fixed counter).

## Chrome long-video stalls

Japan host, driver 615.71.09 (as the 2026-11-17 stall host), old engine
(`?engine=7b20eb2`) vs pool, twice each: 36.5 / 36.3 / 36.6 / 36.3 s, all
blocks 1.0-1.5 s. Still not reproduced on a third host.
