# 2026-11-18: WebGPU buffer pool vs the Chrome long-video stalls (RTX 4090)

Question: were the 25–40 s blocks in Chrome long videos
(`2026-11-17-vast-4090-stalls-f16-phones`, instance 61118463, Japan, driver
615.71.09) caused by buffer churn? Before this change the engine created
~2,772 GPU buffers (32.6 MB) per block and Dawn frees dropped buffers lazily.

Change (`d10f7be`): `fastwan_wgpu::Recycler` keeps dropped storage buffers
and dispatch uniforms per device, moves them to a free list after the next
submit and reuses them for the same (size, usage), zero-filled with
`clear_buffer`. Readback staging buffers are still destroyed.

Method: headless Chromium (Vulkan), Prompt Studio long video, Self-Forcing
f16 weights, 480×272, 20 s (27 blocks), seed 7, `scripts/vast/long_arms.py`.
"old" is the pre-pool engine (`?engine=7b20eb2`, otherwise identical) on the
same host. Full output in `run.log`.

## Results

| host | engine | total | blocks (s) |
|---|---|---|---|
| A: 61167209, Michigan, driver 595.71.05 | old | 45.0 s | 2.6, then 1.5–1.6 flat |
| A | pool f32 | 37.3 s | 1.3, then 1.1–1.3 flat |
| A | pool `?f16=1` | 37.4 s | flat 1.3 |
| A | pool `?halfkv=1` | 37.5 s | flat 1.3 |
| B: 61169799, Nevada, driver 615.71.09 | old | 37.0 s | 1.1–1.4 flat |
| B | pool f32 | 38.7 s | 1.6, then 1.2–1.4 flat |

Buffers per block after the first: old 2,772 created (32.6 MB); pool 3
created (7.1 MB, the staging buffers), 2,769 reused, 6.45 GB live (5.32 GB
with the 16-bit cache), flat.

On host B two arms failed before generating: the first "old" load reported
"Loading failed: network error" (weight streaming), and the second pool arm
timed out waiting for the length options. Neither reached the generate step.

## Verdict

**The stalls did not reproduce on either host, with either engine** —
including driver 615.71.09, the same driver as the stall host. So this run
neither confirms nor refutes buffer churn as their cause: they appear to be
specific to that host (instance 61118463, not on offer today), not to the
engine. The pool is kept: 17% faster on host A (37.3 vs 45.0 s), equal on
host B, pixel-identical on SwiftShader (99.93% exact, max 1 level, same as
before), and it removes ~2,770 allocations per block on every browser.

If a user reports slow blocks again, ask for the page's block times and the
GPU / driver, and compare `?engine=7b20eb2` on that machine.
