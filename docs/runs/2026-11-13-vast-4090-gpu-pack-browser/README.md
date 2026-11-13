# 2026-11-13: frames packed on the GPU, and the browser on a real GPU (RTX 4090)

Vast instance 60522362: RTX 4090, Utah, driver 620.70.06, 450 W, ~$0.30.
Weights: the public f4 file (`fastwan-1.3b-f4.safetensors`), fox gallery
embedding, seed 7. A/B is the same box, same weights: `old` = 638eb0c, `new`
= this change. Output compared frame by frame from the y4m files.

## `pack_rgba8`: what moved, what did not

The new op does TAEHV's and the upscaler's last step on the device
(pixel-shuffle, residual, clamp, 8-bit rounding) and hands back packed RGBA.

| CUDA, 81 frames, warm (run 3 of 3) | old | new | output |
|---|---|---|---|
| decode 832×480 | 0.36 s | **0.33 s** | identical |
| decode 480×272 | 0.14 s | 0.13 s | identical |
| upscale ×4 480×272 → 1920×1088 | 0.99 s | **0.87 s** | identical |

How it got there (each step measured on this box):

1. First version: upscale **0.99 → 1.14 s, slower**, and 16% of upscaled
   pixels off by 1–2 levels. Two causes: the host now did two single-threaded
   passes per pixel (word → bytes by iterator, then RGBA → RGB) where the old
   code converted across 16 threads; and on CUDA the residual (the input
   frame) went through bf16, which has ~1 level of precision near 1.0.
2. The residual is now the input as exact 8-bit words (`upload_words`),
   RGBA → RGB is threaded and the CLI upscale is pipelined: 1.08 s, output
   identical to old.
3. `FASTWAN_PROFILE=1` then showed the upscale's GPU time is **285 ms**
   (convs 272 ms): the rest was host copies. CUDA's `download` of a word
   buffer rebuilt 170 M floats one by one; `download_rgba8` copies the device
   bytes once: **0.87 s**.

So on native CUDA the win is small, and the remaining ~0.6 s of the upscale is
host memory traffic for 1.2 GB of RGB output, not GPU work. The point of the
op is the browser, where it removes a single-threaded wasm loop over every
output pixel and the JS RGB → RGBA loop: frames now go from the GPU straight
into `ImageData`.

## Prompt Studio in the browser, real GPU (first time)

Headless Chromium (Playwright's full build; the headless *shell* build
exposes no `navigator.gpu`) on the same 4090: adapter `nvidia / lovelace`,
`maxStorageBufferBindingSize` 2 GiB, `maxBufferSize` 4 GiB. Page served from
localhost with the weights next to it.

| clip (4-bit weights) | denoise | decode | total |
|---|---|---|---|
| model load | | | 5.7 s (from disk cache) |
| 480×272×81 + upscale ×4 | 8.7 s (2.9 s/step) | 0.8 s | 12.7 s (upscale 2.6 s) |
| 832×480×81 | 32.8 s (10.9 s/step) | 1.8 s | 35.7 s |
| 480×272×33 | 3.3 s | 0.3 s | 3.8 s |

For scale: native WebGPU on another 4090 host was 12.9 s/step at 832×480
(driver 595.99, `docs/runs/2026-11-03-vast-4090-webgpu-perf`); CUDA here is
1.12 s/step. `browser-jellyfish-832x480.jpg`, `browser-fox-480x272-x4.jpg`
(1920×1088, scaled for the page), `browser-studio-market.jpg`.

Found on the way:

- The page treated any `http://127.0.0.1` / `localhost` page as the
  engine's own and hid browser mode, so a locally hosted copy of the site
  could not run in the tab. `fastwan serve` now marks its page
  (`<meta name="fastwan-served" content="engine">`); everything else may run
  in the tab.
- A missing gallery file (the rental had only the fox) now ends the run with
  "Generation failed: … HTTP 404" instead of a stall: the error path from
  638eb0c, seen working.
- The previous rental (Texas) passed only NVIDIA's compute libraries into the
  container; this one exposed Vulkan, and `fastwan check --backend wgpu`
  named the GPU. Check that before trusting any WebGPU number.
