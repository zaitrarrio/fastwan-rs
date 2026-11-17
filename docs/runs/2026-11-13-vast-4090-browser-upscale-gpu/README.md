# 2026-11-13: browser upscale kept on the GPU (RTX 4090)

Vast instance 60556221: RTX 4090, Utah, driver 620.70.06, 450 W, ~$0.15.
Headless Chromium (Playwright's full build) on the GPU, page and weights
served from localhost. Fox, 480×272×81, seed 7, 4-bit weights, "Upscale ×4"
ticked. `old` = d72bacd (every upscaled frame read back and turned into an
`ImageBitmap`), `new` = d700eff (frames stay on the GPU and are drawn onto a
WebGPU canvas, `fastwan_wgpu::CanvasView`). Same box, same session; new was
run before and after old.

| | old | new |
|---|---|---|
| upscale ×4 to 1920×1088, first clip (loads the 2.5 MB upscaler) | 2.6 s | 1.7 s, 1.7 s |
| upscale, second clip | 2.4 s | 1.4 s, 1.5 s |
| end of decode → first upscaled frame on screen | 2.6 s (all at the end) | **0.46–0.55 s** |
| Generate → low-res clip playing | 9.3–9.6 s | same |

Timeline marks (ms after Generate, first clip): old `decode_done 9880`,
`first_upscaled_on_screen 12505`; new `decode_done 9940 / 10240`,
`first_upscaled_on_screen 10400 / 10785`, `all_upscaled 11640 / 11963`.
(The second-clip marks in the raw log are not valid: the recorder saw the
previous clip's state.)

What remains in the 1.4 s is GPU work: the WGSL convs (no tensor cores in
WebGPU) for 81 frames, about 3× CUDA's 0.285 s for the same network on this
card class. Faster WebGPU conv kernels (subgroups) are the next lever.

`screen-upscaled-f40.jpg`: frame 40 on the page as displayed, from the
WebGPU canvas. Reading that canvas back with `drawImage` returned empty
pixels in headless Chrome on the GPU (it worked on SwiftShader), so the
pixel-exact check (canvas = engine frame, maxAbs 0) is from SwiftShader and
the real-GPU check is the screenshot. Nothing in the page reads that canvas
back: PNG export uses `upscaledFrame`, thumbnails use the low-res frames.
