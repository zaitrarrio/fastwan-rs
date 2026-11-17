# 2026-11-15: long videos in Chrome on an RTX 4090 (stalls)

Prompt Studio's long-video mode (`Engine.generateLong`, Self-Forcing f16,
480×272, 20 s = 27 blocks, seed 7) in headless Chromium (Playwright build,
Vulkan) on Vast RTX 4090 hosts (driver 615.71.09, 450 W). Per-block times in
seconds; "denoise" is the latent readback await, so it holds all GPU work.

Correctness: SwiftShader Chrome vs the CPU reference (tiny model, 480×272×165):
99.93% of bytes equal, max 1 level; Stop ends after the block in progress.
Clip mode is unaffected (f4 832×480×81 in 21.9 s in the same session).

| run | total | per block |
|---|---|---|
| native wgpu (`generate --causal --backend wgpu`) | 47.1 s | 2.3, then 1.35-1.62 flat for all 27 |
| Chrome, before | 287.6 s | 1.5-1.7, except 8-10: 21/35/27, 15-17: 16/41/30, 22-24: 16/41/30 |
| Chrome, playback paused | 267.2 s | same pattern: not the page |
| Chrome, `gc()` after each block | 487.4 s | worse: a 24/41/30 triple every 5 blocks |
| Chrome, no keep-alive (staging + uniforms destroyed) | 475.6 s | triples of 21-41 s every ~5 blocks |
| Chrome, keep-alive | 209.7 s, 234.2 s | isolated stalls of 4-36 s, the rest 1.5-1.6 |

- The stalls are waiting, not compute: whole blocks of 1.5 s turn into 16-41 s
  (41.0 s recurs exactly), always inside the readback await, and native wgpu
  on the same GPU never shows them.
- Not JS garbage: forcing GC makes it worse. Destroying dropped buffers,
  dispatch uniforms and readback staging buffers (wgpu's web backend never
  frees them; `Owned` / `retire` now do after the next submit) did not change
  the pattern by itself.
- An empty `queue.submit` every 20 ms while a readback is pending
  (`KeepAlive`, on by default; `?nokeep=1` turns it off) halves the total and
  breaks up the triples. Still not native speed.

Not tried: replacing the K/V cache's per-head `copy_buffer_to_buffer` calls
(~3,600 per block) with one dispatch, fewer and larger submits, desktop
Chrome with a display (all runs here are headless Linux), other GPUs.
