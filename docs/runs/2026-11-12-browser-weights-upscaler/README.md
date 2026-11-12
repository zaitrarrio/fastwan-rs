# 2026-11-12: browser weights, default gallery, ×4 upscaler

Prompt Studio's in-browser mode (`crates/fastwan-web`, WebGPU in the tab),
the files it loads, and Real-ESRGAN's compact video upscaler on every backend.

## Files (R2 bucket `fastwan-public`, prefix `v1/`)

| file | bytes | what |
|---|---|---|
| `manifest.json` | 891 | variants, TAE, upscaler, gallery, sources |
| `fastwan-1.3b-f4.safetensors` | 877,080,928 | `fastwan quantize --weights int4-hqq-g64`: block linears 4-bit, the rest f16 |
| `fastwan-1.3b-f16.safetensors` | 2,979,733,280 | `fastwan quantize --weights native`: everything f16 |
| `taew2_1.safetensors` | 22,642,902 | TAEHV (copied from fv-weights) |
| `realesr-animevideov3.safetensors` | 2,490,032 | `scripts/upscale/convert_esrgan.py` of the official `.pth` (sha256 `b8a83768…`) |
| `gallery/` | 3.9 MB | 16 prompts (`web/gallery-prompts.txt`), f16 UMT5 embeddings, 26–35 tokens |

Both DiT files are block-ordered (head tensors first, then `blocks.N`
contiguous), so the page fetches one block per Range request. The earlier
`fv-weights/…-quant/` files are not, and the page refuses them.

**Not yet public**: the bucket exists with CORS for fastwan.pages.dev and
localhost, but its r2.dev URL has not been enabled, so `PUBLIC_WEIGHTS` in
`web/demo.html` is still empty.

## Rental (Vast 60388658, RTX 4090 48 GB, Texas, driver 615.71.09, 450 W, ~$0.32)

- `fastwan check`: all 52 cases pass on CUDA (sm_89), including the new
  PReLU conv epilogue (rel 4.3e-3) and the tiny upscaler (5.0e-3). The WebGPU
  run on this box was **llvmpipe**, not the GPU (see below).
- `fastwan encode` of the 16 gallery prompts: 4.4–6.1 s each on the CPU.
- CUDA, fox, 480×272×81 from the **f4 browser file**: load 6.1 s, clip 1.08 s
  warm (0.28 s/step, decode 0.15 s); `--upscale` ×4 to 1920×1088: **1.92 s for
  81 frames** (42 fps, including the host-side shuffle). The f16 file: same
  speed, load 8.1 s. `fox-f4-x4-f40.jpg`, `fox-f16-480x272-f40.png`.
- In-browser run on the GPU: **not done**. This host passes only NVIDIA's
  compute libraries into the container (no `libGLX_nvidia` / glcore). After
  installing the 615.71.09 userspace graphics libraries and the ICD, both wgpu
  and Chrome (headless shell, full Chromium headless, and headed under Xvfb)
  still got llvmpipe / SwiftShader or no adapter at all. Earlier 4090 hosts
  did expose Vulkan, so this is per host: check `fastwan check --backend wgpu`
  reports an NVIDIA adapter before trusting a WebGPU number.

## Local (no GPU: Chromium with SwiftShader WebGPU)

- Tiny synthetic model through the page, vs the CPU backend at 128×64×17,
  seed 7: f4 and f16 clips **99.9 % byte-exact, max 1 level**; tiny ×2
  upscaler on that clip 99.95 % exact, max 1. UI flow at 480×272×17 with
  *Upscale* ticked ends with 17 `ImageBitmap` frames at 960×544
  (`studio-browser-mode-tiny.png`).
- **Real weights, loading only**: f4 ready in 10.9 s, f16 in 31.4 s (served
  from localhost), 30 blocks + embedders + TAE + upscaler, 16 gallery chips;
  `maxStorageBufferBindingSize` 1 GiB there. Generation with the real model
  was not run in software.
- Upscaler with the real weights vs an independent numpy SRVGGNetCompact
  (`scripts/upscale/parity.py`, 64×48 crop of a real frame): **100.00 %
  identical** at 8 bits (one pixel off by one level).
  `parity-crop-nearest-vs-x4.png`: nearest ×4 vs ours.
