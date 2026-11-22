# fastwan-rs

FastWan text-to-video on desktop GPUs, NVIDIA or AMD, in Rust. A lean
re-implementation of the Wan path of
[fastvideo-rs](https://github.com/zaitrarrio/fastvideo-rs), scoped to one
model and written for consumer hardware instead of a datacenter.

- **Model:** `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`: Wan 2.1 DiT, 3 DMD
  steps (`[1000, 757, 522]`), no CFG. Decoder: TAEHV `taew2_1`
  (`taew2_2` loads too, for Wan 2.2 latents).
- **Backends** (one op vocabulary, `fastwan_core::Backend`):

| backend | runs on | precision | status |
|---|---|---|---|
| `wgpu` | **NVIDIA, AMD, Intel, Apple**: Vulkan / DX12 / Metal, and browsers | f32 activations, f16 weights | conformance passes on an RTX 4090 (Vulkan); **832×480×81 clip in 42.7 s** (12.9 s per DMD step with VSA + 3.5 s TAEHV) |
| `cuda` | NVIDIA sm_80+ | bf16, cuBLAS GEMMs, tensor-core attention + VSA | conformance passes on an RTX 4090; **832×480×81 clip in 6.9 s** (1.35 s per DMD step with VSA + TAEHV) |
| `cpu` | anywhere | f32 | the reference every backend is checked against |

**VRAM** (measured on an RTX 4090, `docs/runs/2026-11-03-vast-4090-vram`):
CUDA 5.0 GB peak for 832×480×81 (3.6 GB at 480×272×33); WebGPU 7.2 GB
(4.4 GB). The text encoder runs on the CPU (11–22 GB of RAM).

On AMD, Intel or Apple GPUs try `FASTWAN_WGPU_F16=1` (f16 math kernels:
same output to 43.8 dB PSNR; no gain on NVIDIA GeForce, unmeasured elsewhere).

Short on VRAM? `--weights int8` (on `generate`, `bench` and `serve`) stores
the DiT's block weights as int8: peak 3.7 GB on CUDA / 5.6 GB on WebGPU for
832×480×81, same speed, output within the gap between the two backends.

Smaller download: `fastwan quantize --transformer <dir>/transformer --weights
int4-hqq-g64-ffn --out fastwan-int4.safetensors` writes a 1.18 GB checkpoint
(0.88 GB with `int4-hqq-g64`) that `--transformer` loads directly. 4-bit
gives a different sample per seed rather than a worse one; peak VRAM 3.4 GB
(CUDA) / 5.4 GB (WebGPU).

`--backend auto` picks CUDA when it is built in and a device answers, and
WebGPU otherwise.

## Use

```bash
cargo build --release                      # WebGPU + CPU
cargo build --release --features cuda      # + CUDA (no CUDA toolkit needed to build)
```

```bash
./target/release/fastwan check             # every kernel vs the CPU reference, on your GPU
./target/release/fastwan plan --vram-gib 12 --backend wgpu
./target/release/fastwan bench --layers 4  # synthetic weights, real 480p geometry
```

### Tasks

[`Taskfile.yml`](Taskfile.yml) runs this workflow (`task --list`). It needs [Task](https://taskfile.dev).

```bash
task setup            # cargo fetch and a toolchain check
task weights          # checkpoint + TAEHV into ./fastwan
task build            # WebGPU + CPU; task build:cuda on NVIDIA
task clip             # UMT5 embed, then out/<prompt>-832x480x81-s7.mp4
```

`task clip PROMPT="a crowded night market in the rain" SEED=3 BACKEND=wgpu` reuses `./gallery` when that prompt was encoded before. `task embed:gallery` encodes `web/gallery-prompts.txt`. `task serve` is the local studio. `task clip:causal MEMORY=memrope` is History Forcing; `UPSCALE=` and `UPSCALE_X=2` pass the upscaler through.

### Prompt-to-clip demo

```bash
huggingface-cli download FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
    --include "transformer/*" "text_encoder/*" "tokenizer/*" --local-dir fastwan
huggingface-cli download madebyollin/taehv taew2_1.safetensors --local-dir fastwan
./target/release/fastwan serve --model-dir fastwan      # then open http://127.0.0.1:7860/
```

`fastwan serve` keeps the DiT and TAEHV resident on the GPU, encodes prompts
with a native UMT5-XXL port on the CPU (reading the bf16 weights straight from
the mmapped files; budget ~12 GB of RAM for the page cache), and serves
`web/demo.html`: type a prompt, pick a size, length and seed, and watch the
three steps and the decode land, with frames playing as they arrive. Save as
WebM or PNG from the page. `--synthetic` runs a tiny random model, for trying
the engine and page without downloading anything (the clips are noise).

### Hosted pages

<https://fastwan.pages.dev> (Cloudflare Pages; `scripts/deploy-pages.sh`)
serves the same studio plus the WebGPU bench. The hosted studio drives the
engine on the visitor's own machine, so start it with that site allowed:

```bash
./target/release/fastwan serve --model-dir fastwan --allow-origin https://fastwan.pages.dev
```

`--allow-origin` answers CORS and Chrome's private-network preflight for that
one origin only. Generation requests must be JSON and come from the engine's
own page or that origin, so another website cannot queue clips on your GPU.
Safari blocks https pages from calling 127.0.0.1; use Chrome, Edge or Firefox.

### In the browser, no install

Prompt Studio's **In this browser (WebGPU)** mode runs the whole model in the
tab: the engine is the same Rust code compiled to wasm (`crates/fastwan-web`,
built by `scripts/build-web.sh` into `web/pkg/`), and the weights stream from
a public R2 bucket, one transformer block per HTTP Range request, into the
browser's cache (downloaded once). Two weight files:

| variant | file | size |
|---|---|---|
| f4 | int4-hqq-g64 block linears, everything else f16 | 0.88 GB |
| f16 | everything f16 | 2.98 GB |

plus TAEHV (23 MB) and, if *Upscale ×4* is ticked, Real-ESRGAN's compact
video model (2.5 MB), which takes a 480×272 clip to 1920×1088. There is no
text encoder in the browser (UMT5-XXL is 11 GB): prompts come from a
precomputed gallery (`web/gallery-prompts.txt`, embedded with `fastwan
encode`). Rebuild the files with `fastwan quantize --weights int4-hqq-g64`
and `--weights native` (block-ordered on write; the page refuses files that
are not), the upscaler with `scripts/upscale/convert_esrgan.py`. For a local
copy of the weights, open `/demo/?weights=<base URL>`.

### One clip from the command line

Prompt embeddings come from `scripts/encode_prompt.py`:

```bash
huggingface-cli download FastVideo/FastWan2.1-T2V-1.3B-Diffusers --include "transformer/*" --local-dir fastwan
huggingface-cli download madebyollin/taehv taew2_1.safetensors --local-dir fastwan
python scripts/encode_prompt.py "a fox running through snow" fox.safetensors
./target/release/fastwan generate --transformer fastwan/transformer --tae fastwan/taew2_1.safetensors \
    --embeds fox.safetensors --width 832 --height 480 --frames 81 --out fox.y4m
ffmpeg -i fox.y4m fox.mp4
```

Add `--upscale realesr-animevideov3.safetensors` to upscale ×4 after the
decode (convert the official `.pth` with `python3 -I
scripts/upscale/convert_esrgan.py`; it is loaded tensors-only, without torch).

## WebGPU test page

`web/fastwan-webgpu-bench.html` runs the backend's WGSL kernels in the
browser on your GPU: conformance against a JS reference, throughput at Wan
shapes, and a per-resolution clip estimate. Serve it over localhost
(`python3 -m http.server -d web`) and open it in Chrome/Edge. Rebuild it
after a shader change with `python3 web/build.py` (a test fails otherwise).

## Layout

```
crates/fastwan-core    config, geometry, DMD schedule, RoPE, weights, Backend trait, memory plan
crates/fastwan-cpu     reference backend
crates/fastwan-wgpu    WebGPU backend (src/shaders/*.wgsl)
crates/fastwan-cuda    CUDA backend (src/kernels.cu, NVRTC + cuBLAS)
crates/fastwan-model   Wan DiT, TAEHV, ×4 upscaler, DMD pipeline, conformance checker
crates/fastwan-web     the WebGPU engine as wasm, for Prompt Studio's in-browser mode
crates/fastwan-text    tokenizer + UMT5-XXL encoder (CPU, bf16 weights in place)
crates/fastwan-cli     the `fastwan` binary (plan / check / bench / generate / serve)
web/                   demo.html (prompt to clip, served by `fastwan serve`), WebGPU test page
scripts/reference/     fixtures from transformers / diffusers that the Rust ports are tested against
docs/design.md         what is fused and why, what is verified, what is next
```

`cargo test --workspace` runs everything, including the WebGPU conformance
suite when a Vulkan driver is present (`apt install mesa-vulkan-drivers`
gives lavapipe on a machine without a GPU).
