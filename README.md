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
| `wgpu` | **NVIDIA, AMD, Intel, Apple**: Vulkan / DX12 / Metal, and browsers | f32 activations, f16 weights | all conformance checks pass (wgpu/naga on lavapipe, Chrome/Dawn on SwiftShader); no real-GPU timing yet |
| `cuda` | NVIDIA sm_80+ | bf16, cuBLAS tensor-core GEMMs | kernels compile under NVRTC for sm_80/89/120; **never run on a GPU** |
| `cpu` | anywhere | f32 | the reference every backend is checked against |

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
crates/fastwan-model   Wan DiT, TAEHV, DMD pipeline, conformance checker
crates/fastwan-text    tokenizer + UMT5-XXL encoder (CPU, bf16 weights in place)
crates/fastwan-cli     the `fastwan` binary (plan / check / bench / generate / serve)
web/                   demo.html (prompt to clip, served by `fastwan serve`), WebGPU test page
scripts/reference/     fixtures from transformers / diffusers that the Rust ports are tested against
docs/design.md         what is fused and why, what is verified, what is next
```

`cargo test --workspace` runs everything, including the WebGPU conformance
suite when a Vulkan driver is present (`apt install mesa-vulkan-drivers`
gives lavapipe on a machine without a GPU).
