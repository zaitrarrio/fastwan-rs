# fastwan-rs design

## Scope

fastvideo-rs runs ~30 model families on datacenter GPUs, with every
experiment kept behind an env flag. fastwan-rs keeps one path: FastWan 2.1
1.3B, DMD 3-step, batch 1, TAEHV decode. Everything the distilled model
doesn't need (CFG, UniPC, MoE, I2V, TeaCache, sequence parallelism) is gone,
so each op can be a single fused kernel with no branches for other models.

## The op vocabulary

`fastwan_core::Backend` has 13 compute ops. Each is one kernel on a GPU.

| op | what it fuses |
|---|---|
| `linear` | GEMM + bias + optional GELU-tanh; row offsets so the FFN runs in token chunks |
| `layer_norm` | LN + AdaLN modulation |
| `residual_norm` | `h += a·gate` (in place) + LN + affine or AdaLN |
| `gate_residual` | `h += ffn·gate` |
| `head_split` | q/k RMSNorm across heads + RoPE + `[S, H·d] → [H, S, d]` |
| `attention` | flash attention (online softmax, no S×S matrix), writes merged heads |
| `patchify` / `unpatchify` | Conv3d(1,2,2) as a gather, and its inverse |
| `dmd_step` | `x0 = x − σv`, re-noise |
| `conv2d` | implicit GEMM, with the concat and the ×2 upsample done in the address math |
| `tanh_clamp` | TAEHV input clamp |

A Wan block is 14 dispatches: 1 norm, 1 QKV GEMM, 3 head splits, 1
attention, 1 GEMM, 1 residual+norm, 1 GEMM + 1 head split + 1 attention +
1 GEMM (cross), 1 residual+norm, 2 FFN GEMMs per chunk, 1 gated residual.

Work done once instead of per step or per block:

- **Timestep conditioning on the host.** Only 3 timestep values exist. The
  sinusoid → MLP → `time_proj` → `+ scale_shift_table` chain is a few
  mat-vecs per step on the CPU, and the GPU gets finished `[6, dim]` tables.
- **Cross-attention K/V once per prompt.** The text projection and all 30
  blocks' K/V depend only on the prompt; they are computed once and reused
  by all 3 steps.
- **Fused QKV and KV weights**, so one GEMM replaces three (or two).
- **RoPE table at `d/2`.** Both members of a rotated pair share one angle.

TAEHV rewrites (exact; `chunked_decode_equals_one_shot` and the backend
checks cover them):

- TGrow (a 1×1 conv) runs *before* the ×2 upsample. A pointwise op commutes
  with nearest upsampling, so this does ¼ of the work.
- The upsample is folded into the next 3×3 conv's addressing, so the ×2
  tensor never exists in memory.
- The MemBlock `cat([x, past])` is read from two buffers by the conv.
- Chunked decoding carries each MemBlock's last input frame, so memory
  stays constant whatever the clip length.

## Verified, and how

| claim | evidence |
|---|---|
| Wan DiT = Diffusers `WanTransformer3DModel` | one forward on tiny random weights, rel 3.7e-7 (`fastwan-model/tests/hf_parity.rs`, diffusers 0.42) |
| UMT5 encoder = transformers `UMT5EncoderModel` | tiny random weights, rel 1.5e-6, reference run padded + masked as WanPipeline does it, Rust run on the real tokens only (`fastwan-text/tests/hf_parity.rs`, transformers 5.22) |
| a real FastWan clip looks right | yes: coherent, on-prompt 480p clips from the real checkpoint (`docs/runs/2026-11-03-vast-4090/*.mp4`) |
| WebGPU kernels = CPU reference | 25 checks in `fastwan-model::check`, worst rel. error 2.6e-4 (f16 weights), on wgpu/naga + lavapipe; 18 kernel checks in the test page on Chrome/Dawn + SwiftShader |
| tiny DiT 3 DMD steps, WebGPU vs CPU | rel 1.6e-4 |
| FFN chunking and TAEHV chunking change nothing | CPU tests, < 1e-6 |
| CUDA kernels compile | NVRTC 12.9, compute_80/89/120, in `cargo test` when libnvrtc loads |
| CUDA results | all 27 checks pass on an RTX 4090 (`docs/runs/2026-11-03-vast-4090`) |
| WebGPU on a real GPU | all checks pass on the 4090 through NVIDIA Vulkan |
| speed | RTX 4090, 832×480×81, CUDA: 1.35 s per DMD step with VSA (2.48 s dense), **6.9 s per clip** with TAEHV. WebGPU: 12.9 s per step with VSA, **42.7 s per clip** (`docs/runs/2026-11-03-vast-4090-webgpu-perf`) |
| VSA = fastvideo-rs's host reference | CPU path is that oracle; wgpu and CUDA match it on 3 grids (rel ≤ 6e-7 / 5e-3) and end to end on a tiny model, on an RTX 4090 |

## Cost model (480p, 81 frames, 32,760 tokens)

Per DiT forward:

- GEMMs: `30 · 2 · S · (6d² + 2·d·ffn)` ≈ **82 TFLOP**
- self-attention: `30 · 4 · S² · d` ≈ **198 TFLOP**
- cross-attention: ≈ 3 TFLOP

So a clip is ~850 TFLOP with dense attention, and 70% of that is
self-attention. That drives the next steps:

1. ~~**Tensor-core attention on CUDA.**~~ Done (`fw_attention_mma`, bf16
   `mma.sync` m16n8k16, FlashAttention-2 structure): 137 → 2.9 s per 480p step
   on an RTX 4090.
2. ~~**Video sparse attention (VSA).**~~ Done, on all three backends, the
   default for FastWan: 1.83× on the CUDA denoise (2.48 → 1.35 s per step).
3. **WebGPU throughput.** Unrolling the register blocks (named vec4
   accumulators instead of arrays, which NVIDIA's Vulkan compiler kept in
   local memory) took the GEMM to ≈ 11.6 TFLOP/s and a VSA 480p step from
   83 s to 12.9 s (`docs/runs/2026-11-03-vast-4090-webgpu-perf`). Bigger
   tiles alone were 38% slower. f16 math with `shader-f16` is implemented and
   opt-in: same output (43.8 dB PSNR) but 8.5% slower on a 4090, where f16
   has no rate advantage; it awaits an AMD / Intel / Apple measurement. Next:
   subgroups / cooperative matrices where the adapter has them.
4. **AMD fast path.** WebGPU (Vulkan) already runs on AMD. A HIP backend
   would be a mechanical port of `kernels.cu` (hipRTC + hipBLASLt) for ROCm
   cards.

## Other known gaps

- UMT5 runs on the CPU: about 4.6 GMAC per prompt token over 11.4 GB of
  bf16 weights, so it is bound by memory bandwidth (expect a few seconds per
  prompt on a desktop, unmeasured). A GPU path would need ~11.4 GB of VRAM on
  top of the DiT. Wan's ftfy text fixing is not ported; whitespace and HTML
  entities are cleaned.
- On CUDA the latents are bf16; FastVideo keeps them f32. Move `dmd_step`'s
  state to f32 if the 3-step drift shows.
- `linear`'s bias/GELU on CUDA is a second pass; cuBLASLt epilogues would
  fuse it.
- TAEHV allocates per chunk; a buffer pool would remove that churn.
- The full Wan VAE is not implemented (TAEHV only). fastvideo-rs measures the
  full VAE at 2.91 s vs TAEHV 0.26 s on an RTX PRO 6000, with lower quality
  for TAEHV.

## Weight quantization

`--weights int8` stores the block linears (97% of the DiT) as symmetric int8
with an f32 scale per 128 input columns. It is weight-only: activations keep
the backend's precision, and every backend computes `Σ x·(q·s)` with `q·s`
formed in f32, so the CPU's dequantized weights are an exact reference. WebGPU
unpacks codes inside the GEMM's tile load; CUDA expands one weight at a time
into a bf16 scratch and keeps cuBLAS. Measured on a 4090: −1.3 / −1.5 GB of
VRAM, no speed change (the GEMMs are compute bound at 32k tokens), and a
smaller output change than the bf16-vs-f32 backend gap.

Pruna was considered and not adopted: it transforms PyTorch modules at run
time, its distillers cover text-to-image and text-to-text only, and its
step caches (FasterCache, PAB) target 30–50-step samplers, not 3 DMD steps.
Its HQQ quantizer remains the candidate for an offline int4 export if int8
is not small enough.

4-bit (HQQ ported, groups 32–128, FFN-only) gives weight SNR 20–22.5 dB vs
int8's 43.6, and every variant changes the sample a seed produces, without
visible artifacts (`docs/runs/2026-11-03-vast-4090-int4`). It is kept for the
browser download: `fastwan quantize` writes one safetensors file with packed
codes plus f16 scales / zeros (format in `export.rs`), loaded as is by every
backend (real int4 GEMMs: WebGPU unpacks nibbles in the tile load, CUDA
dequantizes into the bf16 scratch). int4-hqq-g64-ffn is 1.18 GB; measured in
`docs/runs/2026-11-03-vast-4090-q4-store`.

Next for the browser: a wasm build that reads this file, and a text-encoder
plan (UMT5-XXL is 11 GB).

## The demo engine (`fastwan serve`)

One engine thread owns the backend and the resident models and runs jobs one
at a time; the HTTP thread only reads a shared job table, so a page polling
for progress never waits on the GPU. The pipeline reports `Prepared`, one
`Step` per DMD step, and `Frames` per TAEHV chunk, so the page plays frames
while the rest of the clip decodes. The server binds 127.0.0.1 and sends no
CORS headers, so another website cannot queue work on your GPU. It keeps the
last eight clips (their RGB frames are the memory cost).

## Decode on tensor cores

TAEHV's convs were a scalar implicit GEMM with four integer divisions per
gathered element: 1.55 s per 81-frame 480p decode on an RTX PRO 6000.
`fw_conv2d_mma` gathers one pixel row per thread with counted (ci, ky, kx),
feeds bf16 `mma.sync` 128×64×32 tiles and prefetches the next K tile into
registers: 0.56 s, the same output to bf16 rounding (58.5 dB). fastvideo-rs
does it in 0.26 s; a double-buffered `cp.async` pipeline and channel-last
activations are the obvious next steps, but the DiT step (3.80 s) is now
the larger gap.

## Reference numbers (fastvideo-rs, not this repo)

`lab/gold/fastwan21-1.3b.toml`, RTX PRO 6000 Blackwell, 832×480×81, VSA:
denoise 1.92 s, full-VAE decode 2.91 s, peak 27.1 GiB. With TAEHV and an
MXFP8 DiT: gen 2.26 s, decode 0.26 s, peak 21.5 GiB.
