# 2026-11-17: History Forcing (MemRoPE), LightTAE and FP8 Self-Forcing on an RTX 4090

Vast instance 61077673 (RTX 4090, Spain, driver 595.99.02, 450 W limit),
~25 min. History Forcing `dmd/model.pt` (`xzxxntxdy/History-Forcing`, EMA
weights) through `scripts/convert_self_forcing.py` and `quantize --weights
native` (11.4 GB f32 pickle → 2.84 GB f16 in 20 s with the FP8 conversion);
Self-Forcing `v1/self-forcing-1.3b-f16.safetensors`. `generate --causal`,
832×480, seed 0, 4 steps (1000 / 937.5 / 833.3 / 625), cache 12 latent
frames with a 3-frame sink, TAEHV decode unless noted.

## Conformance

| backend | result |
|---|---|
| CUDA (sm_89) | 60 / 60 with the VSA check fix (`rope_heads` 3.2e-3, `kv_ema` 1.8e-3, MemRoPE rollout 4.3e-3, window rollout 7.3e-3, VSA model grid 3.1e-3 / 2.6e-3) |
| WebGPU, Vulkan NVIDIA 595.99 | 60 / 60 (`rope_heads` 6e-8, `kv_ema` 6e-8, MemRoPE rollout 1.7e-4) |

Before the fix the two VSA model-grid cases (from main) failed on CUDA at
0.63 / 0.24: CUDA rounds uploads to bf16, and with random q / k the
pooled tile scores are close enough that the rounding alone flips top-k
picks; the CPU oracle fed the same bf16-rounded inputs moves 0.41 / 0.21
from its f32 self. The cases now use bf16-representable inputs.

## 63 s clips (1005 frames, 84 blocks)

| clip | weights | memory | wall | fps | full-cache block |
|---|---|---|---|---|---|
| musician | History Forcing | MemRoPE | 96.1 s | 10.5 | 1.07 + 0.08 s |
| musician | Self-Forcing | window | 93.2 s | 10.8 | 1.04 + 0.08 s |
| musician | History Forcing | window (ablation) | 93.6 s | 10.7 | 1.04 + 0.09 s |
| astronaut | History Forcing | MemRoPE | 95.6 s | 10.5 | 1.07 + 0.08 s |
| astronaut | Self-Forcing | window | 93.1 s | 10.8 | 1.04 + 0.08 s |

First frames at 0.79 s in every run. MemRoPE costs ~3 % per block (the
key rotation of the whole cache each forward, the EMA once per block).
This host's blocks are slower than the 2026-11-14 4090's (1.04 vs 0.84 s
for the same Self-Forcing window run), so compare within this table only.

Strips (one frame every 40, 2.5 s):
- `hf-mem-musician-strip.jpg`: one stable close-up for the whole minute.
- `sf-win-musician-strip.jpg`: stable, a different framing.
- `hf-win-musician-strip.jpg`: **History Forcing weights on the window
  cache degrade**: colour bars at the right edge from ~15 s, a checkered
  guitar texture around 30–45 s, recovering after. With the memory they
  were trained for they do not: the MemRoPE cache is what this model needs.
- `hf-mem-astro-strip.jpg`: large camera motion (push-in to the suit, a
  pod, then following from behind), no artefacts; `sf-win-astro-strip.jpg`:
  a fixed framing with some ground texture crawl.

## LightTAE, FP8, WebGPU (321 / 81 frames)

| comparison | PSNR (avg / Y) |
|---|---|
| TAEHV vs LightTAE (`lighttaew2_1_v2`), same latents | 35.4 / 31.3 dB |
| Self-Forcing bf16 vs LightX2V FP8 (decoded to f16), fox | 21.1 / 17.2 dB |
| MemRoPE rerun vs first run (CUDA) | identical (inf) |
| WebGPU vs CUDA, 480×272×81 fox | 19.4 / 15.3 dB |

LightTAE costs the same as TAEHV here (30.2 vs 30.2 s for 321 frames,
decode 0.08 s per block). FP8 weights make a different sample of the same
scene (the fox turns another way), the band int4 sits in; no artefacts
(`fp8-lighttae-strips.jpg`: FP8 pair, then the decoder pair). WebGPU and
CUDA differ like f32 vs bf16 compounded through a causal rollout; the
WebGPU clip is coherent. WebGPU 480×272: 14.8 s for 81 frames (2.26 s to
first frames), CUDA 1.9 s.
