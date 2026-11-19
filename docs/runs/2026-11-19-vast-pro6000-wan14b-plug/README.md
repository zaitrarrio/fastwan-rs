# 2026-11-19: Wan2.1-T2V-14B with LongLive-Plug's few-step LoRA (RTX PRO 6000 Server Edition)

Vast instance 61346466: RTX PRO 6000 Blackwell Server Edition (sm_120),
600 W, driver 580.178.04, 1 TB RAM; ~20 min. A first rental (a UK Server
Edition) never left "loading" and was destroyed after 30 min.

The binary is e8c120c. It is compared with `main` 387d81e's build (the
History Forcing merge's base).

Weights:
- `Wan-AI/Wan2.1-T2V-14B-Diffusers` `transformer/`: 12 fp32 shards, 54 GB.
  Downloaded with the adapters in 250 s, then `fastwan quantize --weights
  native` wrote 28.4 GB of f16 in 135 s.
- `Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-few-step`
  `generator_lora_lightx2v.safetensors`: rank 128, alpha 128, bf16, Wan names.
- `…-cfg` `adapter_model.safetensors`: PEFT names, f32.

UMT5 and TAEHV (`taew2_1`) are the FastWan tree's. Every clip is 832×480×81,
seed 0, `--schedule longlive-plug`: LightX2V's step-distill ladder from the
adapter's `inference_config.json`, timesteps 1000 / 937.5 / 833.3 / 625,
Euler steps, no CFG.

## Conformance

| backend | result |
|---|---|
| CUDA (sm_120) | **75 / 75**: `add_act` 3.1e-3 / 2.5e-3, two stacked LoRA branches 3.6e-3, VSA 2.3–2.5e-3, MemRoPE 4.3e-3 / 5.7e-3 |
| WebGPU (Vulkan, NVIDIA 580.178.04) | **75 / 75**: stacked LoRA branches 1.7e-4 |

## The 14B with the plug

| clip | adapters | denoise (4 forwards) | total after load | peak GPU | load |
|---|---|---|---|---|---|
| fox | few-step | 22.6 s (5.64–5.68 s each) | 24.9 s | 33.2 GB | 108 s |
| fox | few-step + CFG 0.5 | 23.7 s (5.91–5.94 s) | 25.5 s | 34.8 GB | 109 s |
| fox | none (same 4-step sampler) | 20.7 s (5.16–5.18 s) | 22.5 s | 31.6 GB | 94 s |
| astronaut | few-step | 22.7 s | 24.6 s | 33.2 GB | 95 s |
| musician | few-step | 22.7 s | 24.4 s | 33.2 GB | 97 s |

Prompt encoding is 1.3–1.8 s on the GPU, after 3–6 s of UMT5 on the CPU.
Decode (TAEHV) takes 0.32–0.40 s.

**Branch cost.** The unmerged LoRA branch costs 0.48 s per forward (9 %). The
CFG adapter stacked on it costs a further 0.26 s.

`wan14b-plug-strips.jpg` shows one frame every 16, top to bottom:
1. **Plug alone:** a photographic fox trotting through birch snow at dawn.
2. **Plug + CFG 0.5:** the same scene and motion, with more contrast and
   saturation. It is 21.4 dB from row 1, the strength of a guidance change.
3. **No adapter:** a grey blur. Base Wan 14B is a 50-step model with CFG, and
   4 Euler steps are meaningless without the adapter.
4. **Plug, astronaut:** a clean desert scene with two moons.
5. **Plug, musician:** a clean guitar close-up.

All plug clips are coherent with no artefacts.

## `main` against this build (`docs/runs/2026-11-18-vast-4090-merge-check`'s open item)

| clip | result |
|---|---|
| FastWan 1.3B, 832×480×81, VSA | **byte-identical**; 2.22 s against 2.19 s |
| Self-Forcing window cache, 321 frames (`--causal`) | **byte-identical**; 16.4 s against 17.9 s (the old build had three decode spikes of 0.25–0.40 s; host noise) |

Today's changes leave the FastWan and Self-Forcing paths exactly as they
were: the schedule's fractional timesteps, the LoRA plumbing and `add_act`.
