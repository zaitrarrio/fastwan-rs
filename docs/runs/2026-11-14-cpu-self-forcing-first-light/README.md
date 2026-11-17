# Self-Forcing causal rollout: first light on CPU (2026-11-14)

Weights: `gdhe17/Self-Forcing` `checkpoints/self_forcing_dmd.pt` (Apache-2.0),
converted with `scripts/convert_self_forcing.py` (generator_ema, 825 tensors,
f32 5.68 GB). Prompt: gallery 01 (red fox, snow, birch forest). CPU backend,
4 rayon threads, `generate --causal` (3-frame blocks, cache 12, sink 3,
timesteps 1000 / 937.5 / 833.3 / 625).

- One real-weight forward (block 0, empty cache, 3×12×20 latent, t = 1000)
  against stock diffusers 0.43 `WanTransformer3DModel` loaded from the same
  file: rel 2.9e-4, rms 1.60123 vs 1.60121.
- **320×192, 1 block (9 frames):** coherent fox in a snowy birch forest
  (`fox-320x192-block0.png`). Denoise 941 s, decode 118 s.
- **160×96, 3 blocks (33 frames):** saturated noise from block 0 on, i.e.
  with an empty cache, while the forward matches diffusers: the resolution is
  too far below the 832×480 the model was trained at. Not a rollout bug.
  Denoise ~255 s and decode ~30 s per block.

Not yet measured: multi-block coherence at a real resolution, and any GPU speed.
