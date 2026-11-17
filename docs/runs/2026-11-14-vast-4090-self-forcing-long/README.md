# 2026-11-14: Self-Forcing causal rollout on an RTX 4090 (long videos)

Vast instance 60708595 (RTX 4090, Pennsylvania, driver 580.178.04, 450 W
limit), ~10 min of rental. Weights `self-forcing-1.3b-f16.safetensors`
(`fastwan quantize --weights native` of the converted
`gdhe17/Self-Forcing` `self_forcing_dmd.pt`, Apache-2.0; public at
`weights.fastwan.dev/v1/`). `generate --causal`: 3-latent-frame blocks,
4 steps (1000 / 937.5 / 833.3 / 625), cache 12 latent frames with a 3-frame
sink, dense attention, seed 7, TAEHV streaming decode.

`fastwan check --backend cuda`: all 55 pass, causal case rel 7.3e-3.

| run | backend | size | frames | wall | fps | first frames |
|---|---|---|---|---|---|---|
| fox | CUDA | 832×480 | 321 (20 s) | 24.8 s | 12.9 | 0.75 s |
| night market | CUDA | 832×480 | 321 (20 s) | 24.9 s | 12.9 | 0.75 s |
| astronaut | CUDA | 832×480 | 969 (60 s) | 75.8 s | 12.8 | 0.75 s |
| fox | WebGPU (native) | 480×272 | 81 (5 s) | 13.9 s | 5.8 | 2.27 s |

Per block at 832×480 (CUDA): denoise 0.50 s for block 0 (empty cache),
growing to 0.84 s once the 12-frame cache is full, then flat for the rest of
the video; TAEHV decode 0.10 s per 12 frames. Speed does not depend on video
length: the 60 s video runs at the same 12.8 fps as the 20 s ones. 12 frames
per 0.94 s is 12.8 fps, so real-time playback at 16 fps needs ~25% more.
WebGPU 480×272: ~1.75 s per block.

Quality (strips: one frame every 40, i.e. 2.5 s):
- Fox, 20 s: one continuous shot, the fox keeps walking, background stable.
- Night market, 20 s: the camera moves through the market and the scene
  evolves without cuts.
- Astronaut, 60 s: coherent for the whole minute, same desert, sky and two
  moons. Drift shows: the second astronaut fades out around 15 s and returns
  around 40 s, and a small third moon appears for a while. That is the
  expected failure of a 12-frame window plus sink; LongLive's weights are
  trained for exactly this.

Not measured: Chrome, a phone, and the 3-frame-sink vs no-sink A/B.
