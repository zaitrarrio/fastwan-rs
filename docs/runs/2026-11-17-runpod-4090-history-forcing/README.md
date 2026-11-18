# 2026-11-17: History Forcing vs Self-Forcing, 60 s (RTX 4090)

History Forcing (ziplab/History-Forcing, arXiv 2611.02782; code Apache-2.0,
**weights carry no stated licence**): `dmd/model.pt` from
huggingface.co/xzxxntxdy/History-Forcing (11.35 GB, raw + EMA; sha256
9fd236f48acaa0cc…), EMA weights converted with
`scripts/convert_self_forcing.py` (tensors saved with attributes) and
`quantize --weights native`. Same architecture and recipe as Self-Forcing
(Wan2.1-T2V-1.3B, 4 steps 1000/750/500/250 warped by shift 5, 3-frame
blocks); its generator was trained with `local_attn_size 21, sink_size 3`,
so it runs as `generate --causal --cache-frames 21 --sink-frames 3`. Their
optional MemRoPE demo setting (EMA memory frames, re-roped cache) is not
implemented; its vendored code is CC-BY-NC-SA-4.0.

RunPod pod (Secure, Iceland), RTX 4090, driver 595.112.07, 450 W; CUDA,
832×480, 969 frames (60 s at 16 fps), seed 7. Strips: one frame every 3 s.

| run | cache / sink | time | fps |
|---|---|---|---|
| History Forcing, astronaut | 21 / 3 | 122.8 s | 7.9 |
| History Forcing, fox | 21 / 3 | 122.2 s | 7.9 |
| Self-Forcing, astronaut | 21 / 3 | 122.1 s | 7.9 |
| Self-Forcing, astronaut (Vast 4090, Ireland, driver 570.133.07) | 21 / 3 | 127.5 s | 7.6 |
| Self-Forcing, astronaut and fox (same Vast box) | 12 / 3 | 94.1 s | 10.3 |

A 21-frame cache costs ~30% more time than 12 on the same card.

- **Astronaut** ("an astronaut in a white suit walks across…"): History
  Forcing keeps one astronaut, as prompted, with the same suit and two moons
  all minute, and moves the camera (closer, around the subject); it is more
  saturated ("3D render") and shows glowing vertical light streaks around
  24–30 s and 42–48 s. Self-Forcing adds a second astronaut in every run;
  with a 12-frame cache that one vanishes from ~9 s to ~36 s and a third moon
  appears, with 21 frames it stays all minute (a third moon still flickers in
  around 21–30 s).
- **Fox**: History Forcing walks the fox toward the camera into close-ups,
  same animal throughout, strong backlight that sometimes clips to orange.
  Self-Forcing (12) walks it sideways with several washed-out, ghostly frames.

Verdict: History Forcing is more faithful to the prompt and more dynamic at
equal speed, with artifacts of its own (light streaks, saturation). Not
offered on the site while its weights have no licence.
