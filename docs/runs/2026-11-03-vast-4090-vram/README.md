# 2026-11-03: VRAM peaks and the Prompt Studio on the real model (RTX 4090)

Vast instance 58993079 (offer 34176463): RTX 4090, Michigan, $0.394/h, about
20 minutes. Binary 3b76542 (before the engine-crash fix in eba7cfb).

## Peak VRAM (nvidia-smi every 100 ms; idle 2 MiB; real checkpoint, seed 7)

| backend | attention | clip | peak VRAM | planner | clip time |
|---|---|---|---|---|---|
| CUDA | VSA | 832×480×81 | **5.0 GB** (5020 MiB) | 4.4 GB | 7.05 s |
| CUDA | VSA | 832×480×33 | **4.1 GB** (4060 MiB) | 4.4 GB | 2.73 s |
| CUDA | VSA | 480×272×33 | **3.6 GB** (3580 MiB) | 3.2 GB | 0.84 s |
| CUDA | dense | 832×480×81 | **5.0 GB** (4988 MiB) | 4.4 GB | 10.47 s |
| WebGPU | VSA | 480×272×33 | **4.4 GB** (4396 MiB) | 3.8 GB | 47.3 s |
| WebGPU | VSA | 832×480×81 | **7.2 GB** (7204 MiB) | 6.0 GB | 473.4 s |

The text encoder runs on the CPU and is not in these numbers. The planner
(`fastwan plan`) runs 0.6–1.2 GB low: it leaves out the CUDA context, cuBLAS
workspace and driver allocations. Guidance: 6 GB on CUDA for 480p/81 frames,
8 GB on WebGPU.

**Open question:** WebGPU 480p/81 VSA measured **152 s per step** here, in a
full 30-layer run, against ≈ 65 s per step scaled from a 4-layer `bench` on
another 4090 host earlier today. Not yet explained (host, clocks, or
something per-layer that a 4-layer run hides).

## Prompt Studio on the real model

`fastwan serve --backend cuda --model-dir /w/fastwan`, driven in headless
Chromium (`scripts/vast/page_test.py`): engine ready in **11.1 s**; first clip,
832×480×81, **19.2 s** click-to-done (prompt 12.2 s with the text encoder's
fp32 weights cold in the page cache, 3 × 1.38 s denoise, 2.6 s decode); peak
VRAM 5.0 GB. Screenshot: `prompt-studio-real.png`. The second clip's numbers
were lost to a race in the test script (it read the previous summary), fixed
in `page_test.py`.
