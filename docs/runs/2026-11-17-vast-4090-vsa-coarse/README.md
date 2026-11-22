# 2026-11-17: vsa_coarse variants on an RTX 4090 (WebGPU / Vulkan)

Vast instance 61071604 (RTX 4090, Ireland, driver 570.133.07, 450 W), native
wgpu on Vulkan with timestamp queries (`FASTWAN_PROFILE=1`). Each variant
swapped in with `FASTWAN_WGPU_SHADER_vsa_coarse=<file>`; all four pass
`fastwan check --backend wgpu` (57 checks). Full output in `run.log`; the four
kernels are next to it.

`examples/vsa_bench.rs`: one VSA attention call at the 832×480×81 token grid
(32760 tokens, 624 tiles, top 125), 12 heads, d 128, 20 calls, two rounds:

| variant | vsa_coarse ms/call | whole call ms |
|---|---|---|
| atomic (workgroup atomics; wrong tiles on AMD RDNA1 / Metal) | 5.96 | 93.8 |
| tree (32-pass bit search, tree-summed counts) | 5.92 | 93.8 |
| radix (4-bit radix select, parallel picks, one exp per tile) | 9.38 | 96.8 |
| **qshared** (radix + query mean in workgroup memory, 4 accumulators) | **3.25** | **91.7** |

`fastwan bench --attention vsa --layers 30`, 832×480×81, synthetic weights:
step 0 7.54 s (radix) vs 7.39 s (qshared); vsa_coarse 8.35 → 2.89 ms per call,
3.5% → 1.2% of the denoise. The rest of the step: flash3 38%, the FFN and
attention GEMMs (linear4) 58%.

The score loop, not the selection, dominated: every key tile re-read the
query tile's mean from global memory. lavapipe ranked the variants
differently (radix ≈ atomic, qshared 23% faster than radix), so kernel
timing has to happen on a real GPU.
