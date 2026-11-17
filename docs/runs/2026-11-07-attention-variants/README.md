# 2026-11-07: attention kernel variants (RTX PRO 6000, 600 W)

Vast instance 59516325: RTX PRO 6000 Blackwell Server Edition, Utah, driver
580.178.04, 600 W. `FASTWAN_CUDA_ATTN_VARIANT` picks the MMA attention
kernels on the host (no branch in the kernel). Raw output: `output.txt`.

Registers / shared memory per block from ptxas (`cargo run -p fastwan-cuda
--example ptx -- compute_120`, then `ptxas -v`):

| variant | change | VSA fine, sm_120 | blocks per SM |
|---|---|---|---|
| v0 | the kernel as of 18f27db | 236 regs, 34.9 KB | 2 |
| v1 | 2 barriers per key chunk (was 4); rescale only when a row max moved | 206 regs, 34.9 KB | 2 |
| v2 | v1 + XOR-swizzled unpadded tiles + `__launch_bounds__(128, 3)` | 168 regs (40 B spill), 32.9 KB | 3 |

## Results (bench, 30 layers, 832×480×81; rounds interleaved)

| | v0 | v1 | v2 |
|---|---|---|---|
| VSA denoise, round 1 / 2 | 1.615 / 1.610 s | 1.630 / 1.628 s | **1.585 / 1.582 s** |
| VSA fine per layer (profiled) | 7.12 ms | 7.37 ms | **6.89 ms** |
| dense denoise | 2.94 s | 2.99 s | **2.87 s** |
| real clip total (2nd run) | 2.40 s | 2.32 s | **2.26 s** |

All three pass `check` 50/50. v1 and v2 give the same clip; against v0 it
is 40.2 dB (the rescale moved, so float rounding differs), inside the
33.5 dB bf16-vs-f32 backend gap.

Reading: halving the barriers alone did not help (v1 is slower); the gain
is occupancy (12 warps per SM instead of 8), and it is small: 3% on the VSA
kernel, 2% on the step. v2 is the default on sm_120+ (checked here) and v0
elsewhere until another architecture is checked. The kernel is still far
from its MMA bound; the next structural step would be splitting each
64-query tile across 8 warps (two halves of the key chunk, merged at the
end), which keeps VSA's tile shape and doubles the warps per block.
