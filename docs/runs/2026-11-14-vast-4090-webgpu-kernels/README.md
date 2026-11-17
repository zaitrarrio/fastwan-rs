# 2026-11-14: faster WebGPU GEMM and attention (RTX 4090)

Vast instance 60656164: RTX 4090, Utah, driver 620.70.06, 450 W. Native
numbers are wgpu on Vulkan (`FASTWAN_PROFILE=1` timestamp queries, `bench
--layers 4 --weights int4-hqq-g64`, 832×480×81, VSA); browser numbers are
Prompt Studio in headless Chromium on the same GPU, f4 weights, fox, seed 7,
warm run, kernels chosen with `?kernels=LINEAR,FLASH`. `fastwan check
--backend wgpu` passed (54/54) for every combination below.

## The kernels

- **linear4** (GEMM): 128×128 tiles, 8×8 outputs per thread in sixteen named
  vec4s, BK 16, both operands transposed in shared memory as vec4 so each k
  step is four vec4 loads for 64 FMAs (linear3: eight scalar loads for 16).
  Output is bit-identical to linear3 (same k order; the 832×480 clips match
  byte for byte).
- **flash3** (dense + VSA fine attention): four threads per query row for the
  online softmax instead of one (all 256 threads busy), vec4 shared reads for
  Q, K and V, no dynamically indexed vectors.

## Native (wgpu / Vulkan)

| kernel | per call | step, 4 layers |
|---|---|---|
| linear3 FFN up 1536→8960 | 64.6 ms | 356 ms/layer with flash |
| **linear4** FFN up | **36.5 ms** (1.77×) | 267 ms/layer with flash |
| flash (old) | 69.9 ms | |
| flash2 (first try) | 119.2 ms (1.7× **slower**) | 364 ms/layer |
| **flash3** | **46.7 ms** (1.50×) | **222 ms/layer** |
| flash4 (flash3 + old serial softmax) | 45.7 ms | 220 ms/layer |

Real weights, 832×480×81 (`generate --backend wgpu`, warm): 10.7 s/step,
33.6 s clip → **6.67 s/step, 21.5 s clip** with linear4 + flash3.

flash2 was flash3 with small loops writing `v[j]` into vec4s (loads, S
stores, outputs). Removing only that took it from 119 to 47 ms: NVIDIA's
Vulkan compiler moves dynamically indexed vectors to local memory, the same
trap as the array accumulators of 2026-11-03.

## Chrome (Dawn) on the same GPU

| kernels | 832×480 step | 832×480×81 clip | 480×272×81 clip |
|---|---|---|---|
| linear3 + flash (old default) | 10.9 s | 34.5 s | 9.1 s |
| linear4 + flash | 8.0 s | 25.8 s | 6.2 s |
| linear4 + flash2 | 5.87 s | 19.4 s | 5.3 s |
| **linear4 + flash3 (new default)** | **5.75 s** | **18.9 s** | **5.2 s** |
| linear4 + flash4 | 5.94 s | 19.5 s | 5.3 s |

So **1.9× faster in the browser**. Dawn's shader compiler did not suffer from
flash2's indexing the way the native path did (flash2 was already a win in
Chrome), which is why kernel choices have to be measured in Chrome as well as
natively. flash3 is the best in Chrome and within 1.4% of flash4 natively,
so it is the default on both.

`chrome-832x480-old-vs-new.jpg`: frame 40, old kernels (left) vs new (right);
mean difference 0.7 levels (float summation order in attention).

Not changed: flash2 and flash4 were removed after this run (their numbers are
above). `FASTWAN_WGPU_LINEAR=3` / `FASTWAN_WGPU_FLASH=1` (or `?kernels=3,1`
in the page) select the old kernels for A/B.
