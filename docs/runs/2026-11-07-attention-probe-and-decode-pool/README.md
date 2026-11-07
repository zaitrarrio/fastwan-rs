# 2026-11-07: what limits the attention kernel; a decode buffer pool

## 1. Attention: not bandwidth-bound (RTX PRO 6000, 500 W)

Vast instance 59504156: RTX PRO 6000 Server Edition, New York, driver
580.178.04, **500 W limit** (compare within this run only). A temporary
`FASTWAN_CUDA_ATTN_PROBE` switch made the MMA attention kernels skip their
K/V loads (1) or their MMAs (2); output is meaningless, timing is the point.

| kernel (per layer) | normal | no K/V loads | no MMAs |
|---|---|---|---|
| VSA fine | 8.26 ms | 7.24 ms (−12%) | 5.22 ms (−37%) |
| dense self + cross | 13.3 ms | 10.6 ms (−20%) | 9.9 ms (−25%) |

Dense and VSA both moved K/V at ~4.3 TB/s, which looked like an L2 limit;
it is not: skipping every load saves only 12% (VSA). The MMAs are about a
third; the remaining half is softmax, the four `__syncthreads` per chunk and
latency at 4 warps per block. Next levers are therefore inside the block:
fewer barriers per chunk (warp-specialised producer/consumer, or two chunks
in flight), more warps (for dense, 128-query blocks; VSA's query tile is 64
by construction), and cheaper softmax bookkeeping. Not the memory path.

Tried and reverted: `ldmatrix` for K fragments and `ex2.approx` softmax.
Byte-identical output, 2% slower. The probe switch itself also cost 1.7% of
the denoise on a 4090 (branches in the kernel), so it was removed after the
measurement; reintroduce it as a compile-time variant if needed again.

## 2. Decode: buffer pool (RTX 4090, 350 W)

Vast instance 59508120: RTX 4090, France, driver 615.71.09, 350 W. The CUDA
profiler now times `alloc (zeroed)`. On the PRO 6000 above, the decode made
583 zeroed allocations, 44 ms of its 322 ms GPU time.

`DecodeState` keeps intermediate buffers by length and reuses them in the
next chunk (outputs that are written in full; the first chunk's zero
past-frame still gets a fresh buffer). Stream-ordered execution makes
reuse safe once the last reader is queued.

| | before | after |
|---|---|---|
| allocations per decode | 583 | 40 |
| alloc time (profiled) | 44 ms (PRO 6000) | 3.8 ms (4090) |
| decode, `generate` (4090) | 0.41 s | 0.40 s |

Frames identical to the previous build (PSNR ∞); `check` 50/50 on CUDA,
50/50 on WebGPU (llvmpipe); unit tests cover chunked = one-shot and the
pipelined RGB8 path.

RTX 4090 now (same box, 350 W): **4.09 s per 832×480×81 clip** (denoise
3.57 s at 1.19 s/step, decode 0.41 s), against 6.2–6.7 s before this week's
kernels. The 4090 gained less than the PRO 6000 (1.35 → 1.19 s/step vs
0.88 → 0.54): its bf16 tensor rate with f32 accumulation is half the PRO's,
so more of its step is the MMAs themselves.
