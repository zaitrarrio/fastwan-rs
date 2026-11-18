# 2026-11-18: History Forcing branch merged with main (#4, #5): GPU re-check on an RTX 4090

Vast instance 61318610 (RTX 4090, Indiana, driver 595.99.02, 450 W limit),
~25 min. Binaries: the merge acf28f4 ("merged"), the same plus the check fix
731cd52 ("fix"), main 387d81e ("main") and the History Forcing branch before
the merge (e72ff08 code, "PR"). Weights as in
`docs/runs/2026-11-17-vast-4090-history-forcing` (same driver; sha256 of the
converted History Forcing file identical across conversions is not needed:
every comparison below runs on one file).

## Conformance

| binary | CUDA (sm_89) | WebGPU (Vulkan, NVIDIA 595.99) |
|---|---|---|
| fix | **70 / 70**, three runs identical | |
| merged | 69 / 70: `vsa (5, 6, 9) h2 d128` 0.293 | **70 / 70** (`rope_heads` 7e-8, `kv_ema` 3e-8, MemRoPE rollouts 1.7e-4 / 1.8e-4, f16-cache rollout 1.8e-4) |
| main | 63 / 66: `(5, 6, 9)` 0.293 and both model-grid VSA cases 0.28 | |
| PR | 60 / 60 | |

The failing cases are the check's, not a kernel's: CUDA rounds uploads to
bf16, and when random q / k put the pooled tile scores on a top-k near-tie,
the rounding alone flips a pick. On the CPU, feeding the same draw rounded to
bf16 moves `(5, 6, 9)` by exactly 0.293 from its f32 self. main's new cases
changed the RNG draw `(5, 6, 9)` gets; main's model-grid cases fail for the
same reason (the PR fixed those in e72ff08). 731cd52 gives the small VSA
cases bf16-representable inputs too. All of the PR's cases pass on the merged
binary: `rope_heads` 3.8e-3, `kv_ema` 2.0e-3, MemRoPE rollout d12 / d32
4.3e-3 / 5.7e-3, causal d32 f16 cache 3.6e-3.

## Clips (`generate --causal`, 832×480, seed 0, TAEHV unless noted)

| comparison | result |
|---|---|
| MemRoPE, merged vs PR, 321 frames | **byte-identical** |
| MemRoPE, merged vs a rerun | byte-identical |
| LightTAE, merged (denormalise as a device conv) vs PR (host f32) | 57.9 dB (Y 56.6) |
| TAEHV vs LightTAE on the same latents, merged / PR | 35.40 / 35.40 dB (2026-11-17: 35.4) |
| MemRoPE WebGPU vs CUDA, 480×272×81 | 19.4 dB (2026-11-17: 19.4) |
| MemRoPE 63 s musician (1005 frames) | 96.0 s, 10.5 fps, first frames 0.88 s (2026-11-17: 96.1 s); stable for the whole minute (`hf-mem-musician-63s-strip.jpg`) |
| Self-Forcing window, merged, 321 frames | 29.5 s, 10.9 fps, coherent (`sf-win-merged-strip.jpg`) |

Not measured: the window path against main's binary. main's `generate` has no
`--memory` flag, so both of those runs stopped at argument parsing (my
command's mistake). The window-cache conformance cases pass on both
backends, and the MemRoPE clip, which runs through the same merged cache and
loop, is byte-identical to the PR's.

Per-block accounting changed with main's pipelined loop: merged MemRoPE
blocks print denoise 0.96 s + decode 0.19 s against the PR's 1.06 + 0.10 s,
for the same wall time (30.3 vs 30.6 s for 321 frames). The decode of block
i now overlaps the denoise of block i + 1, so the split moved, not the cost.
WebGPU 480×272×81: 14.7 s (2026-11-17: 14.8 s).
