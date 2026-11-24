# 2026-11-24: FP8 with a per-tensor activation scale (`--weights fp8t`): no gain

Vast instance 62414105 (RTX 4090, Greece, driver 595.99.02, 450 W). Full
output in `run.log`. Follows `docs/runs/2026-11-24-vast-4090-fp8-option`,
where per-token FP8 (`fp8`) was only 6–8% faster than native because two
extra passes over the activations (per-token quantize 189 ms, scale / bias /
GELU epilogue 310 ms per clip) ate most of the halved GEMMs.

Idea: one activation scale per linear input instead of per token, so the
weight's per-row scale and the activation scale combine into one factor per
output channel, which cuBLASLt could take as a device alpha vector
(`CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO`) together with the
bias / GELU epilogue: no separate epilogue pass.

## Result

**cuBLASLt on sm_89 has no FP8 kernel with an alpha vector**: the fused plan
came back empty, the code fell back to the plain FP8 GEMM + `fw_fp8_epilogue`
(the profile still shows `fp8 scale+bias+act` 308 ms and no `gemm fp8t+…`),
and the per-tensor amax is one more pass (147 ms). So `fp8t` is slower than
`fp8`:

| | native | fp8 | fp8t |
|---|---|---|---|
| clip denoise (warm, 832×480×81) | 3.40 s | **3.20 s** | 3.37 s |
| long video, 165 frames | 15.40 s | **14.16 s** | 14.64 s |
| clip PSNR vs native, astronaut / fox | | 21.6 / 27.4 dB | 22.0 / 26.8 dB |
| long PSNR vs native | | 21.9 dB | 20.7 dB |
| peak GPU memory (clip) | 5.05 GB | 4.00 GB | 4.00 GB |

Quality is the same as per-token on these clips. Checks: CUDA 89 / 89 on
both FP8 paths and on the bf16 fallback.

The only full fusion left on this GPU class is a per-tensor *weight* scale
too, so both scales are scalars (`CUBLASLT_MATMUL_DESC_A/B_SCALE_POINTER`)
and the bias / GELU epilogue applies; that trades the per-row weight scale,
which likely costs more quality. Not tried.
