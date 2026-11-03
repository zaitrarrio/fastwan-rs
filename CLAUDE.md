# CLAUDE.md: fastwan-rs

State and invariants only; the reasoning is in `docs/design.md`.

- One op vocabulary (`fastwan_core::Backend`), three implementations. A new op
  means: CPU reference, WGSL kernel, CUDA kernel, and a case in
  `fastwan-model/src/check.rs`, all in the same change.
- The CPU backend is ground truth. GPU code is trusted only after
  `fastwan check --backend <b>` passes on that hardware. CUDA has never run
  on a GPU; WebGPU has only run on software rasterizers.
- WGSL kernels and their CUDA twins share structure (`fw_<op>` in
  `kernels.cu`). Change both. `fw_` because `norm` collides with a CUDA builtin.
- Shader change → `python3 web/build.py` (the test page inlines the shaders;
  `web_page_embeds_current_shaders` fails otherwise).
- WGSL auto layouts drop unused bindings, so every declared binding must be
  statically used (see the guard lines at the end of `conv.wgsl` and
  `elementwise.wgsl`).
- Weight keys are the Diffusers / taehv positional names, checked against
  `weights::wan_manifest` / `tae_manifest`. A missing key is an error, never a
  default.
- No speed claim without a measurement on named hardware.
- The Wan DiT and UMT5 are tested against transformers / diffusers fixtures
  (`scripts/reference/make_fixtures.py`, tiny random weights, checked in). A
  change to either model must keep `hf_parity` green; regenerate fixtures
  only when the reference library changes, and say which version.
- `fastwan serve` binds 127.0.0.1 and sends no CORS headers on purpose (a
  page on another origin must not be able to queue GPU jobs). `web/demo.html`
  is compiled into the binary (`include_str!`); rebuild after editing it.
- Tests: `cargo test --workspace` (needs a Vulkan driver for the WebGPU suite:
  `mesa-vulkan-drivers`); NVRTC kernel compile needs libnvrtc on
  `LD_LIBRARY_PATH` (`pip install nvidia-cuda-nvrtc-cu12`).
