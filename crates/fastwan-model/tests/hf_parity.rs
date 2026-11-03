//! The Rust Wan DiT against diffusers' `WanTransformer3DModel` (tiny random
//! weights, scripts/reference/make_fixtures.py): one forward, same inputs.

use fastwan_core::{
    Backend,
    config::WanConfig,
    weights::{SafetensorsDir, WeightSource},
};
use fastwan_cpu::Cpu;
use fastwan_model::WanDit;

#[test]
fn wan_dit_matches_diffusers() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wan_tiny.safetensors");
    let src = SafetensorsDir::open(&path).unwrap();
    let cfg = WanConfig::tiny();
    let (t, h, w) = (3, 8, 12);
    let x = src.get("input.latent", &[4, t, h, w]).unwrap();
    let ctx = src.get("input.context", &[8, 16]).unwrap();
    let ts = src.get("input.timestep", &[1]).unwrap()[0];
    let want = src.get("expected.velocity", &[4, t, h, w]).unwrap();

    let be = Cpu;
    let dit = WanDit::load(&be, cfg.clone(), &src).unwrap();
    let prompt = dit.prepare_prompt(&be, &ctx);
    let mods = dit.step_mods(&be, ts);
    let ws = dit.workspace(&be, (t, h, w), usize::MAX, None);
    let xb = be.upload(&x);
    let out = be.alloc(x.len());
    dit.forward(&be, &ws, &xb, &mods, &prompt, &out);
    let got = be.download(&out);
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale;
    eprintln!("Wan DiT vs diffusers: rel {err:e}");
    assert!(
        err < 1e-4,
        "Wan DiT differs from diffusers: rel {err:e} (scale {scale})"
    );
}
