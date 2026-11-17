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

/// The causal block forward (KV cache, RoPE offset, rolling eviction with a
/// sink) against strobe's PyTorch `causal_forward` on the same tiny model:
/// four one-frame blocks, a 3-frame cache with a 1-frame sink (the fourth
/// block evicts the second), three forwards per block (t = 757, 500, 0) on
/// the same slots. scripts/reference/make_causal_fixture.py.
#[test]
fn causal_blocks_match_reference() {
    use fastwan_core::rope::rope_table;
    use fastwan_model::causal::{CausalConfig, KvCache};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/wan_causal_tiny.safetensors");
    let src = SafetensorsDir::open(&path).unwrap();
    let cfg = WanConfig::tiny();
    let (h, w) = (8, 12);
    let ctx = src.get("input.context", &[8, 16]).unwrap();
    let be = Cpu;
    let dit = WanDit::load(&be, cfg.clone(), &src).unwrap();
    let prompt = dit.prepare_prompt(&be, &ctx);
    let ws = dit.workspace(
        &be,
        (1, h, w),
        usize::MAX,
        Some(fastwan_model::Attention::Dense),
    );
    let ccfg = CausalConfig {
        block_frames: 1,
        cap_frames: 3,
        sink_frames: 1,
        timesteps: vec![],
    };
    let tpf = (h / cfg.patch) * (w / cfg.patch);
    let mut cache = KvCache::new(&be, cfg.layers, cfg.heads, cfg.head_dim, tpf, &ccfg).unwrap();
    let out = be.alloc(4 * h * w);
    let mut worst = 0f32;
    for b in 0..4 {
        cache.begin_block(b, 1).unwrap();
        let rope = rope_table(&cfg, (1, h / cfg.patch, w / cfg.patch), b);
        let (cos, sin) = (be.upload_table(&rope.cos), be.upload_table(&rope.sin));
        for (i, t) in [757.0f32, 500.0, 0.0].into_iter().enumerate() {
            let x = src.get(&format!("input.x.{b}.{i}"), &[4, 1, h, w]).unwrap();
            let want = src
                .get(&format!("expected.v.{b}.{i}"), &[4, 1, h, w])
                .unwrap();
            let mods = dit.step_mods(&be, t);
            let xb = be.upload(&x);
            dit.forward_causal(&be, &ws, &xb, &mods, &prompt, &out, (&cos, &sin), &cache);
            let got = be.download(&out);
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let err = got
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
                / scale;
            eprintln!("causal block {b} forward {i} (t={t}): rel {err:e}");
            worst = worst.max(err);
        }
    }
    assert!(
        worst < 1e-4,
        "causal forward differs from the reference: rel {worst:e}"
    );
}
