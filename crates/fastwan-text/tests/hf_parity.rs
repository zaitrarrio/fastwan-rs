//! The Rust UMT5 encoder against transformers' `UMT5EncoderModel` (tiny random
//! weights, scripts/reference/make_fixtures.py). The reference ran on the
//! padded, masked prompt as WanPipeline does; we encode only the real tokens.

use fastwan_core::weights::{SafetensorsDir, WeightSource};
use fastwan_text::{Umt5, Umt5Config};

#[test]
fn umt5_matches_transformers() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/umt5_tiny.safetensors");
    let src = SafetensorsDir::open(&path).unwrap();
    let ids: Vec<u32> = src
        .get("input.ids", &[13])
        .unwrap()
        .iter()
        .map(|&v| v as u32)
        .collect();
    let want = src.get("expected.hidden", &[13, 32]).unwrap();
    let cfg = Umt5Config {
        vocab: 64,
        d_model: 32,
        d_kv: 8,
        heads: 4,
        d_ff: 48,
        layers: 2,
        buckets: 32,
        max_distance: 128,
        eps: 1e-6,
    };
    let got = Umt5::new(SafetensorsDir::open(&path).unwrap(), cfg)
        .unwrap()
        .encode(&ids)
        .unwrap();
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale;
    eprintln!("UMT5 vs transformers: rel {err:e}");
    assert!(err < 1e-5, "UMT5 differs from transformers: rel {err:e}");
}
