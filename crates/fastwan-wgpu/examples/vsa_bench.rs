//! Time one VSA attention call at the model's token grid (832×480×81 after the
//! 2×2 patch: 32760 tokens, 624 tiles, 12 heads, d 128) and, with
//! `FASTWAN_PROFILE=1` on an adapter with timestamp queries, each kernel:
//! `cargo run --release -p fastwan-wgpu --example vsa_bench [runs] [heads]`
//! (lavapipe binds at most 128 MiB: use 6 heads there).
use fastwan_core::{Backend, rng::Rng, vsa::VsaSpec};
use fastwan_wgpu::Wgpu;

fn main() {
    let runs: usize = std::env::args()
        .nth(1)
        .and_then(|r| r.parse().ok())
        .unwrap_or(5);
    let be = Wgpu::new().expect("WebGPU adapter");
    println!("{}", be.name());
    let heads: usize = std::env::args()
        .nth(2)
        .and_then(|h| h.parse().ok())
        .unwrap_or(12);
    let spec = VsaSpec::new((21, 30, 52), heads, 128, 0.8);
    let n = spec.heads * spec.plan.seq * spec.head_dim;
    let mut rng = Rng::new(1);
    let (q, k, v, g) = (
        be.upload(&rng.normal(n)),
        be.upload(&rng.normal(n)),
        be.upload(&rng.normal(n)),
        be.upload(&rng.normal(n)),
    );
    let out = be.alloc(n);
    let st = be.vsa_prepare(&spec);
    be.vsa_attention(&q, &k, &v, &g, &st, &out);
    be.sync();
    be.profile_reset();
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        be.vsa_attention(&q, &k, &v, &g, &st, &out);
    }
    be.sync();
    println!(
        "vsa_attention {} tiles, top {}: {:.2} ms per call",
        spec.plan.num_tiles(),
        spec.topk,
        t0.elapsed().as_secs_f64() * 1e3 / runs as f64
    );
    if let Some(r) = be.profile_report() {
        print!("{r}");
    }
}
