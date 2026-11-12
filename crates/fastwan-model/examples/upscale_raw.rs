//! Upscale raw interleaved RGB frames on the CPU backend (the reference for
//! `scripts/upscale/parity.py`):
//! `upscale_raw WEIGHTS.safetensors IN.rgb FRAMES H W OUT.rgb`.
use fastwan_core::weights::SafetensorsDir;
use fastwan_cpu::Cpu;
use fastwan_model::upscale::{UpConfig, Upscaler, rgb8};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let [_, w, input, frames, h, wd, out] = &a[..] else {
        panic!("usage: upscale_raw WEIGHTS IN.rgb FRAMES H W OUT.rgb")
    };
    let (frames, h, w_): (usize, usize, usize) = (
        frames.parse().unwrap(),
        h.parse().unwrap(),
        wd.parse().unwrap(),
    );
    let src = SafetensorsDir::open(Path::new(w)).unwrap();
    let up = Upscaler::load(&Cpu, UpConfig::anime_video_v3(), &src).unwrap();
    let rgb = std::fs::read(input).unwrap();
    let t = std::time::Instant::now();
    let c = up.launch(&Cpu, &rgb, frames, h, w_);
    use fastwan_core::backend::Backend;
    let raw = Cpu.download(&c.out);
    let px = rgb8(&raw, &rgb, frames, h, w_, c.scale);
    eprintln!(
        "{frames}x{h}x{w_} -> x{} in {:.2}s",
        c.scale,
        t.elapsed().as_secs_f64()
    );
    std::fs::write(out, px).unwrap();
}
