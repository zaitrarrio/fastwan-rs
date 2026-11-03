//! Int8 quantization error of every DiT block linear in a checkpoint:
//! `cargo run --release -p fastwan-core --example q8_error -- <transformer dir>`.
//! Reports SNR (signal / quantization-noise power, dB) per matrix kind and
//! the worst single matrix.

use fastwan_core::{quant::Q8, weights::SafetensorsDir, weights::WeightSource};
use std::collections::BTreeMap;

fn main() -> fastwan_core::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: q8_error <transformer dir>");
    let src = SafetensorsDir::open(std::path::Path::new(&dir))?;
    let mut names: Vec<String> = src
        .names()
        .filter(|n| n.starts_with("blocks.") && n.ends_with(".weight"))
        .map(str::to_string)
        .collect();
    names.sort();
    let (mut sig, mut noise) = (BTreeMap::<String, (f64, f64)>::new(), (0.0, 0.0));
    let mut worst = (f64::MAX, String::new());
    let (mut native, mut q8b) = (0usize, 0usize);
    for n in &names {
        let (_, shape, _) = src.raw(n)?;
        if shape.len() != 2 {
            continue; // norms
        }
        let (r, c) = (shape[0], shape[1]);
        let w = src.get(n, &shape)?;
        let q = Q8::quantize(&w, r, c);
        let d = q.dequantize();
        let s: f64 = w.iter().map(|&v| (v as f64).powi(2)).sum();
        let e: f64 = w
            .iter()
            .zip(&d)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let snr = 10.0 * (s / e).log10();
        if snr < worst.0 {
            worst = (snr, n.clone());
        }
        let kind = n.splitn(3, '.').nth(2).unwrap().to_string();
        let k = sig.entry(kind).or_default();
        k.0 += s;
        k.1 += e;
        noise.0 += s;
        noise.1 += e;
        native += 2 * w.len();
        q8b += q.bytes();
    }
    for (k, (s, e)) in &sig {
        println!("{k:40} SNR {:5.1} dB", 10.0 * (s / e).log10());
    }
    println!(
        "all block linears                         SNR {:5.1} dB",
        10.0 * (noise.0 / noise.1).log10()
    );
    println!("worst matrix {} {:.1} dB", worst.1, worst.0);
    println!(
        "bytes: 16-bit {:.2} GB -> int8 {:.2} GB",
        native as f64 / 1e9,
        q8b as f64 / 1e9
    );
    Ok(())
}
