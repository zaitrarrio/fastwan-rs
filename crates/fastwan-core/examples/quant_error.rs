//! Weight error of the quantized formats on a checkpoint's DiT block
//! linears: `cargo run --release -p fastwan-core --example quant_error --
//! <transformer dir>`. SNR = weight power / error power, over all block
//! matrices, plus the worst single matrix and the stored bytes.

use fastwan_core::{
    quant::{Q4, Q8},
    weights::{SafetensorsDir, WeightSource},
};

type Fmt = (
    String,
    Box<dyn Fn(&[f32], usize, usize) -> (Vec<f32>, usize)>,
);

fn main() -> fastwan_core::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: quant_error <transformer dir>");
    let src = SafetensorsDir::open(std::path::Path::new(&dir))?;
    let mut names: Vec<String> = src
        .names()
        .filter(|n| n.starts_with("blocks.") && n.ends_with(".weight"))
        .map(str::to_string)
        .collect();
    names.sort();
    let mut mats = Vec::new();
    for n in names {
        let (_, shape, _) = src.raw(&n)?;
        if shape.len() == 2 {
            let w = src.get(&n, &shape)?;
            mats.push((n, shape[0], shape[1], w));
        }
    }
    let native: usize = mats.iter().map(|m| 2 * m.3.len()).sum();
    println!(
        "{} block matrices, 16-bit {:.2} GB",
        mats.len(),
        native as f64 / 1e9
    );
    let mut fmts: Vec<Fmt> = vec![(
        "int8 g128".into(),
        Box::new(|w: &[f32], r, c| {
            let q = Q8::quantize(w, r, c);
            (q.dequantize(), q.bytes())
        }),
    )];
    for g in [128, 64, 32] {
        for hqq in [false, true] {
            fmts.push((
                format!("int4 g{g}{}", if hqq { " hqq" } else { "" }),
                Box::new(move |w: &[f32], r, c| {
                    let q = Q4::quantize(w, r, c, g, hqq);
                    (q.dequantize(), q.bytes())
                }),
            ));
        }
    }
    for (name, f) in &fmts {
        let (mut s, mut e, mut bytes) = (0.0f64, 0.0f64, 0usize);
        let mut worst = (f64::MAX, String::new());
        for (n, r, c, w) in &mats {
            let (d, b) = f(w, *r, *c);
            let ms: f64 = w.iter().map(|&v| (v as f64).powi(2)).sum();
            let me: f64 = w
                .iter()
                .zip(&d)
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            let snr = 10.0 * (ms / me).log10();
            if snr < worst.0 {
                worst = (snr, n.clone());
            }
            s += ms;
            e += me;
            bytes += b;
        }
        println!(
            "{name:14} SNR {:5.1} dB  worst {:5.1} dB ({})  {:.2} GB",
            10.0 * (s / e).log10(),
            worst.0,
            worst.1,
            bytes as f64 / 1e9
        );
    }
    Ok(())
}
