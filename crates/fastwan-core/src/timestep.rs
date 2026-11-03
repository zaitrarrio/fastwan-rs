//! Timestep conditioning, computed on the host.
//!
//! A DMD run sees three timestep values, so the whole conditioning path —
//! sinusoid → `time_embedder` → `time_proj` → per-block `+ scale_shift_table`
//! — is a handful of mat-vecs per step. It runs once per step on the CPU and
//! the GPU only ever sees the finished `[6, dim]` modulation tables.

/// Diffusers `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`:
/// `[cos(t·f_i), sin(t·f_i)]`, `f_i = exp(−ln(10⁴)·i / (dim/2))`.
pub fn sinusoid(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0; dim];
    for i in 0..half {
        let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        out[i] = (t * f).cos();
        out[half + i] = (t * f).sin();
    }
    out
}

/// `y = W x + b` for a row-major `[out, in]` matrix.
pub fn matvec(w: &[f32], b: Option<&[f32]>, x: &[f32]) -> Vec<f32> {
    let n_in = x.len();
    let n_out = w.len() / n_in;
    (0..n_out)
        .map(|o| {
            let row = &w[o * n_in..(o + 1) * n_in];
            let acc: f64 = row.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum();
            acc as f32 + b.map_or(0.0, |b| b[o])
        })
        .collect()
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}
