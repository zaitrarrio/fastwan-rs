//! 3-axis RoPE tables (Diffusers `WanRotaryPosEmbed`).
//!
//! Each head of width `d` is split `(t, h, w)` (44/42/42 at d = 128) and each
//! axis rotates interleaved pairs `(x[2i], x[2i+1])` by `pos · θ^(−2i/axis)`.
//! Both members of a pair share one angle, so the table stores `d/2` angles
//! per token instead of the reference's repeat-interleaved `d` — half the
//! memory and half the reads.

use crate::config::WanConfig;

/// `cos`/`sin` of shape `[tokens, head_dim / 2]`, tokens in `(t, h, w)` order.
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
}

fn axis_freqs(dim: usize, theta: f64) -> Vec<f64> {
    (0..dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / dim as f64))
        .collect()
}

/// Table for a `(t, h, w)` token grid whose first latent frame is `start_frame`.
pub fn rope_table(cfg: &WanConfig, grid: (usize, usize, usize), start_frame: usize) -> RopeTable {
    let (dt, dh, dw) = cfg.rope_split();
    let (ft, fh, fw) = (
        axis_freqs(dt, cfg.rope_theta),
        axis_freqs(dh, cfg.rope_theta),
        axis_freqs(dw, cfg.rope_theta),
    );
    let half = cfg.head_dim / 2;
    let (gt, gh, gw) = grid;
    let n = gt * gh * gw;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for t in start_frame..start_frame + gt {
        for h in 0..gh {
            for w in 0..gw {
                for (freqs, pos) in [(&ft, t), (&fh, h), (&fw, w)] {
                    for &f in freqs.iter() {
                        // Diffusers computes the outer product in float64 and
                        // casts the angle's cos/sin to float32.
                        let a = pos as f64 * f;
                        cos.push(a.cos() as f32);
                        sin.push(a.sin() as f32);
                    }
                }
            }
        }
    }
    RopeTable { cos, sin, half }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_is_identity_and_shapes_add_up() {
        let cfg = WanConfig::fastwan_1_3b();
        let t = rope_table(&cfg, (2, 3, 4), 0);
        assert_eq!(t.cos.len(), 24 * 64);
        assert!(t.cos[..64].iter().all(|&c| c == 1.0));
        assert!(t.sin[..64].iter().all(|&s| s == 0.0));
        // Token (t=1,0,0): the first temporal pair rotates by 1 rad.
        let row = 12 * 64;
        assert!((t.cos[row] - 1f32.cos()).abs() < 1e-7);
        // ...and the spatial pairs do not.
        assert_eq!(t.cos[row + 22], 1.0);
    }
}
