//! DMD sampling schedule (FastVideo `stages/dmd.py`, fastvideo-rs
//! `schedulers/dmd.rs`).
//!
//! FastWan 2.1 1.3B denoises at `dmd_denoising_steps = [1000, 757, 522]`. Each
//! step maps onto the nearest entry of the *training* noise table — a
//! `FlowMatchEulerDiscreteScheduler(shift = 8)` without `set_timesteps` — and:
//!
//! ```text
//! x0 = x − σ_t · v
//! x  = (1 − σ_next) · x0 + σ_next · ε      (fresh ε; skipped on the last step)
//! ```

pub const FASTWAN_1_3B_STEPS: [i32; 3] = [1000, 757, 522];
/// FastVideo `DMD_TRAINING_NOISE_SHIFT`.
pub const DMD_TRAINING_NOISE_SHIFT: f64 = 8.0;

#[derive(Debug, Clone)]
pub struct DmdSchedule {
    /// Timesteps fed to the DiT, in order.
    pub timesteps: Vec<i32>,
    /// Table sigma for each step, then a trailing `0.0`.
    pub sigmas: Vec<f64>,
}

/// Float32 sigma / timestep training tables, index 0 = timestep N.
fn training_table(n: usize, shift: f64) -> (Vec<f32>, Vec<f32>) {
    let (nf, shift_f, shift_m1) = (n as f32, shift as f32, (shift - 1.0) as f32);
    (0..n)
        .map(|i| {
            let s = (n - i) as f32 / nf;
            let s = (shift_f * s) / (1.0 + shift_m1 * s);
            (s, s * nf)
        })
        .unzip()
}

impl DmdSchedule {
    pub fn new(steps: &[i32]) -> Self {
        let (sig, ts) = training_table(1000, DMD_TRAINING_NOISE_SHIFT);
        let sigma_for = |t: f64| {
            let mut best = (0usize, f64::INFINITY);
            for (i, &v) in ts.iter().enumerate() {
                let d = (f64::from(v) - t).abs();
                if d < best.1 {
                    best = (i, d);
                }
            }
            f64::from(sig[best.0])
        };
        Self {
            timesteps: steps.to_vec(),
            sigmas: steps
                .iter()
                .map(|&t| sigma_for(f64::from(t)))
                .chain(std::iter::once(0.0))
                .collect(),
        }
    }

    pub fn fastwan_1_3b() -> Self {
        Self::new(&FASTWAN_1_3B_STEPS)
    }

    pub fn len(&self) -> usize {
        self.timesteps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timesteps.is_empty()
    }

    /// `(σ_t, σ_next)`; `σ_next` is `None` on the last step (no re-noise).
    pub fn coeffs(&self, i: usize) -> (f32, Option<f32>) {
        let next = (i + 1 < self.len()).then(|| self.sigmas[i + 1] as f32);
        (self.sigmas[i] as f32, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastwan_sigmas() {
        let s = DmdSchedule::fastwan_1_3b();
        assert_eq!(s.sigmas[0], 1.0);
        // Exact-hit timesteps map onto themselves: t/1000.
        assert!((s.sigmas[1] - 0.757).abs() < 1e-3, "{}", s.sigmas[1]);
        assert!((s.sigmas[2] - 0.522).abs() < 1e-3, "{}", s.sigmas[2]);
        assert_eq!(s.coeffs(2).1, None);
    }
}
