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

/// LongLive-Plug's few-step LoRA for Wan2.1-T2V-14B
/// (`Efficient-Large-Model/LongLive-Plug-Wan2.1-T2V-14B-few-step`,
/// `inference_config.json`: LightX2V's `WanStepDistillScheduler`):
/// `denoising_step_list` and `sample_shift`.
pub const LONGLIVE_PLUG_WAN_STEPS: [u32; 4] = [1000, 750, 500, 250];
pub const LONGLIVE_PLUG_WAN_SHIFT: f64 = 5.0;

#[derive(Debug, Clone)]
pub struct DmdSchedule {
    /// Timesteps fed to the DiT, in order (fractional for shifted ladders).
    pub timesteps: Vec<f32>,
    /// Table sigma for each step, then a trailing `0.0`.
    pub sigmas: Vec<f64>,
    /// DMD (FastWan): predict `x0`, then re-noise it to `σ_next` with fresh
    /// noise. Without it (LightX2V's step-distill scheduler) the step is
    /// Euler on the flow, `x ← x − (σ − σ_next)·v`, and the last step
    /// returns `x0 = x − σ·v`.
    pub renoise: bool,
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
            renoise: true,
            timesteps: steps.iter().map(|&t| t as f32).collect(),
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

    /// LightX2V's step-distill ladder: `σ = shift·s / (1 + (shift − 1)·s)` at
    /// `s = step / 1000` (its `linspace(1, 0, 1001)` table, index
    /// `1000 − step`), timesteps `σ·1000`, Euler steps between them.
    pub fn euler_shifted(steps: &[u32], shift: f64) -> Self {
        let sigmas: Vec<f64> = steps
            .iter()
            .map(|&t| {
                let s = f64::from(t) / 1000.0;
                shift * s / (1.0 + (shift - 1.0) * s)
            })
            .collect();
        Self {
            timesteps: sigmas.iter().map(|&s| (s * 1000.0) as f32).collect(),
            sigmas: sigmas.into_iter().chain(std::iter::once(0.0)).collect(),
            renoise: false,
        }
    }

    /// LongLive-Plug's few-step LoRA on Wan2.1-T2V-14B: four forwards,
    /// σ = 1, 0.9375, 0.8333, 0.625.
    pub fn longlive_plug_wan() -> Self {
        Self::euler_shifted(&LONGLIVE_PLUG_WAN_STEPS, LONGLIVE_PLUG_WAN_SHIFT)
    }

    /// The `(sigma_t, sigma_next, noise?)` of [`crate::Backend::dmd_step`] for
    /// step `i`: DMD re-noises to `σ_next`; Euler moves `x` by `σ − σ_next`
    /// (by `σ` on the last step, which is `x0`) with no noise.
    pub fn step_args(&self, i: usize) -> (f32, f32, bool) {
        let (st, next) = self.coeffs(i);
        if self.renoise {
            (st, next.unwrap_or(0.0), next.is_some())
        } else {
            (st - next.unwrap_or(0.0), 0.0, false)
        }
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

    #[test]
    fn longlive_plug_wan_ladder() {
        let s = DmdSchedule::longlive_plug_wan();
        let want = [1.0, 0.9375, 0.833_333_3, 0.625, 0.0];
        for (a, b) in s.sigmas.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{:?}", s.sigmas);
        }
        assert_eq!(s.timesteps[..2], [1000.0, 937.5]);
        assert!((s.timesteps[2] - 833.333).abs() < 1e-3 && s.timesteps[3] == 625.0);
        // Euler: x moves by σ − σ_next, no noise; the last step is x0.
        let (c, sn, noise) = s.step_args(1);
        assert!((c - (0.9375 - 0.833_333_3)).abs() < 1e-6 && sn == 0.0 && !noise);
        assert_eq!(s.step_args(3), (0.625, 0.0, false));
        // DMD keeps its re-noise.
        assert_eq!(DmdSchedule::fastwan_1_3b().step_args(0).2, true);
    }
}
