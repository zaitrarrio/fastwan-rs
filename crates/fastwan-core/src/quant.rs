//! Weight-only int8 quantization of the DiT linears.
//!
//! Symmetric round-to-nearest, one f32 scale per group of [`Q8_GROUP`]
//! consecutive input columns of a row (per row when the row length is not a
//! multiple of it): `w ≈ q · s`, `q ∈ [-127, 127]`, `s = max|w| / 127`.
//! Activations stay in the backend's precision; every backend computes
//! `Σ x · (q · s)` with `q · s` formed in f32, so [`Q8::dequantize`] is the
//! exact reference the GPU kernels are checked against.

/// Columns sharing one scale.
pub const Q8_GROUP: usize = 128;

/// How the DiT's transformer-block weights are stored on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightQuant {
    /// The backend's own precision (bf16 on CUDA, f16 pairs on WebGPU).
    #[default]
    Native,
    /// [`Q8`]: half the bytes of bf16 / f16.
    Int8,
    /// 4-bit [`Q4`] with `group` columns per scale/zero, rounded to nearest
    /// or HQQ-refined. **Evaluation only:** the weights are quantized and
    /// expanded back, then stored at native precision, so this measures the
    /// output of 4-bit weights without saving memory (no int4 kernels yet).
    /// With `ffn_only`, the attention and gate linears use [`Q8`] (real int8)
    /// and only the FFN (≈63% of the block weights) is 4-bit.
    Int4Sim {
        group: usize,
        hqq: bool,
        ffn_only: bool,
    },
}

impl WeightQuant {
    /// The format of the non-FFN block linears.
    pub fn attn(self) -> Self {
        match self {
            Self::Int4Sim { ffn_only: true, .. } => Self::Int8,
            q => q,
        }
    }
}

impl std::str::FromStr for WeightQuant {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "native" | "bf16" | "f16" => Ok(Self::Native),
            "int8" | "q8" => Ok(Self::Int8),
            _ => {
                // int4-sim[-hqq][-g<N>][-ffn]
                let (s, ffn_only) = match s.strip_suffix("-ffn") {
                    Some(r) => (r, true),
                    None => (s, false),
                };
                let rest = s.strip_prefix("int4-sim").ok_or_else(|| {
                    format!(
                        "unknown weight format `{s}` (native | int8 | int4-sim[-hqq][-gN][-ffn])"
                    )
                })?;
                let (hqq, rest) = match rest.strip_prefix("-hqq") {
                    Some(r) => (true, r),
                    None => (false, rest),
                };
                let group = match rest.strip_prefix("-g") {
                    Some(g) => g.parse().map_err(|_| format!("bad group in `{s}`"))?,
                    None if rest.is_empty() => 64,
                    None => return Err(format!("bad weight format `{s}`")),
                };
                Ok(Self::Int4Sim {
                    group,
                    hqq,
                    ffn_only,
                })
            }
        }
    }
}

/// A `[rows, cols]` matrix as int8 codes and per-group scales.
#[derive(Debug, Clone)]
pub struct Q8 {
    pub q: Vec<i8>,
    /// `[rows, cols / group]`.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
}

impl Q8 {
    pub fn quantize(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let group = if cols.is_multiple_of(Q8_GROUP) {
            Q8_GROUP
        } else {
            cols
        };
        let mut q = vec![0i8; data.len()];
        let mut scales = Vec::with_capacity(data.len() / group);
        for (src, dst) in data.chunks(group).zip(q.chunks_mut(group)) {
            let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            for (d, v) in dst.iter_mut().zip(src) {
                *d = (v / s).round().clamp(-127.0, 127.0) as i8;
            }
            scales.push(s);
        }
        Self {
            q,
            scales,
            rows,
            cols,
            group,
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, &q)| q as f32 * self.scales[i / self.group])
            .collect()
    }

    /// Device bytes (codes + scales).
    pub fn bytes(&self) -> usize {
        self.q.len() + 4 * self.scales.len()
    }
}

/// A `[rows, cols]` matrix as unsigned 4-bit codes with an affine
/// scale / zero per group of columns: `w ≈ (q − z) · s`, `q ∈ [0, 15]`.
#[derive(Debug, Clone)]
pub struct Q4 {
    /// One code per byte (packing is a storage concern; see [`Q4::bytes`]).
    pub q: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
}

impl Q4 {
    /// Min/max round-to-nearest; with `hqq`, then refine each group's zero
    /// point with HQQ's half-quadratic proximal solver (Badri & Shaji 2023,
    /// `optimize_weights_proximal`: lp-norm 0.7, β 10, κ 1.01, ≤ 20 iters),
    /// which fits the zero to a sparse-error model instead of the extremes.
    pub fn quantize(data: &[f32], rows: usize, cols: usize, group: usize, hqq: bool) -> Self {
        assert_eq!(data.len(), rows * cols);
        let group = if cols.is_multiple_of(group) {
            group
        } else {
            cols
        };
        let n = data.len() / group;
        let mut q = vec![0u8; data.len()];
        let mut sz = vec![(0.0f32, 0.0f32); n];
        // Groups are independent: split them across threads (HQQ is ~50 flops
        // per weight per iteration; single-threaded it adds minutes to a load).
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
        let per = n.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for ((w, qc), szc) in data
                .chunks(per * group)
                .zip(q.chunks_mut(per * group))
                .zip(sz.chunks_mut(per))
            {
                sc.spawn(move || {
                    for ((w, qg), sz) in w.chunks(group).zip(qc.chunks_mut(group)).zip(szc) {
                        let (lo, hi) = w
                            .iter()
                            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                        // HQQ's convention: inverse scale `is`, codes = round(w·is + z).
                        let is = if hi > lo { 15.0 / (hi - lo) } else { 1.0 };
                        let mut z = -lo * is;
                        let code = |v: f32, z: f32| (v * is + z).round().clamp(0.0, 15.0);
                        if hqq {
                            let (p, mut beta, kappa) = (0.7f32, 10.0f32, 1.01f32);
                            let (mut best, mut best_z) = (f32::MAX, z);
                            for _ in 0..20 {
                                let mut err = 0.0;
                                let mut zsum = 0.0;
                                for &v in w {
                                    let c = code(v, z);
                                    let r = (c - z) / is;
                                    let e = v - r;
                                    err += e.abs();
                                    // shrink: sign(e)·relu(|e| − |e|^(p−1) / β)
                                    let a = e.abs();
                                    let sh = if a > 0.0 {
                                        e.signum() * (a - a.powf(p - 1.0) / beta).max(0.0)
                                    } else {
                                        0.0
                                    };
                                    zsum += c - (v - sh) * is;
                                }
                                let err = err / w.len() as f32;
                                if err >= best {
                                    break;
                                }
                                (best, best_z) = (err, z);
                                z = zsum / w.len() as f32;
                                beta *= kappa;
                            }
                            z = best_z;
                        }
                        for (d, &v) in qg.iter_mut().zip(w) {
                            *d = code(v, z) as u8;
                        }
                        *sz = (1.0 / is, z);
                    }
                });
            }
        });
        let (scales, zeros) = sz.into_iter().unzip();
        Self {
            q,
            scales,
            zeros,
            rows,
            cols,
            group,
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let g = i / self.group;
                (c as f32 - self.zeros[g]) * self.scales[g]
            })
            .collect()
    }

    /// Bytes once packed: two codes per byte, f16 scale and zero per group.
    pub fn bytes(&self) -> usize {
        self.q.len().div_ceil(2) + 4 * self.scales.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_error_is_half_a_step() {
        let (rows, cols) = (8, 256);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * (1 + i / cols) as f32)
            .collect();
        let q = Q8::quantize(&w, rows, cols);
        assert_eq!(q.group, 128);
        assert_eq!(q.scales.len(), rows * 2);
        let d = q.dequantize();
        for (i, (a, b)) in w.iter().zip(&d).enumerate() {
            assert!((a - b).abs() <= q.scales[i / 128] * 0.5 + 1e-6);
        }
    }

    #[test]
    fn int4_round_trip_and_hqq_does_not_hurt() {
        let (rows, cols) = (16, 256);
        // Heavy-tailed rows: mostly small values, a few outliers.
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let u = (i * 7919 % 1009) as f32 / 1009.0 - 0.5;
                if i % 97 == 0 { u * 8.0 } else { u * 0.3 }
            })
            .collect();
        let err = |q: &Q4| -> f32 {
            w.iter()
                .zip(q.dequantize())
                .map(|(a, b)| (a - b).powi(2))
                .sum()
        };
        let rtn = Q4::quantize(&w, rows, cols, 64, false);
        assert!(rtn.q.iter().all(|&c| c <= 15));
        for (i, (a, b)) in w.iter().zip(rtn.dequantize()).enumerate() {
            assert!((a - b).abs() <= rtn.scales[i / 64] * 0.5 + 1e-5);
        }
        let hqq = Q4::quantize(&w, rows, cols, 64, true);
        assert!(hqq.q.iter().all(|&c| c <= 15));
        assert!(
            err(&hqq) <= err(&rtn) * 1.05,
            "{} vs {}",
            err(&hqq),
            err(&rtn)
        );
        assert_eq!(
            "int4-sim-hqq-g32".parse(),
            Ok(WeightQuant::Int4Sim {
                group: 32,
                hqq: true,
                ffn_only: false
            })
        );
        assert_eq!(
            "int4-sim".parse(),
            Ok(WeightQuant::Int4Sim {
                group: 64,
                hqq: false,
                ffn_only: false
            })
        );
        assert_eq!(
            "int4-sim-hqq-g64-ffn"
                .parse::<WeightQuant>()
                .map(|q| q.attn()),
            Ok(WeightQuant::Int8)
        );
    }

    #[test]
    fn odd_rows_use_one_scale_per_row() {
        let q = Q8::quantize(&[0.0; 3 * 100], 3, 100);
        assert_eq!((q.group, q.scales.len()), (100, 3));
        assert!(q.dequantize().iter().all(|&v| v == 0.0));
    }
}
