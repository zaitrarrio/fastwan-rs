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
}

impl std::str::FromStr for WeightQuant {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "native" | "bf16" | "f16" => Ok(Self::Native),
            "int8" | "q8" => Ok(Self::Int8),
            _ => Err(format!("unknown weight format `{s}` (native | int8)")),
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
        let group = if cols % Q8_GROUP == 0 { Q8_GROUP } else { cols };
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
    fn odd_rows_use_one_scale_per_row() {
        let q = Q8::quantize(&[0.0; 3 * 100], 3, 100);
        assert_eq!((q.group, q.scales.len()), (100, 3));
        assert!(q.dequantize().iter().all(|&v| v == 0.0));
    }
}
