//! Quantization of the DiT block linears: weight-only int8 ([`Q8`]) and
//! 4-bit ([`Q4`]), FP8 E4M3 weights *and* activations ([`F8`], W8A8, for
//! CUDA's FP8 tensor cores), and which linears get which
//! ([`WeightQuant::for_key`]).
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
    /// or HQQ-refined. With `ffn_only`, the attention and gate linears use
    /// [`Q8`] and only the FFN (≈59% of the block weights) is 4-bit.
    Int4 {
        group: usize,
        hqq: bool,
        ffn_only: bool,
    },
    /// [`F8`]: E4M3 weights with one scale per output row, and each input
    /// row (token) quantized to E4M3 with its own scale at every linear
    /// ([`fq8_rows`]). On CUDA (sm_89+) the GEMM runs on FP8 tensor cores;
    /// elsewhere the same arithmetic in f32 / bf16. Load-time only: not a
    /// stored file format.
    Fp8,
    /// [`Self::Fp8`] with one activation scale for the whole input of each
    /// linear ([`fq8_tensor`]) instead of one per token: both scales then
    /// fold into one factor per output channel, which cuBLASLt applies with
    /// the bias and GELU inside the GEMM (no separate epilogue pass).
    Fp8Tensor,
}

impl WeightQuant {
    /// The format of the non-FFN block linears.
    pub fn attn(self) -> Self {
        match self {
            Self::Int4 { ffn_only: true, .. } => Self::Int8,
            q => q,
        }
    }

    /// The format of the block linear `key` (`blocks.N.<role>`, no
    /// `.weight`): the FFN gets `self`, attention and gate get [`Self::attn`],
    /// and everything outside the blocks is `Native`. The exporter and the
    /// loader both decide with this, so a stored file loads as it was made.
    pub fn for_key(self, key: &str) -> Self {
        let Some(role) = key
            .strip_prefix("blocks.")
            .and_then(|r| r.split_once('.'))
            .map(|(_, r)| r)
        else {
            return Self::Native;
        };
        match role {
            "ffn.net.0.proj" | "ffn.net.2" => self,
            "attn1.to_q" | "attn1.to_k" | "attn1.to_v" | "attn1.to_out.0" | "attn2.to_q"
            | "attn2.to_k" | "attn2.to_v" | "attn2.to_out.0" | "to_gate_compress" => self.attn(),
            _ => Self::Native,
        }
    }
}

impl std::fmt::Display for WeightQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => f.write_str("native"),
            Self::Int8 => f.write_str("int8"),
            Self::Fp8 => f.write_str("fp8"),
            Self::Fp8Tensor => f.write_str("fp8t"),
            Self::Int4 {
                group,
                hqq,
                ffn_only,
            } => write!(
                f,
                "int4{}-g{group}{}",
                if *hqq { "-hqq" } else { "" },
                if *ffn_only { "-ffn" } else { "" }
            ),
        }
    }
}

impl std::str::FromStr for WeightQuant {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "native" | "bf16" | "f16" => Ok(Self::Native),
            "int8" | "q8" => Ok(Self::Int8),
            "fp8" | "e4m3" => Ok(Self::Fp8),
            "fp8t" | "fp8-tensor" => Ok(Self::Fp8Tensor),
            _ => {
                // int4[-hqq][-g<N>][-ffn]
                let (s, ffn_only) = match s.strip_suffix("-ffn") {
                    Some(r) => (r, true),
                    None => (s, false),
                };
                let rest = s.strip_prefix("int4").ok_or_else(|| {
                    format!(
                        "unknown weight format `{s}` (native | int8 | fp8 | fp8t | int4[-hqq][-gN][-ffn])"
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
                if group == 0 || group % 8 != 0 {
                    return Err(format!("int4 group must be a multiple of 8, got {group}"));
                }
                Ok(Self::Int4 {
                    group,
                    hqq,
                    ffn_only,
                })
            }
        }
    }
}

/// A block linear as stored in a quantized checkpoint
/// ([`crate::weights::WeightSource::stored`]).
#[derive(Debug, Clone)]
pub enum Stored {
    Q8(Q8),
    Q4(Q4),
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

    /// Stack matrices with the same columns and group by rows (fused QKV).
    pub fn concat(parts: Vec<Q8>) -> Self {
        let mut it = parts.into_iter();
        let mut a = it.next().expect("at least one matrix");
        for b in it {
            assert_eq!((a.cols, a.group), (b.cols, b.group));
            a.q.extend(b.q);
            a.scales.extend(b.scales);
            a.rows += b.rows;
        }
        a
    }

    /// Device bytes (codes + scales).
    pub fn bytes(&self) -> usize {
        self.q.len() + 4 * self.scales.len()
    }
}

/// The largest finite E4M3 value (OCP FP8: 1 sign, 4 exponent bits with bias
/// 7, 3 mantissa bits; `S.1111.111` is NaN, so 448 = `0x7e` is the top).
pub const E4M3_MAX: f32 = 448.0;

/// `v` as E4M3, rounded to nearest (ties to even) and saturated to ±448, as
/// CUDA's `cvt.rn.satfinite.e4m3x2.f32` does. NaN gives 0.
pub fn e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs();
    let mag = if a >= E4M3_MAX {
        0x7e
    } else if a < 1.0 / 64.0 {
        // Subnormal: m · 2^-9; m = 8 is the smallest normal (code 8).
        (a * 512.0).round_ties_even() as u8
    } else {
        let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
        let m = (a / 2f32.powi(e - 3)).round_ties_even() as i32 - 8;
        let (e, m) = if m == 8 { (e + 1, 0) } else { (e, m) };
        (((e + 7) << 3) | m).min(0x7e) as u8
    };
    sign | mag
}

/// The value of an E4M3 code (`0x7f` / `0xff`, NaN, are never produced).
pub fn e4m3_decode(b: u8) -> f32 {
    let (e, m) = ((b >> 3) & 15, (b & 7) as f32);
    let v = if e == 0 {
        m / 512.0
    } else {
        (8.0 + m) * 2f32.powi(e as i32 - 10)
    };
    if b & 0x80 != 0 { -v } else { v }
}

/// `(scale, 1 / scale)` for a row whose largest magnitude is `amax`:
/// `scale = amax / 448` (0 for an all-zero row). Every backend computes the
/// codes as `e4m3(v · inv)` with these two f32 values.
pub fn e4m3_row_scale(amax: f32) -> (f32, f32) {
    if amax > 0.0 {
        (amax / E4M3_MAX, E4M3_MAX / amax)
    } else {
        (0.0, 0.0)
    }
}

/// The activation side of an FP8 linear: each of `rows` rows of `cols`
/// values replaced by its E4M3 rounding at the row's own scale (per token).
/// The CPU reference applies this before the matmul; CUDA quantizes for
/// real, WebGPU in a pre-pass, both checked against this.
pub fn fq8_rows(x: &mut [f32], cols: usize) {
    for row in x.chunks_mut(cols) {
        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (s, inv) = e4m3_row_scale(amax);
        for v in row.iter_mut() {
            *v = e4m3_decode(e4m3_encode(*v * inv)) * s;
        }
    }
}

/// [`fq8_rows`] with one scale for the whole slice (every row of one
/// linear's input): `scale = amax(x) / 448` over all of `x`.
pub fn fq8_tensor(x: &mut [f32]) {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (s, inv) = e4m3_row_scale(amax);
    for v in x.iter_mut() {
        *v = e4m3_decode(e4m3_encode(*v * inv)) * s;
    }
}

/// A `[rows, cols]` matrix as E4M3 codes with one f32 scale per row (output
/// channel): `w ≈ e4m3(q) · s`. `per_tensor` says how a linear with it
/// quantizes its input: per token ([`fq8_rows`], false) or with one scale
/// for the whole input ([`fq8_tensor`], true).
#[derive(Debug, Clone)]
pub struct F8 {
    pub q: Vec<u8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub per_tensor: bool,
}

impl F8 {
    pub fn quantize(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let mut q = vec![0u8; data.len()];
        let mut scales = Vec::with_capacity(rows);
        for (src, dst) in data.chunks(cols).zip(q.chunks_mut(cols)) {
            let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let (s, inv) = e4m3_row_scale(amax);
            for (d, v) in dst.iter_mut().zip(src) {
                *d = e4m3_encode(v * inv);
            }
            scales.push(s);
        }
        Self {
            q,
            scales,
            rows,
            cols,
            per_tensor: false,
        }
    }

    /// The same weights, with one activation scale per linear input.
    pub fn per_tensor(mut self) -> Self {
        self.per_tensor = true;
        self
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, &q)| e4m3_decode(q) * self.scales[i / self.cols])
            .collect()
    }

    /// Stack matrices with the same columns by rows (fused QKV).
    pub fn concat(parts: Vec<F8>) -> Self {
        let mut it = parts.into_iter();
        let mut a = it.next().expect("at least one matrix");
        for b in it {
            assert_eq!(a.cols, b.cols);
            a.q.extend(b.q);
            a.scales.extend(b.scales);
            a.rows += b.rows;
        }
        a
    }
}

/// A `[rows, cols]` matrix as unsigned 4-bit codes with an affine
/// scale / zero per group of columns: `w ≈ (q − z) · s`, `q ∈ [0, 15]`.
/// Scales and zeros are rounded to f16 when made, which is how they are
/// stored, so a quantized file and an in-memory quantization are identical.
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
        // wasm has no threads: `available_parallelism` errs there and a
        // spawn panics, so one chunk runs inline.
        let threads = if cfg!(target_arch = "wasm32") {
            1
        } else {
            std::thread::available_parallelism().map_or(1, |t| t.get())
        };
        let per = n.div_ceil(threads).max(1);
        let work = |w: &[f32], qc: &mut [u8], szc: &mut [(f32, f32)]| {
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
                let f = |v: f32| half::f16::from_f32(v).to_f32();
                *sz = (f(1.0 / is), f(z));
            }
        };
        if threads <= 1 {
            work(data, &mut q, &mut sz);
        } else {
            let work = &work;
            std::thread::scope(|sc| {
                for ((w, qc), szc) in data
                    .chunks(per * group)
                    .zip(q.chunks_mut(per * group))
                    .zip(sz.chunks_mut(per))
                {
                    sc.spawn(move || work(w, qc, szc));
                }
            });
        }
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

    /// Two codes per byte, element `2i` in the low nibble (the stored layout;
    /// read as little-endian u32 words, code `j` of a word is bits `4j..4j+4`).
    pub fn packed(&self) -> Vec<u8> {
        self.q
            .chunks(2)
            .map(|c| c[0] | (c.get(1).copied().unwrap_or(0) << 4))
            .collect()
    }

    pub fn from_packed(
        packed: &[u8],
        scales: Vec<f32>,
        zeros: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Self {
        let n = rows * cols;
        assert_eq!(packed.len(), n.div_ceil(2));
        assert!(!scales.is_empty() && scales.len() == zeros.len());
        assert_eq!(n % scales.len(), 0);
        let q = (0..n)
            .map(|i| (packed[i / 2] >> (4 * (i % 2))) & 15)
            .collect();
        let group = n / scales.len();
        Self {
            q,
            scales,
            zeros,
            rows,
            cols,
            group,
        }
    }

    /// Stack matrices with the same columns and group by rows (fused QKV).
    pub fn concat(parts: Vec<Q4>) -> Self {
        let mut it = parts.into_iter();
        let mut a = it.next().expect("at least one matrix");
        for b in it {
            assert_eq!((a.cols, a.group), (b.cols, b.group));
            a.q.extend(b.q);
            a.scales.extend(b.scales);
            a.zeros.extend(b.zeros);
            a.rows += b.rows;
        }
        a
    }

    /// Bytes once packed: two codes per byte, f16 scale and zero per group.
    pub fn bytes(&self) -> usize {
        self.q.len().div_ceil(2) + 4 * self.scales.len()
    }
}

#[cfg(test)]
mod tests {

    /// Every finite code decodes and re-encodes to itself; known values;
    /// saturation; ties to even; and random values land on the nearest code
    /// (checked by brute force over all codes).
    #[test]
    fn e4m3_round_trips_and_rounds_to_nearest() {
        for b in 0u8..=255 {
            if b & 0x7f == 0x7f {
                continue; // NaN
            }
            let v = e4m3_decode(b);
            let back = e4m3_encode(v);
            assert!(
                back == b || (v == 0.0 && back & 0x7f == 0),
                "{b:#x} -> {v} -> {back:#x}"
            );
        }
        assert_eq!(e4m3_encode(448.0), 0x7e);
        assert_eq!(e4m3_encode(1e6), 0x7e);
        assert_eq!(e4m3_encode(-1e6), 0xfe);
        assert_eq!(e4m3_encode(1.0), 0x38);
        assert_eq!(e4m3_decode(0x38), 1.0);
        assert_eq!(e4m3_encode(1.0625), 0x38, "tie to even (1.0)");
        assert_eq!(e4m3_encode(1.1875), 0x3a, "tie to even (1.25)");
        assert_eq!(e4m3_encode(2f32.powi(-10)), 0, "subnormal tie to even (0)");
        assert_eq!(
            e4m3_encode(3.0 * 2f32.powi(-10)),
            2,
            "subnormal tie to even (2)"
        );
        assert_eq!(e4m3_decode(1), 2f32.powi(-9));
        let codes: Vec<(u8, f32)> = (0u8..0x7f).map(|b| (b, e4m3_decode(b))).collect();
        let mut r = crate::rng::Rng::new(3);
        for v in r.normal(20_000) {
            let v = v * 40.0;
            let got = e4m3_decode(e4m3_encode(v)).abs();
            let a = v.abs().min(E4M3_MAX);
            let best = codes
                .iter()
                .map(|&(_, c)| (c - a).abs())
                .fold(f32::MAX, f32::min);
            assert!(
                (got - a).abs() <= best * (1.0 + 1e-6),
                "{v}: {got}, nearest is {best} away"
            );
        }
    }

    /// `F8` keeps each row's largest value exact (it maps to 448) and stays
    /// within E4M3's half-step (1/16 relative) elsewhere; `fq8_rows` likewise.
    #[test]
    fn f8_rows_are_scaled_to_448() {
        let mut r = crate::rng::Rng::new(5);
        let (rows, cols) = (7, 96);
        let w = r.normal(rows * cols);
        let f = F8::quantize(&w, rows, cols);
        let d = f.dequantize();
        for row in 0..rows {
            let src = &w[row * cols..][..cols];
            let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (a, b) in src.iter().zip(&d[row * cols..][..cols]) {
                assert!(
                    (a - b).abs() <= a.abs() / 16.0 + amax / 448.0 / 512.0 * 1.01,
                    "{a} vs {b}"
                );
            }
        }
        let mut x = w.clone();
        fq8_rows(&mut x, cols);
        assert_eq!(
            x, d,
            "per-row activation quantize = per-row weight quantize"
        );
        let mut z = vec![0.0; cols];
        fq8_rows(&mut z, cols);
        assert!(z.iter().all(|&v| v == 0.0));
        // One row: per tensor and per row agree; several: the largest value
        // anywhere is exact and maps to ±448 · scale.
        let mut a = w[..cols].to_vec();
        let mut b = a.clone();
        fq8_rows(&mut a, cols);
        fq8_tensor(&mut b);
        assert_eq!(a, b);
        let mut t = w.clone();
        fq8_tensor(&mut t);
        let (i, amax) =
            w.iter().enumerate().fold(
                (0, 0.0f32),
                |m, (i, v)| if v.abs() > m.1 { (i, v.abs()) } else { m },
            );
        assert_eq!(t[i].abs(), amax);
    }

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
        // half a step, plus the f16 rounding of scale and zero (≤ 0.015 step)
        for (i, (a, b)) in w.iter().zip(rtn.dequantize()).enumerate() {
            assert!((a - b).abs() <= rtn.scales[i / 64] * 0.52 + 1e-5);
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
            "int4-hqq-g32".parse(),
            Ok(WeightQuant::Int4 {
                group: 32,
                hqq: true,
                ffn_only: false
            })
        );
        assert_eq!(
            "int4".parse(),
            Ok(WeightQuant::Int4 {
                group: 64,
                hqq: false,
                ffn_only: false
            })
        );
        assert_eq!(
            "int4-hqq-g64-ffn".parse::<WeightQuant>().map(|q| q.attn()),
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
