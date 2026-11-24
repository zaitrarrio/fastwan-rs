//! The CPU reference backend: plain f32, one obvious loop per op, rayon over
//! rows. It is the ground truth every GPU kernel is compared against, so
//! clarity beats speed here — but it is fast enough to run the tiny graph in
//! unit tests and a short low-resolution clip end to end.

use fastwan_core::backend::{
    Act, Backend, ConvEpilogue, ConvShape, NormKind, PackShape, pack_pixel,
};
use fastwan_core::vsa::{TILE_ELEMS, VsaSpec};
use rayon::prelude::*;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Clone, Default)]
pub struct CpuBuf(Arc<RwLock<Vec<f32>>>);

impl CpuBuf {
    fn r(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.0
            .try_read()
            .expect("buffer aliased with a written output")
    }
    fn w(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.0.try_write().expect("output buffer aliases an input")
    }
}

pub struct CpuWeight {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    /// FP8 W8A8: quantize each input row to E4M3 before the matmul.
    fq8: bool,
}

#[derive(Default)]
pub struct Cpu;

pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// Two-pass mean / inverse std of a row.
fn ln_stats(row: &[f32], eps: f32) -> (f32, f32) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, 1.0 / (var + eps).sqrt())
}

fn norm_row(src: &[f32], dst: &mut [f32], norm: &NormKind<'_, Vec<f32>>, mods: &[f32], eps: f32) {
    let dim = src.len();
    let (mean, inv) = ln_stats(src, eps);
    match norm {
        NormKind::Affine { w, b } => {
            for j in 0..dim {
                dst[j] = (src[j] - mean) * inv * w[j] + b[j];
            }
        }
        NormKind::Mod { scale, shift } => {
            let (sc, sh) = (&mods[scale * dim..][..dim], &mods[shift * dim..][..dim]);
            for j in 0..dim {
                dst[j] = (src[j] - mean) * inv * (1.0 + sc[j]) + sh[j];
            }
        }
    }
}

/// Resolve a `NormKind<CpuBuf>` into one over borrowed vectors.
fn with_norm<R>(norm: NormKind<'_, CpuBuf>, f: impl FnOnce(NormKind<'_, Vec<f32>>) -> R) -> R {
    match norm {
        NormKind::Affine { w, b } => {
            let (w, b) = (w.r(), b.r());
            f(NormKind::Affine { w: &w, b: &b })
        }
        NormKind::Mod { scale, shift } => f(NormKind::Mod { scale, shift }),
    }
}

/// VSA reference (fastvideo-rs `vsa_attention_host`): coarse stage in f64,
/// top-k by score then lower tile index, fine stage with an online softmax.
pub fn vsa_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    spec: &VsaSpec,
    out: &mut [f32],
) {
    let (plan, heads, d) = (&spec.plan, spec.heads, spec.head_dim);
    let (seq, nb, topk) = (
        plan.seq,
        plan.num_tiles(),
        spec.topk.clamp(1, plan.num_tiles()),
    );
    let scale = spec.scale as f64;
    let per_head: Vec<Vec<(usize, Vec<f32>)>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let row =
                |b: &[f32], tok: usize| -> Vec<f32> { b[(h * seq + tok) * d..][..d].to_vec() };
            // 1. Tile means over real tokens.
            let mut mean = vec![vec![0f64; nb * d]; 3];
            for t in 0..nb {
                let n = plan.block_sizes[t] as f64;
                for s in 0..plan.block_sizes[t] as usize {
                    let tok = plan.slot_src[t * TILE_ELEMS + s] as usize;
                    for (m, src) in mean.iter_mut().zip([q, k, v]) {
                        for (c, x) in row(src, tok).iter().enumerate() {
                            m[t * d + c] += *x as f64;
                        }
                    }
                }
                for m in mean.iter_mut() {
                    for c in 0..d {
                        m[t * d + c] /= n;
                    }
                }
            }
            let (qc, kc, vc) = (&mean[0], &mean[1], &mean[2]);
            // 2. Coarse attention + 3. top-k from the same scores.
            let mut coarse = vec![0f32; nb * d];
            let mut sel = vec![0usize; nb * topk];
            for qt in 0..nb {
                let s: Vec<f32> = (0..nb)
                    .map(|kt| {
                        ((0..d)
                            .map(|c| {
                                (qc[qt * d + c] as f32 as f64) * (kc[kt * d + c] as f32 as f64)
                            })
                            .sum::<f64>()
                            * scale) as f32
                    })
                    .collect();
                let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f64> = s.iter().map(|x| ((x - m) as f64).exp()).collect();
                let z: f64 = e.iter().sum();
                for (kt, w) in e.iter().enumerate() {
                    for c in 0..d {
                        coarse[qt * d + c] += (w / z * (vc[kt * d + c] as f32 as f64)) as f32;
                    }
                }
                let mut order: Vec<usize> = (0..nb).collect();
                order.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
                sel[qt * topk..(qt + 1) * topk].copy_from_slice(&order[..topk]);
            }
            // 4. Fine attention inside the selected tiles, 5. combine.
            let mut rows = Vec::with_capacity(seq);
            for qt in 0..nb {
                for qs in 0..plan.block_sizes[qt] as usize {
                    let qtok = plan.slot_src[qt * TILE_ELEMS + qs] as usize;
                    let qr = row(q, qtok);
                    let (mut m, mut z) = (f32::NEG_INFINITY, 0f64);
                    let mut acc = vec![0f64; d];
                    for &kt in &sel[qt * topk..(qt + 1) * topk] {
                        for ks in 0..plan.block_sizes[kt] as usize {
                            let ktok = plan.slot_src[kt * TILE_ELEMS + ks] as usize;
                            let kr = &k[(h * seq + ktok) * d..][..d];
                            let s = ((0..d).map(|c| qr[c] as f64 * kr[c] as f64).sum::<f64>()
                                * scale) as f32;
                            if s > m {
                                let shrink = if m.is_finite() {
                                    ((m - s) as f64).exp()
                                } else {
                                    0.0
                                };
                                z *= shrink;
                                acc.iter_mut().for_each(|a| *a *= shrink);
                                m = s;
                            }
                            let e = ((s - m) as f64).exp();
                            z += e;
                            let vr = &v[(h * seq + ktok) * d..][..d];
                            for c in 0..d {
                                acc[c] += e * vr[c] as f64;
                            }
                        }
                    }
                    let g = row(gate, qtok);
                    let o: Vec<f32> = (0..d)
                        .map(|c| coarse[qt * d + c] * g[c] + (acc[c] / z) as f32)
                        .collect();
                    rows.push((qtok, o));
                }
            }
            rows
        })
        .collect();
    for (h, rows) in per_head.into_iter().enumerate() {
        for (tok, o) in rows {
            out[(tok * heads + h) * d..][..d].copy_from_slice(&o);
        }
    }
}

impl Backend for Cpu {
    type Buf = CpuBuf;
    type Vsa = VsaSpec;
    type Weight = CpuWeight;

    fn name(&self) -> String {
        format!("cpu (rayon, {} threads)", rayon::current_num_threads())
    }

    fn alloc(&self, len: usize) -> CpuBuf {
        CpuBuf(Arc::new(RwLock::new(vec![0.0; len])))
    }
    fn upload(&self, data: &[f32]) -> CpuBuf {
        CpuBuf(Arc::new(RwLock::new(data.to_vec())))
    }
    fn write(&self, buf: &CpuBuf, data: &[f32]) {
        buf.w()[..data.len()].copy_from_slice(data);
    }
    fn download(&self, buf: &CpuBuf) -> Vec<f32> {
        buf.r().clone()
    }
    fn buf_len(&self, buf: &CpuBuf) -> usize {
        buf.r().len()
    }
    fn copy(&self, src: &CpuBuf, so: usize, dst: &CpuBuf, d: usize, len: usize) {
        if Arc::ptr_eq(&src.0, &dst.0) {
            dst.w().copy_within(so..so + len, d);
        } else {
            dst.w()[d..d + len].copy_from_slice(&src.r()[so..so + len]);
        }
    }
    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> CpuWeight {
        assert_eq!(data.len(), rows * cols);
        CpuWeight {
            data: data.to_vec(),
            rows,
            cols,
            fq8: false,
        }
    }
    fn sync(&self) {}

    fn weight_q8(&self, q: &fastwan_core::quant::Q8) -> CpuWeight {
        self.weight(&q.dequantize(), q.rows, q.cols)
    }

    fn weight_q4(&self, q: &fastwan_core::quant::Q4) -> CpuWeight {
        self.weight(&q.dequantize(), q.rows, q.cols)
    }

    fn weight_fp8(&self, q: &fastwan_core::quant::F8) -> CpuWeight {
        CpuWeight {
            fq8: true,
            ..self.weight(&q.dequantize(), q.rows, q.cols)
        }
    }

    fn linear(
        &self,
        x: &CpuBuf,
        x_row: usize,
        rows: usize,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        act: Act,
        y: &CpuBuf,
        y_row: usize,
    ) {
        let (k, n) = (w.cols, w.rows);
        let x = x.r();
        let b = bias.map(|b| b.r());
        let mut y = y.w();
        let x = &x[x_row * k..(x_row + rows) * k];
        // FP8 W8A8: the input rows as E4M3 at their own scales.
        let xq;
        let x = if w.fq8 {
            xq = {
                let mut v = x.to_vec();
                fastwan_core::quant::fq8_rows(&mut v, k);
                v
            };
            &xq[..]
        } else {
            x
        };
        y[y_row * n..(y_row + rows) * n]
            .par_chunks_mut(n)
            .zip(x.par_chunks(k))
            .for_each(|(yr, xr)| {
                for (o, out) in yr.iter_mut().enumerate() {
                    let wr = &w.data[o * k..(o + 1) * k];
                    let mut acc = 0.0f32;
                    for i in 0..k {
                        acc += xr[i] * wr[i];
                    }
                    if let Some(b) = &b {
                        acc += b[o];
                    }
                    *out = match act {
                        Act::None => acc,
                        Act::GeluTanh => gelu_tanh(acc),
                    };
                }
            });
    }

    fn layer_norm(
        &self,
        x: &CpuBuf,
        rows: usize,
        dim: usize,
        norm: NormKind<'_, CpuBuf>,
        mods: &CpuBuf,
        eps: f32,
        out: &CpuBuf,
    ) {
        let (x, m) = (x.r(), mods.r());
        let mut o = out.w();
        with_norm(norm, |norm| {
            o[..rows * dim]
                .par_chunks_mut(dim)
                .zip(x[..rows * dim].par_chunks(dim))
                .for_each(|(d, s)| norm_row(s, d, &norm, &m, eps));
        });
    }

    fn residual_norm(
        &self,
        h: &CpuBuf,
        a: &CpuBuf,
        rows: usize,
        dim: usize,
        mods: &CpuBuf,
        gate: Option<usize>,
        norm: NormKind<'_, CpuBuf>,
        eps: f32,
        normed: &CpuBuf,
    ) {
        let (a, m) = (a.r(), mods.r());
        let mut h = h.w();
        let mut o = normed.w();
        let g = gate.map(|g| &m[g * dim..][..dim]);
        with_norm(norm, |norm| {
            h[..rows * dim]
                .par_chunks_mut(dim)
                .zip(a[..rows * dim].par_chunks(dim))
                .zip(o[..rows * dim].par_chunks_mut(dim))
                .for_each(|((hr, ar), or)| {
                    for j in 0..dim {
                        hr[j] += ar[j] * g.map_or(1.0, |g| g[j]);
                    }
                    norm_row(hr, or, &norm, &m, eps);
                });
        });
    }

    fn gate_residual(
        &self,
        h: &CpuBuf,
        a: &CpuBuf,
        rows: usize,
        dim: usize,
        mods: &CpuBuf,
        gate: usize,
    ) {
        let (a, m) = (a.r(), mods.r());
        let g = &m[gate * dim..][..dim];
        let mut h = h.w();
        h[..rows * dim]
            .par_chunks_mut(dim)
            .zip(a[..rows * dim].par_chunks(dim))
            .for_each(|(hr, ar)| {
                for j in 0..dim {
                    hr[j] += ar[j] * g[j];
                }
            });
    }

    fn head_split(
        &self,
        src: &CpuBuf,
        rows: usize,
        stride: usize,
        col_off: usize,
        heads: usize,
        hd: usize,
        rms_w: Option<&CpuBuf>,
        rope: Option<(&CpuBuf, &CpuBuf)>,
        eps: f32,
        out: &CpuBuf,
    ) {
        let src = src.r();
        let w = rms_w.map(|w| w.r());
        let rope = rope.map(|(c, s)| (c.r(), s.r()));
        let width = heads * hd;
        let half = hd / 2;
        let mut out = out.w();
        // Row-parallel: gather each row, then scatter to [heads, rows, hd].
        let rows_out: Vec<Vec<f32>> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let x = &src[r * stride + col_off..][..width];
                let mut v = x.to_vec();
                if let Some(w) = &w {
                    let ms = x.iter().map(|&a| a * a).sum::<f32>() / width as f32;
                    let inv = 1.0 / (ms + eps).sqrt();
                    for j in 0..width {
                        v[j] = x[j] * inv * w[j];
                    }
                }
                if let Some((c, s)) = &rope {
                    for hh in 0..heads {
                        for i in 0..half {
                            let (cv, sv) = (c[r * half + i], s[r * half + i]);
                            let (x1, x2) = (v[hh * hd + 2 * i], v[hh * hd + 2 * i + 1]);
                            v[hh * hd + 2 * i] = x1 * cv - x2 * sv;
                            v[hh * hd + 2 * i + 1] = x1 * sv + x2 * cv;
                        }
                    }
                }
                v
            })
            .collect();
        for (r, v) in rows_out.iter().enumerate() {
            for hh in 0..heads {
                out[(hh * rows + r) * hd..][..hd].copy_from_slice(&v[hh * hd..][..hd]);
            }
        }
    }

    fn kv_caps(&self, _head_dim: usize) -> (bool, bool) {
        (true, true)
    }

    fn kv_store(
        &self,
        src: &CpuBuf,
        n: usize,
        from: usize,
        len: usize,
        heads: usize,
        d: usize,
        dst: &CpuBuf,
        dst_row: usize,
        stride: usize,
        half: bool,
    ) {
        let src = src.r();
        let mut dst = dst.w();
        for h in 0..heads {
            let s = &src[(h * n + from) * d..][..len * d];
            let o = &mut dst[(h * stride + dst_row) * d..][..len * d];
            if half {
                // The 16-bit cache keeps f16 values: round through f16.
                for (o, &x) in o.iter_mut().zip(s) {
                    *o = half::f16::from_f32(x).to_f32();
                }
            } else {
                o.copy_from_slice(s);
            }
        }
    }

    fn attention_kv(
        &self,
        q: &CpuBuf,
        k: &CpuBuf,
        v: &CpuBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        stride: usize,
        d: usize,
        _half: bool,
        out: &CpuBuf,
    ) {
        // Gather the valid rows of each head and reuse the reference attention.
        let pack = |b: &CpuBuf| {
            let b = b.r();
            let mut v = Vec::with_capacity(heads * sk * d);
            for h in 0..heads {
                v.extend_from_slice(&b[h * stride * d..][..sk * d]);
            }
            CpuBuf(Arc::new(RwLock::new(v)))
        };
        let (pk, pv) = (pack(k), pack(v));
        self.attention(q, &pk, &pv, heads, sq, sk, d, out);
    }

    fn attention(
        &self,
        q: &CpuBuf,
        k: &CpuBuf,
        v: &CpuBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        out: &CpuBuf,
    ) {
        let (q, k, v) = (q.r(), k.r(), v.r());
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = out.w();
        out[..sq * heads * d]
            .par_chunks_mut(heads * d)
            .enumerate()
            .for_each(|(i, orow)| {
                let mut p = vec![0.0f32; sk];
                for h in 0..heads {
                    let qi = &q[(h * sq + i) * d..][..d];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..sk {
                        let kj = &k[(h * sk + j) * d..][..d];
                        let s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                        p[j] = s;
                        mx = mx.max(s);
                    }
                    let mut den = 0.0;
                    for pj in p.iter_mut() {
                        *pj = (*pj - mx).exp();
                        den += *pj;
                    }
                    let o = &mut orow[h * d..][..d];
                    o.fill(0.0);
                    for (j, &pj) in p.iter().enumerate() {
                        let vj = &v[(h * sk + j) * d..][..d];
                        for c in 0..d {
                            o[c] += pj * vj[c];
                        }
                    }
                    for c in o.iter_mut() {
                        *c /= den;
                    }
                }
            });
    }

    fn vsa_prepare(&self, spec: &VsaSpec) -> VsaSpec {
        spec.clone()
    }

    fn vsa_attention(
        &self,
        q: &CpuBuf,
        k: &CpuBuf,
        v: &CpuBuf,
        gate: &CpuBuf,
        spec: &VsaSpec,
        out: &CpuBuf,
    ) {
        let (q, k, v, g) = (q.r(), k.r(), v.r(), gate.r());
        vsa_reference(&q, &k, &v, &g, spec, &mut out.w());
    }

    fn patchify(
        &self,
        latent: &CpuBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &CpuBuf,
    ) {
        let x = latent.r();
        let mut o = out.w();
        let (gh, gw, f) = (h / p, w / p, c * p * p);
        for tt in 0..t {
            for y in 0..gh {
                for xg in 0..gw {
                    let tok = (tt * gh + y) * gw + xg;
                    for cc in 0..c {
                        for kh in 0..p {
                            for kw in 0..p {
                                o[tok * f + cc * p * p + kh * p + kw] =
                                    x[((cc * t + tt) * h + y * p + kh) * w + xg * p + kw];
                            }
                        }
                    }
                }
            }
        }
    }

    fn unpatchify(
        &self,
        xs: &CpuBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &CpuBuf,
    ) {
        let x = xs.r();
        let mut o = out.w();
        let (gh, gw, f) = (h / p, w / p, c * p * p);
        for tt in 0..t {
            for y in 0..gh {
                for xg in 0..gw {
                    let tok = (tt * gh + y) * gw + xg;
                    for kh in 0..p {
                        for kw in 0..p {
                            for cc in 0..c {
                                o[((cc * t + tt) * h + y * p + kh) * w + xg * p + kw] =
                                    x[tok * f + (kh * p + kw) * c + cc];
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_act(&self, y: &CpuBuf, y_off: usize, d: &CpuBuf, d_off: usize, n: usize, act: Act) {
        let d = d.r();
        let mut y = y.w();
        for (o, &a) in y[y_off..y_off + n].iter_mut().zip(&d[d_off..d_off + n]) {
            let s = *o + a;
            *o = match act {
                Act::None => s,
                Act::GeluTanh => gelu_tanh(s),
            };
        }
    }

    fn dmd_step(&self, x: &CpuBuf, v: &CpuBuf, noise: Option<&CpuBuf>, st: f32, sn: f32, n: usize) {
        let v = v.r();
        let noise = noise.map(|b| b.r());
        let mut x = x.w();
        for i in 0..n {
            let x0 = x[i] - st * v[i];
            x[i] = match &noise {
                Some(e) => (1.0 - sn) * x0 + sn * e[i],
                None => x0,
            };
        }
    }

    fn rope_heads(
        &self,
        src: &CpuBuf,
        src_off: usize,
        src_stride: usize,
        dst: &CpuBuf,
        dst_off: usize,
        dst_stride: usize,
        heads: usize,
        rows: usize,
        hd: usize,
        rope: (&CpuBuf, &CpuBuf),
    ) {
        let (src, c, s) = (src.r(), rope.0.r(), rope.1.r());
        let mut dst = dst.w();
        let half = hd / 2;
        for h in 0..heads {
            for r in 0..rows {
                let x = &src[src_off + h * src_stride + r * hd..][..hd];
                let y = &mut dst[dst_off + h * dst_stride + r * hd..][..hd];
                for i in 0..half {
                    let (cv, sv) = (c[r * half + i], s[r * half + i]);
                    let (x1, x2) = (x[2 * i], x[2 * i + 1]);
                    y[2 * i] = x1 * cv - x2 * sv;
                    y[2 * i + 1] = x1 * sv + x2 * cv;
                }
            }
        }
    }

    fn kv_ema(
        &self,
        buf: &CpuBuf,
        heads: usize,
        head_stride: usize,
        hd: usize,
        src_row: usize,
        src_rows: usize,
        dst_row: usize,
        frame_rows: usize,
        alphas: [f32; 2],
    ) {
        assert!(
            src_row + src_rows <= dst_row || dst_row + 2 * frame_rows <= src_row,
            "kv_ema: source and memory rows overlap"
        );
        let mut b = buf.w();
        for h in 0..heads {
            let base = h * head_stride;
            let mut m = vec![0f64; hd];
            for r in 0..src_rows {
                for (j, mj) in m.iter_mut().enumerate() {
                    *mj += b[base + (src_row + r) * hd + j] as f64;
                }
            }
            let m: Vec<f32> = m.iter().map(|&v| (v / src_rows as f64) as f32).collect();
            for (i, &a) in alphas.iter().enumerate() {
                for r in 0..frame_rows {
                    let row = &mut b[base + (dst_row + i * frame_rows + r) * hd..][..hd];
                    for (x, &mj) in row.iter_mut().zip(&m) {
                        *x = a * mj + (1.0 - a) * *x;
                    }
                }
            }
        }
    }

    fn conv2d(
        &self,
        a: &CpuBuf,
        b: Option<&CpuBuf>,
        s: &ConvShape,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        ep: ConvEpilogue<'_, CpuBuf>,
        out: &CpuBuf,
    ) {
        assert_eq!(w.cols, s.k_len());
        assert_eq!(w.rows, s.cout);
        let a = a.r();
        let b = b.map(|b| b.r());
        let bias = bias.map(|b| b.r());
        let res = match ep {
            ConvEpilogue::AddRelu(r) | ConvEpilogue::Prelu(r) => Some(r.r()),
            _ => None,
        };
        let (oh, ow) = s.out_hw();
        let up = if s.upsample { 2 } else { 1 };
        let pad = (s.k / 2) as isize;
        let mut o = out.w();
        o[..s.out_len()]
            .par_chunks_mut(oh * ow)
            .enumerate()
            .for_each(|(fc, plane)| {
                let (f, co) = (fc / s.cout, fc % s.cout);
                let wr = &w.data[co * s.k_len()..][..s.k_len()];
                for y in 0..oh {
                    for xo in 0..ow {
                        let mut acc = bias.as_ref().map_or(0.0, |b| b[co]);
                        for ci in 0..s.cin() {
                            let (src, c, cn): (&[f32], usize, usize) = if ci < s.cin_a {
                                (&a, ci, s.cin_a)
                            } else {
                                (b.as_ref().unwrap(), ci - s.cin_a, s.cin_b)
                            };
                            for ky in 0..s.k {
                                let yy = y as isize + ky as isize - pad;
                                if yy < 0 || yy >= oh as isize {
                                    continue;
                                }
                                for kx in 0..s.k {
                                    let xx = xo as isize + kx as isize - pad;
                                    if xx < 0 || xx >= ow as isize {
                                        continue;
                                    }
                                    let (sy, sx) = (yy as usize / up, xx as usize / up);
                                    acc += wr[(ci * s.k + ky) * s.k + kx]
                                        * src[((f * cn + c) * s.h + sy) * s.w + sx];
                                }
                            }
                        }
                        let i = y * ow + xo;
                        plane[i] = match ep {
                            ConvEpilogue::None => acc,
                            ConvEpilogue::Relu => acc.max(0.0),
                            ConvEpilogue::AddRelu(_) => {
                                (acc + res.as_ref().unwrap()[fc * oh * ow + i]).max(0.0)
                            }
                            ConvEpilogue::Prelu(_) if acc < 0.0 => acc * res.as_ref().unwrap()[co],
                            ConvEpilogue::Prelu(_) => acc,
                        };
                    }
                }
            });
    }

    fn pack_rgba8(&self, src: &CpuBuf, base: Option<&CpuBuf>, s: &PackShape, out: &CpuBuf) {
        let (h, w, p, d) = (s.h, s.w, s.p, s.down);
        let (oh, ow) = s.out_hw();
        let src = src.r();
        let base = base.map(|b| b.r());
        let mut o = out.w();
        o[..s.pixels()]
            .par_chunks_mut(ow)
            .enumerate()
            .for_each(|(row, line)| {
                let (f, oy) = (row / oh + s.skip, row % oh);
                for (ox, px) in line.iter_mut().enumerate() {
                    // One shuffled-image sample, residual added and clamped.
                    let v = |c: usize, y: usize, x: usize| {
                        let (sy, sx) = (y / p, x / p);
                        let k = c * p * p + (y % p) * p + x % p;
                        let mut v = src[((f * 3 * p * p + k) * h + sy) * w + sx];
                        if let Some(b) = &base {
                            let word = b[(f * h + sy) * w + sx].to_bits();
                            v += ((word >> (8 * c)) & 0xff) as f32 / 255.0;
                        }
                        v.clamp(0.0, 1.0)
                    };
                    let avg = |c: usize| {
                        let mut a = 0.0;
                        for dy in 0..d {
                            for dx in 0..d {
                                a += v(c, oy * d + dy, ox * d + dx);
                            }
                        }
                        a / (d * d) as f32
                    };
                    *px = f32::from_bits(pack_pixel(avg(0), avg(1), avg(2)));
                }
            });
    }

    fn unpack_rgba8(&self, src: &CpuBuf, frames: usize, plane: usize, out: &CpuBuf) {
        let src = src.r();
        let mut o = out.w();
        o[..frames * 3 * plane]
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(fc, line)| {
                let (f, c) = (fc / 3, fc % 3);
                for (p, v) in line.iter_mut().enumerate() {
                    *v = ((src[f * plane + p].to_bits() >> (8 * c)) & 0xff) as f32 / 255.0;
                }
            });
    }

    fn tanh_clamp(&self, x: &CpuBuf, n: usize, s: f32) {
        for v in x.w()[..n].iter_mut() {
            *v = (*v / s).tanh() * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastwan_core::rng::Rng;

    /// All tiles selected and a zero gate: VSA is dense attention.
    #[test]
    fn vsa_with_every_tile_is_dense_attention() {
        let (grid, heads, d) = ((5, 6, 9), 2, 16);
        let spec = VsaSpec::new(grid, heads, d, 0.0);
        assert_eq!(spec.topk, spec.plan.num_tiles());
        let s = spec.plan.seq;
        let mut r = Rng::new(1);
        let (q, k, v) = (
            r.normal(heads * s * d),
            r.normal(heads * s * d),
            r.normal(heads * s * d),
        );
        let be = Cpu;
        let (gq, gk, gv) = (be.upload(&q), be.upload(&k), be.upload(&v));
        let dense = be.alloc(s * heads * d);
        be.attention(&gq, &gk, &gv, heads, s, s, d, &dense);
        let sparse = be.alloc(s * heads * d);
        let st = be.vsa_prepare(&spec);
        be.vsa_attention(&gq, &gk, &gv, &be.alloc(heads * s * d), &st, &sparse);
        let (a, b) = (be.download(&dense), be.download(&sparse));
        let err = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(err < 1e-5, "{err}");
    }
}
