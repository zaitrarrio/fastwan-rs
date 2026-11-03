//! The CPU reference backend: plain f32, one obvious loop per op, rayon over
//! rows. It is the ground truth every GPU kernel is compared against, so
//! clarity beats speed here — but it is fast enough to run the tiny graph in
//! unit tests and a short low-resolution clip end to end.

use fastwan_core::backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind};
use rayon::prelude::*;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Clone, Default)]
pub struct CpuBuf(Arc<RwLock<Vec<f32>>>);

impl CpuBuf {
    fn r(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.0.try_read().expect("buffer aliased with a written output")
    }
    fn w(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.0.try_write().expect("output buffer aliases an input")
    }
}

pub struct CpuWeight {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
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

impl Backend for Cpu {
    type Buf = CpuBuf;
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
        CpuWeight { data: data.to_vec(), rows, cols }
    }
    fn sync(&self) {}

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

    fn gate_residual(&self, h: &CpuBuf, a: &CpuBuf, rows: usize, dim: usize, mods: &CpuBuf, gate: usize) {
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

    fn patchify(&self, latent: &CpuBuf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &CpuBuf) {
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

    fn unpatchify(&self, xs: &CpuBuf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &CpuBuf) {
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
            ConvEpilogue::AddRelu(r) => Some(r.r()),
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
                        };
                    }
                }
            });
    }

    fn tanh_clamp(&self, x: &CpuBuf, n: usize, s: f32) {
        for v in x.w()[..n].iter_mut() {
            *v = (*v / s).tanh() * s;
        }
    }
}
