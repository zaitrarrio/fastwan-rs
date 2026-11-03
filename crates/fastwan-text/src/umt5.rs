//! The UMT5 encoder (Hugging Face `UMT5EncoderModel`), host f32 activations
//! over weights read in their stored precision.
//!
//! Per layer (every layer has its own relative-position bias in UMT5, unlike
//! T5, where only the first does):
//!
//! ```text
//! x  = rms(h)·w0;  h += o( softmax(q·kᵀ + bias[bucket(j − i)]) · v )   no 1/√d scaling
//! x  = rms(h)·w1;  h += wo( gelu_tanh(wi_0·x) ⊙ wi_1·x )
//! ```
//! then a final RMS norm.

use fastwan_core::{Error, Result, weights::SafetensorsDir};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Umt5Config {
    pub vocab: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub heads: usize,
    pub d_ff: usize,
    pub layers: usize,
    pub buckets: usize,
    pub max_distance: usize,
    pub eps: f32,
}

impl Umt5Config {
    /// `google/umt5-xxl`, the encoder of every Wan 2.1 / 2.2 checkpoint.
    pub fn xxl() -> Self {
        Self {
            vocab: 256384,
            d_model: 4096,
            d_kv: 64,
            heads: 64,
            d_ff: 10240,
            layers: 24,
            buckets: 32,
            max_distance: 128,
            eps: 1e-6,
        }
    }
}

pub struct Umt5 {
    pub cfg: Umt5Config,
    src: SafetensorsDir,
}

/// A weight matrix as stored: `[rows, cols]`, bf16 / f16 / f32 bytes.
struct W<'a> {
    dtype: safetensors::Dtype,
    data: &'a [u8],
    rows: usize,
    cols: usize,
}

impl W<'_> {
    fn row(&self, r: usize, out: &mut [f32]) {
        use safetensors::Dtype::*;
        let c = self.cols;
        match self.dtype {
            BF16 => {
                for (o, b) in out
                    .iter_mut()
                    .zip(self.data[2 * r * c..2 * (r + 1) * c].chunks_exact(2))
                {
                    *o = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
                }
            }
            F16 => {
                for (o, b) in out
                    .iter_mut()
                    .zip(self.data[2 * r * c..2 * (r + 1) * c].chunks_exact(2))
                {
                    *o = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                }
            }
            _ => {
                for (o, b) in out
                    .iter_mut()
                    .zip(self.data[4 * r * c..4 * (r + 1) * c].chunks_exact(4))
                {
                    *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        }
    }
}

/// `y[t, r] = Σ_k x[t, k]·W[r, k]` for `n` tokens: each weight row is
/// converted once and used for every token (the encoder is bound by reading
/// weights, not by arithmetic).
fn linear(x: &[f32], n: usize, w: &W<'_>) -> Vec<f32> {
    const BLOCK: usize = 32;
    let (rows, k) = (w.rows, w.cols);
    let cols: Vec<Vec<f32>> = (0..rows.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let r0 = b * BLOCK;
            let nr = BLOCK.min(rows - r0);
            let mut wr = vec![0.0f32; k];
            let mut out = vec![0.0f32; nr * n];
            for i in 0..nr {
                w.row(r0 + i, &mut wr);
                for t in 0..n {
                    let xt = &x[t * k..(t + 1) * k];
                    out[i * n + t] = xt.iter().zip(&wr).map(|(a, b)| a * b).sum();
                }
            }
            out
        })
        .collect();
    let mut y = vec![0.0f32; n * rows];
    for (b, blk) in cols.iter().enumerate() {
        let nr = blk.len() / n;
        for i in 0..nr {
            for t in 0..n {
                y[t * rows + b * BLOCK + i] = blk[i * n + t];
            }
        }
    }
    y
}

fn rms_norm(h: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    h.chunks_exact(d)
        .flat_map(|row| {
            let var = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + eps).sqrt();
            row.iter()
                .zip(w)
                .map(move |(v, g)| v * inv * g)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

/// HF `T5Attention._relative_position_bucket`, bidirectional (encoder).
pub fn bucket(rel: i64, buckets: usize, max_distance: usize) -> usize {
    let nb = (buckets / 2) as i64;
    let mut b = if rel > 0 { nb } else { 0 };
    let rp = rel.abs();
    let max_exact = nb / 2;
    b += if rp < max_exact {
        rp
    } else {
        let large = max_exact as f64
            + ((rp as f64 / max_exact as f64).ln() / (max_distance as f64 / max_exact as f64).ln()
                * (nb - max_exact) as f64) as i64 as f64;
        (large as i64).min(nb - 1)
    };
    b as usize
}

impl Umt5 {
    pub fn new(src: SafetensorsDir, cfg: Umt5Config) -> Result<Self> {
        let m = Self { cfg, src };
        // Fail at load, not mid-request, if the checkpoint is not a UMT5 encoder.
        m.w(
            "encoder.block.0.layer.0.SelfAttention.q.weight",
            m.cfg.heads * m.cfg.d_kv,
            m.cfg.d_model,
        )?;
        m.w(
            &format!(
                "encoder.block.{}.layer.1.DenseReluDense.wo.weight",
                m.cfg.layers - 1
            ),
            m.cfg.d_model,
            m.cfg.d_ff,
        )?;
        m.vec("encoder.final_layer_norm.weight", m.cfg.d_model)?;
        Ok(m)
    }

    fn w(&self, name: &str, rows: usize, cols: usize) -> Result<W<'_>> {
        let (dtype, shape, data) = self.src.raw(name)?;
        if shape != [rows, cols] {
            return Err(Error::Weights(format!(
                "`{name}` is {shape:?}, expected [{rows}, {cols}]"
            )));
        }
        Ok(W {
            dtype,
            data,
            rows,
            cols,
        })
    }

    fn vec(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        use fastwan_core::weights::WeightSource;
        self.src.get(name, &[n])
    }

    fn embed_name(&self) -> &'static str {
        if self.src.contains_raw("shared.weight") {
            "shared.weight"
        } else {
            "encoder.embed_tokens.weight"
        }
    }

    /// Hidden states `[ids.len(), d_model]` (the encoder's
    /// `last_hidden_state` for an unpadded prompt).
    pub fn encode(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (n, d, hd, nh) = (ids.len(), c.d_model, c.d_kv, c.heads);
        let inner = nh * hd;
        let emb = self.w(self.embed_name(), c.vocab, d)?;
        let mut h = vec![0.0f32; n * d];
        for (t, &id) in ids.iter().enumerate() {
            if id as usize >= c.vocab {
                return Err(Error::Msg(format!("token id {id} outside the vocabulary")));
            }
            emb.row(id as usize, &mut h[t * d..(t + 1) * d]);
        }
        let buckets: Vec<usize> = (0..n * n)
            .map(|ij| {
                bucket(
                    ij as i64 % n as i64 - ij as i64 / n as i64,
                    c.buckets,
                    c.max_distance,
                )
            })
            .collect();
        for l in 0..c.layers {
            let p = |s: &str| format!("encoder.block.{l}.{s}");
            let x = rms_norm(&h, &self.vec(&p("layer.0.layer_norm.weight"), d)?, c.eps);
            let q = linear(
                &x,
                n,
                &self.w(&p("layer.0.SelfAttention.q.weight"), inner, d)?,
            );
            let k = linear(
                &x,
                n,
                &self.w(&p("layer.0.SelfAttention.k.weight"), inner, d)?,
            );
            let v = linear(
                &x,
                n,
                &self.w(&p("layer.0.SelfAttention.v.weight"), inner, d)?,
            );
            let rel = self.w(
                &p("layer.0.SelfAttention.relative_attention_bias.weight"),
                c.buckets,
                nh,
            )?;
            let mut table = vec![0.0f32; c.buckets * nh];
            for b in 0..c.buckets {
                rel.row(b, &mut table[b * nh..(b + 1) * nh]);
            }
            let mut ctx = vec![0.0f32; n * inner];
            ctx.par_chunks_mut(inner).enumerate().for_each(|(i, out)| {
                let mut s = vec![0.0f32; n];
                for hh in 0..nh {
                    let qi = &q[i * inner + hh * hd..][..hd];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..n {
                        let kj = &k[j * inner + hh * hd..][..hd];
                        s[j] = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>()
                            + table[buckets[i * n + j] * nh + hh];
                        mx = mx.max(s[j]);
                    }
                    let mut den = 0.0;
                    for sj in s.iter_mut() {
                        *sj = (*sj - mx).exp();
                        den += *sj;
                    }
                    let o = &mut out[hh * hd..][..hd];
                    for (j, &sj) in s.iter().enumerate() {
                        let vj = &v[j * inner + hh * hd..][..hd];
                        for (oc, vc) in o.iter_mut().zip(vj) {
                            *oc += sj / den * vc;
                        }
                    }
                }
            });
            let a = linear(
                &ctx,
                n,
                &self.w(&p("layer.0.SelfAttention.o.weight"), d, inner)?,
            );
            h.iter_mut().zip(&a).for_each(|(hv, av)| *hv += av);
            let x = rms_norm(&h, &self.vec(&p("layer.1.layer_norm.weight"), d)?, c.eps);
            let g = linear(
                &x,
                n,
                &self.w(&p("layer.1.DenseReluDense.wi_0.weight"), c.d_ff, d)?,
            );
            let u = linear(
                &x,
                n,
                &self.w(&p("layer.1.DenseReluDense.wi_1.weight"), c.d_ff, d)?,
            );
            let gu: Vec<f32> = g.iter().zip(&u).map(|(a, b)| gelu_tanh(*a) * b).collect();
            let f = linear(
                &gu,
                n,
                &self.w(&p("layer.1.DenseReluDense.wo.weight"), d, c.d_ff)?,
            );
            h.iter_mut().zip(&f).for_each(|(hv, fv)| *hv += fv);
        }
        Ok(rms_norm(
            &h,
            &self.vec("encoder.final_layer_norm.weight", d)?,
            c.eps,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::bucket;

    #[test]
    fn buckets_match_t5() {
        // Values printed by transformers 5.22 `UMT5Attention._relative_position_bucket` (32 buckets, 128).
        let b = |r| bucket(r, 32, 128);
        assert_eq!([b(0), b(1), b(7), b(-1), b(-7)], [0, 17, 23, 1, 7]);
        assert_eq!([b(8), b(-8), b(127), b(-200)], [24, 8, 31, 15]);
        assert_eq!([b(500), b(13), b(-60)], [31, 25, 13]);
    }
}
