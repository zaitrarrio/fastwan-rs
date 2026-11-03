//! TAEHV decoder (madebyollin/taehv `taehv.py`; fastvideo-rs `rt/taehv.rs`) for
//! `taew2_1` (Wan 2.1 latents) and `taew2_2` (Wan 2.2 latents).
//!
//! ```text
//! clamp tanh(x/3)·3 → conv C→n0, relu
//! 3 stages: 3×MemBlock(n_s) → upsample ×2 → TGrow(stride) → conv n_s→n_{s+1}
//! relu → conv n3→3·p² → pixel-shuffle p
//! MemBlock(x, past) = relu(conv(relu(conv(relu(conv(cat[x, past]))))) + x)
//! ```
//!
//! Three rewrites, each exact:
//! * **TGrow before the upsample.** TGrow is a 1×1 conv, and a pointwise op
//!   commutes with nearest upsampling, so it runs at ¼ of the pixels.
//! * **Upsample fused into the next conv.** The ×2 tensor is never stored;
//!   the conv reads its input through `(y/2, x/2)` addressing.
//! * **Concat fused into the MemBlock's first conv** (`cat[x, past]` is read
//!   from two buffers).
//!
//! Frames are decoded in latent chunks; each MemBlock carries its last input
//! frame across a chunk boundary, which is the reference's sequential
//! `memory[i]`, so chunking does not change the output.

use fastwan_core::{
    backend::{Backend, ConvEpilogue, ConvShape},
    config::TaeConfig,
    weights::WeightSource,
    Result,
};

struct ConvW<B: Backend> {
    w: B::Weight,
    b: Option<B::Buf>,
}

struct Stage<B: Backend> {
    mems: Vec<[ConvW<B>; 3]>,
    grow: ConvW<B>,
    conv: ConvW<B>,
}

pub struct TaeHv<B: Backend> {
    pub cfg: TaeConfig,
    first: ConvW<B>,
    stages: Vec<Stage<B>>,
    last: ConvW<B>,
}

/// Carried between chunks: one saved input frame per MemBlock, and how many
/// priming frames are still to be dropped.
pub struct DecodeState<B: Backend> {
    memory: Vec<Option<B::Buf>>,
    to_trim: usize,
}

/// Decoded frames of one chunk: `[frames, 3, H, W]`, values ≈ `[0, 1]`.
pub struct Frames {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

fn conv_w<B: Backend>(be: &B, src: &dyn WeightSource, key: &str, cout: usize, cin: usize, k: usize, bias: bool) -> Result<ConvW<B>> {
    Ok(ConvW {
        w: be.weight(&src.get(&format!("{key}.weight"), &[cout, cin, k, k])?, cout, cin * k * k),
        b: if bias { Some(be.upload(&src.get(&format!("{key}.bias"), &[cout])?)) } else { None },
    })
}

impl<B: Backend> TaeHv<B> {
    pub fn load(be: &B, cfg: TaeConfig, src: &dyn WeightSource) -> Result<Self> {
        let nf = cfg.widths;
        let mut stages = Vec::new();
        for s in 0..3 {
            let base = 3 + 6 * s;
            let mems = (0..cfg.mem_blocks)
                .map(|m| {
                    let p = format!("decoder.{}", base + m);
                    Ok([
                        conv_w(be, src, &format!("{p}.conv.0"), nf[s], 2 * nf[s], 3, true)?,
                        conv_w(be, src, &format!("{p}.conv.2"), nf[s], nf[s], 3, true)?,
                        conv_w(be, src, &format!("{p}.conv.4"), nf[s], nf[s], 3, true)?,
                    ])
                })
                .collect::<Result<Vec<_>>>()?;
            stages.push(Stage {
                mems,
                grow: conv_w(be, src, &format!("decoder.{}.conv", base + 4), nf[s] * cfg.tgrow[s], nf[s], 1, false)?,
                conv: conv_w(be, src, &format!("decoder.{}", base + 5), nf[s + 1], nf[s], 3, false)?,
            });
        }
        Ok(Self {
            first: conv_w(be, src, "decoder.1", nf[0], cfg.latent_channels, 3, true)?,
            last: conv_w(be, src, "decoder.22", cfg.out_channels(), nf[3], 3, true)?,
            stages,
            cfg,
        })
    }

    pub fn state(&self) -> DecodeState<B> {
        DecodeState {
            memory: (0..3 * self.cfg.mem_blocks).map(|_| None).collect(),
            to_trim: self.cfg.frames_to_trim(),
        }
    }

    /// Decode `frames` latent frames given frame-major (`[frames, C, h, w]`).
    pub fn decode_chunk(&self, be: &B, st: &mut DecodeState<B>, z: &[f32], frames: usize, h: usize, w: usize) -> Frames {
        let cfg = &self.cfg;
        let nf = cfg.widths;
        assert_eq!(z.len(), frames * cfg.latent_channels * h * w);
        let (mut f, mut h, mut w) = (frames, h, w);
        let x0 = be.upload(z);
        be.tanh_clamp(&x0, z.len(), 3.0);
        let shape = |f, cin_a, cin_b, cout, h, w, upsample, k| ConvShape { frames: f, cin_a, cin_b, cout, h, w, upsample, k };
        let conv = |x: &B::Buf, b: Option<&B::Buf>, s: ConvShape, cw: &ConvW<B>, ep: ConvEpilogue<'_, B::Buf>| {
            let out = be.alloc(s.out_len());
            be.conv2d(x, b, &s, &cw.w, cw.b.as_ref(), ep, &out);
            out
        };
        let mut x = conv(&x0, None, shape(f, cfg.latent_channels, 0, nf[0], h, w, false, 3), &self.first, ConvEpilogue::Relu);
        for (si, stage) in self.stages.iter().enumerate() {
            let n = nf[si];
            for (mi, [c0, c2, c4]) in stage.mems.iter().enumerate() {
                let fsz = n * h * w;
                let past = be.alloc(f * fsz);
                let slot = &mut st.memory[si * cfg.mem_blocks + mi];
                if let Some(prev) = slot.as_ref() {
                    be.copy(prev, 0, &past, 0, fsz);
                }
                if f > 1 {
                    be.copy(&x, 0, &past, fsz, (f - 1) * fsz);
                }
                let keep = be.alloc(fsz);
                be.copy(&x, (f - 1) * fsz, &keep, 0, fsz);
                *slot = Some(keep);
                let t1 = conv(&x, Some(&past), shape(f, n, n, n, h, w, false, 3), c0, ConvEpilogue::Relu);
                let t2 = conv(&t1, None, shape(f, n, 0, n, h, w, false, 3), c2, ConvEpilogue::Relu);
                x = conv(&t2, None, shape(f, n, 0, n, h, w, false, 3), c4, ConvEpilogue::AddRelu(&x));
            }
            let stride = cfg.tgrow[si];
            // [f, n·stride, h, w] is bit-for-bit [f·stride, n, h, w].
            let g = conv(&x, None, shape(f, n, 0, n * stride, h, w, false, 1), &stage.grow, ConvEpilogue::None);
            f *= stride;
            let ep = if si == 2 { ConvEpilogue::Relu } else { ConvEpilogue::None };
            x = conv(&g, None, shape(f, n, 0, nf[si + 1], h, w, true, 3), &stage.conv, ep);
            h *= 2;
            w *= 2;
        }
        let rgb = conv(&x, None, shape(f, nf[3], 0, cfg.out_channels(), h, w, false, 3), &self.last, ConvEpilogue::None);
        let raw = be.download(&rgb);
        let p = cfg.patch;
        let (oh, ow) = (h * p, w * p);
        let skip = st.to_trim.min(f);
        st.to_trim -= skip;
        let mut data = vec![0.0f32; (f - skip) * 3 * oh * ow];
        for fi in skip..f {
            for c in 0..3 {
                for y in 0..oh {
                    for xx in 0..ow {
                        let ch = c * p * p + (y % p) * p + xx % p;
                        data[(((fi - skip) * 3 + c) * oh + y) * ow + xx] =
                            raw[((fi * 3 * p * p + ch) * h + y / p) * w + xx / p];
                    }
                }
            }
        }
        Frames { frames: f - skip, height: oh, width: ow, data }
    }
}
