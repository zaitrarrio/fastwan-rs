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
    Result,
    backend::{Backend, ConvEpilogue, ConvShape, PackShape, rgba8_bytes},
    config::TaeConfig,
    weights::WeightSource,
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
    /// Intermediate buffers by length, reused by the next chunk (every chunk
    /// but the last has the same shapes). Device work is ordered, so a buffer
    /// handed back once its last reader is queued is safe to overwrite.
    pool: std::collections::HashMap<usize, Vec<B::Buf>>,
}

/// One chunk's decoder output on the device: `[frames, 3·p², h, w]`, of
/// which the first `skip` frames are priming frames to drop.
pub struct Chunk<B: Backend> {
    pub rgb: B::Buf,
    pub frames: usize,
    pub skip: usize,
    pub h: usize,
    pub w: usize,
    pub p: usize,
}

impl<B: Backend> Chunk<B> {
    /// Kept frames and their size: `(frames, H, W)`.
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.frames - self.skip, self.h * self.p, self.w * self.p)
    }

    /// Queue the pixel-shuffle, clamp and 8-bit rounding on the device; the
    /// returned buffer downloads as packed RGBA (see [`Chunk::rgba8`]), 4
    /// bytes per pixel instead of 12 for the f32 map.
    pub fn pack(&self, be: &B) -> B::Buf {
        let s = PackShape {
            frames: self.frames - self.skip,
            skip: self.skip,
            h: self.h,
            w: self.w,
            p: self.p,
        };
        let out = be.alloc_words(s.pixels());
        be.pack_rgba8(&self.rgb, None, &s, &out);
        out
    }

    /// `(frames, H, W, RGBA bytes)` from the downloaded [`Chunk::pack`] buffer.
    pub fn rgba8(&self, words: &[f32]) -> (usize, usize, usize, Vec<u8>) {
        let (f, h, w) = self.dims();
        (f, h, w, rgba8_bytes(&words[..f * h * w]))
    }
}

/// Decoded frames of one chunk: `[frames, 3, H, W]`, values ≈ `[0, 1]`.
pub struct Frames {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

fn conv_w<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    key: &str,
    cout: usize,
    cin: usize,
    k: usize,
    bias: bool,
) -> Result<ConvW<B>> {
    Ok(ConvW {
        w: be.weight(
            &src.get(&format!("{key}.weight"), &[cout, cin, k, k])?,
            cout,
            cin * k * k,
        ),
        b: if bias {
            Some(be.upload(&src.get(&format!("{key}.bias"), &[cout])?))
        } else {
            None
        },
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
                grow: conv_w(
                    be,
                    src,
                    &format!("decoder.{}.conv", base + 4),
                    nf[s] * cfg.tgrow[s],
                    nf[s],
                    1,
                    false,
                )?,
                conv: conv_w(
                    be,
                    src,
                    &format!("decoder.{}", base + 5),
                    nf[s + 1],
                    nf[s],
                    3,
                    false,
                )?,
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
            pool: Default::default(),
        }
    }

    /// Decode `frames` latent frames given frame-major (`[frames, C, h, w]`).
    pub fn decode_chunk(
        &self,
        be: &B,
        st: &mut DecodeState<B>,
        z: &[f32],
        frames: usize,
        h: usize,
        w: usize,
    ) -> Frames {
        let c = self.launch_chunk(be, st, z, frames, h, w);
        let raw = be.download(&c.rgb);
        let (oh, ow) = (c.h * c.p, c.w * c.p);
        let mut data = vec![0.0f32; (c.frames - c.skip) * 3 * oh * ow];
        for fi in c.skip..c.frames {
            for ch in 0..3 {
                for y in 0..oh {
                    for xx in 0..ow {
                        let k = ch * c.p * c.p + (y % c.p) * c.p + xx % c.p;
                        data[(((fi - c.skip) * 3 + ch) * oh + y) * ow + xx] =
                            raw[((fi * 3 * c.p * c.p + k) * c.h + y / c.p) * c.w + xx / c.p];
                    }
                }
            }
        }
        Frames {
            frames: c.frames - c.skip,
            height: oh,
            width: ow,
            data,
        }
    }

    /// Queue one chunk's decode on the device and return without waiting:
    /// [`Chunk::rgb8`] after [`Backend::download`] turns it into frames. The
    /// pipeline launches chunk i+1 before converting chunk i, so the host
    /// work overlaps the GPU's.
    pub fn launch_chunk(
        &self,
        be: &B,
        st: &mut DecodeState<B>,
        z: &[f32],
        frames: usize,
        h: usize,
        w: usize,
    ) -> Chunk<B> {
        let cfg = &self.cfg;
        let nf = cfg.widths;
        assert_eq!(z.len(), frames * cfg.latent_channels * h * w);
        let x0 = if cfg.denormalize {
            assert_eq!(
                cfg.latent_channels, 16,
                "LightTAE denormalises Wan 2.1 latents"
            );
            let plane = h * w;
            let mut d = z.to_vec();
            for (i, v) in d.iter_mut().enumerate() {
                let c = (i / plane) % 16;
                *v = *v * fastwan_core::config::WAN21_LATENT_STD[c]
                    + fastwan_core::config::WAN21_LATENT_MEAN[c];
            }
            be.upload(&d)
        } else {
            be.upload(z)
        };
        let (mut f, mut h, mut w) = (frames, h, w);
        be.tanh_clamp(&x0, z.len(), 3.0);
        let shape = |f, cin_a, cin_b, cout, h, w, upsample, k| ConvShape {
            frames: f,
            cin_a,
            cin_b,
            cout,
            h,
            w,
            upsample,
            k,
        };
        // Pooled buffers: contents are stale, so only for outputs that are
        // written in full (every conv output, `keep`, and `past` once a
        // previous chunk exists).
        let pool = std::cell::RefCell::new(std::mem::take(&mut st.pool));
        let take = |n: usize| {
            let reused = pool.borrow_mut().get_mut(&n).and_then(Vec::pop);
            reused.unwrap_or_else(|| be.alloc(n))
        };
        let give = |b: B::Buf| pool.borrow_mut().entry(be.buf_len(&b)).or_default().push(b);
        let conv = |x: &B::Buf,
                    b: Option<&B::Buf>,
                    s: ConvShape,
                    cw: &ConvW<B>,
                    ep: ConvEpilogue<'_, B::Buf>| {
            let out = take(s.out_len());
            be.conv2d(x, b, &s, &cw.w, cw.b.as_ref(), ep, &out);
            out
        };
        let mut x = conv(
            &x0,
            None,
            shape(f, cfg.latent_channels, 0, nf[0], h, w, false, 3),
            &self.first,
            ConvEpilogue::Relu,
        );
        for (si, stage) in self.stages.iter().enumerate() {
            let n = nf[si];
            for (mi, [c0, c2, c4]) in stage.mems.iter().enumerate() {
                let fsz = n * h * w;
                let slot = &mut st.memory[si * cfg.mem_blocks + mi];
                // The first chunk's past frame is zero: a fresh buffer.
                let past = if slot.is_some() {
                    take(f * fsz)
                } else {
                    be.alloc(f * fsz)
                };
                if let Some(prev) = slot.as_ref() {
                    be.copy(prev, 0, &past, 0, fsz);
                }
                if f > 1 {
                    be.copy(&x, 0, &past, fsz, (f - 1) * fsz);
                }
                let keep = take(fsz);
                be.copy(&x, (f - 1) * fsz, &keep, 0, fsz);
                if let Some(old) = slot.replace(keep) {
                    give(old);
                }
                let t1 = conv(
                    &x,
                    Some(&past),
                    shape(f, n, n, n, h, w, false, 3),
                    c0,
                    ConvEpilogue::Relu,
                );
                give(past);
                let t2 = conv(
                    &t1,
                    None,
                    shape(f, n, 0, n, h, w, false, 3),
                    c2,
                    ConvEpilogue::Relu,
                );
                give(t1);
                let nx = conv(
                    &t2,
                    None,
                    shape(f, n, 0, n, h, w, false, 3),
                    c4,
                    ConvEpilogue::AddRelu(&x),
                );
                give(t2);
                give(std::mem::replace(&mut x, nx));
            }
            let stride = cfg.tgrow[si];
            // [f, n·stride, h, w] is bit-for-bit [f·stride, n, h, w].
            let g = conv(
                &x,
                None,
                shape(f, n, 0, n * stride, h, w, false, 1),
                &stage.grow,
                ConvEpilogue::None,
            );
            f *= stride;
            let ep = if si == 2 {
                ConvEpilogue::Relu
            } else {
                ConvEpilogue::None
            };
            give(std::mem::replace(
                &mut x,
                conv(
                    &g,
                    None,
                    shape(f, n, 0, nf[si + 1], h, w, true, 3),
                    &stage.conv,
                    ep,
                ),
            ));
            give(g);
            h *= 2;
            w *= 2;
        }
        let rgb = conv(
            &x,
            None,
            shape(f, nf[3], 0, cfg.out_channels(), h, w, false, 3),
            &self.last,
            ConvEpilogue::None,
        );
        give(x);
        st.pool = pool.into_inner();
        let skip = st.to_trim.min(f);
        st.to_trim -= skip;
        Chunk {
            rgb,
            frames: f,
            skip,
            h,
            w,
            p: cfg.patch,
        }
    }
}
