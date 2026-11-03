//! Prompt embeddings → DMD denoise → TAEHV decode → RGB frames.

use crate::{dit::WanDit, taehv::TaeHv};
use fastwan_core::{Result, backend::Backend, geometry::Geometry, rng::Rng, schedule::DmdSchedule};
use std::time::Instant;

pub struct Pipeline<B: Backend> {
    pub dit: WanDit<B>,
    pub tae: TaeHv<B>,
    pub schedule: DmdSchedule,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub geometry: Geometry,
    pub seed: u64,
    /// Tokens per FFN chunk (from `fastwan_core::plan`); `usize::MAX` = all.
    pub ffn_chunk: usize,
    /// Latent frames per TAEHV chunk.
    pub tae_chunk: usize,
    /// `None`: VSA on a VSA checkpoint, dense otherwise.
    pub attention: Option<crate::dit::Attention>,
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub prompt_s: f64,
    pub steps_s: Vec<f64>,
    pub decode_s: f64,
}

impl Timings {
    pub fn denoise_s(&self) -> f64 {
        self.steps_s.iter().sum()
    }
}

/// Progress of a generation, in order: `Prepared`, one `Step` per DMD step,
/// then `Frames` once per decoded chunk (8-bit RGB, `count` frames starting
/// at frame `first`).
pub enum Event<'a> {
    Prepared {
        secs: f64,
    },
    Step {
        index: usize,
        total: usize,
        secs: f64,
    },
    Frames {
        first: usize,
        count: usize,
        width: usize,
        height: usize,
        rgb: &'a [u8],
    },
}

/// `frames × height × width × RGB`, 8-bit.
pub struct Video {
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl<B: Backend> Pipeline<B> {
    /// Denoised latent `[C, T, H, W]` on the host.
    pub fn denoise(
        &self,
        be: &B,
        embeds: &[f32],
        opt: &Options,
        t: &mut Timings,
    ) -> Result<Vec<f32>> {
        self.denoise_with(be, embeds, opt, t, &mut |_| {})
    }

    pub fn denoise_with(
        &self,
        be: &B,
        embeds: &[f32],
        opt: &Options,
        t: &mut Timings,
        on: &mut dyn FnMut(Event<'_>),
    ) -> Result<Vec<f32>> {
        let cfg = &self.dit.cfg;
        opt.geometry.validate(cfg)?;
        let n = opt.geometry.latent_len(cfg);
        let mut rng = Rng::new(opt.seed);

        let t0 = Instant::now();
        let prompt = self.dit.prepare_prompt(be, embeds);
        let mods: Vec<_> = self
            .schedule
            .timesteps
            .iter()
            .map(|&ts| self.dit.step_mods(be, ts as f32))
            .collect();
        let ws = self
            .dit
            .workspace(be, opt.geometry.latent(), opt.ffn_chunk, opt.attention);
        be.sync();
        t.prompt_s = t0.elapsed().as_secs_f64();
        on(Event::Prepared { secs: t.prompt_s });

        let x = be.upload(&rng.normal(n));
        let v = be.alloc(n);
        let noise = be.alloc(n);
        for (i, m) in mods.iter().enumerate() {
            let t0 = Instant::now();
            self.dit.forward(be, &ws, &x, m, &prompt, &v);
            let (st, next) = self.schedule.coeffs(i);
            if next.is_some() {
                be.write(&noise, &rng.normal(n));
            }
            be.dmd_step(&x, &v, next.map(|_| &noise), st, next.unwrap_or(0.0), n);
            be.sync();
            t.steps_s.push(t0.elapsed().as_secs_f64());
            on(Event::Step {
                index: i,
                total: mods.len(),
                secs: t0.elapsed().as_secs_f64(),
            });
        }
        Ok(be.download(&x))
    }

    /// TAEHV decode of a `[C, T, H, W]` latent.
    pub fn decode(
        &self,
        be: &B,
        latent: &[f32],
        g: &Geometry,
        chunk: usize,
        t: &mut Timings,
    ) -> Video {
        self.decode_with(be, latent, g, chunk, t, &mut |_| {})
    }

    pub fn decode_with(
        &self,
        be: &B,
        latent: &[f32],
        g: &Geometry,
        chunk: usize,
        t: &mut Timings,
        on: &mut dyn FnMut(Event<'_>),
    ) -> Video {
        let t0 = Instant::now();
        let (lt, lh, lw) = g.latent();
        let c = self.tae.cfg.latent_channels;
        let mut st = self.tae.state();
        let mut out = Vec::new();
        let (mut width, mut height, mut frames) = (0, 0, 0);
        let chunk = chunk.max(1);
        let plane = lh * lw;
        let mut f0 = 0;
        while f0 < lt {
            let nfr = chunk.min(lt - f0);
            // [C, T, h, w] → frame-major [nfr, C, h, w] for this chunk.
            let mut z = Vec::with_capacity(nfr * c * plane);
            for f in f0..f0 + nfr {
                for ch in 0..c {
                    z.extend_from_slice(&latent[(ch * lt + f) * plane..][..plane]);
                }
            }
            let fr = self.tae.decode_chunk(be, &mut st, &z, nfr, lh, lw);
            (width, height) = (fr.width, fr.height);
            frames += fr.frames;
            let px = fr.width * fr.height;
            let start = out.len();
            for f in 0..fr.frames {
                for i in 0..px {
                    for ch in 0..3 {
                        let v = fr.data[(f * 3 + ch) * px + i];
                        out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                    }
                }
            }
            on(Event::Frames {
                first: frames - fr.frames,
                count: fr.frames,
                width,
                height,
                rgb: &out[start..],
            });
            f0 += nfr;
        }
        t.decode_s = t0.elapsed().as_secs_f64();
        Video {
            frames,
            width,
            height,
            rgb: out,
        }
    }

    pub fn generate(&self, be: &B, embeds: &[f32], opt: &Options) -> Result<(Video, Timings)> {
        self.generate_with(be, embeds, opt, &mut |_| {})
    }

    /// [`Pipeline::generate`], reporting progress to `on` as it goes.
    pub fn generate_with(
        &self,
        be: &B,
        embeds: &[f32],
        opt: &Options,
        on: &mut dyn FnMut(Event<'_>),
    ) -> Result<(Video, Timings)> {
        let mut t = Timings::default();
        let latent = self.denoise_with(be, embeds, opt, &mut t, on)?;
        let video = self.decode_with(be, &latent, &opt.geometry, opt.tae_chunk, &mut t, on);
        Ok((video, t))
    }
}
