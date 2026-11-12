//! ×4 video upscaler: Real-ESRGAN's compact network (SRVGGNetCompact,
//! checkpoint `realesr-animevideov3`, BSD-3), per frame.
//!
//! conv3×3 3→F, PReLU → `convs` × (conv3×3 F→F, PReLU) → conv3×3 F→3·s²
//! → pixel-shuffle s, plus the input upsampled ×s (nearest).
//!
//! Every conv runs on the backend (the PReLU is a [`ConvEpilogue`]); the
//! pixel-shuffle, residual add, clamp and rounding run on the host in one
//! pass over the downloaded `[frames, 3·s², h, w]`, like
//! [`crate::taehv::Chunk::rgb8`]. Chosen over the alternatives in
//! `docs/runs/2026-11-12-upscalers`: fastest and steadiest learned option,
//! 2.5 MB of weights.

use fastwan_core::{
    Result,
    backend::{Backend, ConvEpilogue, ConvShape},
    weights::WeightSource,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpConfig {
    /// Feature channels.
    pub feat: usize,
    /// Body convs after the first one.
    pub convs: usize,
    pub scale: usize,
}

impl UpConfig {
    /// `realesr-animevideov3`.
    pub fn anime_video_v3() -> Self {
        Self {
            feat: 64,
            convs: 16,
            scale: 4,
        }
    }
    pub fn tiny() -> Self {
        Self {
            feat: 8,
            convs: 2,
            scale: 2,
        }
    }
    fn layers(&self) -> Vec<(usize, usize, bool)> {
        let mut v = vec![(3, self.feat, true)];
        v.extend((0..self.convs).map(|_| (self.feat, self.feat, true)));
        v.push((self.feat, 3 * self.scale * self.scale, false));
        v
    }
}

/// Checkpoint keys and shapes (the `nn.Sequential` positions of `body`).
pub fn up_manifest(cfg: &UpConfig) -> Vec<(String, Vec<usize>)> {
    let mut v = Vec::new();
    for (i, (cin, cout, prelu)) in cfg.layers().into_iter().enumerate() {
        v.push((format!("body.{}.weight", 2 * i), vec![cout, cin, 3, 3]));
        v.push((format!("body.{}.bias", 2 * i), vec![cout]));
        if prelu {
            v.push((format!("body.{}.weight", 2 * i + 1), vec![cout]));
        }
    }
    v
}

struct Layer<B: Backend> {
    cin: usize,
    cout: usize,
    w: B::Weight,
    b: B::Buf,
    slope: Option<B::Buf>,
}

pub struct Upscaler<B: Backend> {
    pub cfg: UpConfig,
    layers: Vec<Layer<B>>,
}

/// One batch of frames on the device: `[frames, 3·s², h, w]`.
pub struct UpChunk<B: Backend> {
    pub out: B::Buf,
    /// Held until the caller has downloaded `out`.
    _keep: [B::Buf; 3],
    pub frames: usize,
    pub h: usize,
    pub w: usize,
    pub scale: usize,
}

impl<B: Backend> Upscaler<B> {
    pub fn load(be: &B, cfg: UpConfig, src: &dyn WeightSource) -> Result<Self> {
        let mut layers = Vec::new();
        for (i, (cin, cout, prelu)) in cfg.layers().into_iter().enumerate() {
            let w = src.get(&format!("body.{}.weight", 2 * i), &[cout, cin, 3, 3])?;
            layers.push(Layer {
                cin,
                cout,
                w: be.weight(&w, cout, cin * 9),
                b: be.upload(&src.get(&format!("body.{}.bias", 2 * i), &[cout])?),
                slope: if prelu {
                    Some(be.upload(&src.get(&format!("body.{}.weight", 2 * i + 1), &[cout])?))
                } else {
                    None
                },
            });
        }
        Ok(Self { cfg, layers })
    }

    /// Frames per batch so the widest activation (`feat` channels) stays
    /// within `max_bytes` (a device's storage-binding limit); at least 1.
    pub fn batch(&self, h: usize, w: usize, max_bytes: u64) -> usize {
        let per = (self.cfg.feat.max(3 * self.cfg.scale * self.cfg.scale) * h * w * 4) as u64;
        (max_bytes / per).max(1) as usize
    }

    /// Queue the network on `frames` interleaved 8-bit RGB frames
    /// (`[frames, h, w, 3]`). Download `out` and finish with [`rgb8`].
    pub fn launch(&self, be: &B, rgb: &[u8], frames: usize, h: usize, w: usize) -> UpChunk<B> {
        assert_eq!(rgb.len(), frames * h * w * 3);
        let plane = h * w;
        let mut x = vec![0f32; frames * 3 * plane];
        for f in 0..frames {
            for p in 0..plane {
                for c in 0..3 {
                    x[(f * 3 + c) * plane + p] = rgb[(f * plane + p) * 3 + c] as f32 / 255.0;
                }
            }
        }
        let input = be.upload(&x);
        // Ping-pong between two feature buffers; the last conv writes `out`.
        let feat = [
            be.alloc(frames * self.cfg.feat * plane),
            be.alloc(frames * self.cfg.feat * plane),
        ];
        let n = self.layers.len();
        let out = be.alloc(frames * 3 * self.cfg.scale * self.cfg.scale * plane);
        for (i, l) in self.layers.iter().enumerate() {
            let dst = if i + 1 == n { &out } else { &feat[i % 2] };
            let s = ConvShape {
                frames,
                cin_a: l.cin,
                cin_b: 0,
                cout: l.cout,
                h,
                w,
                upsample: false,
                k: 3,
            };
            let ep = match &l.slope {
                Some(sl) => ConvEpilogue::Prelu(sl),
                None => ConvEpilogue::None,
            };
            let src = if i == 0 { &input } else { &feat[(i - 1) % 2] };
            be.conv2d(src, None, &s, &l.w, Some(&l.b), ep, dst);
        }
        let [f0, f1] = feat;
        UpChunk {
            out,
            _keep: [input, f0, f1],
            frames,
            h,
            w,
            scale: self.cfg.scale,
        }
    }
}

/// `[frames, H·s, W·s, 3]` 8-bit RGB from the downloaded network output and
/// the input frames: pixel-shuffle + nearest-upsampled input, clamped.
pub fn rgb8(raw: &[f32], rgb: &[u8], frames: usize, h: usize, w: usize, s: usize) -> Vec<u8> {
    let (oh, ow) = (h * s, w * s);
    let mut out = vec![0u8; frames * oh * ow * 3];
    let row = ow * 3;
    let fill = |r0: usize, dst: &mut [u8]| {
        for (ri, line) in dst.chunks_mut(row).enumerate() {
            let (f, y) = ((r0 + ri) / oh, (r0 + ri) % oh);
            let (sy, iy) = (y / s, y % s);
            for x in 0..ow {
                let (sx, ix) = (x / s, x % s);
                let base = &rgb[((f * h + sy) * w + sx) * 3..][..3];
                for c in 0..3 {
                    let k = c * s * s + iy * s + ix;
                    let v = raw[((f * 3 * s * s + k) * h + sy) * w + sx] + base[c] as f32 / 255.0;
                    line[x * 3 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
    };
    let rows = frames * oh;
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(16);
    if threads <= 1 || rows < 64 || cfg!(target_arch = "wasm32") {
        fill(0, &mut out);
    } else {
        let per = rows.div_ceil(threads);
        std::thread::scope(|sc| {
            for (i, part) in out.chunks_mut(per * row).enumerate() {
                let fill = &fill;
                sc.spawn(move || fill(i * per, part));
            }
        });
    }
    out
}
