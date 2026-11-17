//! ×4 video upscaler: Real-ESRGAN's compact network (SRVGGNetCompact,
//! checkpoint `realesr-animevideov3`, BSD-3), per frame.
//!
//! conv3×3 3→F, PReLU → `convs` × (conv3×3 F→F, PReLU) → conv3×3 F→3·s²
//! → pixel-shuffle s, plus the input upsampled ×s (nearest).
//!
//! Everything runs on the backend: the convs (the PReLU is a
//! [`ConvEpilogue`]), then `pack_rgba8` does the pixel-shuffle, the residual
//! add, clamp and rounding, so a frame leaves the device as 4 bytes per
//! output pixel (the f32 map was 12, and the host pass ran single-threaded in
//! the browser). Chosen over the alternatives in
//! `docs/runs/2026-11-12-upscalers`: fastest and steadiest learned option,
//! 2.5 MB of weights.

use fastwan_core::{
    Result,
    backend::{Backend, ConvEpilogue, ConvShape, PackShape, rgba8_bytes},
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

/// One batch of upscaled frames on the device, packed RGBA (one word per
/// output pixel; download it and read it with [`UpChunk::rgba8`]).
pub struct UpChunk<B: Backend> {
    pub out: B::Buf,
    /// Held until the caller has downloaded `out`.
    _keep: [B::Buf; 5],
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
        let shape = PackShape {
            frames,
            skip: 0,
            h,
            w,
            p: self.cfg.scale,
        };
        let packed = be.alloc_words(shape.pixels());
        // The residual is the input itself, as exact 8-bit words.
        let words: Vec<u32> = rgb
            .chunks_exact(3)
            .map(|p| p[0] as u32 | (p[1] as u32) << 8 | (p[2] as u32) << 16)
            .collect();
        let base = be.upload_words(&words);
        be.pack_rgba8(&out, Some(&base), &shape, &packed);
        UpChunk {
            out: packed,
            _keep: [input, f0, f1, out, base],
            frames,
            h,
            w,
            scale: self.cfg.scale,
        }
    }
}

impl<B: Backend> UpChunk<B> {
    /// The packed frames, dropping the intermediates. Only once the work is
    /// known to be done: in a browser, dropping a buffer destroys it at once.
    pub fn into_out(self) -> B::Buf {
        self.out
    }

    /// `[frames, H·s, W·s, 4]` RGBA bytes from the downloaded `out`.
    pub fn rgba8(&self, words: &[f32]) -> Vec<u8> {
        let n = self.frames * self.h * self.scale * self.w * self.scale;
        rgba8_bytes(&words[..n])
    }
}
