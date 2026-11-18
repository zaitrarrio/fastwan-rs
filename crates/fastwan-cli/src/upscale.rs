//! Upscaling for `generate` and `serve`: the Real-ESRGAN compact model at ×2
//! or ×4, fed packed frames on the device.
use anyhow::{Result, bail};
use fastwan_core::{
    Backend,
    weights::{SafetensorsDir, missing},
};
use std::path::Path;

/// The ×4 Real-ESRGAN compact model from `path`, set to output ×`x` (2 or 4).
pub(crate) fn load<B: Backend>(
    be: &B,
    path: &Path,
    x: usize,
) -> Result<fastwan_model::upscale::Upscaler<B>> {
    use fastwan_model::upscale::{UpConfig, Upscaler, up_manifest};
    let ucfg = UpConfig::anime_video_v3();
    let usrc = SafetensorsDir::open(path)?;
    let miss = missing(&usrc, &up_manifest(&ucfg));
    if !miss.is_empty() {
        bail!(
            "{} is not realesr-animevideov3 (missing {:?})",
            path.display(),
            &miss[..miss.len().min(3)]
        );
    }
    let mut up = Upscaler::load(be, ucfg, &usrc)?;
    up.set_output_scale(x)?;
    Ok(up)
}

/// Upscale `frames` packed RGBA frames on the device (`[frames, h, w]`
/// words) and return them as RGB bytes. Batches of a few frames (the
/// activations are 64 channels at full size), pipelined like the decode:
/// batch i+1 is queued before batch i is read back and converted.
pub(crate) fn frames<B: Backend>(
    be: &B,
    up: &fastwan_model::upscale::Upscaler<B>,
    words: &B::Buf,
    frames: usize,
    h: usize,
    w: usize,
) -> Vec<u8> {
    let s = up.output_scale();
    let batch = up.batch(h, w, 1 << 30).min(8);
    let plane = h * w;
    let launch = |f0: usize| {
        let n = batch.min(frames - f0);
        let part = be.alloc_words(n * plane);
        be.copy(words, f0 * plane, &part, 0, n * plane);
        let c = up.launch_words(be, &part, n, h, w);
        (part, c)
    };
    let starts: Vec<usize> = (0..frames).step_by(batch).collect();
    let mut rgb = Vec::with_capacity(frames * plane * s * s * 3);
    let mut cur = launch(starts[0]);
    for i in 0..starts.len() {
        let next = starts.get(i + 1).map(|&f0| launch(f0));
        let rgba = be.download_rgba8(&cur.1.out);
        rgb.extend(fastwan_core::backend::rgba_to_rgb(&rgba));
        if let Some(n) = next {
            cur = n;
        }
    }
    rgb
}
