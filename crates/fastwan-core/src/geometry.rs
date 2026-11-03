//! Pixel ↔ latent ↔ token shapes for a Wan 2.1 clip.

use crate::{Error, Result, config::WanConfig};

/// Wan 2.1 VAE: 8× spatial, 4× temporal with a causal first frame.
pub const VAE_SPATIAL: usize = 8;
pub const VAE_TEMPORAL: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    /// Pixel frames, `4n + 1`.
    pub frames: usize,
}

impl Geometry {
    /// FastWan 2.1 1.3B's served default: 832×480, 81 frames (5 s at 16 fps).
    pub const FASTWAN_480P: Self = Self {
        width: 832,
        height: 480,
        frames: 81,
    };

    pub fn new(width: usize, height: usize, frames: usize) -> Result<Self> {
        let g = Self {
            width,
            height,
            frames,
        };
        g.validate(&WanConfig::fastwan_1_3b())?;
        Ok(g)
    }

    pub fn validate(&self, cfg: &WanConfig) -> Result<()> {
        let cell = VAE_SPATIAL * cfg.patch;
        if self.width == 0 || self.height == 0 || self.width % cell != 0 || self.height % cell != 0
        {
            return Err(Error::Geometry(format!(
                "{}x{} is not a positive multiple of {cell}",
                self.width, self.height
            )));
        }
        if self.frames == 0 || (self.frames - 1) % VAE_TEMPORAL != 0 {
            return Err(Error::Geometry(format!(
                "{} frames is not 4n+1",
                self.frames
            )));
        }
        Ok(())
    }

    /// Latent `(T, H, W)`.
    pub fn latent(&self) -> (usize, usize, usize) {
        (
            (self.frames - 1) / VAE_TEMPORAL + 1,
            self.height / VAE_SPATIAL,
            self.width / VAE_SPATIAL,
        )
    }

    /// Token grid `(T, H/p, W/p)`.
    pub fn grid(&self, cfg: &WanConfig) -> (usize, usize, usize) {
        let (t, h, w) = self.latent();
        (t, h / cfg.patch, w / cfg.patch)
    }

    pub fn tokens(&self, cfg: &WanConfig) -> usize {
        let (t, h, w) = self.grid(cfg);
        t * h * w
    }

    pub fn latent_len(&self, cfg: &WanConfig) -> usize {
        let (t, h, w) = self.latent();
        cfg.in_channels * t * h * w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastwan_480p_shapes() {
        let g = Geometry::FASTWAN_480P;
        let cfg = WanConfig::fastwan_1_3b();
        assert_eq!(g.latent(), (21, 60, 104));
        assert_eq!(g.grid(&cfg), (21, 30, 52));
        assert_eq!(g.tokens(&cfg), 32760);
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(Geometry::new(830, 480, 81).is_err());
        assert!(Geometry::new(832, 480, 80).is_err());
        assert!(Geometry::new(832, 480, 1).is_ok());
    }
}
