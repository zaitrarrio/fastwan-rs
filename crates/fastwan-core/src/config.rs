//! Architecture constants. Numbers follow the Diffusers `transformer/config.json`
//! of each checkpoint (and fastvideo-rs `fastvideo-models/src/wan/config.rs`).

/// Wan DiT architecture. Batch is always 1: the DMD-distilled FastWan models
/// run without classifier-free guidance, so a second batch row never exists.
#[derive(Debug, Clone, PartialEq)]
pub struct WanConfig {
    /// Spatial patch (the temporal patch is 1 for every Wan checkpoint).
    pub patch: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_dim: usize,
    /// Prompt tokens the text encoder pads to (zeros past the prompt; the DiT
    /// attends to all of them, there is no mask).
    pub text_len: usize,
    pub freq_dim: usize,
    pub ffn_dim: usize,
    pub layers: usize,
    pub eps: f32,
    pub rope_max_seq_len: usize,
    pub rope_theta: f64,
}

impl WanConfig {
    pub fn dim(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Features per token out of `proj_out` (`patch² · out_channels`).
    pub fn patch_features_out(&self) -> usize {
        self.patch * self.patch * self.out_channels
    }

    /// Features per token into the patch embedding (`patch² · in_channels`).
    pub fn patch_features_in(&self) -> usize {
        self.patch * self.patch * self.in_channels
    }

    /// `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` (and Wan 2.1 T2V 1.3B).
    pub fn fastwan_1_3b() -> Self {
        Self {
            patch: 2,
            heads: 12,
            head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            text_len: 512,
            freq_dim: 256,
            ffn_dim: 8960,
            layers: 30,
            eps: 1e-6,
            rope_max_seq_len: 1024,
            rope_theta: 10000.0,
        }
    }

    /// `Wan-AI/Wan2.1-T2V-14B(-Diffusers)`: dim 5120, 40 heads, 40 layers,
    /// FFN 13824 (LongLive-Plug's 14B backbone).
    pub fn wan21_14b() -> Self {
        Self {
            heads: 40,
            ffn_dim: 13824,
            layers: 40,
            ..Self::fastwan_1_3b()
        }
    }

    /// A graph small enough to run on the CPU reference in a unit test. Head
    /// dim stays a multiple of 6 so the 3-axis RoPE split is non-trivial.
    pub fn tiny() -> Self {
        Self {
            patch: 2,
            heads: 2,
            head_dim: 12,
            in_channels: 4,
            out_channels: 4,
            text_dim: 16,
            text_len: 8,
            freq_dim: 16,
            ffn_dim: 40,
            layers: 2,
            eps: 1e-6,
            rope_max_seq_len: 64,
            rope_theta: 10000.0,
        }
    }

    /// [`WanConfig::tiny`] with head_dim 32: the smallest graph the GPU
    /// VSA / flash kernels take (head_dim a multiple of 16).
    pub fn tiny_vsa() -> Self {
        Self {
            head_dim: 32,
            ..Self::tiny()
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "fastwan-1.3b" | "fastwan21-1.3b" | "FastVideo/FastWan2.1-T2V-1.3B-Diffusers" => {
                Some(Self::fastwan_1_3b())
            }
            "wan21-14b" | "wan2.1-14b" | "Wan-AI/Wan2.1-T2V-14B-Diffusers" => {
                Some(Self::wan21_14b())
            }
            "tiny" => Some(Self::tiny()),
            _ => None,
        }
    }

    /// RoPE split of one head: `(t, h, w)` widths. Diffusers `WanRotaryPosEmbed`:
    /// `h = w = 2·(d/6)`, `t = d − h − w` (44/42/42 for d = 128).
    pub fn rope_split(&self) -> (usize, usize, usize) {
        let hw = 2 * (self.head_dim / 6);
        (self.head_dim - 2 * hw, hw, hw)
    }

    /// Parameter count of the DiT (dense; VSA gates excluded).
    pub fn params(&self) -> usize {
        let d = self.dim();
        let lin = |i: usize, o: usize| i * o + o;
        let block = 4 * lin(d, d) // self q k v o
            + 4 * lin(d, d) // cross q k v o
            + 4 * d // norm_q/norm_k ×2
            + 2 * d // norm2
            + lin(d, self.ffn_dim)
            + lin(self.ffn_dim, d)
            + 6 * d;
        self.layers * block
            + lin(self.patch_features_in(), d)
            + lin(self.freq_dim, d)
            + lin(d, d)
            + lin(d, 6 * d)
            + lin(self.text_dim, d)
            + lin(d, d)
            + lin(d, self.patch_features_out())
            + 2 * d
    }
}

/// TAEHV (madebyollin/taehv) decoder for Wan 2.1 latents: `taew2_1.safetensors`.
#[derive(Debug, Clone, PartialEq)]
pub struct TaeConfig {
    pub latent_channels: usize,
    /// Decoder widths `n_f`.
    pub widths: [usize; 4],
    /// TGrow stride per stage (`decoder_time_upscale = (False, True, True)`).
    pub tgrow: [usize; 3],
    /// MemBlocks per stage.
    pub mem_blocks: usize,
    /// Pixel-shuffle factor after the last conv (1 for taew2_1, 2 for taew2_2).
    pub patch: usize,
    /// Undo the Wan 2.1 VAE latent normalisation (`z·std + mean` per channel)
    /// before the decoder: LightX2V's LightTAE (`lighttaew2_1`) was distilled
    /// on raw VAE latents (its `need_scaled`), TAEHV on normalised ones.
    pub denormalize: bool,
}

/// The Wan 2.1 VAE's per-channel latent statistics (`latents_mean` /
/// `latents_std` of its config; LightX2V's `WanVAE_tiny`).
pub const WAN21_LATENT_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
pub const WAN21_LATENT_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

impl TaeConfig {
    pub fn taew2_1() -> Self {
        Self {
            latent_channels: 16,
            widths: [256, 128, 64, 64],
            tgrow: [1, 2, 2],
            mem_blocks: 3,
            patch: 1,
            denormalize: false,
        }
    }

    /// LightX2V's LightTAE for Wan 2.1 (`lightx2v/Autoencoders`,
    /// `lighttaew2_1.safetensors` / `_v2`, Apache-2.0): TAEHV's network and
    /// tensor names, distilled further, decoding raw (denormalised) latents.
    pub fn lighttaew2_1() -> Self {
        Self {
            denormalize: true,
            ..Self::taew2_1()
        }
    }

    /// `taew2_2.safetensors`: Wan 2.2 latents (TI2V-5B / FastWan2.2-5B), 48
    /// channels, 16× spatial — the same network with a ×2 pixel-shuffle head.
    /// It cannot decode Wan 2.1 latents (different latent space).
    pub fn taew2_2() -> Self {
        Self {
            latent_channels: 48,
            patch: 2,
            ..Self::taew2_1()
        }
    }

    pub fn tiny() -> Self {
        Self {
            latent_channels: 4,
            widths: [8, 6, 4, 4],
            tgrow: [1, 2, 2],
            mem_blocks: 3,
            patch: 1,
            denormalize: false,
        }
    }

    /// Output frames per latent frame.
    pub fn t_upscale(&self) -> usize {
        self.tgrow.iter().product()
    }

    /// Priming frames emitted first and trimmed: `4T − 3` frames for Wan.
    pub fn frames_to_trim(&self) -> usize {
        self.t_upscale() - 1
    }

    pub fn spatial_scale(&self) -> usize {
        8 * self.patch
    }

    /// Channels out of the last conv, before the pixel-shuffle.
    pub fn out_channels(&self) -> usize {
        3 * self.patch * self.patch
    }

    pub fn from_file_name(name: &str) -> Option<Self> {
        let n = name.to_ascii_lowercase();
        // LightX2V's files share TAEHV's names after a `light` prefix; their
        // Wan 2.2 LightTAE is not supported (no Wan 2.2 DiT here).
        if n.contains("lighttaew2_1") {
            Some(Self::lighttaew2_1())
        } else if n.contains("lighttae") {
            None
        } else if n.contains("taew2_1") {
            Some(Self::taew2_1())
        } else if n.contains("taew2_2") {
            Some(Self::taew2_2())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tae_kind_follows_the_file_name() {
        assert_eq!(
            TaeConfig::from_file_name("x/taew2_1.safetensors"),
            Some(TaeConfig::taew2_1())
        );
        let l = TaeConfig::from_file_name("x/lighttaew2_1_v2.safetensors").unwrap();
        assert!(l.denormalize && l.latent_channels == 16);
        assert_eq!(TaeConfig::from_file_name("lighttaew2_2.safetensors"), None);
    }

    #[test]
    fn fastwan_1_3b_is_1_4b_params() {
        let p = WanConfig::fastwan_1_3b().params();
        // Wan2.1-T2V-1.3B reports 1.42 B parameters for the DiT.
        assert!((1_400_000_000..1_440_000_000).contains(&p), "{p}");
    }

    #[test]
    fn rope_split_matches_diffusers() {
        assert_eq!(WanConfig::fastwan_1_3b().rope_split(), (44, 42, 42));
        assert_eq!(WanConfig::tiny().rope_split(), (4, 4, 4));
    }

    #[test]
    fn taehv_frames() {
        let t = TaeConfig::taew2_1();
        assert_eq!(t.t_upscale(), 4);
        assert_eq!(21 * t.t_upscale() - t.frames_to_trim(), 81);
    }
}
