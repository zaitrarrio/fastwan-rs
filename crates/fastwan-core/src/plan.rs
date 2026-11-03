//! Memory plan: what a generation keeps resident, for a given device, before
//! anything is allocated. Desktop GPUs (8–32 GB) and WebGPU adapters (a
//! per-binding size cap, often 128 MiB by default and 1–4 GiB when requested)
//! are the targets, so the two knobs are the FFN token chunk — the only
//! activation that scales with `ffn_dim` — and the TAEHV frame chunk.

use crate::{config::{TaeConfig, WanConfig}, geometry::Geometry};

#[derive(Debug, Clone, Copy)]
pub struct Device {
    pub budget_bytes: u64,
    /// Largest single buffer binding the device allows.
    pub max_binding_bytes: u64,
    /// Bytes per stored weight element (2 for f16/bf16).
    pub weight_bytes: u64,
    /// Bytes per activation element (4 for f32, 2 for bf16).
    pub act_bytes: u64,
}

impl Device {
    pub fn gib(budget: f64, weight_bytes: u64, act_bytes: u64) -> Self {
        Self {
            budget_bytes: (budget * (1u64 << 30) as f64) as u64,
            max_binding_bytes: u64::MAX,
            weight_bytes,
            act_bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub tokens: usize,
    pub ffn_chunk: usize,
    pub tae_chunk: usize,
    pub weights: u64,
    pub dit_activations: u64,
    pub cross_kv: u64,
    pub decode_peak: u64,
    /// Largest single buffer the plan binds.
    pub largest_binding: u64,
    pub peak: u64,
    pub fits: bool,
    pub notes: Vec<String>,
}

/// Activation bytes of the DiT at `ffn_chunk`, excluding weights and the
/// cross-attention cache.
fn dit_acts(cfg: &WanConfig, s: u64, latent: u64, chunk: u64, a: u64) -> u64 {
    let d = cfg.dim() as u64;
    let elems = s * d // residual stream h
        + s * d // normed
        + s * 3 * d // fused qkv projection
        + 3 * s * d // q, k, v as [heads, S, d]
        + s * d // attention output (merged heads)
        + s * d // to_out / cross / ffn result
        + chunk * cfg.ffn_dim as u64 // FFN hidden, one chunk
        + s * cfg.patch_features_out().max(cfg.patch_features_in()) as u64
        + 3 * latent; // x, v, noise
    let rope = s * cfg.head_dim as u64 * 4; // cos+sin, f32, d/2 each
    elems * a + rope
}

pub fn plan(cfg: &WanConfig, tae: &TaeConfig, g: &Geometry, dev: &Device) -> Plan {
    let s = g.tokens(cfg) as u64;
    let latent = g.latent_len(cfg) as u64;
    let a = dev.act_bytes;
    let d = cfg.dim() as u64;
    let weights = cfg.params() as u64 * dev.weight_bytes;
    let cross_kv = cfg.layers as u64 * 2 * cfg.text_len as u64 * d * a;
    let mut notes = Vec::new();

    // Largest FFN chunk (multiple of 256 tokens, or all of S) that fits both
    // the binding cap and the budget.
    let fixed = weights + cross_kv;
    let mut chunk = s;
    let fits_chunk = |c: u64| {
        c * cfg.ffn_dim as u64 * a <= dev.max_binding_bytes
            && fixed + dit_acts(cfg, s, latent, c, a) <= dev.budget_bytes
    };
    while chunk > 256 && !fits_chunk(chunk) {
        chunk = (chunk / 2).div_ceil(256) * 256;
    }
    let dit_activations = dit_acts(cfg, s, latent, chunk, a);
    if chunk < s {
        notes.push(format!("FFN runs in {} chunks of {chunk} tokens", s.div_ceil(chunk)));
    }

    // TAEHV: the last stage runs at 4 frames per latent frame, 8× spatial,
    // `n_f[3]` channels; a MemBlock holds x, past, two temporaries and the
    // output. Choose the largest latent chunk that fits next to the weights
    // (the DiT activations are freed before decoding).
    let (t, lh, lw) = g.latent();
    let px = (lh * lw) as u64;
    let nf = tae.widths.map(|w| w as u64);
    let per_latent_peak = |c: u64| {
        // Elements per stage for `c` latent frames: stage 0 at latent
        // resolution, then ×2 spatial per stage, ×tgrow frames per stage.
        let s0 = c * nf[0] * px;
        let s1 = c * tae.tgrow[0] as u64 * nf[1] * 4 * px;
        let s2 = c * (tae.tgrow[0] * tae.tgrow[1]) as u64 * nf[2] * 16 * px;
        let out = c * tae.t_upscale() as u64 * nf[3] * 64 * px;
        // A MemBlock holds x, past, two temporaries and its output; the last
        // conv pair holds its input, the 64-channel map and the RGB output.
        (5 * s0).max(5 * s1).max(5 * s2).max(s2 + 2 * out) * a
    };
    let mut tae_chunk = (t as u64).min(4);
    while tae_chunk > 1 && weights + per_latent_peak(tae_chunk) > dev.budget_bytes {
        tae_chunk -= 1;
    }
    let decode_peak = per_latent_peak(tae_chunk);
    let largest_binding = (s * 3 * d * a)
        .max(chunk * cfg.ffn_dim as u64 * a)
        .max(per_latent_peak(tae_chunk) / 2);
    if largest_binding > dev.max_binding_bytes {
        notes.push(format!(
            "largest buffer {} MiB exceeds the device binding cap {} MiB",
            largest_binding >> 20,
            dev.max_binding_bytes >> 20
        ));
    }
    let peak = (fixed + dit_activations).max(weights + decode_peak);
    Plan {
        tokens: s as usize,
        ffn_chunk: chunk as usize,
        tae_chunk: tae_chunk as usize,
        weights,
        dit_activations,
        cross_kv,
        decode_peak,
        largest_binding,
        peak,
        fits: peak <= dev.budget_bytes && largest_binding <= dev.max_binding_bytes,
        notes,
    }
}

impl std::fmt::Display for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        writeln!(f, "tokens          {}", self.tokens)?;
        writeln!(f, "weights         {:>9.1} MiB", mib(self.weights))?;
        writeln!(f, "cross-attn K/V  {:>9.1} MiB", mib(self.cross_kv))?;
        writeln!(f, "DiT activations {:>9.1} MiB (FFN chunk {})", mib(self.dit_activations), self.ffn_chunk)?;
        writeln!(f, "decode peak     {:>9.1} MiB (TAEHV chunk {} latent frames)", mib(self.decode_peak), self.tae_chunk)?;
        writeln!(f, "largest buffer  {:>9.1} MiB", mib(self.largest_binding))?;
        writeln!(f, "peak            {:>9.1} MiB  fits: {}", mib(self.peak), self.fits)?;
        for n in &self.notes {
            writeln!(f, "note: {n}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_480p_fits_a_16gb_card() {
        let p = plan(
            &WanConfig::fastwan_1_3b(),
            &TaeConfig::taew2_1(),
            &Geometry::FASTWAN_480P,
            &Device::gib(16.0, 2, 2),
        );
        assert!(p.fits, "{p}");
        assert_eq!(p.ffn_chunk, p.tokens);
    }

    #[test]
    fn small_budget_chunks_the_ffn() {
        let mut dev = Device::gib(8.0, 2, 4);
        dev.max_binding_bytes = 1 << 30;
        let p = plan(&WanConfig::fastwan_1_3b(), &TaeConfig::taew2_1(), &Geometry::FASTWAN_480P, &dev);
        assert!(p.ffn_chunk < p.tokens, "{p}");
        assert!(p.ffn_chunk as u64 * 8960 * 4 <= 1 << 30);
    }
}
