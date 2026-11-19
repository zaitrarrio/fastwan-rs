//! LoRA adapters for the Wan DiT's block linears, kept as unmerged branches
//! (`y = W x + b + s·B (A x)`): a few-step adapter's delta is small next to
//! the weight (h3-rs measured LongLive-Plug's at |ΔW|/|W| ≈ 2·10⁻⁴), and
//! merging it into the device's bf16 (CUDA) or f16 (WebGPU) weights keeps
//! only the part that survives their rounding.
//!
//! Key names: Wan's original names (`blocks.N.self_attn.q`, `cross_attn.o`,
//! `ffn.0`, `ffn.2`: LongLive-Plug's and LightX2V's files) or the Diffusers
//! names (`blocks.N.attn1.to_q`, …), with or without a `base_model.model.` /
//! `diffusion_model.` / `transformer.` / `model.` prefix (PEFT, ComfyUI), and
//! `lora_A` / `lora_B` (optionally `.default`) or `lora_down` / `lora_up`.
//! The pair scale is `alpha / rank` when the file stores a per-module
//! `.alpha`, else 1 (alpha = rank, as in every LongLive-Plug release), times
//! the adapter's strength (LongLive-Plug: few-step 1.0, CFG 0.5).

use crate::weights::{SafetensorsDir, WeightSource};
use crate::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// One adapter file: its LoRA pairs indexed by Diffusers module name.
pub struct Adapter {
    src: SafetensorsDir,
    /// Diffusers module → (A key, B key, alpha key).
    pairs: HashMap<String, (String, String, Option<String>)>,
    pub strength: f32,
    pub name: String,
}

/// Wan's original module names → Diffusers' (`WanTransformer3DModel`).
fn diffusers_module(m: &str) -> String {
    let m = [
        "base_model.model.",
        "diffusion_model.",
        "transformer.",
        "model.",
    ]
    .iter()
    .fold(m, |m, p| m.strip_prefix(p).unwrap_or(m));
    let Some(rest) = m.strip_prefix("blocks.") else {
        return m.to_string();
    };
    let Some((n, tail)) = rest.split_once('.') else {
        return m.to_string();
    };
    let tail = match tail {
        "self_attn.q" => "attn1.to_q",
        "self_attn.k" => "attn1.to_k",
        "self_attn.v" => "attn1.to_v",
        "self_attn.o" => "attn1.to_out.0",
        "cross_attn.q" => "attn2.to_q",
        "cross_attn.k" => "attn2.to_k",
        "cross_attn.v" => "attn2.to_v",
        "cross_attn.o" => "attn2.to_out.0",
        "ffn.0" => "ffn.net.0.proj",
        "ffn.2" => "ffn.net.2",
        t => t,
    };
    format!("blocks.{n}.{tail}")
}

impl Adapter {
    /// Open `path` (a `.safetensors` file) at `strength`.
    pub fn open(path: &Path, strength: f32) -> Result<Self> {
        let src = SafetensorsDir::open(path)?;
        let mut a_keys: HashMap<String, String> = HashMap::new();
        let mut b_keys: HashMap<String, String> = HashMap::new();
        let mut alpha: HashMap<String, String> = HashMap::new();
        for n in src.names() {
            let (module, which) = if let Some(m) = n
                .strip_suffix(".lora_A.weight")
                .or_else(|| n.strip_suffix(".lora_A.default.weight"))
                .or_else(|| n.strip_suffix(".lora_down.weight"))
            {
                (m, 'a')
            } else if let Some(m) = n
                .strip_suffix(".lora_B.weight")
                .or_else(|| n.strip_suffix(".lora_B.default.weight"))
                .or_else(|| n.strip_suffix(".lora_up.weight"))
            {
                (m, 'b')
            } else if let Some(m) = n.strip_suffix(".alpha") {
                (m, 'r')
            } else {
                continue;
            };
            let d = diffusers_module(module);
            let map = match which {
                'a' => &mut a_keys,
                'b' => &mut b_keys,
                _ => &mut alpha,
            };
            map.insert(d, n.to_string());
        }
        let mut pairs = HashMap::new();
        for (m, a) in a_keys {
            let b = b_keys.remove(&m).ok_or_else(|| {
                Error::Weights(format!(
                    "{}: `{m}` has lora_A but no lora_B",
                    path.display()
                ))
            })?;
            let r = alpha.remove(&m);
            pairs.insert(m, (a, b, r));
        }
        if let Some(m) = b_keys.keys().next() {
            return Err(Error::Weights(format!(
                "{}: `{m}` has lora_B but no lora_A",
                path.display()
            )));
        }
        if pairs.is_empty() {
            return Err(Error::Weights(format!(
                "{}: no LoRA pairs (lora_A / lora_B)",
                path.display()
            )));
        }
        Ok(Self {
            src,
            pairs,
            strength,
            name: path.display().to_string(),
        })
    }

    /// Modules (Diffusers names) the adapter has a pair for.
    pub fn modules(&self) -> impl Iterator<Item = &str> {
        self.pairs.keys().map(String::as_str)
    }

    pub fn touches(&self, module: &str) -> bool {
        self.pairs.contains_key(module)
    }

    /// The pair on `module` (a `[out, inp]` linear) as `(A [r, inp],
    /// s·B [out, r], r)` with the scale folded into B, or `None`.
    pub fn factors(
        &self,
        module: &str,
        out: usize,
        inp: usize,
    ) -> Result<Option<(Vec<f32>, Vec<f32>, usize)>> {
        let Some((ak, bk, alk)) = self.pairs.get(module) else {
            return Ok(None);
        };
        let (_, shape, _) = self.src.raw(ak)?;
        let r = *shape
            .first()
            .ok_or_else(|| Error::Weights(format!("`{ak}` has no shape")))?;
        let a = self.src.get(ak, &[r, inp])?;
        let mut b = self.src.get(bk, &[out, r])?;
        let alpha = match alk {
            Some(k) => self.src.get(k, &[1])?[0] / r as f32,
            None => 1.0,
        };
        let s = alpha * self.strength;
        if s != 1.0 {
            b.iter_mut().for_each(|x| *x *= s);
        }
        Ok(Some((a, b, r)))
    }
}

/// `"path"` or `"path:strength"` (a Windows drive letter is not a strength).
pub fn parse_spec(spec: &str) -> Result<(std::path::PathBuf, f32)> {
    match spec.rsplit_once(':') {
        Some((p, s)) if !p.is_empty() && s.parse::<f32>().is_ok() => {
            Ok((p.into(), s.parse().expect("checked")))
        }
        _ => Ok((spec.into(), 1.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{f32_bytes, write};

    #[test]
    fn names_map_to_diffusers() {
        for (wan, diff) in [
            ("blocks.3.self_attn.q", "blocks.3.attn1.to_q"),
            ("blocks.3.self_attn.o", "blocks.3.attn1.to_out.0"),
            (
                "base_model.model.blocks.0.cross_attn.v",
                "blocks.0.attn2.to_v",
            ),
            (
                "diffusion_model.blocks.39.ffn.0",
                "blocks.39.ffn.net.0.proj",
            ),
            ("blocks.1.ffn.2", "blocks.1.ffn.net.2"),
            ("blocks.1.attn1.to_k", "blocks.1.attn1.to_k"),
        ] {
            assert_eq!(diffusers_module(wan), diff, "{wan}");
        }
        assert_eq!(
            parse_spec("a/b.safetensors:0.5").unwrap(),
            ("a/b.safetensors".into(), 0.5)
        );
        assert_eq!(
            parse_spec("a/b.safetensors").unwrap(),
            ("a/b.safetensors".into(), 1.0)
        );
    }

    #[test]
    fn factors_fold_alpha_and_strength() {
        let dir = std::env::temp_dir().join(format!("fw-lora-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.safetensors");
        let (r, inp, out) = (2, 3, 4);
        let a: Vec<f32> = (0..r * inp).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..out * r).map(|i| 1.0 + i as f32).collect();
        write(
            &p,
            &[
                (
                    "blocks.0.self_attn.q.lora_A.weight".into(),
                    safetensors::Dtype::F32,
                    vec![r, inp],
                    f32_bytes(&a),
                ),
                (
                    "blocks.0.self_attn.q.lora_B.weight".into(),
                    safetensors::Dtype::F32,
                    vec![out, r],
                    f32_bytes(&b),
                ),
                (
                    "blocks.0.self_attn.q.alpha".into(),
                    safetensors::Dtype::F32,
                    vec![1],
                    f32_bytes(&[4.0]),
                ),
                (
                    "blocks.0.ffn.2.lora_A.weight".into(),
                    safetensors::Dtype::F32,
                    vec![r, inp],
                    f32_bytes(&a),
                ),
                (
                    "blocks.0.ffn.2.lora_B.weight".into(),
                    safetensors::Dtype::F32,
                    vec![out, r],
                    f32_bytes(&b),
                ),
            ],
            Default::default(),
        )
        .unwrap();
        let ad = Adapter::open(&p, 0.5).unwrap();
        assert!(ad.touches("blocks.0.attn1.to_q") && ad.touches("blocks.0.ffn.net.2"));
        // alpha 4 / rank 2 = 2, times strength 0.5 = 1.
        let (fa, fb, rr) = ad
            .factors("blocks.0.attn1.to_q", out, inp)
            .unwrap()
            .unwrap();
        assert_eq!((rr, fa, fb), (r, a.clone(), b.clone()));
        // No alpha: strength only.
        let (_, fb, _) = ad.factors("blocks.0.ffn.net.2", out, inp).unwrap().unwrap();
        assert_eq!(fb, b.iter().map(|x| x * 0.5).collect::<Vec<_>>());
        assert!(
            ad.factors("blocks.0.attn1.to_k", out, inp)
                .unwrap()
                .is_none()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
