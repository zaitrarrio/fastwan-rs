//! Checkpoint access: Diffusers safetensors (mmap'd, any of F32/F16/BF16, any
//! number of shards) or deterministic synthetic weights of the same names and
//! shapes (benchmarks and parity tests without a download).
//!
//! The model asks for every tensor by its exact Diffusers key and shape. A
//! missing key or a shape mismatch is an error: a silently-defaulted weight
//! streams gray haze instead of failing (strobe learned this with the
//! Self-Forcing checkpoint remap).

use crate::{
    Error, Result,
    config::{TaeConfig, WanConfig},
    rng::Rng,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub trait WeightSource {
    /// The tensor as f32, checked against `shape` (size-1 dims are ignored,
    /// so a Conv3d `[o, i, 1, 2, 2]` may be asked for as `[o, i, 2, 2]`).
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>>;
    fn contains(&self, name: &str) -> bool;
    /// The linear `key` (no `.weight`) if the checkpoint stores it quantized
    /// (`<key>.weight.q8` / `.q4` + `.scale` [+ `.zero`], see
    /// [`crate::export`]); `None` for a plain tensor.
    fn stored(&self, _key: &str) -> Result<Option<crate::quant::Stored>> {
        Ok(None)
    }
    /// Whether the linear `key` (no `.weight`) is present, plain or quantized.
    fn has_linear(&self, key: &str) -> bool {
        self.contains(&format!("{key}.weight"))
            || self.contains(&format!("{key}.weight.q8"))
            || self.contains(&format!("{key}.weight.q4"))
    }
}

fn squeeze(s: &[usize]) -> Vec<usize> {
    s.iter().copied().filter(|&d| d != 1).collect()
}

/// A safetensors file's bytes: mapped from disk, or owned (a buffer fetched
/// by the browser).
enum Blob {
    Map(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Blob::Map(m) => m,
            Blob::Owned(v) => v,
        }
    }
}

/// All `*.safetensors` under a directory (or one file), or one in memory.
pub struct SafetensorsDir {
    maps: Vec<Blob>,
    index: HashMap<String, usize>,
}

impl SafetensorsDir {
    /// A complete safetensors file held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let index = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| Error::Weights(format!("in-memory safetensors: {e}")))?
            .names()
            .into_iter()
            .map(|n| (n.to_string(), 0))
            .collect();
        Ok(Self {
            maps: vec![Blob::Owned(bytes)],
            index,
        })
    }

    pub fn open(path: &Path) -> Result<Self> {
        let files: Vec<PathBuf> = if path.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(path)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            v.sort();
            v
        } else {
            vec![path.to_path_buf()]
        };
        if files.is_empty() {
            return Err(Error::Weights(format!(
                "no .safetensors in {}",
                path.display()
            )));
        }
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let file = std::fs::File::open(f)?;
            // SAFETY-free: memmap2's `map` is safe to call; the file must not be
            // truncated while mapped, which is the usual contract for weights.
            let map = map_readonly(&file)?;
            let st = safetensors::SafeTensors::deserialize(&map)
                .map_err(|e| Error::Weights(format!("{}: {e}", f.display())))?;
            for name in st.names() {
                index.insert(name.to_string(), i);
            }
            maps.push(Blob::Map(map));
        }
        Ok(Self { maps, index })
    }

    /// The stored bytes of a tensor, without conversion: `(dtype, shape, data)`.
    /// Lets a host-side model (the UMT5 encoder) read bf16 weights straight
    /// out of the page cache instead of materialising an f32 copy.
    pub fn raw(&self, name: &str) -> Result<(safetensors::Dtype, Vec<usize>, &[u8])> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
        let st = safetensors::SafeTensors::deserialize(&self.maps[i])
            .map_err(|e| Error::Weights(e.to_string()))?;
        let t = st.tensor(name).map_err(|e| Error::Weights(e.to_string()))?;
        Ok((t.dtype(), t.shape().to_vec(), t.data()))
    }

    pub fn contains_raw(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }
}

impl SafetensorsDir {
    /// The Wan architecture a checkpoint holds, read from its shapes: width
    /// from `patch_embedding`, depth from the `blocks.N` indices, FFN width
    /// from block 0 (plain or stored int8 / int4), text width from the text
    /// embedder. Head dim is 128 for every Wan 2.1 DiT (1.3B: 12 heads,
    /// 14B: 40). `None` when the names are not a Wan DiT's.
    pub fn wan_config(&self) -> Option<WanConfig> {
        let shape = |n: &str| self.raw(n).ok().map(|t| t.1);
        let pe = shape("patch_embedding.weight")?;
        let (dim, in_channels) = (pe[0], pe[1]);
        let layers = self
            .names()
            .filter_map(|n| {
                n.strip_prefix("blocks.")?
                    .split('.')
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
            .max()?
            + 1;
        let k = "blocks.0.ffn.net.0.proj.weight";
        let ffn_dim = shape(k)
            .or_else(|| shape(&format!("{k}.q8")))
            .or_else(|| shape(&format!("{k}.q4")))?[0];
        let text_dim = shape("condition_embedder.text_embedder.linear_1.weight")?[1];
        let out = shape("proj_out.weight")?[0];
        let base = WanConfig::fastwan_1_3b();
        if dim % base.head_dim != 0 {
            return None;
        }
        Some(WanConfig {
            heads: dim / base.head_dim,
            in_channels,
            out_channels: out / (base.patch * base.patch),
            text_dim,
            ffn_dim,
            layers,
            ..base
        })
    }
}

#[allow(unsafe_code)]
fn map_readonly(file: &std::fs::File) -> Result<memmap2::Mmap> {
    // SAFETY: weights are read-only inputs; concurrent truncation by another
    // process is outside this program's contract (as for every mmap loader).
    Ok(unsafe { memmap2::Mmap::map(file)? })
}

impl WeightSource for SafetensorsDir {
    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn stored(&self, key: &str) -> Result<Option<crate::quant::Stored>> {
        use crate::quant::{Q4, Q8, Stored};
        use safetensors::Dtype;
        let f32s = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let (_, shape, _) = self.raw(name)?;
            let v = self.get(name, &shape)?;
            Ok((shape, v))
        };
        let bad = |m: String| Error::Weights(format!("`{key}`: {m}"));
        let q8 = format!("{key}.weight.q8");
        let q4 = format!("{key}.weight.q4");
        if self.contains_raw(&q8) {
            let (dt, shape, data) = self.raw(&q8)?;
            if dt != Dtype::I8 || shape.len() != 2 {
                return Err(bad(format!("q8 is {dt:?} {shape:?}")));
            }
            let (sshape, scales) = f32s(&format!("{key}.weight.scale"))?;
            if sshape.len() != 2 || sshape[0] != shape[0] || shape[1] % sshape[1] != 0 {
                return Err(bad(format!("scale {sshape:?} for codes {shape:?}")));
            }
            return Ok(Some(Stored::Q8(Q8 {
                q: data.iter().map(|&b| b as i8).collect(),
                scales,
                rows: shape[0],
                cols: shape[1],
                group: shape[1] / sshape[1],
            })));
        }
        if self.contains_raw(&q4) {
            let (dt, shape, data) = self.raw(&q4)?;
            if dt != Dtype::U8 || shape.len() != 2 {
                return Err(bad(format!("q4 is {dt:?} {shape:?}")));
            }
            let (rows, cols) = (shape[0], 2 * shape[1]);
            let (sshape, scales) = f32s(&format!("{key}.weight.scale"))?;
            let (zshape, zeros) = f32s(&format!("{key}.weight.zero"))?;
            if sshape != zshape || sshape.len() != 2 || sshape[0] != rows || cols % sshape[1] != 0 {
                return Err(bad(format!(
                    "scale {sshape:?} / zero {zshape:?} for codes {shape:?}"
                )));
            }
            return Ok(Some(Stored::Q4(Q4::from_packed(
                data, scales, zeros, rows, cols,
            ))));
        }
        Ok(None)
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
        let st = safetensors::SafeTensors::deserialize(&self.maps[i])
            .map_err(|e| Error::Weights(e.to_string()))?;
        let t = st.tensor(name).map_err(|e| Error::Weights(e.to_string()))?;
        if squeeze(t.shape()) != squeeze(shape) {
            return Err(Error::Weights(format!(
                "`{name}` is {:?}, expected {shape:?}",
                t.shape()
            )));
        }
        let b = t.data();
        use safetensors::Dtype;
        Ok(match t.dtype() {
            Dtype::F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F16 => b
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::BF16 => b
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => {
                return Err(Error::Weights(format!(
                    "`{name}`: unsupported dtype {other:?}"
                )));
            }
        })
    }
}

/// Deterministic weights for every name the model asks for: `U(−a, a)` with
/// `a = 1/√fan_in` for matrices, ones for norm weights, small values for the
/// modulation tables. Same name → same values on every backend.
pub struct Synthetic {
    pub seed: u64,
}

impl WeightSource for Synthetic {
    fn contains(&self, _name: &str) -> bool {
        true
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let n: usize = shape.iter().product();
        let mut h = self.seed;
        for b in name.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3);
        }
        let mut rng = Rng::new(h);
        let norm_weight = name.ends_with("norm_q.weight")
            || name.ends_with("norm_k.weight")
            || name.ends_with("norm2.weight");
        Ok(if norm_weight {
            rng.uniform(n, 0.1).into_iter().map(|v| 1.0 + v).collect()
        } else if name.contains("scale_shift_table") || name.ends_with(".bias") {
            rng.uniform(n, 0.1)
        } else {
            let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
            rng.uniform(n, 1.0 / (fan_in as f32).sqrt())
        })
    }
}

/// Every tensor the Wan DiT reads, with its Diffusers shape.
pub fn wan_manifest(cfg: &WanConfig) -> Vec<(String, Vec<usize>)> {
    let d = cfg.dim();
    let mut v: Vec<(String, Vec<usize>)> = vec![
        (
            "patch_embedding.weight".into(),
            vec![d, cfg.in_channels, 1, cfg.patch, cfg.patch],
        ),
        ("patch_embedding.bias".into(), vec![d]),
        (
            "condition_embedder.time_embedder.linear_1.weight".into(),
            vec![d, cfg.freq_dim],
        ),
        (
            "condition_embedder.time_embedder.linear_1.bias".into(),
            vec![d],
        ),
        (
            "condition_embedder.time_embedder.linear_2.weight".into(),
            vec![d, d],
        ),
        (
            "condition_embedder.time_embedder.linear_2.bias".into(),
            vec![d],
        ),
        ("condition_embedder.time_proj.weight".into(), vec![6 * d, d]),
        ("condition_embedder.time_proj.bias".into(), vec![6 * d]),
        (
            "condition_embedder.text_embedder.linear_1.weight".into(),
            vec![d, cfg.text_dim],
        ),
        (
            "condition_embedder.text_embedder.linear_1.bias".into(),
            vec![d],
        ),
        (
            "condition_embedder.text_embedder.linear_2.weight".into(),
            vec![d, d],
        ),
        (
            "condition_embedder.text_embedder.linear_2.bias".into(),
            vec![d],
        ),
        ("scale_shift_table".into(), vec![1, 2, d]),
        ("proj_out.weight".into(), vec![cfg.patch_features_out(), d]),
        ("proj_out.bias".into(), vec![cfg.patch_features_out()]),
    ];
    for l in 0..cfg.layers {
        let p = format!("blocks.{l}");
        for a in ["attn1", "attn2"] {
            for m in ["to_q", "to_k", "to_v", "to_out.0"] {
                v.push((format!("{p}.{a}.{m}.weight"), vec![d, d]));
                v.push((format!("{p}.{a}.{m}.bias"), vec![d]));
            }
            v.push((format!("{p}.{a}.norm_q.weight"), vec![d]));
            v.push((format!("{p}.{a}.norm_k.weight"), vec![d]));
        }
        v.push((format!("{p}.norm2.weight"), vec![d]));
        v.push((format!("{p}.norm2.bias"), vec![d]));
        v.push((format!("{p}.ffn.net.0.proj.weight"), vec![cfg.ffn_dim, d]));
        v.push((format!("{p}.ffn.net.0.proj.bias"), vec![cfg.ffn_dim]));
        v.push((format!("{p}.ffn.net.2.weight"), vec![d, cfg.ffn_dim]));
        v.push((format!("{p}.ffn.net.2.bias"), vec![d]));
        v.push((format!("{p}.scale_shift_table"), vec![1, 6, d]));
    }
    v
}

/// Every tensor the TAEHV decoder reads (positional `decoder.N` keys, exactly
/// as `taehv.py`'s `nn.Sequential` numbers them).
pub fn tae_manifest(cfg: &TaeConfig) -> Vec<(String, Vec<usize>)> {
    let nf = cfg.widths;
    let mut v = vec![
        (
            "decoder.1.weight".to_string(),
            vec![nf[0], cfg.latent_channels, 3, 3],
        ),
        ("decoder.1.bias".to_string(), vec![nf[0]]),
    ];
    for s in 0..3 {
        let base = 3 + 6 * s;
        for m in 0..cfg.mem_blocks {
            let p = format!("decoder.{}", base + m);
            v.push((format!("{p}.conv.0.weight"), vec![nf[s], 2 * nf[s], 3, 3]));
            v.push((format!("{p}.conv.0.bias"), vec![nf[s]]));
            v.push((format!("{p}.conv.2.weight"), vec![nf[s], nf[s], 3, 3]));
            v.push((format!("{p}.conv.2.bias"), vec![nf[s]]));
            v.push((format!("{p}.conv.4.weight"), vec![nf[s], nf[s], 3, 3]));
            v.push((format!("{p}.conv.4.bias"), vec![nf[s]]));
        }
        v.push((
            format!("decoder.{}.conv.weight", base + 4),
            vec![nf[s] * cfg.tgrow[s], nf[s], 1, 1],
        ));
        v.push((
            format!("decoder.{}.weight", base + 5),
            vec![nf[s + 1], nf[s], 3, 3],
        ));
    }
    v.push((
        "decoder.22.weight".into(),
        vec![cfg.out_channels(), nf[3], 3, 3],
    ));
    v.push(("decoder.22.bias".into(), vec![cfg.out_channels()]));
    v
}

/// Names in `manifest` that `src` lacks.
pub fn missing(src: &dyn WeightSource, manifest: &[(String, Vec<usize>)]) -> Vec<String> {
    manifest
        .iter()
        .filter(|(n, _)| {
            // A quantized checkpoint stores `X.weight` as `X.weight.q8` / `.q4`.
            !(src.contains(n)
                || src.contains(&format!("{n}.q8"))
                || src.contains(&format!("{n}.q4")))
        })
        .map(|(n, _)| n.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn wan_config_from_shapes() {
        use crate::export::{f32_bytes, write};
        let dir = std::env::temp_dir().join(format!("fw-infer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("dit.safetensors");
        let z = |n: usize| f32_bytes(&vec![0.0; n]);
        let (dim, ffn, text, layers) = (256, 64, 16, 3);
        let mut t = vec![
            (
                "patch_embedding.weight".to_string(),
                safetensors::Dtype::F32,
                vec![dim, 16, 1, 2, 2],
                z(dim * 64),
            ),
            (
                "condition_embedder.text_embedder.linear_1.weight".into(),
                safetensors::Dtype::F32,
                vec![dim, text],
                z(dim * text),
            ),
            (
                "proj_out.weight".into(),
                safetensors::Dtype::F32,
                vec![64, dim],
                z(64 * dim),
            ),
            (
                "blocks.0.ffn.net.0.proj.weight".into(),
                safetensors::Dtype::F32,
                vec![ffn, dim],
                z(ffn * dim),
            ),
        ];
        for l in 0..layers {
            t.push((
                format!("blocks.{l}.norm2.weight"),
                safetensors::Dtype::F32,
                vec![dim],
                z(dim),
            ));
        }
        write(&p, &t, Default::default()).unwrap();
        let c = SafetensorsDir::open(&p).unwrap().wan_config().unwrap();
        assert_eq!(
            (
                c.heads,
                c.head_dim,
                c.layers,
                c.ffn_dim,
                c.text_dim,
                c.in_channels,
                c.out_channels
            ),
            (2, 128, 3, 64, 16, 16, 16)
        );
        assert_eq!(WanConfig::wan21_14b().dim(), 5120);
        std::fs::remove_dir_all(&dir).ok();
    }

    use super::*;

    #[test]
    fn manifest_matches_param_count() {
        let cfg = WanConfig::fastwan_1_3b();
        let n: usize = wan_manifest(&cfg)
            .iter()
            .map(|(_, s)| s.iter().product::<usize>())
            .sum();
        assert_eq!(n, cfg.params());
    }

    #[test]
    fn tae_manifest_has_the_positional_keys() {
        let m = tae_manifest(&TaeConfig::taew2_1());
        assert!(
            m.iter()
                .any(|(n, s)| n == "decoder.19.conv.weight" && s == &vec![128, 64, 1, 1])
        );
        assert!(
            m.iter()
                .any(|(n, s)| n == "decoder.7.conv.weight" && s == &vec![256, 256, 1, 1])
        );
        assert!(m.iter().any(|(n, _)| n == "decoder.17.conv.4.bias"));
    }

    #[test]
    fn synthetic_is_deterministic() {
        let s = Synthetic { seed: 1 };
        assert_eq!(
            s.get("a.weight", &[4, 4]).unwrap(),
            s.get("a.weight", &[4, 4]).unwrap()
        );
        assert_ne!(
            s.get("a.weight", &[4, 4]).unwrap(),
            s.get("b.weight", &[4, 4]).unwrap()
        );
    }

    #[test]
    fn safetensors_roundtrip_bf16() {
        let dir = std::env::temp_dir().join(format!("fastwan-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vals = [1.0f32, -2.5, 0.125, 3.0];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2, 2], &bytes)
                .unwrap();
        safetensors::serialize_to_file([("w", view)], None, &dir.join("m.safetensors")).unwrap();
        let src = SafetensorsDir::open(&dir).unwrap();
        assert_eq!(src.get("w", &[2, 2]).unwrap(), vals);
        assert!(src.get("w", &[4]).is_err());
        assert!(src.get("nope", &[1]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
