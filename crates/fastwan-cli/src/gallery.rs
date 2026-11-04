//! A prompt gallery: UMT5 embeddings computed once (`fastwan encode`) so the
//! engine can run without the 11 GB text encoder (`fastwan serve --gallery`).
//!
//! A directory with `gallery.json` and one safetensors file per prompt:
//!
//! ```text
//! gallery.json        {"prompts": [{"prompt": "...", "tokens": 33, "file": "01-a-red-fox-….safetensors"}]}
//! 01-….safetensors    prompt_embeds [tokens, 4096], F16 (F32 if a value does not fit);
//!                     metadata {"prompt": "...", "tokens": "33"}
//! ```
//!
//! Only the prompt's real tokens are stored (~8 KB each in f16): the model
//! pads the rest of its 512 positions with zeros, as Diffusers does.

use anyhow::{Context, Result, bail};
use fastwan_core::{config::WanConfig, export, weights::SafetensorsDir, weights::WeightSource};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Serialize, Deserialize)]
pub struct Entry {
    pub prompt: String,
    pub tokens: usize,
    pub file: String,
}

#[derive(Default, Serialize, Deserialize)]
struct Index {
    prompts: Vec<Entry>,
}

pub struct Gallery {
    dir: PathBuf,
    pub entries: Vec<Entry>,
}

/// Whitespace-insensitive key: the page may reflow a prompt.
fn norm(p: &str) -> String {
    p.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn slug(p: &str) -> String {
    let mut s = String::new();
    for w in p
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        if s.len() + w.len() > 40 {
            break;
        }
        if !s.is_empty() {
            s.push('-');
        }
        s.push_str(&w.to_ascii_lowercase());
    }
    if s.is_empty() { "prompt".into() } else { s }
}

impl Gallery {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("gallery.json");
        let idx: Index = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        if idx.prompts.is_empty() {
            bail!("{} lists no prompts", path.display());
        }
        for e in &idx.prompts {
            if !dir.join(&e.file).is_file() {
                bail!("{}: missing {}", path.display(), e.file);
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            entries: idx.prompts,
        })
    }

    pub fn find(&self, prompt: &str) -> Option<&Entry> {
        let k = norm(prompt);
        self.entries.iter().find(|e| norm(&e.prompt) == k)
    }

    /// `[text_len, text_dim]`, zero past the prompt's tokens.
    pub fn embeds(&self, e: &Entry, cfg: &WanConfig) -> Result<Vec<f32>> {
        load_embeds(&self.dir.join(&e.file), cfg)
    }

    /// Store one prompt's embeddings (the first `tokens` rows of `embeds`)
    /// and list it in `gallery.json`, replacing an entry with the same prompt.
    pub fn add(
        dir: &Path,
        prompt: &str,
        tokens: usize,
        embeds: &[f32],
        dim: usize,
    ) -> Result<Entry> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("gallery.json");
        let mut idx: Index = match std::fs::read(&path) {
            Ok(b) => {
                serde_json::from_slice(&b).with_context(|| format!("parse {}", path.display()))?
            }
            Err(_) => Index::default(),
        };
        let rows = &embeds[..tokens * dim];
        let (dt, bytes) = if rows.iter().all(|v| v.is_finite() && v.abs() <= 65504.0) {
            (safetensors::Dtype::F16, export::f16_bytes(rows))
        } else {
            (safetensors::Dtype::F32, export::f32_bytes(rows))
        };
        let pos = idx
            .prompts
            .iter()
            .position(|e| norm(&e.prompt) == norm(prompt));
        let n = pos.map_or(idx.prompts.len() + 1, |i| i + 1);
        let file = format!("{n:02}-{}.safetensors", slug(prompt));
        export::write(
            &dir.join(&file),
            &[("prompt_embeds".into(), dt, vec![tokens, dim], bytes)],
            [
                ("prompt".to_string(), prompt.to_string()),
                ("tokens".to_string(), tokens.to_string()),
            ]
            .into(),
        )?;
        let e = Entry {
            prompt: prompt.to_string(),
            tokens,
            file,
        };
        match pos {
            Some(i) => {
                let old = std::mem::replace(&mut idx.prompts[i], e.clone());
                if old.file != e.file {
                    let _ = std::fs::remove_file(dir.join(old.file));
                }
            }
            None => idx.prompts.push(e.clone()),
        }
        std::fs::write(&path, serde_json::to_string_pretty(&idx)?)?;
        Ok(e)
    }
}

/// `prompt_embeds` `[L ≤ text_len, text_dim]` (F32 / F16 / BF16, or
/// `[1, L, D]`), zero-padded to `[text_len, text_dim]`.
pub fn load_embeds(path: &Path, cfg: &WanConfig) -> Result<Vec<f32>> {
    let st = SafetensorsDir::open(path)?;
    let name = st
        .names()
        .find(|n| n.contains("embed"))
        .map(str::to_string)
        .context("no `prompt_embeds` tensor in the embeddings file")?;
    let mut out: Vec<f32> = (1..=cfg.text_len)
        .rev()
        .find_map(|l| st.get(&name, &[l, cfg.text_dim]).ok())
        .context("prompt_embeds must be [L <= 512, 4096]")?;
    out.resize(cfg.text_len * cfg.text_dim, 0.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_find_and_load_round_trip() {
        let cfg = WanConfig::tiny();
        let dir = std::env::temp_dir().join(format!("fastwan-gallery-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dim = cfg.text_dim;
        let e1: Vec<f32> = (0..cfg.text_len * dim)
            .map(|i| (i % 97) as f32 * 0.01 - 0.4)
            .collect();
        Gallery::add(&dir, "A red fox, in snow!", 3, &e1, dim).unwrap();
        Gallery::add(&dir, "a  market", 2, &e1, dim).unwrap();
        // Same prompt again replaces, not appends.
        Gallery::add(&dir, "A red fox,   in snow!", 4, &e1, dim).unwrap();
        let g = Gallery::open(&dir).unwrap();
        assert_eq!(g.entries.len(), 2);
        let e = g.find("A red fox, in\nsnow!").unwrap();
        assert_eq!(
            (e.tokens, e.file.as_str()),
            (4, "01-a-red-fox-in-snow.safetensors")
        );
        let got = g.embeds(e, &cfg).unwrap();
        assert_eq!(got.len(), cfg.text_len * dim);
        for (i, (a, b)) in got.iter().zip(&e1).enumerate() {
            if i < 4 * dim {
                assert!((a - b).abs() <= 1e-3 * b.abs().max(1.0), "{i}: {a} vs {b}");
            } else {
                assert_eq!(*a, 0.0);
            }
        }
        assert!(g.find("a dog").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
