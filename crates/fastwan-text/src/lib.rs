//! Prompt → UMT5-XXL embeddings, natively: the text encoder Wan conditions on.
//!
//! It runs on the CPU and reads the checkpoint's bf16 weights straight out of
//! the memory-mapped safetensors (the 11.4 GB never becomes an f32 copy, and
//! the GPU keeps its memory for the DiT). Only the prompt's real tokens are
//! encoded: Diffusers' `WanPipeline` masks the padding and then zeroes it, and
//! T5's relative-position bias depends only on distances, so encoding
//! `n` tokens and zero-padding to 512 is the same result for a fraction of the
//! work.

pub mod umt5;

use fastwan_core::{Error, Result, weights::SafetensorsDir};
use std::path::Path;

pub use umt5::{Umt5, Umt5Config};

pub struct PromptEncoder {
    tokenizer: tokenizers::Tokenizer,
    model: Umt5,
    /// Rows the DiT expects (`WanConfig::text_len`).
    pub text_len: usize,
}

/// Wan's `prompt_clean` without ftfy: unescape the common HTML entities and
/// collapse whitespace.
pub fn clean_prompt(p: &str) -> String {
    let p = p
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    p.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl PromptEncoder {
    /// `tokenizer`: a `tokenizer.json`; `text_encoder`: the checkpoint's
    /// `text_encoder/` directory (or one safetensors file).
    pub fn open(tokenizer: &Path, text_encoder: &Path, cfg: Umt5Config) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer)
            .map_err(|e| Error::Msg(format!("tokenizer {}: {e}", tokenizer.display())))?;
        let model = Umt5::new(SafetensorsDir::open(text_encoder)?, cfg)?;
        Ok(Self {
            tokenizer,
            model,
            text_len: 512,
        })
    }

    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(clean_prompt(prompt), true)
            .map_err(|e| Error::Msg(format!("tokenize: {e}")))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(self.text_len);
        Ok(ids)
    }

    /// `[text_len, d_model]`, rows past the prompt zeroed.
    pub fn encode(&self, prompt: &str) -> Result<Vec<f32>> {
        let ids = self.tokenize(prompt)?;
        let h = self.model.encode(&ids)?;
        let mut out = vec![0.0; self.text_len * self.model.cfg.d_model];
        out[..h.len()].copy_from_slice(&h);
        Ok(out)
    }
}
