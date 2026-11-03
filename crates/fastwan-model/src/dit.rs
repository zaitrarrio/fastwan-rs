//! The Wan 2.1 DiT (Diffusers `WanTransformer3DModel`), batch 1, no CFG.
//!
//! Per block (slot order of the `[6, dim]` modulation: shift, scale, gate of
//! the self-attention, then of the FFN):
//!
//! ```text
//! n   = LN(h)·(1+scale_msa)+shift_msa                    layer_norm
//! qkv = n·W_qkvᵀ                                         linear (q|k|v fused)
//! q,k = rope(rms(q|k)), v                                head_split ×3
//! h  += to_out(attn(q,k,v))·gate_msa;  n = norm2(h)      attention, linear, residual_norm
//! h  += to_out(attn(rms(n·W_qᵀ), K_text, V_text))        linear, head_split, attention, linear
//! n   = LN(h)·(1+scale_ffn)+shift_ffn                    ...fused into the residual above
//! h  += ffn(n)·gate_ffn                                  linear ×2 per token chunk, gate_residual
//! ```
//!
//! What is precomputed, and why it is safe:
//! * the timestep path (sinusoid → MLP → `time_proj` → `+ scale_shift_table`)
//!   runs on the host once per DMD step ([`WanDit::step_mods`]);
//! * the text projection and every block's cross-attention K/V run once per
//!   prompt ([`WanDit::prepare_prompt`]) — they depend on nothing else.

use fastwan_core::{
    Result,
    backend::{Act, Backend, NormKind},
    config::WanConfig,
    rope::rope_table,
    timestep::{matvec, silu, sinusoid},
    weights::WeightSource,
};

const SHIFT_MSA: usize = 0;
const SCALE_MSA: usize = 1;
const GATE_MSA: usize = 2;
const SHIFT_FFN: usize = 3;
const SCALE_FFN: usize = 4;
const GATE_FFN: usize = 5;

struct Linear<B: Backend> {
    w: B::Weight,
    b: B::Buf,
}

type Shapes = std::collections::HashMap<String, Vec<usize>>;

/// Several `nn.Linear`s stacked by rows into one matrix (fused QKV / KV), each
/// checked against its manifest shape.
fn load_linear<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    shapes: &Shapes,
    keys: &[&str],
) -> Result<Linear<B>> {
    let (mut w, mut b, mut rows, mut cols) = (Vec::new(), Vec::new(), 0, 0);
    for k in keys {
        let shape = shapes
            .get(&format!("{k}.weight"))
            .ok_or_else(|| fastwan_core::Error::Weights(format!("`{k}` is not in the manifest")))?;
        let (o, i) = (shape[0], shape[1..].iter().product::<usize>());
        w.extend(src.get(&format!("{k}.weight"), shape)?);
        b.extend(src.get(&format!("{k}.bias"), &[o])?);
        rows += o;
        cols = i;
    }
    Ok(Linear {
        w: be.weight(&w, rows, cols),
        b: be.upload(&b),
    })
}

struct Block<B: Backend> {
    qkv: Linear<B>,
    norm_q1: B::Buf,
    norm_k1: B::Buf,
    out1: Linear<B>,
    q2: Linear<B>,
    kv2: Linear<B>,
    norm_q2: B::Buf,
    norm_k2: B::Buf,
    out2: Linear<B>,
    norm2_w: B::Buf,
    norm2_b: B::Buf,
    ffn1: Linear<B>,
    ffn2: Linear<B>,
    /// `to_gate_compress`: present on VSA-distilled checkpoints (FastWan).
    gate: Option<Linear<B>>,
    /// `[6·dim]`, host: added to `timestep_proj` once per step.
    table: Vec<f32>,
}

/// Host copies of the timestep MLP.
struct TimeEmbed {
    l1w: Vec<f32>,
    l1b: Vec<f32>,
    l2w: Vec<f32>,
    l2b: Vec<f32>,
    pw: Vec<f32>,
    pb: Vec<f32>,
    head_table: Vec<f32>,
}

pub struct WanDit<B: Backend> {
    pub cfg: WanConfig,
    patch: Linear<B>,
    text1: Linear<B>,
    text2: Linear<B>,
    blocks: Vec<Block<B>>,
    proj_out: Linear<B>,
    time: TimeEmbed,
}

/// Per-step conditioning, on the device: one `[6, dim]` table per block and
/// the `[2, dim]` output-head table.
pub struct StepMods<B: Backend> {
    pub blocks: Vec<B::Buf>,
    pub head: B::Buf,
}

/// Per-prompt cross-attention K/V of every block, `[heads, text_len, d]`.
pub struct Prompt<B: Backend> {
    kv: Vec<(B::Buf, B::Buf)>,
}

/// Activation buffers for one token count, reused by every block and step.
/// Self-attention algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Attention {
    Dense,
    /// Video sparse attention keeping `1 − sparsity` of the key tiles.
    Vsa {
        sparsity: f64,
    },
}

pub struct Workspace<B: Backend> {
    pub attention: Attention,
    vsa: Option<B::Vsa>,
    /// The VSA gate, `[heads, S, d]`.
    g: B::Buf,
    pub tokens: usize,
    pub grid: (usize, usize, usize),
    pub ffn_chunk: usize,
    h: B::Buf,
    normed: B::Buf,
    qkv: B::Buf,
    q: B::Buf,
    k: B::Buf,
    v: B::Buf,
    attn: B::Buf,
    tmp: B::Buf,
    hid: B::Buf,
    patches: B::Buf,
    rope_cos: B::Buf,
    rope_sin: B::Buf,
}

impl<B: Backend> WanDit<B> {
    pub fn load(be: &B, cfg: WanConfig, src: &dyn WeightSource) -> Result<Self> {
        let d = cfg.dim();
        let mut shapes: Shapes = fastwan_core::weights::wan_manifest(&cfg)
            .into_iter()
            .collect();
        for l in 0..cfg.layers {
            shapes.insert(format!("blocks.{l}.to_gate_compress.weight"), vec![d, d]);
        }
        let lin = |keys: &[&str]| load_linear(be, src, &shapes, keys);
        let vec = |k: &str, n: usize| -> Result<B::Buf> { Ok(be.upload(&src.get(k, &[n])?)) };
        let mut blocks = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            blocks.push(Block {
                qkv: lin(&[&p("attn1.to_q"), &p("attn1.to_k"), &p("attn1.to_v")])?,
                norm_q1: vec(&p("attn1.norm_q.weight"), d)?,
                norm_k1: vec(&p("attn1.norm_k.weight"), d)?,
                out1: lin(&[&p("attn1.to_out.0")])?,
                q2: lin(&[&p("attn2.to_q")])?,
                kv2: lin(&[&p("attn2.to_k"), &p("attn2.to_v")])?,
                norm_q2: vec(&p("attn2.norm_q.weight"), d)?,
                norm_k2: vec(&p("attn2.norm_k.weight"), d)?,
                out2: lin(&[&p("attn2.to_out.0")])?,
                norm2_w: vec(&p("norm2.weight"), d)?,
                norm2_b: vec(&p("norm2.bias"), d)?,
                ffn1: lin(&[&p("ffn.net.0.proj")])?,
                ffn2: lin(&[&p("ffn.net.2")])?,
                gate: if src.contains(&p("to_gate_compress.weight")) {
                    Some(lin(&[&p("to_gate_compress")])?)
                } else {
                    None
                },
                table: src.get(&p("scale_shift_table"), &[1, 6, d])?,
            });
        }
        let ce = "condition_embedder";
        let time = TimeEmbed {
            l1w: src.get(
                &format!("{ce}.time_embedder.linear_1.weight"),
                &[d, cfg.freq_dim],
            )?,
            l1b: src.get(&format!("{ce}.time_embedder.linear_1.bias"), &[d])?,
            l2w: src.get(&format!("{ce}.time_embedder.linear_2.weight"), &[d, d])?,
            l2b: src.get(&format!("{ce}.time_embedder.linear_2.bias"), &[d])?,
            pw: src.get(&format!("{ce}.time_proj.weight"), &[6 * d, d])?,
            pb: src.get(&format!("{ce}.time_proj.bias"), &[6 * d])?,
            head_table: src.get("scale_shift_table", &[1, 2, d])?,
        };
        let patch = lin(&["patch_embedding"])?;
        Ok(Self {
            text1: lin(&[&format!("{ce}.text_embedder.linear_1")])?,
            text2: lin(&[&format!("{ce}.text_embedder.linear_2")])?,
            proj_out: lin(&["proj_out"])?,
            patch,
            blocks,
            time,
            cfg,
        })
    }

    /// Host timestep conditioning for one DMD timestep, uploaded.
    pub fn step_mods(&self, be: &B, timestep: f32) -> StepMods<B> {
        let t = &self.time;
        let emb = sinusoid(timestep, self.cfg.freq_dim);
        let h = silu(&matvec(&t.l1w, Some(&t.l1b), &emb));
        let temb = matvec(&t.l2w, Some(&t.l2b), &h);
        let proj = matvec(&t.pw, Some(&t.pb), &silu(&temb));
        let d = self.cfg.dim();
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                let m: Vec<f32> = proj.iter().zip(&b.table).map(|(a, c)| a + c).collect();
                be.upload(&m)
            })
            .collect();
        let head: Vec<f32> = (0..2 * d).map(|i| t.head_table[i] + temb[i % d]).collect();
        StepMods {
            blocks,
            head: be.upload(&head),
        }
    }

    /// Text projection + every block's cross K/V for `[text_len, text_dim]`
    /// prompt embeddings (UMT5 output, zero-padded to `text_len`).
    pub fn prepare_prompt(&self, be: &B, embeds: &[f32]) -> Prompt<B> {
        let (l, d, c) = (self.cfg.text_len, self.cfg.dim(), &self.cfg);
        assert_eq!(
            embeds.len(),
            l * c.text_dim,
            "prompt embeds must be [text_len, text_dim]"
        );
        let x = be.upload(embeds);
        let mid = be.alloc(l * d);
        let ctx = be.alloc(l * d);
        be.linear(
            &x,
            0,
            l,
            &self.text1.w,
            Some(&self.text1.b),
            Act::GeluTanh,
            &mid,
            0,
        );
        be.linear(
            &mid,
            0,
            l,
            &self.text2.w,
            Some(&self.text2.b),
            Act::None,
            &ctx,
            0,
        );
        let kv = be.alloc(l * 2 * d);
        let kv = self
            .blocks
            .iter()
            .map(|b| {
                be.linear(&ctx, 0, l, &b.kv2.w, Some(&b.kv2.b), Act::None, &kv, 0);
                let (k, v) = (be.alloc(l * d), be.alloc(l * d));
                be.head_split(
                    &kv,
                    l,
                    2 * d,
                    0,
                    c.heads,
                    c.head_dim,
                    Some(&b.norm_k2),
                    None,
                    c.eps,
                    &k,
                );
                be.head_split(&kv, l, 2 * d, d, c.heads, c.head_dim, None, None, c.eps, &v);
                (k, v)
            })
            .collect();
        Prompt { kv }
    }

    /// Whether every block carries `to_gate_compress` (a VSA checkpoint).
    pub fn has_vsa_gates(&self) -> bool {
        self.blocks.iter().all(|b| b.gate.is_some())
    }

    /// `attention: None` picks VSA at the default sparsity on a VSA checkpoint
    /// (what it was distilled with) when the backend can run it, and dense
    /// attention otherwise.
    pub fn workspace(
        &self,
        be: &B,
        latent: (usize, usize, usize),
        ffn_chunk: usize,
        attention: Option<Attention>,
    ) -> Workspace<B> {
        let c = &self.cfg;
        let grid = (latent.0, latent.1 / c.patch, latent.2 / c.patch);
        let s = grid.0 * grid.1 * grid.2;
        let d = c.dim();
        let chunk = ffn_chunk.clamp(1, s);
        let rope = rope_table(c, grid, 0);
        let attention =
            attention.unwrap_or(if self.has_vsa_gates() && be.vsa_supported(c.head_dim) {
                Attention::Vsa {
                    sparsity: fastwan_core::vsa::DEFAULT_SPARSITY,
                }
            } else {
                Attention::Dense
            });
        let vsa = match attention {
            Attention::Vsa { sparsity } => {
                assert!(
                    self.has_vsa_gates(),
                    "VSA needs a checkpoint with to_gate_compress in every block"
                );
                Some(be.vsa_prepare(&fastwan_core::vsa::VsaSpec::new(
                    grid, c.heads, c.head_dim, sparsity,
                )))
            }
            Attention::Dense => None,
        };
        Workspace {
            attention,
            vsa,
            g: be.alloc(s * d),
            tokens: s,
            grid,
            ffn_chunk: chunk,
            h: be.alloc(s * d),
            normed: be.alloc(s * d),
            qkv: be.alloc(s * 3 * d),
            q: be.alloc(s * d),
            k: be.alloc(s * d),
            v: be.alloc(s * d),
            attn: be.alloc(s * d),
            tmp: be.alloc(s * d),
            hid: be.alloc(chunk * c.ffn_dim),
            patches: be.alloc(s * c.patch_features_in().max(c.patch_features_out())),
            rope_cos: be.upload_table(&rope.cos),
            rope_sin: be.upload_table(&rope.sin),
        }
    }

    /// Flow prediction `v` for latent `x` (`[C, T, H, W]`), written to `out`.
    pub fn forward(
        &self,
        be: &B,
        ws: &Workspace<B>,
        x: &B::Buf,
        mods: &StepMods<B>,
        prompt: &Prompt<B>,
        out: &B::Buf,
    ) {
        let c = &self.cfg;
        let (s, d, hd, nh, eps) = (ws.tokens, c.dim(), c.head_dim, c.heads, c.eps);
        let (t, lh, lw) = (ws.grid.0, ws.grid.1 * c.patch, ws.grid.2 * c.patch);
        be.patchify(x, c.in_channels, t, lh, lw, c.patch, &ws.patches);
        be.linear(
            &ws.patches,
            0,
            s,
            &self.patch.w,
            Some(&self.patch.b),
            Act::None,
            &ws.h,
            0,
        );
        let rope = Some((&ws.rope_cos, &ws.rope_sin));
        for (li, (bl, m)) in self.blocks.iter().zip(&mods.blocks).enumerate() {
            // Self-attention.
            be.layer_norm(
                &ws.h,
                s,
                d,
                NormKind::Mod {
                    scale: SCALE_MSA,
                    shift: SHIFT_MSA,
                },
                m,
                eps,
                &ws.normed,
            );
            be.linear(
                &ws.normed,
                0,
                s,
                &bl.qkv.w,
                Some(&bl.qkv.b),
                Act::None,
                &ws.qkv,
                0,
            );
            be.head_split(
                &ws.qkv,
                s,
                3 * d,
                0,
                nh,
                hd,
                Some(&bl.norm_q1),
                rope,
                eps,
                &ws.q,
            );
            be.head_split(
                &ws.qkv,
                s,
                3 * d,
                d,
                nh,
                hd,
                Some(&bl.norm_k1),
                rope,
                eps,
                &ws.k,
            );
            be.head_split(&ws.qkv, s, 3 * d, 2 * d, nh, hd, None, None, eps, &ws.v);
            match (&ws.vsa, &bl.gate) {
                (Some(vsa), Some(gate)) => {
                    // The gate is to_gate_compress of the same normed input,
                    // split into heads with no norm and no RoPE.
                    be.linear(
                        &ws.normed,
                        0,
                        s,
                        &gate.w,
                        Some(&gate.b),
                        Act::None,
                        &ws.tmp,
                        0,
                    );
                    be.head_split(&ws.tmp, s, d, 0, nh, hd, None, None, eps, &ws.g);
                    be.vsa_attention(&ws.q, &ws.k, &ws.v, &ws.g, vsa, &ws.attn);
                }
                _ => be.attention(&ws.q, &ws.k, &ws.v, nh, s, s, hd, &ws.attn),
            }
            be.linear(
                &ws.attn,
                0,
                s,
                &bl.out1.w,
                Some(&bl.out1.b),
                Act::None,
                &ws.tmp,
                0,
            );
            let norm2 = NormKind::Affine {
                w: &bl.norm2_w,
                b: &bl.norm2_b,
            };
            be.residual_norm(
                &ws.h,
                &ws.tmp,
                s,
                d,
                m,
                Some(GATE_MSA),
                norm2,
                eps,
                &ws.normed,
            );
            // Cross-attention against the prompt's cached K/V.
            let (k2, v2) = &prompt.kv[li];
            be.linear(
                &ws.normed,
                0,
                s,
                &bl.q2.w,
                Some(&bl.q2.b),
                Act::None,
                &ws.tmp,
                0,
            );
            be.head_split(
                &ws.tmp,
                s,
                d,
                0,
                nh,
                hd,
                Some(&bl.norm_q2),
                None,
                eps,
                &ws.q,
            );
            be.attention(&ws.q, k2, v2, nh, s, c.text_len, hd, &ws.attn);
            be.linear(
                &ws.attn,
                0,
                s,
                &bl.out2.w,
                Some(&bl.out2.b),
                Act::None,
                &ws.tmp,
                0,
            );
            let ffn_norm = NormKind::Mod {
                scale: SCALE_FFN,
                shift: SHIFT_FFN,
            };
            be.residual_norm(&ws.h, &ws.tmp, s, d, m, None, ffn_norm, eps, &ws.normed);
            // FFN in token chunks: the `ffn_dim`-wide hidden never exceeds
            // one chunk.
            let mut r = 0;
            while r < s {
                let n = ws.ffn_chunk.min(s - r);
                be.linear(
                    &ws.normed,
                    r,
                    n,
                    &bl.ffn1.w,
                    Some(&bl.ffn1.b),
                    Act::GeluTanh,
                    &ws.hid,
                    0,
                );
                be.linear(
                    &ws.hid,
                    0,
                    n,
                    &bl.ffn2.w,
                    Some(&bl.ffn2.b),
                    Act::None,
                    &ws.tmp,
                    r,
                );
                r += n;
            }
            be.gate_residual(&ws.h, &ws.tmp, s, d, m, GATE_FFN);
        }
        // Output head: AdaLN with the [2, dim] table (shift = row 0, scale = row 1).
        be.layer_norm(
            &ws.h,
            s,
            d,
            NormKind::Mod { scale: 1, shift: 0 },
            &mods.head,
            eps,
            &ws.normed,
        );
        be.linear(
            &ws.normed,
            0,
            s,
            &self.proj_out.w,
            Some(&self.proj_out.b),
            Act::None,
            &ws.patches,
            0,
        );
        be.unpatchify(&ws.patches, c.out_channels, t, lh, lw, c.patch, out);
    }
}
