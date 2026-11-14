//! Causal (Self-Forcing style) generation: the DiT run block by block over a
//! per-layer K/V cache, so a video of any length is generated in pieces with
//! memory fixed by the cache size, not the video length.
//!
//! Per block of `block_frames` latent frames (Self-Forcing: 3): noise, the
//! few-step DMD denoise (each step one forward that writes this block's keys
//! and values into the cache and attends to the cache), then one forward on
//! the clean result at timestep 0 that rewrites the block's cache entries
//! with clean keys/values for the blocks after it. Within a block attention
//! is full; across blocks it is causal (a block only sees the cache).
//!
//! The cache keeps `sink_frames` first frames forever and a rolling window of
//! the most recent frames up to `cap_frames` in all (LongLive's "frame sink +
//! short window"). It is a ring: attention does not care about key order and
//! RoPE is applied before a key is stored, so an evicted slot is simply
//! overwritten instead of shifting the cache (WebGPU cannot copy a buffer
//! onto itself). Checked against strobe's PyTorch `causal_forward`
//! (tests/hf_parity.rs, scripts/reference/make_causal_fixture.py).

use crate::dit::{Prompt, StepMods, WanDit, Workspace};
use fastwan_core::{Backend, Error, Result, rng::Rng};

/// The rollout's shape and schedule.
#[derive(Debug, Clone)]
pub struct CausalConfig {
    /// Latent frames generated per block.
    pub block_frames: usize,
    /// Latent frames the cache holds, sink and current block included.
    pub cap_frames: usize,
    /// First latent frames kept in the cache for the whole video.
    pub sink_frames: usize,
    /// DiT timesteps of the few-step denoise, in order; σ = t / 1000.
    pub timesteps: Vec<f32>,
}

impl CausalConfig {
    /// Self-Forcing's DMD checkpoint (Wan2.1-T2V-1.3B): 3-frame blocks, the
    /// 4-step list 1000 / 750 / 500 / 250 warped onto the shift-5 flow
    /// schedule (`timesteps[1000 − step]`), and LongLive's 3-frame sink with a
    /// 12-frame cache.
    pub fn self_forcing() -> Self {
        let warp = |step: f64| {
            // σ = linspace(1, 1/1000, 1000)[1000 − step], shifted by 5.
            let i = 1000.0 - step;
            let s = 1.0 - i * (1.0 - 1.0 / 1000.0) / 999.0;
            let s = 5.0 * s / (1.0 + 4.0 * s);
            (s * 1000.0) as f32
        };
        Self {
            block_frames: 3,
            cap_frames: 12,
            sink_frames: 3,
            timesteps: [1000.0, 750.0, 500.0, 250.0].map(warp).to_vec(),
        }
    }

    /// Set the cache size and sink, in latent frames.
    pub fn cache_frames(&mut self, cap: usize, sink: usize) -> &mut Self {
        self.cap_frames = cap;
        self.sink_frames = sink;
        self
    }
}

/// One run of consecutive block frames stored in consecutive cache slots.
#[derive(Debug, Clone, Copy)]
struct Run {
    /// First frame within the block.
    from: usize,
    /// First cache slot.
    slot: usize,
    frames: usize,
}

/// Per-layer K/V cache, `[heads, cap_frames · tokens_per_frame, head_dim]`.
pub struct KvCache<B: Backend> {
    k: Vec<B::Buf>,
    v: Vec<B::Buf>,
    /// The valid prefix packed with its own stride while the cache fills.
    pack_k: B::Buf,
    pack_v: B::Buf,
    heads: usize,
    head_dim: usize,
    tpf: usize,
    cap: usize,
    sink: usize,
    /// Absolute latent frame held by each slot.
    slot_frame: Vec<Option<usize>>,
    runs: Vec<Run>,
    block_frames: usize,
    last_start: Option<usize>,
}

impl<B: Backend> KvCache<B> {
    pub fn new(
        be: &B,
        layers: usize,
        heads: usize,
        head_dim: usize,
        tpf: usize,
        cfg: &CausalConfig,
    ) -> Result<Self> {
        if cfg.cap_frames < cfg.sink_frames + cfg.block_frames {
            return Err(Error::Geometry(format!(
                "cache of {} frames cannot hold a {}-frame sink and a {}-frame block",
                cfg.cap_frames, cfg.sink_frames, cfg.block_frames
            )));
        }
        let n = heads * cfg.cap_frames * tpf * head_dim;
        Ok(Self {
            k: (0..layers).map(|_| be.alloc(n)).collect(),
            v: (0..layers).map(|_| be.alloc(n)).collect(),
            pack_k: be.alloc(n),
            pack_v: be.alloc(n),
            heads,
            head_dim,
            tpf,
            cap: cfg.cap_frames,
            sink: cfg.sink_frames,
            slot_frame: vec![None; cfg.cap_frames],
            runs: Vec::new(),
            block_frames: 0,
            last_start: None,
        })
    }

    /// Choose the slots for the block starting at latent frame `start`. The
    /// same block again (the next denoise step, the clean rewrite) keeps its
    /// slots. A new block fills free slots first, then the oldest non-sink.
    pub fn begin_block(&mut self, start: usize, frames: usize) -> Result<()> {
        if self.last_start == Some(start) && self.block_frames == frames {
            return Ok(());
        }
        let mut slots = Vec::with_capacity(frames);
        for j in 0..frames {
            let free = (0..self.cap).find(|&s| self.slot_frame[s].is_none() && !slots.contains(&s));
            let slot = match free {
                Some(s) => s,
                None => (self.sink..self.cap)
                    .filter(|s| !slots.contains(s))
                    .min_by_key(|&s| self.slot_frame[s])
                    .ok_or_else(|| Error::Geometry("KV cache too small for the block".into()))?,
            };
            self.slot_frame[slot] = Some(start + j);
            slots.push(slot);
        }
        self.runs.clear();
        for (j, &s) in slots.iter().enumerate() {
            match self.runs.last_mut() {
                Some(r) if r.slot + r.frames == s && r.from + r.frames == j => r.frames += 1,
                _ => self.runs.push(Run {
                    from: j,
                    slot: s,
                    frames: 1,
                }),
            }
        }
        self.last_start = Some(start);
        self.block_frames = frames;
        Ok(())
    }

    /// Valid frames (all of them once the cache is full).
    pub fn frames(&self) -> usize {
        self.slot_frame.iter().filter(|s| s.is_some()).count()
    }

    /// Write layer `li`'s block keys/values (`[heads, n, d]`) into their
    /// slots, then attend `q` to every valid cache entry into `out`.
    pub(crate) fn attend(
        &self,
        be: &B,
        li: usize,
        q: &B::Buf,
        k: &B::Buf,
        v: &B::Buf,
        n: usize,
        out: &B::Buf,
    ) {
        let (hd, tpf) = (self.head_dim, self.tpf);
        let stride = self.cap * tpf * hd;
        for r in &self.runs {
            let len = r.frames * tpf * hd;
            for h in 0..self.heads {
                let src = (h * n + r.from * tpf) * hd;
                let dst = h * stride + r.slot * tpf * hd;
                be.copy(k, src, &self.k[li], dst, len);
                be.copy(v, src, &self.v[li], dst, len);
            }
        }
        let valid = self.frames();
        if valid == self.cap {
            be.attention(
                q,
                &self.k[li],
                &self.v[li],
                self.heads,
                n,
                self.cap * tpf,
                hd,
                out,
            );
        } else {
            // Filling: the valid frames are the prefix [0, valid) (nothing has
            // been evicted yet); pack it so each head is contiguous.
            let len = valid * tpf * hd;
            for h in 0..self.heads {
                be.copy(&self.k[li], h * stride, &self.pack_k, h * len, len);
                be.copy(&self.v[li], h * stride, &self.pack_v, h * len, len);
            }
            be.attention(
                q,
                &self.pack_k,
                &self.pack_v,
                self.heads,
                n,
                valid * tpf,
                hd,
                out,
            );
        }
    }
}

/// Block-by-block generation state: the DiT workspace for one block, the
/// cache, the per-step modulation and the noise source.
pub struct CausalRollout<'a, B: Backend> {
    dit: &'a WanDit<B>,
    cfg: CausalConfig,
    ws: Workspace<B>,
    cache: KvCache<B>,
    mods: Vec<StepMods<B>>,
    clean_mods: StepMods<B>,
    x: B::Buf,
    v: B::Buf,
    noise: B::Buf,
    rng: Rng,
    next_frame: usize,
    /// Latent `(height, width)`.
    pub latent_hw: (usize, usize),
}

impl<'a, B: Backend> CausalRollout<'a, B> {
    /// `latent_hw` is the latent grid (`height / 8, width / 8`). `ffn_chunk`
    /// bounds the FFN's hidden buffer as in [`WanDit::workspace`].
    pub fn new(
        be: &B,
        dit: &'a WanDit<B>,
        cfg: CausalConfig,
        latent_hw: (usize, usize),
        ffn_chunk: usize,
        seed: u64,
    ) -> Result<Self> {
        let c = &dit.cfg;
        let (lh, lw) = latent_hw;
        if lh % c.patch != 0 || lw % c.patch != 0 {
            return Err(Error::Geometry(format!(
                "latent {lh}×{lw} is not a multiple of the patch"
            )));
        }
        let tpf = (lh / c.patch) * (lw / c.patch);
        let ws = dit.workspace(
            be,
            (cfg.block_frames, lh, lw),
            ffn_chunk,
            Some(crate::dit::Attention::Dense),
        );
        let cache = KvCache::new(be, c.layers, c.heads, c.head_dim, tpf, &cfg)?;
        let mods = cfg
            .timesteps
            .iter()
            .map(|&t| dit.step_mods(be, t))
            .collect();
        let clean_mods = dit.step_mods(be, 0.0);
        let n = c.in_channels * cfg.block_frames * lh * lw;
        Ok(Self {
            dit,
            ws,
            cache,
            mods,
            clean_mods,
            x: be.alloc(n),
            v: be.alloc(n),
            noise: be.alloc(n),
            rng: Rng::new(seed),
            next_frame: 0,
            latent_hw,
            cfg,
        })
    }

    pub fn config(&self) -> &CausalConfig {
        &self.cfg
    }

    /// Latent frames generated so far.
    pub fn frames_done(&self) -> usize {
        self.next_frame
    }

    /// Queue the next block: denoise from fresh noise, then rewrite its cache
    /// entries from the clean result. Returns the block's clean latent on the
    /// device, `[C, block_frames, h, w]`, valid until the next call.
    pub fn next_block(&mut self, be: &B, prompt: &Prompt<B>) -> Result<&B::Buf> {
        let c = &self.dit.cfg;
        let (lh, lw) = self.latent_hw;
        let nf = self.cfg.block_frames;
        let start = self.next_frame;
        let n = c.in_channels * nf * lh * lw;
        self.cache.begin_block(start, nf)?;
        let rope = fastwan_core::rope::rope_table(c, (nf, lh / c.patch, lw / c.patch), start);
        let (cos, sin) = (be.upload_table(&rope.cos), be.upload_table(&rope.sin));
        be.write(&self.x, &self.rng.normal(n));
        let steps = self.cfg.timesteps.len();
        for i in 0..steps {
            self.dit.forward_causal(
                be,
                &self.ws,
                &self.x,
                &self.mods[i],
                prompt,
                &self.v,
                (&cos, &sin),
                &self.cache,
            );
            let st = self.cfg.timesteps[i] / 1000.0;
            let next = (i + 1 < steps).then(|| self.cfg.timesteps[i + 1] / 1000.0);
            if next.is_some() {
                be.write(&self.noise, &self.rng.normal(n));
            }
            be.dmd_step(
                &self.x,
                &self.v,
                next.map(|_| &self.noise),
                st,
                next.unwrap_or(0.0),
                n,
            );
        }
        // The clean rewrite: same slots, timestep 0, output unused.
        self.dit.forward_causal(
            be,
            &self.ws,
            &self.x,
            &self.clean_mods,
            prompt,
            &self.v,
            (&cos, &sin),
            &self.cache,
        );
        self.next_frame += nf;
        Ok(&self.x)
    }
}

/// `[C, T, h, w]` → frame-major `[T, C, h, w]` (what TAEHV decodes).
pub fn frame_major(latent: &[f32], channels: usize, frames: usize, plane: usize) -> Vec<f32> {
    let mut z = Vec::with_capacity(latent.len());
    for f in 0..frames {
        for ch in 0..channels {
            z.extend_from_slice(&latent[(ch * frames + f) * plane..][..plane]);
        }
    }
    z
}
