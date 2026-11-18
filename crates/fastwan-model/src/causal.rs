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
//!
//! [`Memory::MemRope`] is History Forcing's long-video memory (MemRoPE):
//! the cache is laid out `[sink | long EMA | short EMA | recent | block]`
//! in slot order, keys are stored without RoPE and rotated at their slot
//! index on every forward (block-relative RoPE: the current block's queries
//! sit at the last slots, so positions never grow), and the frames a new
//! block evicts from the front of the recent window are averaged (over all
//! their tokens, per head and channel) into the two memory frames,
//! `m ← α·mean + (1 − α)·m` with α 0.01 (long) and 0.1 (short); the first
//! eviction sets both to the mean. Checked against MemRoPE's PyTorch
//! attention (`scripts/reference/make_memrope_fixture.py`).

use crate::dit::{Prompt, StepMods, WanDit, Workspace};
use fastwan_core::{Backend, Error, Result, rng::Rng};

/// What the cache keeps once it is full.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Memory {
    /// A sink and a rolling window of the most recent frames, keys rotated
    /// at their absolute frame before they are stored (Self-Forcing,
    /// LongLive).
    Window,
    /// MemRoPE (History Forcing): a sink, two EMA memory frames and a
    /// recent window, keys rotated at their slot index on every forward.
    MemRope { alpha_long: f32, alpha_short: f32 },
}

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
    /// Store the K/V cache as 16-bit values where the backend can
    /// ([`Backend::kv_caps`]): half the cache memory (phones), f16 rounding
    /// of keys and values. Ignored where unsupported, and with MemRoPE (its
    /// `kv_ema` / `rope_heads` / shifts work on f32 caches).
    pub half_kv: bool,
    /// The cache policy once it is full.
    pub memory: Memory,
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
            half_kv: false,
            memory: Memory::Window,
        }
    }

    /// History Forcing's DMD checkpoint (the same Wan2.1-T2V-1.3B causal
    /// generator and 4-step schedule as Self-Forcing) with its MemRoPE
    /// memory: a 12-frame cache of 3 sink frames, a long and a short EMA
    /// frame (α 0.01 / 0.1), 4 recent frames and the 3-frame block
    /// (`configs/long_video/inference.yaml`).
    pub fn history_forcing() -> Self {
        Self {
            memory: Memory::MemRope {
                alpha_long: 0.01,
                alpha_short: 0.1,
            },
            ..Self::self_forcing()
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
    /// The valid prefix packed with its own stride while the cache fills,
    /// for backends whose attention cannot read a cache in place.
    pack: Option<(B::Buf, B::Buf)>,
    /// Attention reads the cache in place ([`Backend::attention_kv`]).
    strided: bool,
    /// 16-bit storage.
    half: bool,
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
    memory: Memory,
    /// MemRoPE: valid frames (a prefix of the slots), the current block's
    /// first slot, whether the memory frames hold an EMA yet, and the key
    /// RoPE table for slot positions `0..cap`.
    valid: usize,
    block_slot: usize,
    ema_ready: bool,
    key_rope: Option<(B::Buf, B::Buf)>,
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
        if let Memory::MemRope { .. } = cfg.memory
            && cfg.cap_frames < cfg.sink_frames + 2 + cfg.block_frames
        {
            return Err(Error::Geometry(format!(
                "MemRoPE cache of {} frames cannot hold a {}-frame sink, two memory frames and a {}-frame block",
                cfg.cap_frames, cfg.sink_frames, cfg.block_frames
            )));
        }
        let n = heads * cfg.cap_frames * tpf * head_dim;
        let (strided, can_half) = be.kv_caps(head_dim);
        let memrope = matches!(cfg.memory, Memory::MemRope { .. });
        // MemRoPE's ops (kv_ema, rope_heads, the recent-window shift) read
        // and write the cache as f32 (bf16 on CUDA): no 16-bit cache there.
        let half = cfg.half_kv && can_half && !memrope;
        Ok(Self {
            k: (0..layers).map(|_| be.alloc_kv(n, half)).collect(),
            v: (0..layers).map(|_| be.alloc_kv(n, half)).collect(),
            // MemRoPE always needs a scratch (rotated keys, the shift).
            pack: (!strided || memrope).then(|| (be.alloc(n), be.alloc(n))),
            strided,
            half,
            heads,
            head_dim,
            tpf,
            cap: cfg.cap_frames,
            sink: cfg.sink_frames,
            slot_frame: vec![None; cfg.cap_frames],
            runs: Vec::new(),
            block_frames: 0,
            last_start: None,
            memory: cfg.memory,
            valid: 0,
            block_slot: 0,
            ema_ready: false,
            key_rope: None,
        })
    }

    /// Whether keys are stored without RoPE and rotated at their slot on
    /// every forward (MemRoPE); `head_split` then leaves K unrotated.
    pub fn rotates_keys(&self) -> bool {
        matches!(self.memory, Memory::MemRope { .. })
    }

    /// The RoPE table for slot positions `0..cap` (MemRoPE), `[cap · tpf,
    /// head_dim / 2]`: `rope_table(cfg, (cap, gh, gw), 0)`.
    pub fn set_key_rope(&mut self, cos: B::Buf, sin: B::Buf) {
        self.key_rope = Some((cos, sin));
    }

    /// The temporal RoPE position of the block starting at latent frame
    /// `start` (call after [`KvCache::begin_block`]): the frame itself, or
    /// with MemRoPE its first slot.
    pub fn query_frame(&self, start: usize) -> usize {
        match self.memory {
            Memory::Window => start,
            Memory::MemRope { .. } => self.block_slot,
        }
    }

    /// Choose the slots for the block starting at latent frame `start`. The
    /// same block again (the next denoise step, the clean rewrite) keeps its
    /// slots. A new block fills free slots first, then the oldest non-sink;
    /// with MemRoPE it is appended, and once the cache is full the frames it
    /// evicts are folded into the memory frames and the rest of the recent
    /// window moves up (on the device, every layer, before the block runs).
    pub fn begin_block(&mut self, be: &B, start: usize, frames: usize) -> Result<()> {
        if self.last_start == Some(start) && self.block_frames == frames {
            return Ok(());
        }
        if let Memory::MemRope {
            alpha_long,
            alpha_short,
        } = self.memory
        {
            self.memrope_block(be, frames, [alpha_long, alpha_short])?;
            self.runs = vec![Run {
                from: 0,
                slot: self.block_slot,
                frames,
            }];
            self.last_start = Some(start);
            self.block_frames = frames;
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

    fn memrope_block(&mut self, be: &B, frames: usize, alphas: [f32; 2]) -> Result<()> {
        if self.valid + frames <= self.cap {
            self.block_slot = self.valid;
            self.valid += frames;
            return Ok(());
        }
        let (tpf, hd) = (self.tpf, self.head_dim);
        let stride = self.cap * tpf * hd;
        let mem = self.sink + 2;
        let evict = self.valid + frames - self.cap;
        if self.valid < mem + evict {
            return Err(Error::Geometry(format!(
                "MemRoPE: {} valid frames cannot lose {evict} after a {}-frame sink and two memory frames",
                self.valid, self.sink
            )));
        }
        let kept = self.valid - mem - evict;
        let alphas = if self.ema_ready { alphas } else { [1.0, 1.0] };
        let (pk, pv) = self.pack.as_ref().expect("a MemRoPE scratch");
        for (bufs, scratch) in [(&self.k, pk), (&self.v, pv)] {
            for buf in bufs {
                be.kv_ema(
                    buf,
                    self.heads,
                    stride,
                    hd,
                    mem * tpf,
                    evict * tpf,
                    self.sink * tpf,
                    tpf,
                    alphas,
                );
                if kept > 0 {
                    // Shift the kept recent frames up through the scratch
                    // (a buffer cannot be copied onto itself on WebGPU).
                    let len = kept * tpf * hd;
                    for h in 0..self.heads {
                        be.copy(
                            buf,
                            h * stride + (mem + evict) * tpf * hd,
                            scratch,
                            h * len,
                            len,
                        );
                    }
                    for h in 0..self.heads {
                        be.copy(scratch, h * len, buf, h * stride + mem * tpf * hd, len);
                    }
                }
            }
        }
        self.ema_ready = true;
        self.block_slot = mem + kept;
        self.valid = self.block_slot + frames;
        Ok(())
    }

    /// Valid frames (all of them once the cache is full).
    pub fn frames(&self) -> usize {
        match self.memory {
            Memory::Window => self.slot_frame.iter().filter(|s| s.is_some()).count(),
            Memory::MemRope { .. } => self.valid,
        }
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
        let (hd, tpf, heads) = (self.head_dim, self.tpf, self.heads);
        let stride = self.cap * tpf;
        // One store per run of slots, all heads at once.
        for r in &self.runs {
            let (from, len, row) = (r.from * tpf, r.frames * tpf, r.slot * tpf);
            be.kv_store(
                k,
                n,
                from,
                len,
                heads,
                hd,
                &self.k[li],
                row,
                stride,
                self.half,
            );
            be.kv_store(
                v,
                n,
                from,
                len,
                heads,
                hd,
                &self.v[li],
                row,
                stride,
                self.half,
            );
        }
        // The valid frames are the prefix [0, valid) while the cache fills
        // (nothing evicted yet), all of it once full.
        let valid = self.frames();
        if let Some((cos, sin)) = &self.key_rope
            && self.rotates_keys()
        {
            // Rotate the valid prefix at slot positions into the scratch.
            // Where attention reads a strided cache (or the cache is full)
            // the scratch keeps the cache's layout and V is read in place;
            // otherwise both are packed per head.
            let rows = valid * tpf;
            let (pk, pv) = self.pack.as_ref().expect("a MemRoPE scratch");
            let in_place = self.strided || valid == self.cap;
            let kstride = if in_place { stride } else { rows };
            be.rope_heads(
                &self.k[li],
                0,
                stride * hd,
                pk,
                0,
                kstride * hd,
                heads,
                rows,
                hd,
                (cos, sin),
            );
            let v = if in_place {
                &self.v[li]
            } else {
                for h in 0..heads {
                    be.copy(&self.v[li], h * stride * hd, pv, h * rows * hd, rows * hd);
                }
                pv
            };
            be.attention_kv(q, pk, v, heads, n, rows, kstride, hd, false, out);
            return;
        }
        assert!(
            !self.rotates_keys(),
            "MemRoPE cache without its key RoPE table"
        );
        if valid == self.cap || self.strided {
            be.attention_kv(
                q,
                &self.k[li],
                &self.v[li],
                heads,
                n,
                valid * tpf,
                stride,
                hd,
                self.half,
                out,
            );
        } else {
            // Pack the prefix so each head is contiguous (f32 caches only).
            let (pk, pv) = self.pack.as_ref().expect("a packing cache");
            let len = valid * tpf * hd;
            for h in 0..heads {
                be.copy(&self.k[li], h * stride * hd, pk, h * len, len);
                be.copy(&self.v[li], h * stride * hd, pv, h * len, len);
            }
            be.attention(q, pk, pv, heads, n, valid * tpf, hd, out);
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
        let mut cache = KvCache::new(be, c.layers, c.heads, c.head_dim, tpf, &cfg)?;
        if cache.rotates_keys() {
            let t =
                fastwan_core::rope::rope_table(c, (cfg.cap_frames, lh / c.patch, lw / c.patch), 0);
            cache.set_key_rope(be.upload_table(&t.cos), be.upload_table(&t.sin));
        }
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
        self.cache.begin_block(be, start, nf)?;
        let rope = fastwan_core::rope::rope_table(
            c,
            (nf, lh / c.patch, lw / c.patch),
            self.cache.query_frame(start),
        );
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
