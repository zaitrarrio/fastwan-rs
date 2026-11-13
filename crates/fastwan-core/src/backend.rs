//! The op vocabulary every backend implements.
//!
//! It is deliberately *fused* and deliberately small: each op is one kernel on
//! a GPU, and the set is exactly what a FastWan DMD step plus a TAEHV decode
//! needs — nothing a general tensor library would add. Shapes are passed as
//! integers, never stored; buffers are flat and contiguous.
//!
//! Activation precision is the backend's business (`f32` on CPU and WebGPU,
//! `bf16` on CUDA): [`Backend::upload`] / [`Backend::download`] convert at the
//! boundary and nothing above this trait sees a dtype.
//!
//! Aliasing rule: an output buffer never aliases an input, except the
//! arguments documented as *in place*.

/// Epilogue activation of [`Backend::linear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    /// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` (PyTorch `gelu(approximate="tanh")`).
    GeluTanh,
}

/// The norm half of a fused residual + LayerNorm.
#[derive(Debug, Clone, Copy)]
pub enum NormKind<'a, T> {
    /// `LN(x) · w + b` (Wan `norm2`, the cross-attention norm).
    Affine { w: &'a T, b: &'a T },
    /// `LN(x) · (1 + mods[scale]) + mods[shift]` (AdaLN, no affine).
    Mod { scale: usize, shift: usize },
}

/// Epilogue of [`Backend::conv2d`], after the bias.
#[derive(Debug, Clone, Copy)]
pub enum ConvEpilogue<'a, T> {
    None,
    Relu,
    /// `relu(y + r)` with `r` shaped like the output (TAEHV MemBlock skip).
    AddRelu(&'a T),
    /// PReLU: `y ≥ 0 ? y : y · slope[c]`, one slope per output channel
    /// (Real-ESRGAN's compact upscaler).
    Prelu(&'a T),
}

/// [`Backend::pack_rgba8`]: `src` is `[skip + frames, 3·p², h, w]` (a
/// pixel-shuffled RGB map, as TAEHV and the upscaler end); the output is
/// `frames` images of `h·p × w·p` packed pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackShape {
    /// Frames written (the first `skip` source frames are dropped).
    pub frames: usize,
    pub skip: usize,
    pub h: usize,
    pub w: usize,
    pub p: usize,
}

impl PackShape {
    pub fn pixels(&self) -> usize {
        self.frames * self.h * self.p * self.w * self.p
    }
}

/// One packed pixel as [`Backend::pack_rgba8`] writes it: `R | G<<8 | B<<16
/// | 255<<24` from values in `[0, 1]`, the same rounding as the host paths it
/// replaced (`(clamp(v) · 255 + 0.5) as u8`).
pub fn pack_pixel(r: f32, g: f32, b: f32) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    q(r) | q(g) << 8 | q(b) << 16 | 0xff00_0000
}

/// The bytes of a downloaded [`Backend::alloc_words`] buffer: each f32's bit
/// pattern is one packed pixel, so this is RGBA, 4 bytes per pixel.
pub fn rgba8_bytes(words: &[f32]) -> Vec<u8> {
    words
        .iter()
        .flat_map(|w| w.to_bits().to_le_bytes())
        .collect()
}

/// RGBA → RGB (drops alpha).
pub fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

/// A 2-D convolution over `frames` independent images, NCHW, stride 1,
/// "same" padding (`k / 2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvShape {
    pub frames: usize,
    /// Channels read from the first input.
    pub cin_a: usize,
    /// Channels read from the optional second input, concatenated after the
    /// first (`cat([a, b], dim=1)` without materialising it).
    pub cin_b: usize,
    pub cout: usize,
    /// Input height / width as stored.
    pub h: usize,
    pub w: usize,
    /// Read the inputs through a nearest ×2 upsample (output is `2h × 2w`),
    /// without materialising the upsampled tensor.
    pub upsample: bool,
    /// 1 or 3.
    pub k: usize,
}

impl ConvShape {
    pub fn cin(&self) -> usize {
        self.cin_a + self.cin_b
    }
    pub fn out_hw(&self) -> (usize, usize) {
        let s = if self.upsample { 2 } else { 1 };
        (self.h * s, self.w * s)
    }
    pub fn out_len(&self) -> usize {
        let (h, w) = self.out_hw();
        self.frames * self.cout * h * w
    }
    /// Reduction length of the implicit GEMM (`cin · k²`).
    pub fn k_len(&self) -> usize {
        self.cin() * self.k * self.k
    }
}

pub trait Backend {
    /// A flat device buffer of activations.
    type Buf;
    /// Device state of [`Backend::vsa_attention`] for one token grid: tile
    /// tables and the coarse-stage scratch.
    type Vsa;
    /// A `[rows, cols]` row-major matrix (`nn.Linear` layout: `[out, in]`) in
    /// the backend's storage precision.
    type Weight;

    fn name(&self) -> String;

    // ---- memory ----------------------------------------------------------
    /// Zero-filled.
    fn alloc(&self, len: usize) -> Self::Buf;
    fn upload(&self, data: &[f32]) -> Self::Buf;
    /// A lookup table the kernels read in full precision even on a backend
    /// whose activations are narrower (the RoPE cos/sin: bf16 has 8 bits of
    /// mantissa, too few for rotation angles). Only [`Backend::head_split`]
    /// reads these.
    fn upload_table(&self, data: &[f32]) -> Self::Buf {
        self.upload(data)
    }
    /// Overwrite the first `data.len()` elements.
    fn write(&self, buf: &Self::Buf, data: &[f32]);
    /// Blocking read of the whole buffer (native targets).
    fn download(&self, buf: &Self::Buf) -> Vec<f32>;
    fn buf_len(&self, buf: &Self::Buf) -> usize;
    /// `dst[dst_off..][..len] = src[src_off..][..len]`.
    fn copy(&self, src: &Self::Buf, src_off: usize, dst: &Self::Buf, dst_off: usize, len: usize);
    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> Self::Weight;
    /// An int8 weight-only matrix for [`Backend::linear`]. The result must
    /// equal `linear` with `weight(&q.dequantize())` up to the backend's own
    /// activation rounding.
    fn weight_q8(&self, q: &crate::quant::Q8) -> Self::Weight;
    /// A 4-bit weight-only matrix: the same contract as [`Backend::weight_q8`]
    /// with `weight(&q.dequantize())`.
    fn weight_q4(&self, q: &crate::quant::Q4) -> Self::Weight;
    /// Wait for queued work (timing only; downloads synchronise themselves).
    fn sync(&self);

    /// Per-kernel GPU time since the last [`Backend::profile_reset`], when the
    /// backend was built with profiling on (`FASTWAN_PROFILE=1` on WebGPU).
    fn profile_report(&self) -> Option<String> {
        None
    }
    fn profile_reset(&self) {}

    // ---- transformer -----------------------------------------------------
    /// `y[y_row + i, :] = act(x[x_row + i, :] · Wᵀ + b)` for `i < rows`.
    /// Row offsets let the FFN run in token chunks.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        x: &Self::Buf,
        x_row: usize,
        rows: usize,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        act: Act,
        y: &Self::Buf,
        y_row: usize,
    );

    /// `out = norm(x)` over rows of width `dim`. `mods` is the `[slots, dim]`
    /// modulation table ([`NormKind::Mod`] reads it).
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        x: &Self::Buf,
        rows: usize,
        dim: usize,
        norm: NormKind<'_, Self::Buf>,
        mods: &Self::Buf,
        eps: f32,
        out: &Self::Buf,
    );

    /// In place on `h`: `h += a · mods[gate]` (or `h += a` without a gate),
    /// then `normed = norm(h)`. The norm reads the unrounded sum.
    #[allow(clippy::too_many_arguments)]
    fn residual_norm(
        &self,
        h: &Self::Buf,
        a: &Self::Buf,
        rows: usize,
        dim: usize,
        mods: &Self::Buf,
        gate: Option<usize>,
        norm: NormKind<'_, Self::Buf>,
        eps: f32,
        normed: &Self::Buf,
    );

    /// In place on `h`: `h += a · mods[gate]`.
    fn gate_residual(
        &self,
        h: &Self::Buf,
        a: &Self::Buf,
        rows: usize,
        dim: usize,
        mods: &Self::Buf,
        gate: usize,
    );

    /// Columns `[col_off, col_off + heads·head_dim)` of `src` (`rows` rows of
    /// width `src_stride`) → `out` as `[heads, rows, head_dim]`, optionally
    /// RMS-normalised across all heads (`x · rsqrt(mean(x²) + eps) · w`) and
    /// rotated by RoPE (`cos`/`sin` are `[rows, head_dim / 2]`, interleaved
    /// pairs).
    #[allow(clippy::too_many_arguments)]
    fn head_split(
        &self,
        src: &Self::Buf,
        rows: usize,
        src_stride: usize,
        col_off: usize,
        heads: usize,
        head_dim: usize,
        rms_w: Option<&Self::Buf>,
        rope: Option<(&Self::Buf, &Self::Buf)>,
        eps: f32,
        out: &Self::Buf,
    );

    /// Softmax attention, scale `1/√head_dim`, no mask. `q` is
    /// `[heads, sq, d]`, `k`/`v` `[heads, sk, d]`; `out` is written with heads
    /// merged, `[sq, heads·d]`, ready for `to_out`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        out: &Self::Buf,
    );

    /// Whether [`Backend::vsa_attention`] can run at this head dim on this
    /// device (WebGPU needs its flash kernel: head_dim % 16 == 0, ≤ 128, and
    /// enough workgroup memory).
    fn vsa_supported(&self, _head_dim: usize) -> bool {
        true
    }

    /// Build the VSA state for one token grid ([`crate::vsa::VsaSpec`]).
    fn vsa_prepare(&self, spec: &crate::vsa::VsaSpec) -> Self::Vsa;

    /// Video sparse attention ([`crate::vsa`]). `q`, `k`, `v`, `gate` are
    /// `[heads, seq, d]`; `out` is `[seq, heads·d]`, as from
    /// [`Backend::attention`].
    #[allow(clippy::too_many_arguments)]
    fn vsa_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        gate: &Self::Buf,
        vsa: &Self::Vsa,
        out: &Self::Buf,
    );

    /// Latent `[c, t, h, w]` → tokens `[t·(h/p)·(w/p), c·p²]`, feature index
    /// `c·p² + kh·p + kw` (the Conv3d `(1, p, p)` weight flattened).
    #[allow(clippy::too_many_arguments)]
    fn patchify(
        &self,
        latent: &Self::Buf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &Self::Buf,
    );

    /// Tokens `[t·(h/p)·(w/p), p²·c]` (feature `(kh·p + kw)·c + c'`, Diffusers
    /// unpatchify) → latent `[c, t, h, w]`.
    #[allow(clippy::too_many_arguments)]
    fn unpatchify(
        &self,
        x: &Self::Buf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &Self::Buf,
    );

    /// In place on `x` (`n` elements): `x0 = x − σ_t·v`, then
    /// `x = (1 − σ_n)·x0 + σ_n·noise` when `noise` is given.
    fn dmd_step(
        &self,
        x: &Self::Buf,
        v: &Self::Buf,
        noise: Option<&Self::Buf>,
        sigma_t: f32,
        sigma_next: f32,
        n: usize,
    );

    // ---- decoder ---------------------------------------------------------
    /// `out = epilogue(conv(cat([a, b]), W) + bias)`; `W` is `[cout, cin·k²]`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        a: &Self::Buf,
        b: Option<&Self::Buf>,
        shape: &ConvShape,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        epilogue: ConvEpilogue<'_, Self::Buf>,
        out: &Self::Buf,
    );

    /// In place: `x = tanh(x / s) · s` over the first `n` elements.
    fn tanh_clamp(&self, x: &Self::Buf, n: usize, s: f32);

    // ---- frames out ------------------------------------------------------
    /// A buffer of `n` 32-bit words that `download` returns bit-exactly (each
    /// f32's bits are the word). Activations may be narrower (CUDA: bf16).
    fn alloc_words(&self, n: usize) -> Self::Buf {
        self.alloc(n)
    }
    /// Pixel-shuffle `src`, add `base` upsampled ×p (nearest; `[skip +
    /// frames, 3, h, w]`) when given, clamp, round to 8 bits and pack as
    /// [`pack_pixel`] into `out` (from [`Backend::alloc_words`], one word per
    /// pixel). Frames leave the device at 4 bytes per pixel, ready to show.
    fn pack_rgba8(&self, src: &Self::Buf, base: Option<&Self::Buf>, s: &PackShape, out: &Self::Buf);
}
