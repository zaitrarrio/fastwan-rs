//! CUDA backend: cudarc (driver loaded at run time, so this builds on a
//! machine without CUDA), kernels compiled by NVRTC at start-up for the
//! device's own SM, GEMMs on cuBLAS in bf16 with f32 accumulation (tensor
//! cores on every RTX card since Turing… bf16 needs Ampere, sm_80+).
//!
//! Activations and weights are bf16; RoPE tables stay f32
//! ([`Backend::upload_table`]). The non-GEMM kernels are line-for-line twins
//! of the WGSL kernels that `fastwan-model::check` verifies on every CI run;
//! run `fastwan check --backend cuda` on a GPU before trusting a number.

#![allow(clippy::too_many_arguments)]

use cudarc::cublas::{CudaBlas, sys as blas_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use fastwan_core::backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind};
use half::bf16;
use std::sync::Arc;

const KERNELS: &str = include_str!("kernels.cu");

#[derive(Clone)]
pub struct CuBuf {
    s: Arc<CudaSlice<u16>>,
    len: usize,
    /// Stored as f32 (2 u16 words per element) — RoPE tables only.
    f32: bool,
}

/// VSA state: tile tables (u32), f32 pooled means / coarse output, u32 picks.
pub struct CuVsa {
    slot: CuBuf,
    bsz: CuBuf,
    pooled: CuBuf,
    coarse: CuBuf,
    sel: CuBuf,
    nb: usize,
    topk: usize,
    heads: usize,
    d: usize,
    seq: usize,
    scale: f32,
}

pub struct CuWeight {
    buf: CuBuf,
    rows: usize,
    cols: usize,
    /// Weight-only quantized: `buf` holds the codes (int8: 2 per word, int4:
    /// 4 per word), this table the f32 per-group scales (int8) or (scale,
    /// zero) pairs (int4), then the group. [`Backend::linear`] expands them
    /// into the shared bf16 scratch before the GEMM.
    quant: Option<(Quant, CuBuf, usize)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Quant {
    Q8,
    Q4,
}

pub struct Cuda {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    f: Kernels,
    name: String,
    /// Compute capability major ≥ 8: bf16 mma.sync is available.
    ampere: bool,
    /// bf16 scratch for the int8 weight being multiplied (grown on demand;
    /// one stream, so consecutive GEMMs reuse it safely).
    q8_scratch: std::sync::Mutex<Option<CuBuf>>,
}

struct Kernels {
    bias_act: CudaFunction,
    norm: CudaFunction,
    head_split: CudaFunction,
    attention: CudaFunction,
    attention_mma: CudaFunction,
    vsa_pool: CudaFunction,
    vsa_coarse: CudaFunction,
    vsa_fine: CudaFunction,
    vsa_fine_mma: CudaFunction,
    gate_residual: CudaFunction,
    dmd_step: CudaFunction,
    tanh_clamp: CudaFunction,
    patch: CudaFunction,
    conv2d: CudaFunction,
    dequant_q8: CudaFunction,
    dequant_q4: CudaFunction,
}

#[derive(Debug)]
pub struct CudaError(pub String);
impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CudaError {}

fn err(e: impl std::fmt::Debug) -> CudaError {
    CudaError(format!("{e:?}"))
}

/// Compile the kernels to PTX for `arch` (e.g. `compute_89`). Needs only
/// libnvrtc, not a GPU — CI uses this to keep `kernels.cu` compiling.
pub fn compile_kernels(arch: &'static str) -> Result<cudarc::nvrtc::Ptx, CudaError> {
    cudarc::nvrtc::compile_ptx_with_opts(
        KERNELS,
        cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec!["--std=c++17".into()],
            ..Default::default()
        },
    )
    .map_err(err)
}

enum Arg {
    P(u64),
    I(i32),
    L(i64),
    F(f32),
}

impl Cuda {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal).map_err(err)?;
        let (maj, min) = ctx.compute_capability().map_err(err)?;
        let arch: &'static str = Box::leak(format!("compute_{maj}{min}").into_boxed_str());
        let ptx = compile_kernels(arch)?;
        let module = ctx.load_module(ptx).map_err(err)?;
        let get = |n: &str| module.load_function(n).map_err(err);
        let f = Kernels {
            bias_act: get("fw_bias_act")?,
            norm: get("fw_norm")?,
            head_split: get("fw_head_split")?,
            attention: get("fw_attention")?,
            attention_mma: get("fw_attention_mma")?,
            vsa_pool: get("fw_vsa_pool")?,
            vsa_coarse: get("fw_vsa_coarse")?,
            vsa_fine: get("fw_vsa_fine")?,
            vsa_fine_mma: get("fw_vsa_fine_mma")?,
            gate_residual: get("fw_gate_residual")?,
            dmd_step: get("fw_dmd_step")?,
            tanh_clamp: get("fw_tanh_clamp")?,
            patch: get("fw_patch")?,
            conv2d: get("fw_conv2d")?,
            dequant_q8: get("fw_dequant_q8")?,
            dequant_q4: get("fw_dequant_q4")?,
        };
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(err)?;
        let name = format!("cuda {} (sm_{maj}{min})", ctx.name().unwrap_or_default());
        Ok(Self {
            ctx,
            stream,
            blas,
            module,
            f,
            name,
            ampere: maj >= 8,
            q8_scratch: std::sync::Mutex::new(None),
        })
    }

    fn ptr(&self, b: &CuBuf) -> u64 {
        let (p, _sync) = b.s.device_ptr(&self.stream);
        p
    }

    fn opt(&self, b: Option<&CuBuf>) -> u64 {
        b.map_or(0, |b| self.ptr(b))
    }

    fn launch(
        &self,
        f: &CudaFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[Arg],
    ) {
        let mut lb = self.stream.launch_builder(f);
        for a in args {
            match a {
                Arg::P(v) => lb.arg(v),
                Arg::I(v) => lb.arg(v),
                Arg::L(v) => lb.arg(v),
                Arg::F(v) => lb.arg(v),
            };
        }
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
        // SAFETY: every kernel's parameter list is matched by the `args`
        // built next to its launch, and every pointer is a live allocation of
        // at least the extent the kernel indexes (lengths asserted by callers).
        unsafe { lb.launch(cfg) }.expect("kernel launch");
    }

    fn elems(n: usize) -> (u32, u32, u32) {
        ((n.div_ceil(256)) as u32, 1, 1)
    }
}

impl Cuda {
    /// 32-bit words (u32 tables or f32 scratch), via the f32 table path.
    fn words(&self, bits: &[u32]) -> CuBuf {
        let v: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        Backend::upload_table(self, if v.is_empty() { &[0.0][..] } else { &v })
    }
}

impl Backend for Cuda {
    type Buf = CuBuf;
    type Vsa = CuVsa;
    type Weight = CuWeight;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn alloc(&self, len: usize) -> CuBuf {
        CuBuf {
            s: Arc::new(
                self.stream
                    .alloc_zeros::<u16>(len.max(1))
                    .expect("cuda alloc"),
            ),
            len,
            f32: false,
        }
    }

    fn upload(&self, data: &[f32]) -> CuBuf {
        let h: Vec<u16> = data.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let h = if h.is_empty() { vec![0] } else { h };
        CuBuf {
            s: Arc::new(self.stream.clone_htod(&h).expect("upload")),
            len: data.len(),
            f32: false,
        }
    }

    fn upload_table(&self, data: &[f32]) -> CuBuf {
        let h: Vec<u16> = data
            .iter()
            .flat_map(|v| {
                let b = v.to_bits();
                [(b & 0xffff) as u16, (b >> 16) as u16]
            })
            .collect();
        CuBuf {
            s: Arc::new(self.stream.clone_htod(&h).expect("upload")),
            len: data.len(),
            f32: true,
        }
    }

    fn write(&self, buf: &CuBuf, data: &[f32]) {
        assert!(!buf.f32 && data.len() <= buf.len);
        let h: Vec<u16> = data.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        // SAFETY: `h` is a host slice of `data.len()` u16s and the
        // destination holds at least that many; the copy is ordered on this
        // stream and `h` is kept alive until the stream is synchronised.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(self.ptr(buf), &h, self.stream.cu_stream())
                .expect("write");
        }
        self.stream.synchronize().expect("sync");
    }

    fn download(&self, buf: &CuBuf) -> Vec<f32> {
        let h = self.stream.clone_dtoh(&*buf.s).expect("download");
        if buf.f32 {
            h.chunks_exact(2)
                .map(|p| f32::from_bits(p[0] as u32 | (p[1] as u32) << 16))
                .collect()
        } else {
            h[..buf.len]
                .iter()
                .map(|&b| bf16::from_bits(b).to_f32())
                .collect()
        }
    }

    fn buf_len(&self, buf: &CuBuf) -> usize {
        buf.len
    }

    fn copy(&self, src: &CuBuf, so: usize, dst: &CuBuf, d: usize, len: usize) {
        if len == 0 {
            return;
        }
        assert!(!src.f32 && !dst.f32 && so + len <= src.len && d + len <= dst.len);
        // SAFETY: both ranges are inside their allocations (asserted above).
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                self.ptr(dst) + (d * 2) as u64,
                self.ptr(src) + (so * 2) as u64,
                len * 2,
                self.stream.cu_stream(),
            )
            .expect("dtod");
        }
    }

    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> CuWeight {
        assert_eq!(data.len(), rows * cols);
        CuWeight {
            buf: self.upload(data),
            rows,
            cols,
            quant: None,
        }
    }

    fn weight_q8(&self, q: &fastwan_core::quant::Q8) -> CuWeight {
        let mut words: Vec<u16> =
            q.q.chunks(2)
                .map(|c| c[0] as u8 as u16 | (c.get(1).map_or(0, |&v| v as u8 as u16) << 8))
                .collect();
        if words.is_empty() {
            words.push(0);
        }
        CuWeight {
            buf: CuBuf {
                s: Arc::new(self.stream.clone_htod(&words).expect("upload")),
                len: words.len(),
                f32: false,
            },
            rows: q.rows,
            cols: q.cols,
            quant: Some((Quant::Q8, Backend::upload_table(self, &q.scales), q.group)),
        }
    }

    fn weight_q4(&self, q: &fastwan_core::quant::Q4) -> CuWeight {
        let bytes = q.packed();
        let mut words: Vec<u16> = bytes
            .chunks(2)
            .map(|c| c[0] as u16 | (c.get(1).map_or(0, |&v| v as u16) << 8))
            .collect();
        if words.is_empty() {
            words.push(0);
        }
        let sz: Vec<f32> = q
            .scales
            .iter()
            .zip(&q.zeros)
            .flat_map(|(&s, &z)| [s, z])
            .collect();
        CuWeight {
            buf: CuBuf {
                s: Arc::new(self.stream.clone_htod(&words).expect("upload")),
                len: words.len(),
                f32: false,
            },
            rows: q.rows,
            cols: q.cols,
            quant: Some((Quant::Q4, Backend::upload_table(self, &sz), q.group)),
        }
    }

    fn sync(&self) {
        self.stream.synchronize().expect("sync");
    }

    fn linear(
        &self,
        x: &CuBuf,
        x_row: usize,
        rows: usize,
        w: &CuWeight,
        bias: Option<&CuBuf>,
        act: Act,
        y: &CuBuf,
        y_row: usize,
    ) {
        let (n, k) = (w.rows, w.cols);
        assert!((x_row + rows) * k <= x.len && (y_row + rows) * n <= y.len);
        let mut scratch = self.q8_scratch.lock().unwrap();
        let wp = match &w.quant {
            None => self.ptr(&w.buf),
            Some((kind, scales, group)) => {
                if scratch.as_ref().is_none_or(|b| b.len < n * k) {
                    *scratch = Some(self.alloc(n * k));
                }
                let dst = scratch.as_ref().unwrap();
                self.launch(
                    match kind {
                        Quant::Q8 => &self.f.dequant_q8,
                        Quant::Q4 => &self.f.dequant_q4,
                    },
                    Self::elems(n * k),
                    (256, 1, 1),
                    &[
                        Arg::P(self.ptr(dst)),
                        Arg::P(self.ptr(&w.buf)),
                        Arg::P(self.ptr(scales)),
                        Arg::L((n * k) as i64),
                        Arg::I(*group as i32),
                    ],
                );
                self.ptr(dst)
            }
        };
        let (alpha, beta) = (1.0f32, 0.0f32);
        let yp = self.ptr(y) + (y_row * n * 2) as u64;
        // Row-major y[m,n] = x[m,k]·W[n,k]ᵀ is column-major yᵀ = W·xᵀ:
        // A = W (transposed), B = x, C = y, all with leading dims k, k, n.
        // SAFETY: pointers and extents checked above; alpha/beta outlive the call.
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                blas_sys::cublasOperation_t::CUBLAS_OP_T,
                blas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                rows as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                wp as *const _,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (self.ptr(x) + (x_row * k * 2) as u64) as *const _,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                k as i32,
                (&beta) as *const f32 as *const _,
                yp as *mut _,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                n as i32,
                blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .expect("cublas gemm");
        }
        if bias.is_some() || act != Act::None {
            let total = rows * n;
            self.launch(
                &self.f.bias_act,
                Self::elems(total),
                (256, 1, 1),
                &[
                    Arg::P(yp),
                    Arg::P(self.opt(bias)),
                    Arg::L(total as i64),
                    Arg::I(n as i32),
                    Arg::I(bias.is_some() as i32),
                    Arg::I((act == Act::GeluTanh) as i32),
                ],
            );
        }
    }

    fn layer_norm(
        &self,
        x: &CuBuf,
        rows: usize,
        dim: usize,
        norm: NormKind<'_, CuBuf>,
        mods: &CuBuf,
        eps: f32,
        out: &CuBuf,
    ) {
        self.norm_impl(x, None, rows, dim, mods, None, norm, eps, out);
    }

    fn residual_norm(
        &self,
        h: &CuBuf,
        a: &CuBuf,
        rows: usize,
        dim: usize,
        mods: &CuBuf,
        gate: Option<usize>,
        norm: NormKind<'_, CuBuf>,
        eps: f32,
        normed: &CuBuf,
    ) {
        self.norm_impl(h, Some(a), rows, dim, mods, gate, norm, eps, normed);
    }

    fn gate_residual(
        &self,
        h: &CuBuf,
        a: &CuBuf,
        rows: usize,
        dim: usize,
        mods: &CuBuf,
        gate: usize,
    ) {
        let n = rows * dim;
        self.launch(
            &self.f.gate_residual,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(h)),
                Arg::P(self.ptr(a)),
                Arg::P(self.ptr(mods)),
                Arg::L(n as i64),
                Arg::I(dim as i32),
                Arg::I(gate as i32),
            ],
        );
    }

    fn head_split(
        &self,
        src: &CuBuf,
        rows: usize,
        stride: usize,
        col_off: usize,
        heads: usize,
        hd: usize,
        rms_w: Option<&CuBuf>,
        rope: Option<(&CuBuf, &CuBuf)>,
        eps: f32,
        out: &CuBuf,
    ) {
        if let Some((c, s)) = rope {
            assert!(c.f32 && s.f32, "RoPE tables must come from upload_table");
        }
        self.launch(
            &self.f.head_split,
            (rows as u32, 1, 1),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(src)),
                Arg::P(self.opt(rms_w)),
                Arg::P(self.opt(rope.map(|r| r.0))),
                Arg::P(self.opt(rope.map(|r| r.1))),
                Arg::P(self.ptr(out)),
                Arg::I(rows as i32),
                Arg::I(stride as i32),
                Arg::I(col_off as i32),
                Arg::I(heads as i32),
                Arg::I(hd as i32),
                Arg::I(rms_w.is_some() as i32),
                Arg::I(rope.is_some() as i32),
                Arg::F(eps),
            ],
        );
    }

    fn attention(
        &self,
        q: &CuBuf,
        k: &CuBuf,
        v: &CuBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        out: &CuBuf,
    ) {
        assert!(d <= 128);
        let scale = 1.0 / (d as f32).sqrt();
        if d == 128 && self.ampere && std::env::var("FASTWAN_CUDA_SCALAR_ATTN").is_err() {
            self.launch(
                &self.f.attention_mma,
                (sq.div_ceil(64) as u32, heads as u32, 1),
                (128, 1, 1),
                &[
                    Arg::P(self.ptr(q)),
                    Arg::P(self.ptr(k)),
                    Arg::P(self.ptr(v)),
                    Arg::P(self.ptr(out)),
                    Arg::I(heads as i32),
                    Arg::I(sq as i32),
                    Arg::I(sk as i32),
                    Arg::F(scale),
                ],
            );
            return;
        }
        self.launch(
            &self.f.attention,
            (sq.div_ceil(64) as u32, heads as u32, 1),
            (64, 1, 1),
            &[
                Arg::P(self.ptr(q)),
                Arg::P(self.ptr(k)),
                Arg::P(self.ptr(v)),
                Arg::P(self.ptr(out)),
                Arg::I(heads as i32),
                Arg::I(sq as i32),
                Arg::I(sk as i32),
                Arg::I(d as i32),
                Arg::F(scale),
            ],
        );
    }

    fn vsa_prepare(&self, spec: &fastwan_core::vsa::VsaSpec) -> CuVsa {
        let (nb, h, d) = (spec.plan.num_tiles(), spec.heads, spec.head_dim);
        assert!(
            nb <= 2048,
            "VSA coarse stage holds at most 2048 tiles per block"
        );
        CuVsa {
            slot: self.words(&spec.plan.slot_src),
            bsz: self.words(&spec.plan.block_sizes),
            pooled: self.words(&vec![0; 3 * h * nb * d]),
            coarse: self.words(&vec![0; h * nb * d]),
            sel: self.words(&vec![0; h * nb * spec.topk]),
            nb,
            topk: spec.topk,
            heads: h,
            d,
            seq: spec.plan.seq,
            scale: spec.scale,
        }
    }

    fn vsa_attention(&self, q: &CuBuf, k: &CuBuf, v: &CuBuf, gate: &CuBuf, s: &CuVsa, out: &CuBuf) {
        let grid = (s.nb as u32, s.heads as u32, 1);
        self.launch(
            &self.f.vsa_pool,
            grid,
            (128, 1, 1),
            &[
                Arg::P(self.ptr(q)),
                Arg::P(self.ptr(k)),
                Arg::P(self.ptr(v)),
                Arg::P(self.ptr(&s.slot)),
                Arg::P(self.ptr(&s.bsz)),
                Arg::P(self.ptr(&s.pooled)),
                Arg::I(s.heads as i32),
                Arg::I(s.seq as i32),
                Arg::I(s.d as i32),
                Arg::I(s.nb as i32),
            ],
        );
        self.launch(
            &self.f.vsa_coarse,
            grid,
            (256, 1, 1),
            &[
                Arg::P(self.ptr(&s.pooled)),
                Arg::P(self.ptr(&s.coarse)),
                Arg::P(self.ptr(&s.sel)),
                Arg::I(s.heads as i32),
                Arg::I(s.d as i32),
                Arg::I(s.nb as i32),
                Arg::I(s.topk as i32),
                Arg::F(s.scale),
            ],
        );
        let tail = [
            Arg::P(self.ptr(&s.slot)),
            Arg::P(self.ptr(&s.sel)),
            Arg::P(self.ptr(&s.coarse)),
            Arg::P(self.ptr(gate)),
            Arg::I(s.nb as i32),
            Arg::I(s.topk as i32),
        ];
        let head = [
            Arg::P(self.ptr(q)),
            Arg::P(self.ptr(k)),
            Arg::P(self.ptr(v)),
            Arg::P(self.ptr(out)),
            Arg::I(s.heads as i32),
            Arg::I(s.seq as i32),
        ];
        if s.d == 128 && self.ampere && std::env::var("FASTWAN_CUDA_SCALAR_ATTN").is_err() {
            let args: Vec<Arg> = head
                .into_iter()
                .chain([Arg::F(s.scale)])
                .chain(tail)
                .collect();
            self.launch(&self.f.vsa_fine_mma, grid, (128, 1, 1), &args);
        } else {
            let args: Vec<Arg> = head
                .into_iter()
                .chain([Arg::I(s.d as i32), Arg::F(s.scale)])
                .chain(tail)
                .collect();
            self.launch(&self.f.vsa_fine, grid, (64, 1, 1), &args);
        }
    }

    fn patchify(
        &self,
        latent: &CuBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &CuBuf,
    ) {
        self.patch_impl(out, latent, c, t, h, w, p, 0);
    }

    fn unpatchify(&self, x: &CuBuf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &CuBuf) {
        self.patch_impl(out, x, c, t, h, w, p, 1);
    }

    fn dmd_step(&self, x: &CuBuf, v: &CuBuf, noise: Option<&CuBuf>, st: f32, sn: f32, n: usize) {
        self.launch(
            &self.f.dmd_step,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(v)),
                Arg::P(self.opt(noise)),
                Arg::L(n as i64),
                Arg::I(noise.is_some() as i32),
                Arg::F(st),
                Arg::F(sn),
            ],
        );
    }

    fn conv2d(
        &self,
        a: &CuBuf,
        b: Option<&CuBuf>,
        s: &ConvShape,
        w: &CuWeight,
        bias: Option<&CuBuf>,
        ep: ConvEpilogue<'_, CuBuf>,
        out: &CuBuf,
    ) {
        assert_eq!((w.rows, w.cols), (s.cout, s.k_len()));
        assert!(w.quant.is_none(), "conv weights are not quantized");
        let (epc, res) = match ep {
            ConvEpilogue::None => (0, None),
            ConvEpilogue::Relu => (1, None),
            ConvEpilogue::AddRelu(r) => (2, Some(r)),
        };
        let (oh, ow) = s.out_hw();
        let mt = (s.frames * oh * ow).div_ceil(64) as u32;
        let (gy, gz) = if mt <= 65535 {
            (mt, 1)
        } else {
            let z = mt.div_ceil(65535);
            (mt.div_ceil(z), z)
        };
        self.launch(
            &self.f.conv2d,
            (s.cout.div_ceil(64) as u32, gy, gz),
            (16, 16, 1),
            &[
                Arg::P(self.ptr(a)),
                Arg::P(self.opt(b)),
                Arg::P(self.ptr(&w.buf)),
                Arg::P(self.opt(bias)),
                Arg::P(self.opt(res)),
                Arg::P(self.ptr(out)),
                Arg::I(s.frames as i32),
                Arg::I(s.cin_a as i32),
                Arg::I(s.cin_b as i32),
                Arg::I(s.cout as i32),
                Arg::I(s.h as i32),
                Arg::I(s.w as i32),
                Arg::I(if s.upsample { 2 } else { 1 }),
                Arg::I(s.k as i32),
                Arg::I(bias.is_some() as i32),
                Arg::I(epc),
            ],
        );
    }

    fn tanh_clamp(&self, x: &CuBuf, n: usize, s: f32) {
        self.launch(
            &self.f.tanh_clamp,
            Self::elems(n),
            (256, 1, 1),
            &[Arg::P(self.ptr(x)), Arg::L(n as i64), Arg::F(s)],
        );
    }
}

impl Cuda {
    fn norm_impl(
        &self,
        h: &CuBuf,
        a: Option<&CuBuf>,
        rows: usize,
        dim: usize,
        mods: &CuBuf,
        gate: Option<usize>,
        norm: NormKind<'_, CuBuf>,
        eps: f32,
        out: &CuBuf,
    ) {
        let (affine, scale, shift, w, b) = match norm {
            NormKind::Affine { w, b } => (1, 0, 0, Some(w), Some(b)),
            NormKind::Mod { scale, shift } => (0, scale as i32, shift as i32, None, None),
        };
        self.launch(
            &self.f.norm,
            (rows as u32, 1, 1),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(h)),
                Arg::P(self.opt(a)),
                Arg::P(self.ptr(mods)),
                Arg::P(self.opt(w)),
                Arg::P(self.opt(b)),
                Arg::P(self.ptr(out)),
                Arg::I(rows as i32),
                Arg::I(dim as i32),
                Arg::I(a.is_some() as i32),
                Arg::I(gate.unwrap_or(0) as i32),
                Arg::I(gate.is_some() as i32),
                Arg::I(affine),
                Arg::I(scale),
                Arg::I(shift),
                Arg::F(eps),
            ],
        );
    }

    fn patch_impl(
        &self,
        dst: &CuBuf,
        src: &CuBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        dir: i32,
    ) {
        let n = c * t * h * w;
        self.launch(
            &self.f.patch,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(dst)),
                Arg::P(self.ptr(src)),
                Arg::L(n as i64),
                Arg::I(c as i32),
                Arg::I(t as i32),
                Arg::I(h as i32),
                Arg::I(w as i32),
                Arg::I(p as i32),
                Arg::I(dir),
            ],
        );
    }
}

#[cfg(test)]
mod tests {
    /// Needs libnvrtc only (no GPU): `pip install nvidia-cuda-nvrtc-cu12` and
    /// point `LD_LIBRARY_PATH` at it. Skipped when NVRTC cannot be loaded.
    #[test]
    fn kernels_compile_for_ampere_ada_and_blackwell() {
        for arch in ["compute_80", "compute_89", "compute_120"] {
            match std::panic::catch_unwind(|| super::compile_kernels(arch)) {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => panic!("{arch}: {e}"),
                Err(_) => {
                    eprintln!("skipping: libnvrtc not loadable");
                    return;
                }
            }
        }
    }
}
