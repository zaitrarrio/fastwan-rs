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
use cudarc::cublaslt::{result as lt_result, sys as lt_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use fastwan_core::backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind, PackShape};
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
    /// cuBLASLt with the bias / GELU epilogue fused into the GEMM; `None`
    /// when unavailable or `FASTWAN_CUDA_NO_LT=1` (then GEMM + `fw_bias_act`).
    lt: Option<Lt>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    f: Kernels,
    name: String,
    /// Compute capability major ≥ 8: bf16 mma.sync is available.
    ampere: bool,
    /// bf16 scratch for the int8 weight being multiplied (grown on demand;
    /// one stream, so consecutive GEMMs reuse it safely).
    q8_scratch: std::sync::Mutex<Option<CuBuf>>,
    /// `FASTWAN_CUDA_SCALAR_CONV=1`: the pre-tensor-core conv, for A/B.
    scalar_conv: bool,
    /// `FASTWAN_CUDA_ATTN_VARIANT` = 0 | 1 | 2: which MMA attention kernels
    /// run (see `attn_mma_v` in kernels.cu). Default 2 on sm_120+ (measured
    /// and checked on an RTX PRO 6000: 3% faster VSA fine), 0 elsewhere until
    /// a run checks it there.
    attn_variant: usize,
    /// `FASTWAN_PROFILE=1`: seconds and calls per kernel / GEMM shape. Each
    /// timed op synchronises the stream before and after (≈10 µs each), so a
    /// profiled run is a little slower than a plain one.
    prof: Option<std::sync::Mutex<std::collections::HashMap<String, (f64, usize)>>>,
}

struct Kernels {
    bias_act: CudaFunction,
    norm: CudaFunction,
    head_split: CudaFunction,
    attention: CudaFunction,
    attention_mma: CudaFunction,
    attention_mma_v: [CudaFunction; 2],
    vsa_pool: CudaFunction,
    vsa_coarse: CudaFunction,
    vsa_fine: CudaFunction,
    vsa_fine_mma: CudaFunction,
    vsa_fine_mma_v: [CudaFunction; 2],
    gate_residual: CudaFunction,
    dmd_step: CudaFunction,
    tanh_clamp: CudaFunction,
    add_act: CudaFunction,
    rope_heads: CudaFunction,
    kv_ema: CudaFunction,
    pack_rgba8: CudaFunction,
    unpack_rgba8: CudaFunction,
    patch: CudaFunction,
    conv2d: CudaFunction,
    conv2d_mma: CudaFunction,
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

/// The real SM target (`sm_XY`) NVRTC can emit SASS for, or `None` for an SM
/// it cannot target directly (then the `compute_XY` PTX path is used).
pub fn sass_arch(maj: i32, min: i32) -> Option<&'static str> {
    Some(match (maj, min) {
        (7, 5) => "sm_75",
        (8, 0) => "sm_80",
        (8, 6) => "sm_86",
        (8, 7) => "sm_87",
        (8, 9) => "sm_89",
        (9, 0) => "sm_90",
        (10, 0) => "sm_100",
        (10, 3) => "sm_103",
        (11, 0) => "sm_110",
        (12, 0) => "sm_120",
        (12, 1) => "sm_121",
        _ => return None,
    })
}

/// Compile the kernels to SASS (a cubin) for a real target (`sm_120`), the
/// way fastvideo-rs's `nvrtc_sass` does: a cubin needs no PTX JIT, so an
/// NVRTC newer than the driver (13.4 against R595 rejected the PTX with
/// CUDA_ERROR_UNSUPPORTED_PTX_VERSION) still loads. Needs only libnvrtc.
pub fn compile_cubin(arch: &str) -> Result<Vec<u8>, CudaError> {
    compile_cubin_src(KERNELS, arch)
}

/// [`compile_cubin`] for any CUDA source (examples and benchmarks).
pub fn compile_cubin_src(source: &str, arch: &str) -> Result<Vec<u8>, CudaError> {
    use cudarc::nvrtc::{result, sys};
    use std::ffi::{CStr, CString};
    let src = CString::new(source).map_err(|e| CudaError(format!("nvrtc source: {e}")))?;
    let prog = result::create_program(&src, None)
        .map_err(|e| CudaError(format!("nvrtc create: {e:?}")))?;
    let options = vec![
        "--std=c++17".to_string(),
        format!("--gpu-architecture={arch}"),
    ];
    // SAFETY: `prog` comes from `create_program`, `src` outlives it, and it is
    // destroyed exactly once below whatever happens in between.
    let out = unsafe {
        match result::compile_program(prog, &options) {
            Err(e) => {
                let log = result::get_program_log(prog)
                    .ok()
                    .map(|l| CStr::from_ptr(l.as_ptr()).to_string_lossy().into_owned())
                    .unwrap_or_default();
                Err(CudaError(format!("nvrtc {arch}: {e:?}\n{log}")))
            }
            Ok(()) => {
                let mut size = 0usize;
                match sys::nvrtcGetCUBINSize(prog, &mut size).result() {
                    Err(e) => Err(CudaError(format!("nvrtcGetCUBINSize {arch}: {e:?}"))),
                    Ok(()) if size == 0 => Err(CudaError(format!("nvrtc {arch}: empty cubin"))),
                    Ok(()) => {
                        let mut buf = vec![0u8; size];
                        sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr().cast())
                            .result()
                            .map(|()| buf)
                            .map_err(|e| CudaError(format!("nvrtcGetCUBIN {arch}: {e:?}")))
                    }
                }
            }
        }
    };
    // SAFETY: created above, not destroyed yet.
    let _ = unsafe { result::destroy_program(prog) };
    out
}

/// One cached cuBLASLt plan per (n, k, rows, epilogue).
struct LtPlan {
    desc: lt_sys::cublasLtMatmulDesc_t,
    a: lt_sys::cublasLtMatrixLayout_t,
    b: lt_sys::cublasLtMatrixLayout_t,
    c: lt_sys::cublasLtMatrixLayout_t,
    algo: lt_sys::cublasLtMatmulAlgo_t,
}

struct Lt {
    handle: lt_sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    plans: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize, u8), Option<LtPlan>>>,
}

// SAFETY: the handles are only used from the thread that owns the backend,
// on its one stream; the Mutex serialises the plan cache.
unsafe impl Send for Lt {}
unsafe impl Sync for Lt {}

const LT_WORKSPACE: usize = 32 << 20;

impl Lt {
    fn new(stream: &Arc<CudaStream>) -> Result<Self, CudaError> {
        let handle = lt_result::create_handle().map_err(err)?;
        let workspace = stream.alloc_zeros::<u8>(LT_WORKSPACE).map_err(err)?;
        Ok(Self {
            handle,
            workspace,
            plans: Default::default(),
        })
    }

    /// Column-major yᵀ[n, rows] = Wᵀ… as in the cuBLAS path: A = W (op T,
    /// stored k × n), B = x (k × rows), C = D = y (n × rows), all bf16, f32
    /// accumulate. `epi` 0 = bias, 1 = GELU(·+bias). `None` = not supported.
    fn plan(&self, n: usize, k: usize, rows: usize, epi: u8) -> Option<LtPlan> {
        use lt_sys::cudaDataType_t::{CUDA_R_16BF, CUDA_R_32F};
        let a = lt_result::create_matrix_layout(CUDA_R_16BF, k as u64, n as u64, k as i64).ok()?;
        let b =
            lt_result::create_matrix_layout(CUDA_R_16BF, k as u64, rows as u64, k as i64).ok()?;
        let c =
            lt_result::create_matrix_layout(CUDA_R_16BF, n as u64, rows as u64, n as i64).ok()?;
        let desc = lt_result::create_matmul_desc(
            lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
        )
        .ok()?;
        let t: i32 = 1;
        let epilogue = if epi == 1 {
            lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS
        } else {
            lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS
        };
        let pref = lt_result::create_matmul_pref().ok()?;
        let ws = LT_WORKSPACE as u64;
        // SAFETY: freshly created descriptors; attribute buffers outlive the calls.
        unsafe {
            lt_result::set_matmul_desc_attribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&t) as *const i32 as *const _,
                4,
            )
            .ok()?;
            lt_result::set_matmul_desc_attribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                (&epilogue) as *const _ as *const _,
                std::mem::size_of::<lt_sys::cublasLtEpilogue_t>(),
            )
            .ok()?;
            lt_result::set_matmul_pref_attribute(
                pref,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws) as *const u64 as *const _,
                8,
            )
            .ok()?;
            // A placeholder bias pointer for the heuristic; set per call.
            let dummy: u64 = self.workspace_ptr();
            lt_result::set_matmul_desc_attribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                (&dummy) as *const u64 as *const _,
                8,
            )
            .ok()?;
            let h = lt_result::get_matmul_algo_heuristic(self.handle, desc, a, b, c, c, pref);
            let _ = lt_result::destroy_matmul_pref(pref);
            Some(LtPlan {
                desc,
                a,
                b,
                c,
                algo: h.ok()?.algo,
            })
        }
    }

    fn workspace_ptr(&self) -> u64 {
        // The workspace lives as long as `self`; no stream ordering is needed
        // for a pointer value.
        let (p, _sync) = self.workspace.device_ptr(self.workspace.stream());
        p
    }
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
        // SASS for the device's own SM first (no PTX JIT: an NVRTC newer than
        // the driver still loads); PTX for `compute_XY` when NVRTC has no
        // real target for it or the cubin failed.
        let module = match sass_arch(maj, min).map(compile_cubin) {
            Some(Ok(cubin)) => ctx
                .load_module(cudarc::nvrtc::Ptx::from_binary(cubin))
                .map_err(err)?,
            other => {
                if let Some(Err(e)) = other {
                    eprintln!("cuda: SASS compile failed ({}); falling back to PTX", e.0);
                }
                let arch: &'static str = Box::leak(format!("compute_{maj}{min}").into_boxed_str());
                ctx.load_module(compile_kernels(arch)?).map_err(err)?
            }
        };
        let get = |n: &str| module.load_function(n).map_err(err);
        let f = Kernels {
            bias_act: get("fw_bias_act")?,
            norm: get("fw_norm")?,
            head_split: get("fw_head_split")?,
            attention: get("fw_attention")?,
            attention_mma: get("fw_attention_mma")?,
            attention_mma_v: [get("fw_attention_mma_v1")?, get("fw_attention_mma_v2")?],
            vsa_pool: get("fw_vsa_pool")?,
            vsa_coarse: get("fw_vsa_coarse")?,
            vsa_fine: get("fw_vsa_fine")?,
            vsa_fine_mma: get("fw_vsa_fine_mma")?,
            vsa_fine_mma_v: [get("fw_vsa_fine_mma_v1")?, get("fw_vsa_fine_mma_v2")?],
            gate_residual: get("fw_gate_residual")?,
            dmd_step: get("fw_dmd_step")?,
            tanh_clamp: get("fw_tanh_clamp")?,
            add_act: get("fw_add_act")?,
            rope_heads: get("fw_rope_heads")?,
            kv_ema: get("fw_kv_ema")?,
            pack_rgba8: get("fw_pack_rgba8")?,
            unpack_rgba8: get("fw_unpack_rgba8")?,
            patch: get("fw_patch")?,
            conv2d: get("fw_conv2d")?,
            conv2d_mma: get("fw_conv2d_mma")?,
            dequant_q8: get("fw_dequant_q8")?,
            dequant_q4: get("fw_dequant_q4")?,
        };
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(err)?;
        let lt = if std::env::var("FASTWAN_CUDA_NO_LT").is_ok_and(|v| v == "1") {
            None
        } else {
            Lt::new(&stream).ok()
        };
        let name = format!("cuda {} (sm_{maj}{min})", ctx.name().unwrap_or_default());
        Ok(Self {
            ctx,
            stream,
            blas,
            lt,
            module,
            f,
            name,
            ampere: maj >= 8,
            q8_scratch: std::sync::Mutex::new(None),
            scalar_conv: std::env::var("FASTWAN_CUDA_SCALAR_CONV").is_ok_and(|v| v == "1"),
            attn_variant: std::env::var("FASTWAN_CUDA_ATTN_VARIANT")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| *v <= 2)
                .unwrap_or(if maj >= 12 { 2 } else { 0 }),
            prof: std::env::var("FASTWAN_PROFILE")
                .is_ok_and(|v| v == "1")
                .then(Default::default),
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
        self.timed(
            || self.kernel_name(f).to_string(),
            || unsafe { lb.launch(cfg) }.expect("kernel launch"),
        );
    }

    /// Run `op`; with profiling on, wall time between stream syncs under `tag`.
    fn timed<R>(&self, tag: impl FnOnce() -> String, op: impl FnOnce() -> R) -> R {
        let Some(prof) = &self.prof else {
            return op();
        };
        self.stream.synchronize().expect("sync");
        let t = std::time::Instant::now();
        let r = op();
        self.stream.synchronize().expect("sync");
        let dt = t.elapsed().as_secs_f64();
        let mut m = prof.lock().unwrap();
        let e = m.entry(tag()).or_default();
        e.0 += dt;
        e.1 += 1;
        r
    }

    fn kernel_name(&self, f: &CudaFunction) -> &'static str {
        let k = &self.f;
        let all: [(&CudaFunction, &'static str); 25] = [
            (&k.attention_mma_v[0], "attention (mma v1)"),
            (&k.attention_mma_v[1], "attention (mma v2)"),
            (&k.vsa_fine_mma_v[0], "vsa fine (mma v1)"),
            (&k.vsa_fine_mma_v[1], "vsa fine (mma v2)"),
            (&k.bias_act, "bias_act"),
            (&k.norm, "norm"),
            (&k.head_split, "head_split (qk-norm + rope)"),
            (&k.attention, "attention (scalar)"),
            (&k.attention_mma, "attention (mma)"),
            (&k.vsa_pool, "vsa pool"),
            (&k.vsa_coarse, "vsa coarse + top-k"),
            (&k.vsa_fine, "vsa fine (scalar)"),
            (&k.vsa_fine_mma, "vsa fine (mma)"),
            (&k.gate_residual, "gate_residual"),
            (&k.dmd_step, "dmd_step"),
            (&k.tanh_clamp, "tanh_clamp"),
            (&k.add_act, "add_act"),
            (&k.rope_heads, "rope_heads"),
            (&k.kv_ema, "kv_ema"),
            (&k.patch, "patchify"),
            (&k.conv2d, "conv2d (scalar)"),
            (&k.conv2d_mma, "conv2d (mma)"),
            (&k.dequant_q8, "dequant int8"),
            (&k.dequant_q4, "dequant int4"),
            (&k.bias_act, "bias_act"),
        ];
        all.iter()
            .find(|(g, _)| std::ptr::eq(*g, f))
            .map_or("kernel", |(_, n)| n)
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

impl Cuda {
    fn profile_table(&self) -> Option<String> {
        let prof = self.prof.as_ref()?;
        let m = prof.lock().unwrap();
        let flops = m.get("gemm FLOP (total)").map_or(0.0, |e| e.0);
        let mut rows: Vec<(&String, &(f64, usize))> = m
            .iter()
            .filter(|(k, _)| k.as_str() != "gemm FLOP (total)")
            .collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let total: f64 = rows.iter().map(|r| r.1.0).sum();
        let gemm: f64 = rows
            .iter()
            .filter(|r| r.0.starts_with("gemm"))
            .map(|r| r.1.0)
            .sum();
        let mut out = format!(
            "profile (CUDA, synced per op): {:.1} ms in ops; GEMMs {:.1} ms at {:.1} TFLOP/s\n",
            total * 1e3,
            gemm * 1e3,
            if gemm > 0.0 { flops / gemm / 1e12 } else { 0.0 }
        );
        for (k, (t, n)) in rows {
            out += &format!(
                "  {k:40} {:9.1} ms {:5.1}%  {n:6} calls  {:8.3} ms/call\n",
                t * 1e3,
                100.0 * t / total.max(1e-12),
                t * 1e3 / *n as f64
            );
        }
        Some(out)
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
        let s = self.timed(
            || "alloc (zeroed)".into(),
            || {
                self.stream
                    .alloc_zeros::<u16>(len.max(1))
                    .expect("cuda alloc")
            },
        );
        CuBuf {
            s: Arc::new(s),
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
            // bf16 → f32 is a 16-bit shift; split big buffers (decoded frames)
            // across the cores.
            let src = &h[..buf.len];
            let mut out = vec![0.0f32; src.len()];
            let threads = std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .min(16);
            let conv = |s: &[u16], d: &mut [f32]| {
                for (o, &b) in d.iter_mut().zip(s) {
                    *o = f32::from_bits((b as u32) << 16);
                }
            };
            if threads <= 1 || src.len() < 1 << 20 {
                conv(src, &mut out);
            } else {
                let per = src.len().div_ceil(threads);
                std::thread::scope(|sc| {
                    for (s, d) in src.chunks(per).zip(out.chunks_mut(per)) {
                        sc.spawn(move || conv(s, d));
                    }
                });
            }
            out
        }
    }

    fn buf_len(&self, buf: &CuBuf) -> usize {
        buf.len
    }

    fn copy(&self, src: &CuBuf, so: usize, dst: &CuBuf, d: usize, len: usize) {
        if len == 0 {
            return;
        }
        // bf16 activations (2 bytes) or f32-flagged buffers such as packed
        // RGBA words (4 bytes); both sides must be the same kind.
        assert!(
            src.f32 == dst.f32 && so + len <= src.len && d + len <= dst.len,
            "copy: mixed bf16 / f32 buffers or out of range"
        );
        let e = if src.f32 { 4 } else { 2 };
        // SAFETY: both ranges are inside their allocations (asserted above).
        self.timed(
            || "copy (dtod)".into(),
            || unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    self.ptr(dst) + (d * e) as u64,
                    self.ptr(src) + (so * e) as u64,
                    len * e,
                    self.stream.cu_stream(),
                )
                .expect("dtod");
            },
        );
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

    fn profile_report(&self) -> Option<String> {
        self.profile_table()
    }

    fn profile_reset(&self) {
        if let Some(p) = &self.prof {
            self.stream.synchronize().expect("sync");
            p.lock().unwrap().clear();
        }
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
        let xp = self.ptr(x) + (x_row * k * 2) as u64;
        // cuBLASLt with the bias (+ GELU) in the epilogue: no second pass.
        if let (Some(lt), Some(b)) = (&self.lt, bias) {
            let epi = (act == Act::GeluTanh) as u8;
            let mut plans = lt.plans.lock().unwrap();
            let plan = plans
                .entry((n, k, rows, epi))
                .or_insert_with(|| lt.plan(n, k, rows, epi));
            if let Some(p) = plan {
                let bp = self.ptr(b);
                let ws = lt.workspace_ptr();
                self.timed(
                    || {
                        format!(
                            "gemm+bias{} n={n} k={k}{}",
                            if epi == 1 { "+gelu" } else { "" },
                            if rows < 1024 { " (text)" } else { "" }
                        )
                    },
                    // SAFETY: the plan's layouts match these extents (checked
                    // above); bias holds n values; workspace is LT_WORKSPACE.
                    || unsafe {
                        lt_result::set_matmul_desc_attribute(
                            p.desc,
                            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                            (&bp) as *const u64 as *const _,
                            8,
                        )
                        .expect("lt bias pointer");
                        lt_result::matmul(
                            lt.handle,
                            p.desc,
                            (&alpha) as *const f32 as *const _,
                            (&beta) as *const f32 as *const _,
                            wp as *const _,
                            p.a,
                            xp as *const _,
                            p.b,
                            yp as *const _,
                            p.c,
                            yp as *mut _,
                            p.c,
                            &p.algo,
                            ws as *mut _,
                            LT_WORKSPACE,
                            self.stream.cu_stream() as *mut _,
                        )
                        .expect("cublasLt matmul");
                    },
                );
                if let Some(pr) = &self.prof {
                    let mut m = pr.lock().unwrap();
                    let e = m.entry("gemm FLOP (total)".into()).or_default();
                    e.0 += 2.0 * rows as f64 * n as f64 * k as f64;
                    e.1 += 1;
                }
                return;
            }
        }
        // Row-major y[m,n] = x[m,k]·W[n,k]ᵀ is column-major yᵀ = W·xᵀ:
        // A = W (transposed), B = x, C = y, all with leading dims k, k, n.
        // SAFETY: pointers and extents checked above; alpha/beta outlive the call.
        let flops = 2.0 * rows as f64 * n as f64 * k as f64;
        self.timed(
            || {
                format!(
                    "gemm n={n} k={k}{}",
                    if rows < 1024 { " (text)" } else { "" }
                )
            },
            || unsafe {
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
                    xp as *const _,
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
            },
        );
        if let Some(p) = &self.prof {
            let mut m = p.lock().unwrap();
            let e = m.entry("gemm FLOP (total)".into()).or_default();
            e.0 += flops;
            e.1 += 1;
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
                match self.attn_variant {
                    0 => &self.f.attention_mma,
                    v => &self.f.attention_mma_v[v - 1],
                },
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
            let f = match self.attn_variant {
                0 => &self.f.vsa_fine_mma,
                v => &self.f.vsa_fine_mma_v[v - 1],
            };
            self.launch(f, grid, (128, 1, 1), &args);
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
            ConvEpilogue::Prelu(r) => (3, Some(r)),
        };
        let (oh, ow) = s.out_hw();
        // Tensor-core implicit GEMM on sm_80+ (FASTWAN_CUDA_SCALAR_CONV=1 for
        // the scalar kernel, to A/B); tiles of 128 pixels instead of 64.
        let mma = self.ampere && !self.scalar_conv;
        let tile = if mma { 128 } else { 64 };
        let mt = (s.frames * oh * ow).div_ceil(tile) as u32;
        let (gy, gz) = if mt <= 65535 {
            (mt, 1)
        } else {
            let z = mt.div_ceil(65535);
            (mt.div_ceil(z), z)
        };
        self.launch(
            if mma {
                &self.f.conv2d_mma
            } else {
                &self.f.conv2d
            },
            (s.cout.div_ceil(64) as u32, gy, gz),
            if mma { (128, 1, 1) } else { (16, 16, 1) },
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

    fn download_rgba8(&self, buf: &CuBuf) -> Vec<u8> {
        assert!(buf.f32, "download_rgba8: buffer must come from alloc_words");
        let h = self.stream.clone_dtoh(&*buf.s).expect("download");
        // Each word is two little-endian u16 halves: in memory, already the
        // packed RGBA bytes. One copy, no per-element work.
        let n = buf.len * 4;
        let mut out = vec![0u8; n];
        // SAFETY: `h` holds at least `2 * buf.len` u16s = `n` bytes; the
        // regions do not overlap and any bit pattern is a valid u8.
        unsafe { std::ptr::copy_nonoverlapping(h.as_ptr() as *const u8, out.as_mut_ptr(), n) };
        out
    }

    fn alloc_words(&self, n: usize) -> CuBuf {
        // Two u16 per word, flagged f32 so `download` returns the bits as-is.
        let s = self
            .stream
            .alloc_zeros::<u16>(2 * n.max(1))
            .expect("cuda alloc");
        CuBuf {
            s: Arc::new(s),
            len: n,
            f32: true,
        }
    }

    fn pack_rgba8(&self, src: &CuBuf, base: Option<&CuBuf>, s: &PackShape, out: &CuBuf) {
        let n = s.pixels();
        assert!(
            out.f32 && out.len >= n,
            "pack_rgba8: out must come from alloc_words"
        );
        assert!(!src.f32 && base.is_none_or(|b| b.f32), "base: upload_words");
        self.launch(
            &self.f.pack_rgba8,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(src)),
                Arg::P(self.opt(base)),
                Arg::P(self.ptr(out)),
                Arg::L(n as i64),
                Arg::I(s.skip as i32),
                Arg::I(s.h as i32),
                Arg::I(s.w as i32),
                Arg::I(s.p as i32),
                Arg::I(base.is_some() as i32),
                Arg::I(s.down as i32),
            ],
        );
    }

    fn unpack_rgba8(&self, src: &CuBuf, frames: usize, plane: usize, out: &CuBuf) {
        let n = frames * 3 * plane;
        assert!(
            src.f32 && src.len >= frames * plane && !out.f32 && out.len >= n,
            "unpack_rgba8: src from alloc_words / upload_words, out an activation"
        );
        self.launch(
            &self.f.unpack_rgba8,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(src)),
                Arg::P(self.ptr(out)),
                Arg::L(n as i64),
                Arg::L(plane as i64),
            ],
        );
    }

    fn rope_heads(
        &self,
        src: &CuBuf,
        src_off: usize,
        src_stride: usize,
        dst: &CuBuf,
        dst_off: usize,
        dst_stride: usize,
        heads: usize,
        rows: usize,
        hd: usize,
        rope: (&CuBuf, &CuBuf),
    ) {
        assert!(
            rope.0.f32 && rope.1.f32,
            "RoPE tables must come from upload_table"
        );
        self.launch(
            &self.f.rope_heads,
            Self::elems(heads * rows * hd / 2),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(src)),
                Arg::P(self.ptr(dst)),
                Arg::P(self.ptr(rope.0)),
                Arg::P(self.ptr(rope.1)),
                Arg::I(heads as i32),
                Arg::I(rows as i32),
                Arg::I(hd as i32),
                Arg::L(src_off as i64),
                Arg::L(src_stride as i64),
                Arg::L(dst_off as i64),
                Arg::L(dst_stride as i64),
            ],
        );
    }

    fn kv_ema(
        &self,
        buf: &CuBuf,
        heads: usize,
        head_stride: usize,
        hd: usize,
        src_row: usize,
        src_rows: usize,
        dst_row: usize,
        frame_rows: usize,
        alphas: [f32; 2],
    ) {
        assert!(
            src_row + src_rows <= dst_row || dst_row + 2 * frame_rows <= src_row,
            "kv_ema: source and memory rows overlap"
        );
        self.launch(
            &self.f.kv_ema,
            ((heads * hd.div_ceil(32)) as u32, 1, 1),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(buf)),
                Arg::I(heads as i32),
                Arg::I(hd as i32),
                Arg::L(head_stride as i64),
                Arg::I(src_row as i32),
                Arg::I(src_rows as i32),
                Arg::I(dst_row as i32),
                Arg::I(frame_rows as i32),
                Arg::F(alphas[0]),
                Arg::F(alphas[1]),
            ],
        );
    }

    fn add_act(&self, y: &CuBuf, y_off: usize, d: &CuBuf, d_off: usize, n: usize, act: Act) {
        self.launch(
            &self.f.add_act,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(y)),
                Arg::L(y_off as i64),
                Arg::P(self.ptr(d)),
                Arg::L(d_off as i64),
                Arg::L(n as i64),
                Arg::I(matches!(act, Act::GeluTanh) as i32),
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
    fn kernels_compile_to_sass_for_ada_and_blackwell() {
        for arch in ["sm_89", "sm_120"] {
            match std::panic::catch_unwind(|| super::compile_cubin(arch)) {
                Ok(Ok(cubin)) => assert!(cubin.len() > 1024, "{arch}: cubin too small"),
                Ok(Err(e)) => panic!("{arch}: {}", e.0),
                Err(_) => {
                    eprintln!("skipping: libnvrtc not loadable");
                    return;
                }
            }
        }
    }

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
