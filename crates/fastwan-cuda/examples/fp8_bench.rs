//! Is an FP8 path worth building? cuBLASLt bf16 vs FP8 (E4M3 in, bf16 out,
//! f32 accumulate) at the DiT's GEMM shapes, plus the per-token activation
//! quantize an FP8 linear would add. Needs a GPU with FP8 tensor cores
//! (sm_89+): `cargo run --release -p fastwan-cuda --example fp8_bench`.
use cudarc::cublaslt::{result as lt, sys};
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use std::time::Instant;

const WS: usize = 32 << 20;

/// Quantize each row of x (bf16, rows × k) to E4M3 with its own scale
/// (amax / 448), one block per row: the extra pass of a W8A8 linear.
const QUANT: &str = r#"
extern "C" __global__ void quant_rows(const unsigned short* x, unsigned char* q, float* s, int k) {
    const int r = blockIdx.x;
    const unsigned short* xr = x + (long)r * k;
    __shared__ float red[256];
    float m = 0.f;
    for (int i = threadIdx.x; i < k; i += 256) m = fmaxf(m, fabsf(__uint_as_float(((unsigned)xr[i]) << 16)));
    red[threadIdx.x] = m; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    const float sc = fmaxf(red[0], 1e-12f) / 448.f;
    if (threadIdx.x == 0) s[r] = sc;
    const float inv = 1.f / sc;
    for (int i = threadIdx.x; i < k; i += 256) {
        unsigned short pair;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.f), "f"(__uint_as_float(((unsigned)xr[i]) << 16) * inv));
        q[(long)r * k + i] = (unsigned char)(pair & 0xff);
    }
}
"#;

fn main() {
    let ctx = CudaContext::new(0).expect("cuda");
    let st = ctx.default_stream();
    let h = lt::create_handle().expect("cublasLt");
    let ws = st.alloc_zeros::<u8>(WS).unwrap();
    let (wsp, _g) = ws.device_ptr(&st);
    let (maj, min) = (
        ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap(),
        ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .unwrap(),
    );
    let arch = fastwan_cuda::sass_arch(maj, min).unwrap_or("sm_89");
    let quant = match fastwan_cuda::compile_cubin_src(QUANT, arch) {
        Ok(c) => Some(
            ctx.load_module(cudarc::nvrtc::Ptx::from_binary(c))
                .unwrap()
                .load_function("quant_rows")
                .unwrap(),
        ),
        Err(e) => {
            println!("(quant kernel not compiled: {e:?})");
            None
        }
    };
    println!("device {} ({arch})", ctx.name().unwrap_or_default());
    // (n, k) of the DiT's linears and token counts: a long-video block at
    // 480×272 and 832×480, a full 832×480×81 clip.
    let shapes = [
        (1536usize, 1536usize, "attn q/k/v/o"),
        (8960, 1536, "ffn up"),
        (1536, 8960, "ffn down"),
    ];
    println!(
        "{:<14} {:>6}  {:>9} {:>9} {:>9}  {:>6} {:>6}  {:>8}",
        "linear", "rows", "bf16 ms", "fp8 ms", "fp8f ms", "x", "x fast", "quant ms"
    );
    for rows in [1530usize, 4680, 32760] {
        for &(n, k, name) in &shapes {
            let a8 = st.alloc_zeros::<u8>(n * k).unwrap();
            let b8 = st.alloc_zeros::<u8>(k * rows).unwrap();
            let a16 = st.alloc_zeros::<u16>(n * k).unwrap();
            let b16 = st.alloc_zeros::<u16>(k * rows).unwrap();
            let d = st.alloc_zeros::<u16>(n * rows).unwrap();
            let sc = st.alloc_zeros::<f32>(rows).unwrap();
            let t16 = time_gemm(
                &st,
                h,
                wsp,
                n,
                k,
                rows,
                sys::cudaDataType_t::CUDA_R_16BF,
                ptr(&a16, &st),
                ptr(&b16, &st),
                ptr(&d, &st),
                false,
            );
            let t8 = time_gemm(
                &st,
                h,
                wsp,
                n,
                k,
                rows,
                sys::cudaDataType_t::CUDA_R_8F_E4M3,
                ptr(&a8, &st),
                ptr(&b8, &st),
                ptr(&d, &st),
                false,
            );
            let t8f = time_gemm(
                &st,
                h,
                wsp,
                n,
                k,
                rows,
                sys::cudaDataType_t::CUDA_R_8F_E4M3,
                ptr(&a8, &st),
                ptr(&b8, &st),
                ptr(&d, &st),
                true,
            );
            let tq = quant.as_ref().map(|f| {
                let run = || {
                    let mut lb = st.launch_builder(f);
                    let (x, q, s, kk) = (ptr(&b16, &st), ptr(&b8, &st), ptr(&sc, &st), k as i32);
                    lb.arg(&x).arg(&q).arg(&s).arg(&kk);
                    unsafe {
                        lb.launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .unwrap();
                };
                time(&st, run)
            });
            let f = |t: Option<f64>| t.map_or("n/a".to_string(), |v| format!("{v:.3}"));
            let r = |a: Option<f64>, b: Option<f64>| match (a, b) {
                (Some(a), Some(b)) => format!("{:.2}", a / b),
                _ => "-".into(),
            };
            println!(
                "{name:<14} {rows:>6}  {:>9} {:>9} {:>9}  {:>6} {:>6}  {:>8}",
                f(t16),
                f(t8),
                f(t8f),
                r(t16, t8),
                r(t16, t8f),
                f(tq)
            );
        }
    }
}

fn ptr<T>(
    b: &cudarc::driver::CudaSlice<T>,
    st: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> u64 {
    let (p, _g) = b.device_ptr(st);
    p
}

/// Mean ms of `run` over 20 calls after 3 warm-ups.
fn time(st: &std::sync::Arc<cudarc::driver::CudaStream>, run: impl Fn()) -> f64 {
    for _ in 0..3 {
        run();
    }
    st.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..20 {
        run();
    }
    st.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1000.0 / 20.0
}

/// Column-major Dᵀ[n, rows] = Wᵀ·x as the backend does (A = W op T, k × n;
/// B = x, k × rows), inputs of type `t`, D bf16, f32 accumulate. None if
/// cuBLASLt has no algorithm for it.
#[allow(clippy::too_many_arguments)]
fn time_gemm(
    st: &std::sync::Arc<cudarc::driver::CudaStream>,
    h: sys::cublasLtHandle_t,
    wsp: u64,
    n: usize,
    k: usize,
    rows: usize,
    t: sys::cudaDataType_t,
    a: u64,
    b: u64,
    d: u64,
    fast: bool,
) -> Option<f64> {
    use sys::cudaDataType_t::{CUDA_R_16BF, CUDA_R_32F};
    let la = lt::create_matrix_layout(t, k as u64, n as u64, k as i64).ok()?;
    let lb = lt::create_matrix_layout(t, k as u64, rows as u64, k as i64).ok()?;
    let lc = lt::create_matrix_layout(CUDA_R_16BF, n as u64, rows as u64, n as i64).ok()?;
    let desc =
        lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, CUDA_R_32F).ok()?;
    let tr: i32 = 1;
    let fa: i8 = fast as i8;
    let pref = lt::create_matmul_pref().ok()?;
    let wsz = WS as u64;
    unsafe {
        lt::set_matmul_desc_attribute(
            desc,
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            (&tr) as *const i32 as *const _,
            4,
        )
        .ok()?;
        if t == sys::cudaDataType_t::CUDA_R_8F_E4M3 {
            lt::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                (&fa) as *const i8 as *const _,
                1,
            )
            .ok()?;
        }
        lt::set_matmul_pref_attribute(
            pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&wsz) as *const u64 as *const _,
            8,
        )
        .ok()?;
        let algo = lt::get_matmul_algo_heuristic(h, desc, la, lb, lc, lc, pref)
            .ok()?
            .algo;
        let (alpha, beta) = (1f32, 0f32);
        let run = || {
            lt::matmul(
                h,
                desc,
                (&alpha) as *const f32 as *const _,
                (&beta) as *const f32 as *const _,
                a as *const _,
                la,
                b as *const _,
                lb,
                d as *const _,
                lc,
                d as *mut _,
                lc,
                &algo,
                wsp as *mut _,
                WS,
                st.cu_stream() as *mut _,
            )
            .expect("matmul");
        };
        Some(time(st, run))
    }
}
