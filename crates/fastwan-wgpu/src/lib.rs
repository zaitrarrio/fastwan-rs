//! WebGPU backend (wgpu): one WGSL kernel per [`Backend`] op.
//!
//! Runs natively on Vulkan / Metal / DX12 (any desktop GPU, no CUDA) and in
//! the browser on `wasm32`. Activations are f32; weights are f16 pairs packed
//! in `u32` and unpacked with core `unpack2x16float`, which halves weight
//! memory and bandwidth without depending on the optional `shader-f16`
//! feature. A weight whose values do not fit f16 (or whose row length is odd)
//! stays f32 — never silently clipped.
//!
//! Dispatches are recorded into one command encoder and submitted in batches;
//! a host write or read flushes first, so queue ordering always matches
//! program order.

#[cfg(target_arch = "wasm32")]
mod canvas;
#[cfg(target_arch = "wasm32")]
pub use canvas::CanvasView;

use fastwan_core::backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind, PackShape};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const COMMON: &str = include_str!("shaders/common.wgsl");
const LINEAR: &str = include_str!("shaders/linear.wgsl");
const LINEAR2: &str = include_str!("shaders/linear2.wgsl");
const LINEAR3: &str = include_str!("shaders/linear3.wgsl");
const LINEAR3H: &str = include_str!("shaders/linear3h.wgsl");
const LINEAR4: &str = include_str!("shaders/linear4.wgsl");
const LINEAR4H: &str = include_str!("shaders/linear4h.wgsl");
const FLASH_H: &str = include_str!("shaders/flash_h.wgsl");
/// Token-sized GEMMs only take the f16 path: the 512-row text projections
/// carry encoder activations that can exceed f16's ±65504.
const F16_MIN_ROWS: usize = 1024;
const CONV: &str = include_str!("shaders/conv.wgsl");
const ATTN: &str = include_str!("shaders/attention.wgsl");
const NORM: &str = include_str!("shaders/norm.wgsl");
const HEADS: &str = include_str!("shaders/head_split.wgsl");
const ELEM: &str = include_str!("shaders/elementwise.wgsl");
const MEMORY: &str = include_str!("shaders/memory.wgsl");
const PACK: &str = include_str!("shaders/pack.wgsl");
const UNPACK: &str = include_str!("shaders/unpack.wgsl");
const KV: &str = include_str!("shaders/kv.wgsl");
const FLASH: &str = include_str!("shaders/flash.wgsl");
const FLASH3: &str = include_str!("shaders/flash3.wgsl");
const FLASH3H: &str = include_str!("shaders/flash3h.wgsl");
/// Shared memory flash3 declares (vec4 panels, S tile, partials, row state).
const FLASH3_SHARED: u32 = 512 * 16 + 4160 * 4 + 2 * 256 * 4 + 5 * 64 * 4;
const VSA_POOL: &str = include_str!("shaders/vsa_pool.wgsl");
const VSA_COARSE: &str = include_str!("shaders/vsa_coarse.wgsl");
/// Shared memory the flash kernel declares (Q|K panel, S tile, row state).
const FLASH_SHARED: u32 = 2048 * 4 + 4160 * 4 + 5 * 64 * 4;

/// Submit after this many recorded dispatches.
const BATCH: usize = 64;
/// Max workgroups per dispatch dimension (WebGPU default limit).
const MAX_DIM: u32 = 65535;

#[derive(Clone)]
pub struct GBuf {
    buf: Arc<Owned>,
    len: usize,
}

/// A device buffer that goes back to its device's [`Recycler`] when the last
/// [`GBuf`] handle goes.
///
/// Why not create and free per use: a long video creates and drops thousands
/// of activation buffers and dispatch uniforms per block (several GB at
/// 480×272). wgpu's WebGPU backend never frees a dropped buffer (the JS
/// garbage collector does, rarely), and even `destroy()` only releases the
/// memory once the GPU has finished every submission that used it. With the
/// next block queued before the previous one's frames are read back, Chrome
/// then holds about two blocks of activations at once, and on an RTX 4090
/// blocks ran ~20× slower in bursts (35.5 s vs 1.6 s) while native wgpu stayed
/// flat (docs/runs/2026-11-17-vast-4090-stalls-f16-phones). So dropped
/// buffers are recycled instead: retired on drop, moved to a free list right
/// after the next submission (every command that could use them is then
/// submitted, and later commands and queue writes are ordered after it), and
/// handed out again for the same size and usage, cleared, so a buffer is
/// zero-filled as from `create_buffer`.
pub(crate) struct Owned(Option<wgpu::Buffer>, Arc<Recycler>);

impl std::ops::Deref for Owned {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.0.as_ref().expect("live buffer")
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        if let Some(b) = self.0.take() {
            self.1.retire(b);
        }
    }
}

/// Buffer accounting for diagnosing allocation churn: buffers created, bytes
/// created, buffers reused from the free list, bytes held (live + free list).
static MEM_CREATED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MEM_CREATED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MEM_REUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MEM_HELD_BYTES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

fn tracked(b: wgpu::Buffer) -> wgpu::Buffer {
    use std::sync::atomic::Ordering::Relaxed;
    MEM_CREATED.fetch_add(1, Relaxed);
    MEM_CREATED_BYTES.fetch_add(b.size(), Relaxed);
    MEM_HELD_BYTES.fetch_add(b.size() as i64, Relaxed);
    b
}

/// Allocation counters since start: buffers created, bytes created, buffers
/// reused, bytes held (in use or on a free list).
#[derive(Debug, Clone, Copy)]
pub struct MemStats {
    pub created: u64,
    pub created_bytes: u64,
    pub reused: u64,
    pub held_bytes: i64,
}

pub fn mem_stats() -> MemStats {
    use std::sync::atomic::Ordering::Relaxed;
    MemStats {
        created: MEM_CREATED.load(Relaxed),
        created_bytes: MEM_CREATED_BYTES.load(Relaxed),
        reused: MEM_REUSED.load(Relaxed),
        held_bytes: MEM_HELD_BYTES.load(Relaxed),
    }
}

const STORAGE_RW: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);
const UNIFORM_W: wgpu::BufferUsages =
    wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST);

/// Per-device buffer recycling (see [`Owned`]).
#[derive(Default)]
pub(crate) struct Recycler {
    retired: Mutex<Vec<wgpu::Buffer>>,
    free: Mutex<HashMap<(u64, u32), Vec<wgpu::Buffer>>>,
}

impl Recycler {
    /// `b` is no longer referenced by anything not yet recorded.
    fn retire(&self, b: wgpu::Buffer) {
        self.retired.lock().unwrap().push(b);
    }

    /// Call only right after the pending encoder was submitted: nothing
    /// unsubmitted can refer to a retired buffer then. Activation and uniform
    /// buffers go to the free list, the rest (readback staging) are freed.
    fn recycle(&self) {
        let retired = std::mem::take(&mut *self.retired.lock().unwrap());
        if retired.is_empty() {
            return;
        }
        let mut free = self.free.lock().unwrap();
        for b in retired {
            let u = b.usage();
            if u == STORAGE_RW || u == UNIFORM_W {
                free.entry((b.size(), u.bits())).or_default().push(b);
            } else {
                MEM_HELD_BYTES.fetch_sub(b.size() as i64, std::sync::atomic::Ordering::Relaxed);
                b.destroy();
            }
        }
    }

    fn take(&self, size: u64, usage: wgpu::BufferUsages) -> Option<wgpu::Buffer> {
        let b = self
            .free
            .lock()
            .unwrap()
            .get_mut(&(size, usage.bits()))?
            .pop()?;
        MEM_REUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(b)
    }

    /// Free every buffer on the free list (between runs: the next one may use
    /// other sizes).
    fn trim(&self) {
        for (_, v) in self.free.lock().unwrap().drain() {
            for b in v {
                MEM_HELD_BYTES.fetch_sub(b.size() as i64, std::sync::atomic::Ordering::Relaxed);
                b.destroy();
            }
        }
    }
}

/// VSA state for one token grid.
pub struct GVsa {
    slot: GBuf,
    bsz: GBuf,
    pooled: GBuf,
    coarse: GBuf,
    sel: GBuf,
    nb: usize,
    topk: usize,
    heads: usize,
    d: usize,
    seq: usize,
    scale: f32,
}

impl GVsa {
    /// The intermediates of the last `vsa_attention` with this state:
    /// `(pooled, coarse, sel)`, as `fastwan_core::vsa::coarse_reference`
    /// lays them out (`sel` holds u32 words). For the browser self-test.
    pub fn stages(&self) -> (&GBuf, &GBuf, &GBuf) {
        (&self.pooled, &self.coarse, &self.sel)
    }
}

/// GPU timestamps around every dispatch, summed per kernel.
struct Prof {
    qs: wgpu::QuerySet,
    cap: u32,
    slots: Vec<String>,
    totals: HashMap<String, (f64, u64)>,
    period_ns: f64,
}

/// Construction options; [`WgpuOptions::from_env`] reads them from the environment.
#[derive(Debug, Clone, Copy, Default)]
pub struct WgpuOptions {
    /// Use the f16 math kernels where the adapter has `shader-f16`
    /// (`FASTWAN_WGPU_F16=1`, or `=attn` for attention only).
    pub f16: bool,
    /// With `f16`: f16 attention only, f32 GEMMs. On an AMD RDNA1 Mac
    /// (Chrome → Metal) f16 attention is 1.33× faster but the f16 GEMM 0.90×.
    pub f16_attn_only: bool,
    /// GEMM kernel (`FASTWAN_WGPU_LINEAR`, see `Backend::linear`); `None` =
    /// the default.
    pub linear: Option<u32>,
    /// Attention kernel (`FASTWAN_WGPU_FLASH`): 0 = the plain kernel, 1 =
    /// flash.wgsl, 3 = flash3.wgsl; `None` = the default (3). In the browser,
    /// where there is no environment, the page passes these.
    pub flash: Option<u32>,
    /// Submit after this many recorded dispatches (`None` = 64). Phones get
    /// fewer per submission, so no one command buffer runs long enough for
    /// the OS's GPU watchdog.
    pub submit_every: Option<usize>,
}

impl WgpuOptions {
    pub fn from_env() -> Self {
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u32>().ok());
        Self {
            f16: std::env::var("FASTWAN_WGPU_F16").is_ok_and(|v| v == "1" || v == "attn"),
            f16_attn_only: std::env::var("FASTWAN_WGPU_F16").is_ok_and(|v| v == "attn"),
            linear: num("FASTWAN_WGPU_LINEAR"),
            flash: num("FASTWAN_WGPU_FLASH"),
            submit_every: None,
        }
    }
}

pub struct GWeight {
    buf: GBuf,
    rows: usize,
    cols: usize,
    f16: bool,
    /// Weight-only quantized: `buf` holds the codes, this buffer the
    /// per-group scales (Q8) or interleaved scale / zero pairs (Q4), then the
    /// group size.
    quant: Option<(Quant, GBuf, usize)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Quant {
    /// [`fastwan_core::quant::Q8`]: 4 codes per word.
    Q8,
    /// [`fastwan_core::quant::Q4`]: 8 codes per word.
    Q4,
}

struct Pending {
    enc: Option<wgpu::CommandEncoder>,
    n: usize,
}

pub struct Wgpu {
    /// Free lists of dropped buffers, reused instead of reallocated.
    recycler: Arc<Recycler>,
    /// For canvas surfaces (browser only).
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    instance: wgpu::Instance,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    attn_bk: usize,
    /// The register-tiled flash kernel fits this adapter's shared memory.
    flash_ok: bool,
    /// Attention kernel when `flash_ok`: 3 = flash3.wgsl (the default, if
    /// shared memory fits), 1 = flash.wgsl (`FASTWAN_WGPU_FLASH=1`).
    flash_variant: u32,
    /// GEMM kernel variant (see `Backend::linear`).
    linear_variant: u32,
    /// Dispatches per submission.
    batch: usize,
    /// f16 math kernels (adapter has `shader-f16` and f16 mode is on).
    f16: bool,
    /// f16 GEMMs too (f16 mode is not attention-only).
    f16_linear: bool,
    pipelines: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
    /// Pipeline name by address, for the profiler.
    names: Mutex<HashMap<usize, String>>,
    /// Shape detail for the next dispatch's profile entry (linear / conv).
    tag: Mutex<Option<String>>,
    /// `FASTWAN_PROFILE=1` and the adapter has timestamp queries.
    prof: Option<Mutex<Prof>>,
    pending: Mutex<Pending>,
    /// In the browser: the first uncaptured WebGPU error or the device-loss
    /// reason. wgpu's default handler panics, which inside a JS callback
    /// leaves the caller's promise pending forever (a silent hang).
    fault: Arc<Mutex<Option<String>>>,
    /// Browser: tick the device while a readback is pending (see `KeepAlive`).
    keepalive: std::sync::atomic::AtomicBool,
    dummy: GBuf,
    pub allow_f16_weights: bool,
}

#[derive(Debug)]
pub struct InitError(pub String);

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for InitError {}

impl Wgpu {
    /// The first high-performance adapter, with the adapter's own limits
    /// requested (the WebGPU defaults cap a binding at 128 MiB; the DiT's
    /// fused QKV buffer at 480p is ~600 MiB in f32).
    pub async fn new_async() -> Result<Self, InitError> {
        Self::with_options_async(WgpuOptions::from_env()).await
    }

    pub async fn with_options_async(opts: WgpuOptions) -> Result<Self, InitError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("no WebGPU adapter: {e}")))?;
        let limits = adapter.limits();
        let profile = std::env::var("FASTWAN_PROFILE").is_ok_and(|v| v == "1")
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let f16 = opts.f16 && adapter.features().contains(wgpu::Features::SHADER_F16);
        let mut features = wgpu::Features::empty();
        if profile {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if f16 {
            features |= wgpu::Features::SHADER_F16;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fastwan"),
                required_features: features,
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("request_device: {e}")))?;
        let fault = Arc::new(Mutex::new(None::<String>));
        #[cfg(target_arch = "wasm32")]
        {
            let f = fault.clone();
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                let mut f = f.lock().unwrap();
                if f.is_none() {
                    *f = Some(format!("WebGPU error: {e}"));
                }
            }));
            let f = fault.clone();
            device.set_device_lost_callback(move |reason, msg| {
                let mut f = f.lock().unwrap();
                if f.is_none() {
                    *f = Some(format!("the GPU device was lost ({reason:?}): {msg}"));
                }
            });
        }
        // K and V tiles of BK keys × 128 dims, f32, must fit shared memory.
        let wgs = device.limits().max_compute_workgroup_storage_size as usize;
        let attn_bk = if wgs >= 2 * 32 * 128 * 4 { 32 } else { 16 };
        let flash_ok = wgs >= FLASH_SHARED as usize && opts.flash != Some(0);
        // Measured on an RTX 4090 (docs/runs/2026-11-14-vast-4090-webgpu-kernels):
        // flash3 + linear4 halve the step in Chrome and cut it 1.6× natively.
        let flash_variant = match opts.flash {
            Some(1) => 1,
            _ if wgs >= FLASH3_SHARED as usize => 3,
            _ => 1,
        };
        let linear_variant = opts.linear.unwrap_or(4);
        let recycler = Arc::new(Recycler::default());
        let dummy = GBuf {
            buf: Arc::new(Owned(
                Some(tracked(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("dummy"),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }))),
                recycler.clone(),
            )),
            len: 4,
        };
        let prof = profile.then(|| {
            let cap = 2048;
            Mutex::new(Prof {
                qs: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("profile"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2 * cap,
                }),
                cap,
                slots: Vec::new(),
                totals: HashMap::new(),
                period_ns: queue.get_timestamp_period() as f64,
            })
        });
        Ok(Self {
            info: adapter.get_info(),
            instance,
            adapter,
            device,
            queue,
            recycler,
            attn_bk,
            flash_ok,
            flash_variant,
            linear_variant,
            batch: opts.submit_every.unwrap_or(BATCH).max(1),
            f16,
            f16_linear: f16 && !opts.f16_attn_only,
            pipelines: Mutex::new(HashMap::new()),
            names: Mutex::new(HashMap::new()),
            tag: Mutex::new(None),
            prof,
            pending: Mutex::new(Pending { enc: None, n: 0 }),
            fault,
            keepalive: std::sync::atomic::AtomicBool::new(true),
            dummy,
            allow_f16_weights: true,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, InitError> {
        pollster::block_on(Self::new_async())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_options(opts: WgpuOptions) -> Result<Self, InitError> {
        pollster::block_on(Self::with_options_async(opts))
    }

    /// Whether the f16 math kernels are in use.
    pub fn f16(&self) -> bool {
        self.f16
    }

    /// A CPU rasterizer (llvmpipe / lavapipe / SwiftShader / WARP). Fine for
    /// conformance checks, meaningless for timing: wgpu falls back to one
    /// silently when the GPU's Vulkan driver fails to load.
    pub fn is_software(&self) -> bool {
        self.info.device_type == wgpu::DeviceType::Cpu
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    fn pipeline(&self, key: &str, src: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let mut map = self.pipelines.lock().unwrap();
        if let Some(p) = map.get(key) {
            return p.clone();
        }
        // `enable f16;` must open the module, before the shared helpers.
        // Kernel A/B without a rebuild (native only):
        // FASTWAN_WGPU_SHADER_<key>=file.wgsl replaces the kernel's source.
        #[cfg(not(target_arch = "wasm32"))]
        let body = match std::env::var(format!("FASTWAN_WGPU_SHADER_{key}")) {
            Ok(path) => std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("FASTWAN_WGPU_SHADER_{key}={path}: {e}")),
            Err(_) => src(),
        };
        #[cfg(target_arch = "wasm32")]
        let body = src();
        let code = if body.contains("vec4<f16>") {
            format!("enable f16;\n{COMMON}\n{body}")
        } else {
            format!("{COMMON}\n{body}")
        };
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(key),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
        let p = Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(key),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );
        map.insert(key.to_string(), p.clone());
        self.names
            .lock()
            .unwrap()
            .insert(Arc::as_ptr(&p) as usize, key.to_string());
        p
    }

    /// Per-kernel GPU time since the last reset (`FASTWAN_PROFILE=1`).
    pub fn profile_table(&self) -> Option<String> {
        let prof = self.prof.as_ref()?;
        self.flush();
        let p = prof.lock().unwrap();
        let total: f64 = p.totals.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = p.totals.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let mut out = format!(
            "GPU time by kernel ({:.1} ms total, timestamp queries):\n",
            total / 1e6
        );
        for (name, (ns, n)) in rows {
            out += &format!(
                "  {:>9.1} ms {:>5.1}%  {:>6} calls {:>9.3} ms/call  {name}\n",
                ns / 1e6,
                100.0 * ns / total.max(1.0),
                n,
                ns / 1e6 / *n as f64
            );
        }
        Some(out)
    }

    pub fn profile_reset(&self) {
        if let Some(prof) = &self.prof {
            self.flush();
            prof.lock().unwrap().totals.clear();
        }
    }

    fn weight_src(src: &str, f16: bool) -> String {
        if f16 {
            src.replace("{W_TYPE}", "u32").replace(
                "{W_LOAD}",
                "let v = unpack2x16float(w[i >> 1u]); return select(v.x, v.y, (i & 1u) == 1u);",
            )
        } else {
            src.replace("{W_TYPE}", "f32")
                .replace("{W_LOAD}", "return w[i];")
        }
    }

    /// A GEMM shader reading int8 codes packed 4 per word, sign-extended and
    /// scaled per group (`p._p` columns per scale, binding 5).
    fn q8_src(src: &str) -> String {
        src.replace("{W_TYPE}", "u32")
            .replace(
                "fn load_w(",
                "@group(0) @binding(5) var<storage, read> wscale: array<f32>;\nfn load_w(",
            )
            .replace(
                "{W_LOAD}",
                "let b = (w[i >> 2u] >> ((i & 3u) * 8u)) << 24u; \
                 return f32(bitcast<i32>(b) >> 24u) * wscale[i / p._p];",
            )
    }

    /// The 4-bit twin: 8 codes per word, `(q − zero) · scale` per group with
    /// (scale, zero) pairs at binding 5 — `fw_dequant_q4` on CUDA.
    fn q4_src(src: &str) -> String {
        src.replace("{W_TYPE}", "u32")
            .replace(
                "fn load_w(",
                "@group(0) @binding(5) var<storage, read> wsz: array<vec2<f32>>;\nfn load_w(",
            )
            .replace(
                "{W_LOAD}",
                "let c = f32((w[i >> 3u] >> ((i & 7u) * 4u)) & 15u); \
                 let g = wsz[i / p._p]; return (c - g.y) * g.x;",
            )
    }

    /// Record one dispatch. `params` is the uniform block as 32-bit words.
    fn dispatch(
        &self,
        pipe: &wgpu::ComputePipeline,
        params: &[u32],
        bufs: &[&GBuf],
        groups: (u32, u32, u32),
    ) {
        let tag = self.tag.lock().unwrap().take();
        let slot = self.prof.as_ref().map(|prof| {
            let full = {
                let p = prof.lock().unwrap();
                p.slots.len() as u32 >= p.cap
            };
            if full {
                self.flush();
            }
            let mut p = prof.lock().unwrap();
            let base = self
                .names
                .lock()
                .unwrap()
                .get(&(pipe as *const _ as usize))
                .cloned()
                .unwrap_or_else(|| "?".into());
            p.slots.push(match tag {
                Some(t) => format!("{base} {t}"),
                None => base,
            });
            (p.slots.len() - 1) as u32
        });
        let mut words = params.to_vec();
        while words.len() % 4 != 0 {
            words.push(0);
        }
        let bytes = bytemuck_cast(&words);
        let ubuf = match self.recycler.take(bytes.len() as u64, UNIFORM_W) {
            // Ordered before the pending encoder, which is the first to use it
            // again (it was retired only after the last one was submitted).
            Some(b) => {
                self.queue.write_buffer(&b, 0, bytes);
                b
            }
            None => tracked(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytes,
                        usage: UNIFORM_W,
                    }),
            ),
        };
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.buf.as_entire_binding(),
            });
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pend = self.pending.lock().unwrap();
        let enc = pend
            .enc
            .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        {
            let guard = self.prof.as_ref().map(|p| p.lock().unwrap());
            let tw = match (&guard, slot) {
                (Some(p), Some(i)) => Some(wgpu::ComputePassTimestampWrites {
                    query_set: &p.qs,
                    beginning_of_pass_write_index: Some(2 * i),
                    end_of_pass_write_index: Some(2 * i + 1),
                }),
                _ => None,
            };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: tw,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.recycler.retire(ubuf);
        pend.n += 1;
        if pend.n >= self.batch {
            let enc = pend.enc.take().unwrap();
            pend.n = 0;
            drop(pend);
            self.queue.submit([enc.finish()]);
            self.recycler.recycle();
        }
    }

    pub(crate) fn flush(&self) {
        let mut pend = self.pending.lock().unwrap();
        let mut enc = pend.enc.take();
        pend.n = 0;
        drop(pend);
        // Profiling: resolve this batch's timestamps and fold them into the totals.
        let mut readback = None;
        if let Some(prof) = &self.prof {
            let p = prof.lock().unwrap();
            let n = p.slots.len() as u32;
            if n > 0 {
                let bytes = (2 * n as u64) * 8;
                let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let e = enc
                    .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
                e.resolve_query_set(&p.qs, 0..2 * n, &resolve, 0);
                e.copy_buffer_to_buffer(&resolve, 0, &read, 0, bytes);
                readback = Some(read);
            }
        }
        if let Some(enc) = enc {
            self.queue.submit([enc.finish()]);
        }
        self.recycler.recycle();
        #[cfg(not(target_arch = "wasm32"))]
        if let (Some(read), Some(prof)) = (readback, &self.prof) {
            let slice = read.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            let ticks: Vec<u64> =
                bytemuck::cast_slice::<u8, u64>(&slice.get_mapped_range().expect("timestamps"))
                    .to_vec();
            let mut p = prof.lock().unwrap();
            let period = p.period_ns;
            let slots = std::mem::take(&mut p.slots);
            for (i, name) in slots.into_iter().enumerate() {
                let ns = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period;
                let e = p.totals.entry(name).or_insert((0.0, 0));
                e.0 += ns;
                e.1 += 1;
            }
        }
    }

    /// `n` workgroups as a (x, y) grid within the per-dimension limit.
    fn grid(n: usize) -> (u32, u32, u32) {
        let n = n.max(1) as u32;
        if n <= MAX_DIM {
            (n, 1, 1)
        } else {
            let y = n.div_ceil(MAX_DIM);
            (n.div_ceil(y), y, 1)
        }
    }

    fn elementwise(
        &self,
        op: u32,
        x: &GBuf,
        y: Option<&GBuf>,
        aux: Option<&GBuf>,
        n: usize,
        extra: [u32; 8],
        s: [f32; 2],
    ) {
        let pipe = self.pipeline("elementwise", || ELEM.to_string());
        let mut p = vec![n as u32, op];
        p.extend_from_slice(&extra);
        p.push(s[0].to_bits());
        p.push(s[1].to_bits());
        self.dispatch(
            &pipe,
            &p,
            &[x, y.unwrap_or(&self.dummy), aux.unwrap_or(&self.dummy)],
            Self::grid(n.div_ceil(256)),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        h: &GBuf,
        a: Option<&GBuf>,
        rows: usize,
        dim: usize,
        mods: &GBuf,
        gate: Option<usize>,
        norm: NormKind<'_, GBuf>,
        eps: f32,
        out: &GBuf,
    ) {
        let pipe = self.pipeline("norm", || NORM.to_string());
        let (affine, scale, shift, w, b) = match norm {
            NormKind::Affine { w, b } => (1, 0, 0, w, b),
            NormKind::Mod { scale, shift } => {
                (0, scale as u32, shift as u32, &self.dummy, &self.dummy)
            }
        };
        let p = [
            rows as u32,
            dim as u32,
            a.is_some() as u32,
            gate.unwrap_or(0) as u32,
            gate.is_some() as u32,
            affine,
            scale,
            shift,
            eps.to_bits(),
        ];
        self.dispatch(
            &pipe,
            &p,
            &[h, a.unwrap_or(&self.dummy), mods, w, b, out],
            Self::grid(rows),
        );
    }
}

fn bytemuck_cast(w: &[u32]) -> &[u8] {
    bytemuck::cast_slice(w)
}

impl Wgpu {
    fn owned(&self, b: wgpu::Buffer) -> Arc<Owned> {
        Arc::new(Owned(Some(tracked(b)), self.recycler.clone()))
    }

    /// Free the buffers kept for reuse (call between runs: the next one may
    /// use other sizes).
    pub fn trim_pool(&self) {
        // Submitting moves everything dropped so far onto the free list.
        self.flush();
        self.recycler.trim();
    }

    fn upload_u32(&self, data: &[u32]) -> GBuf {
        let data = if data.is_empty() { &[0u32][..] } else { data };
        GBuf {
            buf: self.owned(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    }),
            ),
            len: data.len(),
        }
    }

    /// Flash kernel dispatch: mode 0 dense, mode 1 VSA fine stage.
    #[allow(clippy::too_many_arguments)]
    fn flash(
        &self,
        mode: u32,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        out: &GBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        scale: f32,
        vsa: Option<(&GVsa, &GBuf)>,
    ) {
        let pipe = if self.flash_variant == 3 && d <= 128 && self.f16 {
            self.pipeline("flash3h", || FLASH3H.to_string())
        } else if self.flash_variant == 3 && d <= 128 {
            self.pipeline("flash3", || FLASH3.to_string())
        } else if self.f16 {
            self.pipeline("flash_h", || FLASH_H.to_string())
        } else {
            self.pipeline("flash", || FLASH.to_string())
        };
        let (nb, topk) = vsa.map_or((0, 0), |(s, _)| (s.nb, s.topk));
        let p = [
            heads as u32,
            sq as u32,
            sk as u32,
            d as u32,
            (scale * std::f32::consts::LOG2_E).to_bits(),
            mode,
            nb as u32,
            topk as u32,
        ];
        let blocks = if mode == 0 { sq.div_ceil(64) } else { nb };
        let (slot, sel, coarse, gate) = match vsa {
            Some((s, g)) => (&s.slot, &s.sel, &s.coarse, g),
            None => (&self.dummy, &self.dummy, &self.dummy, &self.dummy),
        };
        self.dispatch(
            &pipe,
            &p,
            &[q, k, v, out, slot, sel, coarse, gate],
            (blocks as u32, heads as u32, 1),
        );
    }
}

impl Wgpu {
    /// flash3.wgsl reading a K/V cache: keys and values of head h start at
    /// row h · kstride (not h · sk), and with `half` they are f16 pairs in
    /// u32 (unpacked with core `unpack2x16float`; no `shader-f16` needed).
    fn flash3_kv_src(half: bool, f16: bool) -> String {
        let mut s = (if f16 { FLASH3H } else { FLASH3 }).replace(
            "struct P { heads: u32, sq: u32, sk: u32, d: u32, scale_log2: f32, mode: u32, nb: u32, topk: u32 }",
            "struct P { heads: u32, sq: u32, sk: u32, d: u32, scale_log2: f32, mode: u32, nb: u32, topk: u32, kstride: u32, _a: u32, _b: u32, _c: u32 }",
        );
        let k_load = "    return k[(h * p.sk + tok) * p.d + col];";
        let v_load = "                    let b = (h * p.sk + ktok) * d + 4u * cq;\n                    vv = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]);";
        assert!(s.contains("struct P { heads: u32, sq: u32, sk: u32, d: u32, scale_log2: f32, mode: u32, nb: u32, topk: u32, kstride"));
        assert!(
            s.contains(k_load) && s.contains(v_load),
            "flash3.wgsl changed: update flash3_kv_src"
        );
        if half {
            s = s
                .replace("var<storage, read> k: array<f32>;", "var<storage, read> k: array<u32>;")
                .replace("var<storage, read> v: array<f32>;", "var<storage, read> v: array<u32>;")
                .replace(
                    k_load,
                    "    let e = (h * p.kstride + tok) * p.d + col;\n    let w = unpack2x16float(k[e / 2u]);\n    return select(w.x, w.y, (e & 1u) != 0u);",
                )
                .replace(
                    v_load,
                    "                    let b = (h * p.kstride + ktok) * d + 4u * cq;\n                    let w0 = unpack2x16float(v[b / 2u]);\n                    let w1 = unpack2x16float(v[b / 2u + 1u]);\n                    vv = vec4<f32>(w0.x, w0.y, w1.x, w1.y);",
                );
        } else {
            s = s
                .replace(k_load, "    return k[(h * p.kstride + tok) * p.d + col];")
                .replace(
                    v_load,
                    "                    let b = (h * p.kstride + ktok) * d + 4u * cq;\n                    vv = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]);",
                );
        }
        s
    }

    fn kv_flash_ok(&self, d: usize) -> bool {
        self.flash_ok && self.flash_variant == 3 && d % 16 == 0 && d <= 128
    }
}

impl Backend for Wgpu {
    type Buf = GBuf;
    type Vsa = GVsa;
    type Weight = GWeight;

    fn name(&self) -> String {
        format!(
            "wgpu {:?}: {} ({}{}){}",
            self.info.backend,
            self.info.name,
            self.info.driver,
            if self.info.driver_info.is_empty() {
                String::new()
            } else {
                format!(" {}", self.info.driver_info)
            },
            if !self.f16 {
                ""
            } else if self.f16_linear {
                " +f16"
            } else {
                " attn+f16"
            }
        )
    }

    fn alloc(&self, len: usize) -> GBuf {
        let size = (len.max(1) * 4) as u64;
        let b = match self.recycler.take(size, STORAGE_RW) {
            // Zero it as `create_buffer` would, in command order: after every
            // submitted use, before anything recorded from now on.
            Some(b) => {
                let mut pend = self.pending.lock().unwrap();
                pend.enc
                    .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()))
                    .clear_buffer(&b, 0, None);
                b
            }
            None => tracked(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: STORAGE_RW,
                mapped_at_creation: false,
            })),
        };
        GBuf {
            buf: Arc::new(Owned(Some(b), self.recycler.clone())),
            len,
        }
    }

    fn upload(&self, data: &[f32]) -> GBuf {
        let padded;
        let bytes: &[u8] = if data.is_empty() {
            padded = [0f32];
            bytemuck::cast_slice(&padded)
        } else {
            bytemuck::cast_slice(data)
        };
        GBuf {
            buf: self.owned(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                    }),
            ),
            len: data.len(),
        }
    }

    fn write(&self, buf: &GBuf, data: &[f32]) {
        self.flush();
        self.queue
            .write_buffer(&buf.buf, 0, bytemuck::cast_slice(data));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn download(&self, buf: &GBuf) -> Vec<f32> {
        self.read_with(buf, |d| {
            bytemuck::cast_slice::<u8, f32>(d)[..buf.len].to_vec()
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn download_rgba8(&self, buf: &GBuf) -> Vec<u8> {
        self.read_with(buf, |d| d[..buf.len * 4].to_vec())
    }

    #[cfg(target_arch = "wasm32")]
    fn download(&self, _buf: &GBuf) -> Vec<f32> {
        panic!("blocking readback is unavailable in the browser; use Wgpu::download_async")
    }

    fn buf_len(&self, buf: &GBuf) -> usize {
        buf.len
    }

    fn copy(&self, src: &GBuf, so: usize, dst: &GBuf, d: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut pend = self.pending.lock().unwrap();
        let enc = pend
            .enc
            .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        enc.copy_buffer_to_buffer(
            &src.buf,
            (so * 4) as u64,
            &dst.buf,
            (d * 4) as u64,
            (len * 4) as u64,
        );
    }

    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> GWeight {
        assert_eq!(data.len(), rows * cols);
        let fits = data.iter().all(|v| v.abs() <= 65000.0);
        if self.allow_f16_weights && cols % 2 == 0 && fits {
            let packed: Vec<u32> = data
                .chunks_exact(2)
                .map(|p| {
                    let lo = half::f16::from_f32(p[0]).to_bits() as u32;
                    let hi = half::f16::from_f32(p[1]).to_bits() as u32;
                    lo | (hi << 16)
                })
                .collect();
            let buf = GBuf {
                buf: self.owned(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&packed),
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                )),
                len: packed.len(),
            };
            GWeight {
                buf,
                rows,
                cols,
                f16: true,
                quant: None,
            }
        } else {
            GWeight {
                buf: self.upload(data),
                rows,
                cols,
                f16: false,
                quant: None,
            }
        }
    }

    fn weight_q8(&self, q: &fastwan_core::quant::Q8) -> GWeight {
        let packed: Vec<u32> =
            q.q.chunks(4)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .fold(0u32, |a, (j, &v)| a | ((v as u8 as u32) << (8 * j)))
                })
                .collect();
        GWeight {
            buf: self.upload_u32(&packed),
            rows: q.rows,
            cols: q.cols,
            f16: false,
            quant: Some((Quant::Q8, self.upload(&q.scales), q.group)),
        }
    }

    fn weight_q4(&self, q: &fastwan_core::quant::Q4) -> GWeight {
        let mut bytes = q.packed();
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sz: Vec<f32> = q
            .scales
            .iter()
            .zip(&q.zeros)
            .flat_map(|(&s, &z)| [s, z])
            .collect();
        GWeight {
            buf: self.upload_u32(&words),
            rows: q.rows,
            cols: q.cols,
            f16: false,
            quant: Some((Quant::Q4, self.upload(&sz), q.group)),
        }
    }

    fn sync(&self) {
        self.flush();
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    fn linear(
        &self,
        x: &GBuf,
        x_row: usize,
        rows: usize,
        w: &GWeight,
        bias: Option<&GBuf>,
        act: Act,
        y: &GBuf,
        y_row: usize,
    ) {
        let f16 = w.f16;
        if self.prof.is_some() {
            *self.tag.lock().unwrap() = Some(format!(
                "{}x{}{}",
                w.rows,
                w.cols,
                if act == Act::GeluTanh { " gelu" } else { "" }
            ));
        }
        // 128×128 tiles for the big GEMMs (FASTWAN_WGPU_LINEAR=1 forces the
        // 64×64 kernel, for A/B); small row counts keep the 64×64 kernel.
        // FASTWAN_WGPU_LINEAR selects the GEMM kernel for A/B runs:
        // 1 = 64×64 array accumulators (linear.wgsl),
        // 2 = 128×128 (linear2.wgsl: measured 38% slower on an RTX 4090),
        // 3 = 64×64 with hand-unrolled vec4 accumulators (linear3.wgsl:
        //     5.75× faster than 1 on an RTX 4090, 446 → 77.5 ms per FFN
        //     GEMM; the arrays of 1 and 2 were spilled to local memory),
        // 4 = 128×128, 8×8 per thread in named vec4s, vec4 shared reads
        //     (linear4.wgsl, the default: 1.75× faster than 3 on an RTX 4090,
        //     same output bit for bit; rows < 128 fall back to 3).
        let variant = self.linear_variant;
        let variant = match variant {
            2 if rows < 128 => 1,
            4 if rows < 128 => 3,
            v => v,
        };
        let (src, tile, name) = match variant {
            2 => (LINEAR2, 128, "linear2"),
            4 => (LINEAR4, 128, "linear4"),
            1 => (LINEAR, 64, "linear"),
            _ => (LINEAR3, 64, "linear3"),
        };
        let half = self.f16_linear && f16 && (variant == 3 || variant == 4) && rows >= F16_MIN_ROWS;
        let pipe = if let Some((kind, ..)) = &w.quant {
            match kind {
                Quant::Q8 => self.pipeline(&format!("{name}_q8"), || Self::q8_src(src)),
                Quant::Q4 => self.pipeline(&format!("{name}_q4"), || Self::q4_src(src)),
            }
        } else if half && variant == 4 {
            self.pipeline("linear4h", || LINEAR4H.to_string())
        } else if half {
            self.pipeline("linear3h", || LINEAR3H.to_string())
        } else {
            let key = format!("{name}_{}", if f16 { "f16" } else { "f32" });
            self.pipeline(&key, || Self::weight_src(src, f16))
        };
        let p = [
            rows as u32,
            w.rows as u32,
            w.cols as u32,
            x_row as u32,
            y_row as u32,
            bias.is_some() as u32,
            (act == Act::GeluTanh) as u32,
            w.quant.as_ref().map_or(0, |q| q.2) as u32,
        ];
        let groups = (w.rows.div_ceil(tile) as u32, rows.div_ceil(tile) as u32, 1);
        let b = bias.unwrap_or(&self.dummy);
        match &w.quant {
            Some((_, scales, _)) => self.dispatch(&pipe, &p, &[x, &w.buf, b, y, scales], groups),
            None => self.dispatch(&pipe, &p, &[x, &w.buf, b, y], groups),
        }
    }

    fn layer_norm(
        &self,
        x: &GBuf,
        rows: usize,
        dim: usize,
        norm: NormKind<'_, GBuf>,
        mods: &GBuf,
        eps: f32,
        out: &GBuf,
    ) {
        self.norm(x, None, rows, dim, mods, None, norm, eps, out);
    }

    fn residual_norm(
        &self,
        h: &GBuf,
        a: &GBuf,
        rows: usize,
        dim: usize,
        mods: &GBuf,
        gate: Option<usize>,
        norm: NormKind<'_, GBuf>,
        eps: f32,
        normed: &GBuf,
    ) {
        self.norm(h, Some(a), rows, dim, mods, gate, norm, eps, normed);
    }

    fn gate_residual(&self, h: &GBuf, a: &GBuf, rows: usize, dim: usize, mods: &GBuf, gate: usize) {
        self.elementwise(
            0,
            h,
            Some(a),
            Some(mods),
            rows * dim,
            [dim as u32, gate as u32, 0, 0, 0, 0, 0, 0],
            [0.0; 2],
        );
    }

    fn head_split(
        &self,
        src: &GBuf,
        rows: usize,
        stride: usize,
        col_off: usize,
        heads: usize,
        hd: usize,
        rms_w: Option<&GBuf>,
        rope: Option<(&GBuf, &GBuf)>,
        eps: f32,
        out: &GBuf,
    ) {
        let pipe = self.pipeline("head_split", || HEADS.to_string());
        let p = [
            rows as u32,
            stride as u32,
            col_off as u32,
            heads as u32,
            hd as u32,
            rms_w.is_some() as u32,
            rope.is_some() as u32,
            0,
            eps.to_bits(),
        ];
        let (c, s) = rope.unwrap_or((&self.dummy, &self.dummy));
        self.dispatch(
            &pipe,
            &p,
            &[src, rms_w.unwrap_or(&self.dummy), c, s, out],
            Self::grid(rows),
        );
    }

    fn kv_caps(&self, d: usize) -> (bool, bool) {
        let ok = self.kv_flash_ok(d);
        (ok, ok)
    }

    fn alloc_kv(&self, n: usize, half: bool) -> GBuf {
        if half {
            assert!(n % 2 == 0, "a 16-bit K/V cache holds pairs");
            self.alloc(n / 2)
        } else {
            self.alloc(n)
        }
    }

    fn kv_store(
        &self,
        src: &GBuf,
        n: usize,
        from: usize,
        len: usize,
        heads: usize,
        d: usize,
        dst: &GBuf,
        dst_row: usize,
        stride: usize,
        half: bool,
    ) {
        if len == 0 {
            return;
        }
        assert!(!half || d % 2 == 0, "16-bit K/V rows hold pairs");
        let pipe = self.pipeline("kv_store", || KV.to_string());
        let items = heads * len * if half { d / 2 } else { d };
        self.dispatch(
            &pipe,
            &[
                n as u32,
                from as u32,
                len as u32,
                heads as u32,
                d as u32,
                dst_row as u32,
                stride as u32,
                half as u32,
            ],
            &[src, dst],
            Self::grid(items.div_ceil(256)),
        );
    }

    fn attention_kv(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        stride: usize,
        d: usize,
        half: bool,
        out: &GBuf,
    ) {
        if !self.kv_flash_ok(d) {
            assert!(
                stride == sk && !half,
                "attention_kv needs flash3 here (kv_caps)"
            );
            return self.attention(q, k, v, heads, sq, sk, d, out);
        }
        let f16 = self.f16;
        let key = format!(
            "flash3{}_kv{}",
            if f16 { "h" } else { "" },
            if half { "h" } else { "" }
        );
        let pipe = self.pipeline(&key, || Self::flash3_kv_src(half, f16));
        let scale = 1.0 / (d as f32).sqrt();
        let p = [
            heads as u32,
            sq as u32,
            sk as u32,
            d as u32,
            (scale * std::f32::consts::LOG2_E).to_bits(),
            0,
            0,
            0,
            stride as u32,
            0,
            0,
            0,
        ];
        let d0 = &self.dummy;
        self.dispatch(
            &pipe,
            &p,
            &[q, k, v, out, d0, d0, d0, d0],
            (sq.div_ceil(64) as u32, heads as u32, 1),
        );
    }

    fn attention(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        out: &GBuf,
    ) {
        assert!(d <= 128, "attention kernel supports head_dim <= 128");
        if self.flash_ok && d % 16 == 0 {
            self.flash(
                0,
                q,
                k,
                v,
                out,
                heads,
                sq,
                sk,
                d,
                1.0 / (d as f32).sqrt(),
                None,
            );
            return;
        }
        let bk = self.attn_bk;
        let pipe = self.pipeline(&format!("attention_bk{bk}"), || {
            ATTN.replace("{BK_D}", &(bk * 128).to_string())
                .replace("{BK}", &bk.to_string())
        });
        let p = [
            heads as u32,
            sq as u32,
            sk as u32,
            d as u32,
            (1.0 / (d as f32).sqrt()).to_bits(),
        ];
        assert!(heads as u32 <= MAX_DIM);
        self.dispatch(
            &pipe,
            &p,
            &[q, k, v, out],
            (sq.div_ceil(64) as u32, heads as u32, 1),
        );
    }

    fn profile_report(&self) -> Option<String> {
        self.profile_table()
    }
    fn profile_reset(&self) {
        Wgpu::profile_reset(self)
    }

    fn vsa_supported(&self, head_dim: usize) -> bool {
        self.flash_ok && head_dim % 16 == 0 && head_dim <= 128
    }

    fn vsa_prepare(&self, spec: &fastwan_core::vsa::VsaSpec) -> GVsa {
        let (nb, h, d) = (spec.plan.num_tiles(), spec.heads, spec.head_dim);
        assert!(
            nb <= 2048,
            "VSA coarse stage holds at most 2048 tiles per workgroup"
        );
        GVsa {
            slot: self.upload_u32(&spec.plan.slot_src),
            bsz: self.upload_u32(&spec.plan.block_sizes),
            pooled: self.alloc(3 * h * nb * d),
            coarse: self.alloc(h * nb * d),
            sel: self.upload_u32(&vec![0u32; h * nb * spec.topk]),
            nb,
            topk: spec.topk,
            heads: h,
            d,
            seq: spec.plan.seq,
            scale: spec.scale,
        }
    }

    fn vsa_attention(&self, q: &GBuf, k: &GBuf, v: &GBuf, gate: &GBuf, s: &GVsa, out: &GBuf) {
        assert!(
            self.flash_ok && s.d % 16 == 0 && s.d <= 128,
            "VSA needs the flash kernel (head_dim multiple of 16, <= 128)"
        );
        let pool = self.pipeline("vsa_pool", || VSA_POOL.to_string());
        self.dispatch(
            &pool,
            &[s.heads as u32, s.seq as u32, s.d as u32, s.nb as u32],
            &[q, k, v, &s.slot, &s.bsz, &s.pooled],
            (s.nb as u32, s.heads as u32, 1),
        );
        let coarse = self.pipeline("vsa_coarse", || VSA_COARSE.to_string());
        let p = [
            s.heads as u32,
            s.d as u32,
            s.nb as u32,
            s.topk as u32,
            s.scale.to_bits(),
        ];
        self.dispatch(
            &coarse,
            &p,
            &[&s.pooled, &s.coarse, &s.sel],
            (s.nb as u32, s.heads as u32, 1),
        );
        self.flash(
            1,
            q,
            k,
            v,
            out,
            s.heads,
            s.seq,
            s.seq,
            s.d,
            s.scale,
            Some((s, gate)),
        );
    }

    fn patchify(
        &self,
        latent: &GBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        p: usize,
        out: &GBuf,
    ) {
        let n = c * t * h * w;
        self.elementwise(
            3,
            out,
            Some(latent),
            None,
            n,
            [0, 0, 0, c as u32, t as u32, h as u32, w as u32, p as u32],
            [0.0; 2],
        );
    }

    fn unpatchify(&self, x: &GBuf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &GBuf) {
        let n = c * t * h * w;
        self.elementwise(
            4,
            out,
            Some(x),
            None,
            n,
            [0, 0, 0, c as u32, t as u32, h as u32, w as u32, p as u32],
            [0.0; 2],
        );
    }

    fn dmd_step(&self, x: &GBuf, v: &GBuf, noise: Option<&GBuf>, st: f32, sn: f32, n: usize) {
        self.elementwise(
            1,
            x,
            Some(v),
            noise,
            n,
            [0, 0, noise.is_some() as u32, 0, 0, 0, 0, 0],
            [st, sn],
        );
    }

    fn rope_heads(
        &self,
        src: &GBuf,
        src_off: usize,
        src_stride: usize,
        dst: &GBuf,
        dst_off: usize,
        dst_stride: usize,
        heads: usize,
        rows: usize,
        hd: usize,
        rope: (&GBuf, &GBuf),
    ) {
        let pipe = self.pipeline("memory", || MEMORY.to_string());
        let p = [
            0,
            heads as u32,
            rows as u32,
            hd as u32,
            src_off as u32,
            src_stride as u32,
            dst_off as u32,
            dst_stride as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let n = heads * rows * hd / 2;
        self.dispatch(
            &pipe,
            &p,
            &[src, rope.0, rope.1, dst],
            Self::grid(n.div_ceil(256)),
        );
    }

    fn kv_ema(
        &self,
        buf: &GBuf,
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
        let pipe = self.pipeline("memory", || MEMORY.to_string());
        let p = [
            1,
            heads as u32,
            0,
            hd as u32,
            0,
            0,
            0,
            head_stride as u32,
            src_row as u32,
            src_rows as u32,
            dst_row as u32,
            frame_rows as u32,
            alphas[0].to_bits(),
            alphas[1].to_bits(),
            0,
            0,
        ];
        self.dispatch(
            &pipe,
            &p,
            &[&self.dummy, &self.dummy, &self.dummy, buf],
            Self::grid(heads * hd.div_ceil(32)),
        );
    }

    fn conv2d(
        &self,
        a: &GBuf,
        b: Option<&GBuf>,
        s: &ConvShape,
        w: &GWeight,
        bias: Option<&GBuf>,
        ep: ConvEpilogue<'_, GBuf>,
        out: &GBuf,
    ) {
        assert_eq!((w.rows, w.cols), (s.cout, s.k_len()), "conv weight shape");
        assert!(w.quant.is_none(), "conv weights are not quantized");
        let f16 = w.f16;
        if self.prof.is_some() {
            *self.tag.lock().unwrap() = Some(format!("{}->{} k{}", s.cin(), s.cout, s.k));
        }
        let pipe = self.pipeline(if f16 { "conv_f16" } else { "conv_f32" }, || {
            Self::weight_src(CONV, f16)
        });
        let (epc, res) = match ep {
            ConvEpilogue::None => (0, &self.dummy),
            ConvEpilogue::Relu => (1, &self.dummy),
            ConvEpilogue::AddRelu(r) => (2, r),
            ConvEpilogue::Prelu(r) => (3, r),
        };
        let p = [
            s.frames as u32,
            s.cin_a as u32,
            s.cin_b as u32,
            s.cout as u32,
            s.h as u32,
            s.w as u32,
            if s.upsample { 2 } else { 1 },
            s.k as u32,
            b.is_some() as u32,
            bias.is_some() as u32,
            epc,
        ];
        let (oh, ow) = s.out_hw();
        let mt = (s.frames * oh * ow).div_ceil(64);
        let (gy, gz) = if mt as u32 <= MAX_DIM {
            (mt as u32, 1)
        } else {
            let z = (mt as u32).div_ceil(MAX_DIM);
            ((mt as u32).div_ceil(z), z)
        };
        let groups = (s.cout.div_ceil(64) as u32, gy, gz);
        self.dispatch(
            &pipe,
            &p,
            &[
                a,
                b.unwrap_or(&self.dummy),
                &w.buf,
                bias.unwrap_or(&self.dummy),
                res,
                out,
            ],
            groups,
        );
    }

    fn pack_rgba8(&self, src: &GBuf, base: Option<&GBuf>, s: &PackShape, out: &GBuf) {
        let n = s.pixels();
        assert!(
            out.len >= n,
            "pack_rgba8: out holds {} words, needs {n}",
            out.len
        );
        let pipe = self.pipeline("pack", || PACK.to_string());
        let p = [
            n as u32,
            s.skip as u32,
            s.h as u32,
            s.w as u32,
            s.p as u32,
            base.is_some() as u32,
            s.down as u32,
            0,
        ];
        self.dispatch(
            &pipe,
            &p,
            &[out, src, base.unwrap_or(&self.dummy)],
            Self::grid(n.div_ceil(256)),
        );
    }

    fn unpack_rgba8(&self, src: &GBuf, frames: usize, plane: usize, out: &GBuf) {
        let n = frames * 3 * plane;
        assert!(
            src.len >= frames * plane && out.len >= n,
            "unpack_rgba8: sizes"
        );
        let pipe = self.pipeline("unpack", || UNPACK.to_string());
        self.dispatch(
            &pipe,
            &[n as u32, plane as u32, 0, 0],
            &[out, src],
            Self::grid(n.div_ceil(256)),
        );
    }

    fn add_act(&self, y: &GBuf, y_off: usize, d: &GBuf, d_off: usize, n: usize, act: Act) {
        let gelu = matches!(act, Act::GeluTanh) as u32;
        let extra = [0, gelu, 0, y_off as u32, d_off as u32, 0, 0, 0];
        self.elementwise(5, y, Some(d), None, n, extra, [0.0, 0.0]);
    }

    fn tanh_clamp(&self, x: &GBuf, n: usize, s: f32) {
        self.elementwise(2, x, None, None, n, [0; 8], [s, 0.0]);
    }
}

impl Wgpu {
    /// The largest storage-buffer binding this device accepts (the memory
    /// planner splits the FFN and decoder to stay under it).
    pub fn max_binding(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size as u64
    }

    /// Blocking readback (native only): `f` sees the mapped bytes.
    #[cfg(not(target_arch = "wasm32"))]
    fn read_with<T>(&self, buf: &GBuf, f: impl FnOnce(&[u8]) -> T) -> T {
        let size = (buf.len.max(1) * 4) as u64;
        let staging = tracked(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        {
            let mut pend = self.pending.lock().unwrap();
            let enc = pend
                .enc
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        }
        self.flush();
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device lost while reading back");
        rx.recv().unwrap().expect("map failed");
        let data = slice.get_mapped_range().expect("mapped range");
        let out = f(&data);
        drop(data);
        staging.unmap();
        self.recycler.retire(staging);
        out
    }

    /// The first WebGPU error or device loss seen so far (browser only;
    /// natively wgpu panics on them).
    pub fn fault(&self) -> Option<String> {
        self.fault.lock().unwrap().clone()
    }

    /// Adapter limits that decide what fits, for messages.
    pub fn limits_text(&self) -> String {
        let l = self.device.limits();
        format!(
            "maxStorageBufferBindingSize {} MiB, maxBufferSize {} MiB",
            l.max_storage_buffer_binding_size as u64 >> 20,
            l.max_buffer_size >> 20
        )
    }

    /// [`Self::download_async`] that reports a failed map or a recorded
    /// WebGPU error instead of panicking.
    pub async fn try_download_async(&self, buf: &GBuf) -> Result<Vec<f32>, String> {
        self.readback_begin(buf)?.wait(self).await
    }

    /// Start reading `buf` back: the copy into a staging buffer is submitted
    /// now and mapped asynchronously, so work queued after this call does not
    /// delay it. Await [`Readback::wait`] later; the long-video loop queues
    /// the next block in between, so the GPU is never idle while the page
    /// waits for frames.
    pub fn readback_begin(&self, buf: &GBuf) -> Result<Readback, String> {
        if let Some(f) = self.fault() {
            return Err(f);
        }
        let size = (buf.len.max(1) * 4) as u64;
        let staging = tracked(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        {
            let mut pend = self.pending.lock().unwrap();
            let enc = pend
                .enc
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        }
        self.flush();
        let (tx, rx) = futures_oneshot();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r));
        Ok(Readback {
            staging,
            rx,
            len: buf.len,
        })
    }

    /// Non-blocking readback for the browser (and usable natively).
    pub async fn download_async(&self, buf: &GBuf) -> Vec<f32> {
        let size = (buf.len.max(1) * 4) as u64;
        let staging = tracked(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        {
            let mut pend = self.pending.lock().unwrap();
            let enc = pend
                .enc
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        }
        self.flush();
        let (tx, rx) = futures_oneshot();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r));
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.await.expect("map failed");
        let data = staging.slice(..).get_mapped_range().expect("mapped range");
        let out = bytemuck::cast_slice::<u8, f32>(&data)[..buf.len].to_vec();
        drop(data);
        staging.unmap();
        self.recycler.retire(staging);
        out
    }
}

/// A minimal one-shot channel future (no executor dependency).
/// A readback started by [`Wgpu::readback_begin`].
pub struct Readback {
    staging: wgpu::Buffer,
    rx: OneRx<Result<(), wgpu::BufferAsyncError>>,
    len: usize,
}

impl Readback {
    /// The buffer's contents once the GPU has written them.
    pub async fn wait(self, be: &Wgpu) -> Result<Vec<f32>, String> {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = be.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        #[cfg(target_arch = "wasm32")]
        let _tick = be
            .keepalive
            .load(std::sync::atomic::Ordering::Relaxed)
            .then(|| KeepAlive::start(be.queue.clone()))
            .flatten();
        if let Err(e) = self.rx.await {
            return Err(be
                .fault()
                .unwrap_or_else(|| format!("GPU readback failed: {e}")));
        }
        if let Some(f) = be.fault() {
            return Err(f);
        }
        let data = self
            .staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| format!("GPU readback failed: {e:?}"))?;
        let out = bytemuck::cast_slice::<u8, f32>(&data)[..self.len].to_vec();
        drop(data);
        self.staging.unmap();
        be.recycler.retire(self.staging);
        Ok(out)
    }
}

/// While a browser readback is pending, submit an empty batch every 20 ms.
/// Chrome delivered `mapAsync` callbacks for long videos up to ~40 s late
/// (blocks of 1.5 s took 16-41 s in runs, natively flat): an empty submit makes
/// its GPU process tick the device, which is when callbacks are delivered.
#[cfg(target_arch = "wasm32")]
struct KeepAlive {
    id: i32,
    _cb: wasm_bindgen::closure::Closure<dyn FnMut()>,
}

#[cfg(target_arch = "wasm32")]
impl KeepAlive {
    fn start(queue: wgpu::Queue) -> Option<Self> {
        use wasm_bindgen::JsCast;
        let cb = wasm_bindgen::closure::Closure::<dyn FnMut()>::new(move || {
            queue.submit(std::iter::empty());
        });
        let id = web_sys::window()?
            .set_interval_with_callback_and_timeout_and_arguments_0(cb.as_ref().unchecked_ref(), 20)
            .ok()?;
        Some(Self { id, _cb: cb })
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for KeepAlive {
    fn drop(&mut self) {
        if let Some(w) = web_sys::window() {
            w.clear_interval_with_handle(self.id);
        }
    }
}

impl Wgpu {
    /// Browser: whether readbacks keep the device ticking (default on).
    pub fn set_keepalive(&self, on: bool) {
        self.keepalive
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }
}

fn futures_oneshot<T>() -> (OneTx<T>, OneRx<T>) {
    let s = Arc::new(Mutex::new((None, None::<std::task::Waker>)));
    (OneTx(s.clone()), OneRx(s))
}
struct OneTx<T>(Arc<Mutex<(Option<T>, Option<std::task::Waker>)>>);
struct OneRx<T>(Arc<Mutex<(Option<T>, Option<std::task::Waker>)>>);
impl<T> OneTx<T> {
    fn send(self, v: T) {
        let mut g = self.0.lock().unwrap();
        g.0 = Some(v);
        if let Some(w) = g.1.take() {
            w.wake();
        }
    }
}
impl<T> std::future::Future for OneRx<T> {
    type Output = T;
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<T> {
        let mut g = self.0.lock().unwrap();
        match g.0.take() {
            Some(v) => std::task::Poll::Ready(v),
            None => {
                g.1 = Some(cx.waker().clone());
                std::task::Poll::Pending
            }
        }
    }
}
