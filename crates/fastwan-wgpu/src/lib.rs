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

use fastwan_core::backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const COMMON: &str = include_str!("shaders/common.wgsl");
const LINEAR: &str = include_str!("shaders/linear.wgsl");
const CONV: &str = include_str!("shaders/conv.wgsl");
const ATTN: &str = include_str!("shaders/attention.wgsl");
const NORM: &str = include_str!("shaders/norm.wgsl");
const HEADS: &str = include_str!("shaders/head_split.wgsl");
const ELEM: &str = include_str!("shaders/elementwise.wgsl");

/// Submit after this many recorded dispatches.
const BATCH: usize = 64;
/// Max workgroups per dispatch dimension (WebGPU default limit).
const MAX_DIM: u32 = 65535;

#[derive(Clone)]
pub struct GBuf {
    buf: Arc<wgpu::Buffer>,
    len: usize,
}

pub struct GWeight {
    buf: GBuf,
    rows: usize,
    cols: usize,
    f16: bool,
}

struct Pending {
    enc: Option<wgpu::CommandEncoder>,
    n: usize,
}

pub struct Wgpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    attn_bk: usize,
    pipelines: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
    pending: Mutex<Pending>,
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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fastwan"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("request_device: {e}")))?;
        // K and V tiles of BK keys × 128 dims, f32, must fit shared memory.
        let wgs = device.limits().max_compute_workgroup_storage_size as usize;
        let attn_bk = if wgs >= 2 * 32 * 128 * 4 { 32 } else { 16 };
        let dummy = GBuf {
            buf: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dummy"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })),
            len: 4,
        };
        Ok(Self {
            info: adapter.get_info(),
            device,
            queue,
            attn_bk,
            pipelines: Mutex::new(HashMap::new()),
            pending: Mutex::new(Pending { enc: None, n: 0 }),
            dummy,
            allow_f16_weights: true,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, InitError> {
        pollster::block_on(Self::new_async())
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
        let code = format!("{COMMON}\n{}", src());
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let p = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        map.insert(key.to_string(), p.clone());
        p
    }

    fn weight_src(src: &str, f16: bool) -> String {
        if f16 {
            src.replace("{W_TYPE}", "u32")
                .replace("{W_LOAD}", "let v = unpack2x16float(w[i >> 1u]); return select(v.x, v.y, (i & 1u) == 1u);")
        } else {
            src.replace("{W_TYPE}", "f32").replace("{W_LOAD}", "return w[i];")
        }
    }

    /// Record one dispatch. `params` is the uniform block as 32-bit words.
    fn dispatch(&self, pipe: &wgpu::ComputePipeline, params: &[u32], bufs: &[&GBuf], groups: (u32, u32, u32)) {
        let mut words = params.to_vec();
        while words.len() % 4 != 0 {
            words.push(0);
        }
        let ubuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck_cast(&words),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.buf.as_entire_binding() });
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
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        pend.n += 1;
        if pend.n >= BATCH {
            let enc = pend.enc.take().unwrap();
            pend.n = 0;
            drop(pend);
            self.queue.submit([enc.finish()]);
        }
    }

    fn flush(&self) {
        let mut pend = self.pending.lock().unwrap();
        if let Some(enc) = pend.enc.take() {
            pend.n = 0;
            drop(pend);
            self.queue.submit([enc.finish()]);
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

    fn elementwise(&self, op: u32, x: &GBuf, y: Option<&GBuf>, aux: Option<&GBuf>, n: usize, extra: [u32; 8], s: [f32; 2]) {
        let pipe = self.pipeline("elementwise", || ELEM.to_string());
        let mut p = vec![n as u32, op];
        p.extend_from_slice(&extra);
        p.push(s[0].to_bits());
        p.push(s[1].to_bits());
        self.dispatch(&pipe, &p, &[x, y.unwrap_or(&self.dummy), aux.unwrap_or(&self.dummy)], Self::grid(n.div_ceil(256)));
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(&self, h: &GBuf, a: Option<&GBuf>, rows: usize, dim: usize, mods: &GBuf, gate: Option<usize>, norm: NormKind<'_, GBuf>, eps: f32, out: &GBuf) {
        let pipe = self.pipeline("norm", || NORM.to_string());
        let (affine, scale, shift, w, b) = match norm {
            NormKind::Affine { w, b } => (1, 0, 0, w, b),
            NormKind::Mod { scale, shift } => (0, scale as u32, shift as u32, &self.dummy, &self.dummy),
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
        self.dispatch(&pipe, &p, &[h, a.unwrap_or(&self.dummy), mods, w, b, out], Self::grid(rows));
    }
}

fn bytemuck_cast(w: &[u32]) -> &[u8] {
    bytemuck::cast_slice(w)
}

impl Backend for Wgpu {
    type Buf = GBuf;
    type Weight = GWeight;

    fn name(&self) -> String {
        format!("wgpu {:?}: {} ({})", self.info.backend, self.info.name, self.info.driver)
    }

    fn alloc(&self, len: usize) -> GBuf {
        GBuf {
            buf: Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (len.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
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
            buf: Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })),
            len: data.len(),
        }
    }

    fn write(&self, buf: &GBuf, data: &[f32]) {
        self.flush();
        self.queue.write_buffer(&buf.buf, 0, bytemuck::cast_slice(data));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn download(&self, buf: &GBuf) -> Vec<f32> {
        let size = (buf.len.max(1) * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("device lost while reading back");
        rx.recv().unwrap().expect("map failed");
        let data = slice.get_mapped_range().expect("mapped range");
        let out = bytemuck::cast_slice::<u8, f32>(&data)[..buf.len].to_vec();
        drop(data);
        staging.unmap();
        out
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
        enc.copy_buffer_to_buffer(&src.buf, (so * 4) as u64, &dst.buf, (d * 4) as u64, (len * 4) as u64);
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
                buf: Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&packed),
                    usage: wgpu::BufferUsages::STORAGE,
                })),
                len: packed.len(),
            };
            GWeight { buf, rows, cols, f16: true }
        } else {
            GWeight { buf: self.upload(data), rows, cols, f16: false }
        }
    }

    fn sync(&self) {
        self.flush();
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    fn linear(&self, x: &GBuf, x_row: usize, rows: usize, w: &GWeight, bias: Option<&GBuf>, act: Act, y: &GBuf, y_row: usize) {
        let f16 = w.f16;
        let pipe = self.pipeline(if f16 { "linear_f16" } else { "linear_f32" }, || Self::weight_src(LINEAR, f16));
        let p = [
            rows as u32,
            w.rows as u32,
            w.cols as u32,
            x_row as u32,
            y_row as u32,
            bias.is_some() as u32,
            (act == Act::GeluTanh) as u32,
        ];
        let groups = (w.rows.div_ceil(64) as u32, rows.div_ceil(64) as u32, 1);
        self.dispatch(&pipe, &p, &[x, &w.buf, bias.unwrap_or(&self.dummy), y], groups);
    }

    fn layer_norm(&self, x: &GBuf, rows: usize, dim: usize, norm: NormKind<'_, GBuf>, mods: &GBuf, eps: f32, out: &GBuf) {
        self.norm(x, None, rows, dim, mods, None, norm, eps, out);
    }

    fn residual_norm(&self, h: &GBuf, a: &GBuf, rows: usize, dim: usize, mods: &GBuf, gate: Option<usize>, norm: NormKind<'_, GBuf>, eps: f32, normed: &GBuf) {
        self.norm(h, Some(a), rows, dim, mods, gate, norm, eps, normed);
    }

    fn gate_residual(&self, h: &GBuf, a: &GBuf, rows: usize, dim: usize, mods: &GBuf, gate: usize) {
        self.elementwise(0, h, Some(a), Some(mods), rows * dim, [dim as u32, gate as u32, 0, 0, 0, 0, 0, 0], [0.0; 2]);
    }

    fn head_split(&self, src: &GBuf, rows: usize, stride: usize, col_off: usize, heads: usize, hd: usize, rms_w: Option<&GBuf>, rope: Option<(&GBuf, &GBuf)>, eps: f32, out: &GBuf) {
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
        self.dispatch(&pipe, &p, &[src, rms_w.unwrap_or(&self.dummy), c, s, out], Self::grid(rows));
    }

    fn attention(&self, q: &GBuf, k: &GBuf, v: &GBuf, heads: usize, sq: usize, sk: usize, d: usize, out: &GBuf) {
        assert!(d <= 128, "attention kernel supports head_dim <= 128");
        let bk = self.attn_bk;
        let pipe = self.pipeline(&format!("attention_bk{bk}"), || {
            ATTN.replace("{BK_D}", &(bk * 128).to_string()).replace("{BK}", &bk.to_string())
        });
        let p = [heads as u32, sq as u32, sk as u32, d as u32, (1.0 / (d as f32).sqrt()).to_bits()];
        assert!(heads as u32 <= MAX_DIM);
        self.dispatch(&pipe, &p, &[q, k, v, out], (sq.div_ceil(64) as u32, heads as u32, 1));
    }

    fn patchify(&self, latent: &GBuf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &GBuf) {
        let n = c * t * h * w;
        self.elementwise(3, out, Some(latent), None, n, [0, 0, 0, c as u32, t as u32, h as u32, w as u32, p as u32], [0.0; 2]);
    }

    fn unpatchify(&self, x: &GBuf, c: usize, t: usize, h: usize, w: usize, p: usize, out: &GBuf) {
        let n = c * t * h * w;
        self.elementwise(4, out, Some(x), None, n, [0, 0, 0, c as u32, t as u32, h as u32, w as u32, p as u32], [0.0; 2]);
    }

    fn dmd_step(&self, x: &GBuf, v: &GBuf, noise: Option<&GBuf>, st: f32, sn: f32, n: usize) {
        self.elementwise(1, x, Some(v), noise, n, [0, 0, noise.is_some() as u32, 0, 0, 0, 0, 0], [st, sn]);
    }

    fn conv2d(&self, a: &GBuf, b: Option<&GBuf>, s: &ConvShape, w: &GWeight, bias: Option<&GBuf>, ep: ConvEpilogue<'_, GBuf>, out: &GBuf) {
        assert_eq!((w.rows, w.cols), (s.cout, s.k_len()), "conv weight shape");
        let f16 = w.f16;
        let pipe = self.pipeline(if f16 { "conv_f16" } else { "conv_f32" }, || Self::weight_src(CONV, f16));
        let (epc, res) = match ep {
            ConvEpilogue::None => (0, &self.dummy),
            ConvEpilogue::Relu => (1, &self.dummy),
            ConvEpilogue::AddRelu(r) => (2, r),
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
        let (gy, gz) = if mt as u32 <= MAX_DIM { (mt as u32, 1) } else { let z = (mt as u32).div_ceil(MAX_DIM); ((mt as u32).div_ceil(z), z) };
        let groups = (s.cout.div_ceil(64) as u32, gy, gz);
        self.dispatch(&pipe, &p, &[a, b.unwrap_or(&self.dummy), &w.buf, bias.unwrap_or(&self.dummy), res, out], groups);
    }

    fn tanh_clamp(&self, x: &GBuf, n: usize, s: f32) {
        self.elementwise(2, x, None, None, n, [0; 8], [s, 0.0]);
    }
}

impl Wgpu {
    /// Non-blocking readback for the browser (and usable natively).
    pub async fn download_async(&self, buf: &GBuf) -> Vec<f32> {
        let size = (buf.len.max(1) * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut pend = self.pending.lock().unwrap();
            let enc = pend
                .enc
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        }
        self.flush();
        let (tx, rx) = futures_oneshot();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r));
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.await.expect("map failed");
        let data = staging.slice(..).get_mapped_range().expect("mapped range");
        let out = bytemuck::cast_slice::<u8, f32>(&data)[..buf.len].to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

/// A minimal one-shot channel future (no executor dependency).
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
