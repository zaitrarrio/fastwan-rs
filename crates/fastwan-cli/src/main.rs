//! `fastwan`: plan, check, bench and generate FastWan on CPU, WebGPU or CUDA.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fastwan_core::{
    config::{TaeConfig, WanConfig},
    geometry::Geometry,
    plan::{plan, Device},
    schedule::DmdSchedule,
    weights::{missing, tae_manifest, wan_manifest, SafetensorsDir, Synthetic, WeightSource as _},
    Backend,
};
use fastwan_model::{dit::WanDit, pipeline::{Options, Pipeline}, taehv::TaeHv, Video};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "fastwan", about = "FastWan text-to-video on CPU, WebGPU or CUDA")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum BackendKind {
    /// CUDA when built in and a device is present, else WebGPU (Vulkan/DX12/Metal: NVIDIA, AMD, Intel, Apple).
    Auto,
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(clap::Args, Clone)]
struct Shape {
    #[arg(long, default_value_t = 832)]
    width: usize,
    #[arg(long, default_value_t = 480)]
    height: usize,
    #[arg(long, default_value_t = 81)]
    frames: usize,
}

impl Shape {
    fn geometry(&self) -> Result<Geometry> {
        let g = Geometry { width: self.width, height: self.height, frames: self.frames };
        g.validate(&WanConfig::fastwan_1_3b())?;
        Ok(g)
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Memory plan for a device, before allocating anything.
    Plan {
        #[command(flatten)]
        shape: Shape,
        /// Device memory budget.
        #[arg(long, default_value_t = 24.0)]
        vram_gib: f64,
        /// Largest buffer binding (WebGPU adapters cap this), MiB.
        #[arg(long)]
        max_binding_mib: Option<u64>,
        #[arg(long, value_enum, default_value_t = BackendKind::Cuda)]
        backend: BackendKind,
    },
    /// Every op and the tiny model against the CPU reference.
    Check {
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
    },
    /// Time the DMD steps (and decode) with synthetic weights at a real geometry.
    Bench {
        #[command(flatten)]
        shape: Shape,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// DiT layers to build (30 for the real model; fewer for a quick look).
        #[arg(long, default_value_t = 30)]
        layers: usize,
        #[arg(long, default_value_t = false)]
        decode: bool,
    },
    /// Generate a clip from precomputed UMT5 prompt embeddings.
    Generate {
        #[command(flatten)]
        shape: Shape,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// The checkpoint's `transformer/` directory (Diffusers safetensors).
        #[arg(long)]
        transformer: PathBuf,
        /// `taew2_1.safetensors`.
        #[arg(long)]
        tae: PathBuf,
        /// safetensors holding `prompt_embeds` `[L, 4096]` (scripts/encode_prompt.py).
        #[arg(long)]
        embeds: PathBuf,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value = "out.y4m")]
        out: PathBuf,
        #[arg(long)]
        ffn_chunk: Option<usize>,
        #[arg(long, default_value_t = 2)]
        tae_chunk: usize,
    },
}

/// Run `f` with the selected backend.
macro_rules! with_backend {
    ($kind:expr, |$be:ident| $body:expr) => {{
        let kind = resolve($kind);
        match kind {
            BackendKind::Auto => unreachable!(),
            BackendKind::Cpu => {
                let $be = fastwan_cpu::Cpu;
                $body
            }
            BackendKind::Wgpu => {
                let $be = fastwan_wgpu::Wgpu::new().map_err(|e| anyhow::anyhow!("{e}"))?;
                $body
            }
            BackendKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let $be = fastwan_cuda::Cuda::new(0).map_err(|e| anyhow::anyhow!("{e}"))?;
                    $body
                }
                #[cfg(not(feature = "cuda"))]
                {
                    bail!("built without CUDA: cargo build --release --features cuda")
                }
            }
        }
    }};
}

fn resolve(kind: BackendKind) -> BackendKind {
    if kind != BackendKind::Auto {
        return kind;
    }
    #[cfg(feature = "cuda")]
    if fastwan_cuda::Cuda::new(0).is_ok() {
        return BackendKind::Cuda;
    }
    BackendKind::Wgpu
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Plan { shape, vram_gib, max_binding_mib, backend } => {
            let act = if backend == BackendKind::Cuda { 2 } else { 4 };
            let mut dev = Device::gib(vram_gib, 2, act);
            if let Some(m) = max_binding_mib {
                dev.max_binding_bytes = m << 20;
            }
            let p = plan(&WanConfig::fastwan_1_3b(), &TaeConfig::taew2_1(), &shape.geometry()?, &dev);
            print!("{p}");
        }
        Cmd::Check { backend } => with_backend!(backend, |be| check(&be, resolve(backend))?),
        Cmd::Bench { shape, backend, layers, decode } => {
            with_backend!(backend, |be| bench(&be, &shape.geometry()?, layers, decode)?)
        }
        Cmd::Generate { shape, backend, transformer, tae, embeds, seed, out, ffn_chunk, tae_chunk } => {
            let g = shape.geometry()?;
            with_backend!(backend, |be| generate(&be, &g, &transformer, &tae, &embeds, seed, &out, ffn_chunk, tae_chunk)?)
        }
    }
    Ok(())
}

fn check<B: Backend>(be: &B, kind: BackendKind) -> Result<()> {
    println!("{}", be.name());
    // bf16 activations on CUDA: ~3 significant digits per op.
    let tol = if kind == BackendKind::Cuda { 3e-2 } else { 2e-3 };
    let res = fastwan_model::check::run_all(be, tol);
    let mut bad = 0;
    for r in &res {
        println!("{:<48} rel {:.2e}  {}", r.name, r.rel_err, if r.ok { "ok" } else { "FAIL" });
        bad += (!r.ok) as usize;
    }
    if bad > 0 {
        bail!("{bad} of {} checks failed (tolerance {tol:e})", res.len());
    }
    println!("all {} checks pass (tolerance {tol:e})", res.len());
    Ok(())
}

fn bench<B: Backend>(be: &B, g: &Geometry, layers: usize, decode: bool) -> Result<()> {
    let cfg = WanConfig { layers, ..WanConfig::fastwan_1_3b() };
    println!("{}  |  {}x{}x{}  {} tokens  {} layers", be.name(), g.width, g.height, g.frames, g.tokens(&cfg), layers);
    let src = Synthetic { seed: 1 };
    let t0 = std::time::Instant::now();
    let pipe = Pipeline {
        dit: WanDit::load(be, cfg.clone(), &src)?,
        tae: TaeHv::load(be, TaeConfig::taew2_1(), &src)?,
        schedule: DmdSchedule::fastwan_1_3b(),
    };
    println!("load (synthetic) {:.2}s", t0.elapsed().as_secs_f64());
    let embeds = vec![0.0; cfg.text_len * cfg.text_dim];
    let opt = Options { geometry: *g, seed: 7, ffn_chunk: usize::MAX, tae_chunk: 2 };
    let mut t = fastwan_model::Timings::default();
    let latent = pipe.denoise(be, &embeds, &opt, &mut t)?;
    println!("prompt prep {:.3}s", t.prompt_s);
    for (i, s) in t.steps_s.iter().enumerate() {
        println!("step {i} {:.3}s  ({:.1} ms/layer)", s, s * 1e3 / layers as f64);
    }
    println!("denoise {:.3}s; scaled to 30 layers ≈ {:.2}s", t.denoise_s(), t.denoise_s() * 30.0 / layers as f64);
    if decode {
        let v = pipe.decode(be, &latent, g, opt.tae_chunk, &mut t);
        println!("decode (TAEHV) {:.3}s -> {} frames {}x{}", t.decode_s, v.frames, v.width, v.height);
    }
    Ok(())
}

fn load_embeds(path: &Path, cfg: &WanConfig) -> Result<Vec<f32>> {
    let st = SafetensorsDir::open(path)?;
    let name = st.names().find(|n| n.contains("embed")).map(str::to_string)
        .context("no `prompt_embeds` tensor in the embeddings file")?;
    // Accept [L, D] or [1, L, D]; pad with zeros (or trim) to text_len.
    let total: Vec<f32> = (1..=cfg.text_len)
        .rev()
        .find_map(|l| st.get(&name, &[l, cfg.text_dim]).ok())
        .context("prompt_embeds must be [L <= 512, 4096]")?;
    let mut out = total;
    out.resize(cfg.text_len * cfg.text_dim, 0.0);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn generate<B: Backend>(be: &B, g: &Geometry, transformer: &Path, tae: &Path, embeds: &Path, seed: u64, out: &Path, ffn_chunk: Option<usize>, tae_chunk: usize) -> Result<()> {
    let cfg = WanConfig::fastwan_1_3b();
    let tcfg = TaeConfig::from_file_name(&tae.to_string_lossy()).unwrap_or_else(TaeConfig::taew2_1);
    if tcfg.latent_channels != cfg.out_channels {
        bail!("{} decodes {}-channel latents; FastWan 2.1 makes {} (use taew2_1)", tae.display(), tcfg.latent_channels, cfg.out_channels);
    }
    println!("{}", be.name());
    let t0 = std::time::Instant::now();
    let wsrc = SafetensorsDir::open(transformer)?;
    let miss = missing(&wsrc, &wan_manifest(&cfg));
    if !miss.is_empty() {
        bail!("{} tensors missing from {}, e.g. {:?}", miss.len(), transformer.display(), &miss[..miss.len().min(3)]);
    }
    let tsrc = SafetensorsDir::open(tae)?;
    let miss = missing(&tsrc, &tae_manifest(&tcfg));
    if !miss.is_empty() {
        bail!("{} tensors missing from {}, e.g. {:?}", miss.len(), tae.display(), &miss[..miss.len().min(3)]);
    }
    let pipe = Pipeline {
        dit: WanDit::load(be, cfg.clone(), &wsrc)?,
        tae: TaeHv::load(be, tcfg, &tsrc)?,
        schedule: DmdSchedule::fastwan_1_3b(),
    };
    let e = load_embeds(embeds, &cfg)?;
    be.sync();
    println!("load {:.1}s", t0.elapsed().as_secs_f64());
    let opt = Options { geometry: *g, seed, ffn_chunk: ffn_chunk.unwrap_or(usize::MAX), tae_chunk };
    let (video, t) = pipe.generate(be, &e, &opt)?;
    println!("prompt {:.2}s  denoise {:.2}s ({:?})  decode {:.2}s", t.prompt_s, t.denoise_s(),
        t.steps_s.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>(), t.decode_s);
    write_y4m(out, &video, 16)?;
    println!("wrote {} ({} frames {}x{})", out.display(), video.frames, video.width, video.height);
    Ok(())
}

/// YUV4MPEG2, 4:4:4, BT.601 full range: plays in mpv/ffplay, converts with
/// `ffmpeg -i out.y4m out.mp4`, and needs no dependency to write.
fn write_y4m(path: &Path, v: &Video, fps: u32) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "YUV4MPEG2 W{} H{} F{fps}:1 Ip A1:1 C444 XCOLORRANGE=FULL", v.width, v.height)?;
    let px = v.width * v.height;
    let mut planes = vec![0u8; 3 * px];
    for fr in 0..v.frames {
        let rgb = &v.rgb[fr * px * 3..(fr + 1) * px * 3];
        for i in 0..px {
            let (r, g, b) = (rgb[3 * i] as f32, rgb[3 * i + 1] as f32, rgb[3 * i + 2] as f32);
            planes[i] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
            planes[px + i] = (128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b).round().clamp(0.0, 255.0) as u8;
            planes[2 * px + i] = (128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b).round().clamp(0.0, 255.0) as u8;
        }
        f.write_all(b"FRAME\n")?;
        f.write_all(&planes)?;
    }
    Ok(())
}

