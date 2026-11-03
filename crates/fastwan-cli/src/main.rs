//! `fastwan`: plan, check, bench and generate FastWan on CPU, WebGPU or CUDA.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use fastwan_core::{
    Backend,
    config::{TaeConfig, WanConfig},
    geometry::Geometry,
    plan::{Device, plan},
    schedule::DmdSchedule,
    weights::{SafetensorsDir, Synthetic, WeightSource as _, missing, tae_manifest, wan_manifest},
};
use fastwan_model::{
    Video,
    dit::WanDit,
    pipeline::{Options, Pipeline},
    taehv::TaeHv,
};
use std::path::{Path, PathBuf};

mod serve;

#[derive(Parser)]
#[command(
    name = "fastwan",
    about = "FastWan text-to-video on CPU, WebGPU or CUDA"
)]
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

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum AttnKind {
    /// VSA on a checkpoint that carries to_gate_compress (FastWan), dense otherwise.
    Auto,
    Dense,
    Vsa,
}

#[derive(clap::Args, Clone, Copy)]
struct AttnArgs {
    #[arg(long, value_enum, default_value_t = AttnKind::Auto)]
    attention: AttnKind,
    /// Fraction of key tiles VSA skips (FastVideo's default 0.8).
    #[arg(long, default_value_t = fastwan_core::vsa::DEFAULT_SPARSITY)]
    vsa_sparsity: f64,
}

impl AttnArgs {
    fn get(&self) -> Option<fastwan_model::Attention> {
        match self.attention {
            AttnKind::Auto => None,
            AttnKind::Dense => Some(fastwan_model::Attention::Dense),
            AttnKind::Vsa => Some(fastwan_model::Attention::Vsa {
                sparsity: self.vsa_sparsity,
            }),
        }
    }
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
        let g = Geometry {
            width: self.width,
            height: self.height,
            frames: self.frames,
        };
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
        #[command(flatten)]
        attn: AttnArgs,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// DiT layers to build (30 for the real model; fewer for a quick look).
        #[arg(long, default_value_t = 30)]
        layers: usize,
        #[arg(long, default_value_t = false)]
        decode: bool,
    },
    /// Run the local engine and the prompt-to-clip demo page.
    Serve {
        #[arg(long, default_value = "127.0.0.1:7860")]
        addr: String,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        #[command(flatten)]
        attn: AttnArgs,
        /// A FastVideo/FastWan2.1-T2V-1.3B-Diffusers download (transformer/,
        /// text_encoder/, tokenizer/); TAEHV is read from <dir>/taew2_1.safetensors
        /// unless --tae says otherwise.
        #[arg(long, required_unless_present = "synthetic")]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        tae: Option<PathBuf>,
        /// Tiny random model: test the engine and page without weights.
        #[arg(long, default_value_t = false)]
        synthetic: bool,
        /// Largest frame the page may ask for, in pixels (832×480 = 399360).
        #[arg(long, default_value_t = 832 * 480)]
        max_pixels: usize,
        #[arg(long, default_value_t = 81)]
        max_frames: usize,
        #[arg(long, default_value_t = 2)]
        tae_chunk: usize,
        /// Let one web origin call this engine from the browser, e.g.
        /// https://fastwan.pages.dev for the hosted demo page.
        #[arg(long)]
        allow_origin: Option<String>,
    },
    /// Generate a clip from precomputed UMT5 prompt embeddings.
    Generate {
        #[command(flatten)]
        shape: Shape,
        #[command(flatten)]
        attn: AttnArgs,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// The checkpoint's `transformer/` directory (Diffusers safetensors).
        #[arg(long)]
        transformer: PathBuf,
        /// `taew2_1.safetensors`.
        #[arg(long)]
        tae: PathBuf,
        /// safetensors holding `prompt_embeds` `[L, 4096]` (scripts/encode_prompt.py).
        #[arg(long, required_unless_present = "prompt")]
        embeds: Option<PathBuf>,
        /// Encode this prompt with the native UMT5 (needs --text-encoder and --tokenizer).
        #[arg(long, requires_all = ["text_encoder", "tokenizer"])]
        prompt: Option<String>,
        /// The checkpoint's `text_encoder/` directory.
        #[arg(long)]
        text_encoder: Option<PathBuf>,
        /// `tokenizer/tokenizer.json`.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Generate this many times (same seed) and report each; the first run
        /// includes shader / kernel compilation.
        #[arg(long, default_value_t = 1)]
        runs: usize,
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
                if $be.is_software() && !allow_software() {
                    bail!(
                        "WebGPU found only a software rasterizer ({}): the GPU's Vulkan/DX12/Metal driver did not load. \
                         Fix the driver, or set FASTWAN_ALLOW_SOFTWARE=1 to run on the CPU rasterizer anyway",
                        $be.adapter_info().name
                    );
                }
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

/// Software WebGPU adapters are refused unless asked for; `check` always
/// allows them (it compares against the CPU reference, not a clock).
fn allow_software() -> bool {
    std::env::var("FASTWAN_ALLOW_SOFTWARE").is_ok_and(|v| v == "1")
        || ALLOW_SW.load(std::sync::atomic::Ordering::Relaxed)
}
static ALLOW_SW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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
    if matches!(cli.cmd, Cmd::Check { .. } | Cmd::Plan { .. }) {
        ALLOW_SW.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    match cli.cmd {
        Cmd::Plan {
            shape,
            vram_gib,
            max_binding_mib,
            backend,
        } => {
            let act = if backend == BackendKind::Cuda { 2 } else { 4 };
            let mut dev = Device::gib(vram_gib, 2, act);
            if let Some(m) = max_binding_mib {
                dev.max_binding_bytes = m << 20;
            }
            let p = plan(
                &WanConfig::fastwan_1_3b(),
                &TaeConfig::taew2_1(),
                &shape.geometry()?,
                &dev,
            );
            print!("{p}");
        }
        Cmd::Check { backend } => with_backend!(backend, |be| check(&be, resolve(backend))?),
        Cmd::Bench {
            attn,
            shape,
            backend,
            layers,
            decode,
        } => {
            with_backend!(backend, |be| bench(
                &be,
                &shape.geometry()?,
                layers,
                decode,
                attn.get()
            )?)
        }
        Cmd::Serve {
            attn,
            addr,
            backend,
            model_dir,
            tae,
            synthetic,
            max_pixels,
            max_frames,
            tae_chunk,
            allow_origin,
        } => {
            let weights = if synthetic {
                serve::Weights::Synthetic
            } else {
                let d = model_dir.expect("clap requires --model-dir");
                serve::Weights::Real {
                    transformer: d.join("transformer"),
                    text_encoder: d.join("text_encoder"),
                    tokenizer: d.join("tokenizer").join("tokenizer.json"),
                    tae: tae.unwrap_or_else(|| d.join("taew2_1.safetensors")),
                }
            };
            let settings = serve::Settings {
                addr,
                weights,
                max_pixels,
                max_frames,
                tae_chunk,
                allow_origin,
                attention: attn.get(),
            };
            serve::serve(settings, move |s, shared, rx| {
                with_backend!(backend, |be| serve::engine(be, s, shared, rx))
            })?;
        }
        Cmd::Generate {
            attn,
            shape,
            backend,
            transformer,
            tae,
            embeds,
            prompt,
            text_encoder,
            tokenizer,
            runs,
            seed,
            out,
            ffn_chunk,
            tae_chunk,
        } => {
            let g = shape.geometry()?;
            let cfg = WanConfig::fastwan_1_3b();
            let e = match (prompt, embeds) {
                (Some(p), _) => {
                    let t0 = std::time::Instant::now();
                    let enc = fastwan_text::PromptEncoder::open(
                        tokenizer.as_deref().expect("clap requires --tokenizer"),
                        text_encoder
                            .as_deref()
                            .expect("clap requires --text-encoder"),
                        fastwan_text::Umt5Config::xxl(),
                    )?;
                    let n = enc.tokenize(&p)?.len();
                    let e = enc.encode(&p)?;
                    println!(
                        "umt5 (cpu) {n} tokens in {:.2}s",
                        t0.elapsed().as_secs_f64()
                    );
                    e
                }
                (None, Some(path)) => load_embeds(&path, &cfg)?,
                (None, None) => bail!("give --prompt or --embeds"),
            };
            with_backend!(backend, |be| generate(
                &be,
                &g,
                &transformer,
                &tae,
                &e,
                seed,
                &out,
                ffn_chunk,
                tae_chunk,
                runs,
                attn.get()
            )?)
        }
    }
    Ok(())
}

fn check<B: Backend>(be: &B, kind: BackendKind) -> Result<()> {
    println!("{}", be.name());
    // bf16 activations on CUDA: ~3 significant digits per op.
    let tol = if kind == BackendKind::Cuda {
        3e-2
    } else {
        2e-3
    };
    let res = fastwan_model::check::run_all(be, tol);
    let mut bad = 0;
    for r in &res {
        println!(
            "{:<48} rel {:.2e}  {}",
            r.name,
            r.rel_err,
            if r.ok { "ok" } else { "FAIL" }
        );
        bad += (!r.ok) as usize;
    }
    if bad > 0 {
        bail!("{bad} of {} checks failed (tolerance {tol:e})", res.len());
    }
    println!("all {} checks pass (tolerance {tol:e})", res.len());
    Ok(())
}

fn bench<B: Backend>(
    be: &B,
    g: &Geometry,
    layers: usize,
    decode: bool,
    attention: Option<fastwan_model::Attention>,
) -> Result<()> {
    let cfg = WanConfig {
        layers,
        ..WanConfig::fastwan_1_3b()
    };
    println!(
        "{}  |  {}x{}x{}  {} tokens  {} layers",
        be.name(),
        g.width,
        g.height,
        g.frames,
        g.tokens(&cfg),
        layers
    );
    let src = Synthetic { seed: 1 };
    let t0 = std::time::Instant::now();
    let pipe = Pipeline {
        dit: WanDit::load(be, cfg.clone(), &src)?,
        tae: TaeHv::load(be, TaeConfig::taew2_1(), &src)?,
        schedule: DmdSchedule::fastwan_1_3b(),
    };
    println!("load (synthetic) {:.2}s", t0.elapsed().as_secs_f64());
    let embeds = vec![0.0; cfg.text_len * cfg.text_dim];
    let opt = Options {
        geometry: *g,
        seed: 7,
        ffn_chunk: usize::MAX,
        tae_chunk: 2,
        attention,
    };
    let mut t = fastwan_model::Timings::default();
    let latent = pipe.denoise(be, &embeds, &opt, &mut t)?;
    println!("prompt prep {:.3}s", t.prompt_s);
    for (i, s) in t.steps_s.iter().enumerate() {
        println!(
            "step {i} {:.3}s  ({:.1} ms/layer)",
            s,
            s * 1e3 / layers as f64
        );
    }
    println!(
        "denoise {:.3}s; scaled to 30 layers ≈ {:.2}s",
        t.denoise_s(),
        t.denoise_s() * 30.0 / layers as f64
    );
    if decode {
        let v = pipe.decode(be, &latent, g, opt.tae_chunk, &mut t);
        println!(
            "decode (TAEHV) {:.3}s -> {} frames {}x{}",
            t.decode_s, v.frames, v.width, v.height
        );
    }
    Ok(())
}

fn load_embeds(path: &Path, cfg: &WanConfig) -> Result<Vec<f32>> {
    let st = SafetensorsDir::open(path)?;
    let name = st
        .names()
        .find(|n| n.contains("embed"))
        .map(str::to_string)
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
fn generate<B: Backend>(
    be: &B,
    g: &Geometry,
    transformer: &Path,
    tae: &Path,
    e: &[f32],
    seed: u64,
    out: &Path,
    ffn_chunk: Option<usize>,
    tae_chunk: usize,
    runs: usize,
    attention: Option<fastwan_model::Attention>,
) -> Result<()> {
    let cfg = WanConfig::fastwan_1_3b();
    let tcfg = TaeConfig::from_file_name(&tae.to_string_lossy()).unwrap_or_else(TaeConfig::taew2_1);
    if tcfg.latent_channels != cfg.out_channels {
        bail!(
            "{} decodes {}-channel latents; FastWan 2.1 makes {} (use taew2_1)",
            tae.display(),
            tcfg.latent_channels,
            cfg.out_channels
        );
    }
    println!("{}", be.name());
    let t0 = std::time::Instant::now();
    let wsrc = SafetensorsDir::open(transformer)?;
    let miss = missing(&wsrc, &wan_manifest(&cfg));
    if !miss.is_empty() {
        bail!(
            "{} tensors missing from {}, e.g. {:?}",
            miss.len(),
            transformer.display(),
            &miss[..miss.len().min(3)]
        );
    }
    let tsrc = SafetensorsDir::open(tae)?;
    let miss = missing(&tsrc, &tae_manifest(&tcfg));
    if !miss.is_empty() {
        bail!(
            "{} tensors missing from {}, e.g. {:?}",
            miss.len(),
            tae.display(),
            &miss[..miss.len().min(3)]
        );
    }
    let pipe = Pipeline {
        dit: WanDit::load(be, cfg.clone(), &wsrc)?,
        tae: TaeHv::load(be, tcfg, &tsrc)?,
        schedule: DmdSchedule::fastwan_1_3b(),
    };
    be.sync();
    println!("load {:.1}s", t0.elapsed().as_secs_f64());
    let opt = Options {
        geometry: *g,
        seed,
        ffn_chunk: ffn_chunk.unwrap_or(usize::MAX),
        tae_chunk,
        attention,
    };
    let mut last = None;
    for run in 0..runs.max(1) {
        let t0 = std::time::Instant::now();
        let (video, t) = pipe.generate(be, e, &opt)?;
        println!(
            "run {run}: total {:.2}s  prompt {:.2}s  denoise {:.2}s ({:?})  decode {:.2}s",
            t0.elapsed().as_secs_f64(),
            t.prompt_s,
            t.denoise_s(),
            t.steps_s
                .iter()
                .map(|s| format!("{s:.2}"))
                .collect::<Vec<_>>(),
            t.decode_s
        );
        last = Some(video);
    }
    let video = last.expect("at least one run");
    write_y4m(out, &video, 16)?;
    println!(
        "wrote {} ({} frames {}x{})",
        out.display(),
        video.frames,
        video.width,
        video.height
    );
    Ok(())
}

/// YUV4MPEG2, 4:4:4, BT.601 full range: plays in mpv/ffplay, converts with
/// `ffmpeg -i out.y4m out.mp4`, and needs no dependency to write.
fn write_y4m(path: &Path, v: &Video, fps: u32) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        f,
        "YUV4MPEG2 W{} H{} F{fps}:1 Ip A1:1 C444 XCOLORRANGE=FULL",
        v.width, v.height
    )?;
    let px = v.width * v.height;
    let mut planes = vec![0u8; 3 * px];
    for fr in 0..v.frames {
        let rgb = &v.rgb[fr * px * 3..(fr + 1) * px * 3];
        for i in 0..px {
            let (r, g, b) = (
                rgb[3 * i] as f32,
                rgb[3 * i + 1] as f32,
                rgb[3 * i + 2] as f32,
            );
            planes[i] = (0.299 * r + 0.587 * g + 0.114 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
            planes[px + i] = (128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
            planes[2 * px + i] = (128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        f.write_all(b"FRAME\n")?;
        f.write_all(&planes)?;
    }
    Ok(())
}
