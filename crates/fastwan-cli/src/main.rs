//! `fastwan`: plan, check, bench and generate FastWan on CPU, WebGPU or CUDA.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use fastwan_core::{
    Backend,
    config::{TaeConfig, WanConfig},
    geometry::Geometry,
    plan::{Device, plan},
    schedule::DmdSchedule,
    weights::{SafetensorsDir, Synthetic, missing, tae_manifest, wan_manifest},
};
use fastwan_model::{
    Video,
    dit::WanDit,
    pipeline::{Options, Pipeline},
    taehv::TaeHv,
};
use std::path::{Path, PathBuf};

mod gallery;
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
    /// VSA on a checkpoint that carries to_gate_compress (FastWan) at 81+ frames
    /// (shorter clips decay at the end under VSA), dense otherwise.
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
    /// DiT block weights on the device: `native` (bf16 / f16) or `int8`
    /// (weight-only, per-128-column scales: half the memory).
    #[arg(long, default_value = "native")]
    weights: fastwan_core::quant::WeightQuant,
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
        /// The DiT checkpoint to load instead of <model-dir>/transformer, e.g. a
        /// `fastwan quantize` file.
        #[arg(long)]
        transformer: Option<PathBuf>,
        /// Run without the text encoder: only these precomputed prompts
        /// (`fastwan encode`); <model-dir> then needs no text_encoder/.
        #[arg(long)]
        gallery: Option<PathBuf>,
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
    /// Write a quantized copy of the DiT: block linears as int8 / 4-bit codes,
    /// everything else f16 (the browser download format; see
    /// `fastwan_core::export`). Load it with --transformer like the original.
    Quantize {
        /// The checkpoint's `transformer/` directory.
        #[arg(long)]
        transformer: PathBuf,
        /// native (all f16) | int8 | int4[-hqq][-gN][-ffn], e.g. int4-hqq-g64.
        #[arg(long)]
        weights: fastwan_core::quant::WeightQuant,
        #[arg(long)]
        out: PathBuf,
    },
    /// Encode prompts once with the native UMT5 and store their embeddings as
    /// a gallery, so `serve --gallery` / `generate --embeds` run without the
    /// 11 GB text encoder.
    Encode {
        /// A FastWan download with text_encoder/ and tokenizer/.
        #[arg(long)]
        model_dir: PathBuf,
        /// A prompt (repeat for several).
        #[arg(long = "prompt")]
        prompts: Vec<String>,
        /// A text file with one prompt per line (blank lines and `#` lines skipped).
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// The gallery directory (created; existing prompts are kept).
        #[arg(long)]
        out_dir: PathBuf,
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
        /// Upscale ×4 after decoding with Real-ESRGAN's compact video model
        /// (`realesr-animevideov3.safetensors`; scripts/upscale/convert_esrgan.py).
        #[arg(long)]
        upscale: Option<PathBuf>,
        /// Causal rollout (a Self-Forcing checkpoint: scripts/convert_self_forcing.py):
        /// 3-latent-frame blocks over a rolling K/V cache, any --frames, streamed
        /// to --out as each block decodes.
        #[arg(long, default_value_t = false)]
        causal: bool,
        /// Causal: latent frames the K/V cache holds (sink and block included).
        #[arg(long, default_value_t = 12)]
        cache_frames: usize,
        /// Causal: first latent frames kept in the cache for the whole video.
        #[arg(long, default_value_t = 3)]
        sink_frames: usize,
        /// Causal: what the full cache keeps. `memrope` is History Forcing's
        /// memory (its checkpoint: scripts/convert_self_forcing.py on
        /// dmd/model.pt): 2 EMA frames after the sink, unrotated keys
        /// rotated at their slot (block-relative RoPE).
        #[arg(long, value_enum, default_value_t = CausalMemory::Window)]
        memory: CausalMemory,
    },
}

/// `generate --causal --memory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum CausalMemory {
    /// Sink + rolling window (Self-Forcing, LongLive).
    Window,
    /// MemRoPE: sink + long / short EMA memory frames + recent window.
    Memrope,
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
                attn.get(),
                attn.weights
            )?)
        }
        Cmd::Serve {
            attn,
            addr,
            backend,
            model_dir,
            tae,
            transformer,
            gallery,
            synthetic,
            max_pixels,
            max_frames,
            tae_chunk,
            allow_origin,
        } => {
            let weights = if synthetic {
                serve::Weights::Synthetic { gallery }
            } else {
                let d = model_dir.expect("clap requires --model-dir");
                serve::Weights::Real {
                    transformer: transformer.unwrap_or_else(|| d.join("transformer")),
                    text: match gallery {
                        Some(g) => serve::TextSource::Gallery(g),
                        None => serve::TextSource::Umt5 {
                            text_encoder: d.join("text_encoder"),
                            tokenizer: d.join("tokenizer").join("tokenizer.json"),
                        },
                    },
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
                quant: attn.weights,
            };
            serve::serve(settings, move |s, shared, rx| {
                with_backend!(backend, |be| serve::engine(be, s, shared, rx))
            })?;
        }
        Cmd::Encode {
            model_dir,
            mut prompts,
            prompt_file,
            out_dir,
        } => {
            if let Some(f) = prompt_file {
                let text =
                    std::fs::read_to_string(&f).with_context(|| format!("read {}", f.display()))?;
                prompts.extend(
                    text.lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty() && !l.starts_with('#'))
                        .map(str::to_string),
                );
            }
            if prompts.is_empty() {
                bail!("give --prompt (repeatable) or --prompt-file");
            }
            let t0 = std::time::Instant::now();
            let enc = fastwan_text::PromptEncoder::open(
                &model_dir.join("tokenizer").join("tokenizer.json"),
                &model_dir.join("text_encoder"),
                fastwan_text::Umt5Config::xxl(),
            )?;
            println!("umt5 opened in {:.1}s", t0.elapsed().as_secs_f64());
            let dim = WanConfig::fastwan_1_3b().text_dim;
            for p in &prompts {
                let t = std::time::Instant::now();
                let tokens = enc.tokenize(p)?.len();
                let e = enc.encode(p)?;
                let entry = gallery::Gallery::add(&out_dir, p, tokens, &e, dim)?;
                println!(
                    "{} ({tokens} tokens, {:.1}s): {p}",
                    entry.file,
                    t.elapsed().as_secs_f64()
                );
            }
            println!("gallery: {}", out_dir.join("gallery.json").display());
        }
        Cmd::Quantize {
            transformer,
            weights,
            out,
        } => {
            // `native` writes every tensor as f16 (the browser's f16 option).
            let cfg = WanConfig::fastwan_1_3b();
            let t0 = std::time::Instant::now();
            let src = SafetensorsDir::open(&transformer)?;
            let miss = missing(&src, &wan_manifest(&cfg));
            if !miss.is_empty() {
                bail!(
                    "{} tensors missing, e.g. {:?}",
                    miss.len(),
                    &miss[..miss.len().min(3)]
                );
            }
            let tensors = fastwan_core::export::quantize_dit(&src, &cfg, weights)?;
            let bytes: usize = tensors.iter().map(|t| t.3.len()).sum();
            fastwan_core::export::write(&out, &tensors, fastwan_core::export::metadata(weights))?;
            let src_bytes: u64 = if transformer.is_file() {
                std::fs::metadata(&transformer)?.len()
            } else {
                std::fs::read_dir(&transformer)?
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            };
            println!(
                "{weights}: {} tensors, {:.3} GB (source {:.3} GB) -> {} in {:.1}s",
                tensors.len(),
                bytes as f64 / 1e9,
                src_bytes as f64 / 1e9,
                out.display(),
                t0.elapsed().as_secs_f64()
            );
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
            upscale,
            causal,
            cache_frames,
            sink_frames,
            memory,
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
                (None, Some(path)) => gallery::load_embeds(&path, &cfg)?,
                (None, None) => bail!("give --prompt or --embeds"),
            };
            if causal {
                if upscale.is_some() || runs > 1 {
                    bail!("--causal does not take --upscale or --runs yet");
                }
                let mut ccfg = match memory {
                    CausalMemory::Window => fastwan_model::causal::CausalConfig::self_forcing(),
                    CausalMemory::Memrope => fastwan_model::causal::CausalConfig::history_forcing(),
                };
                ccfg.cache_frames(cache_frames, sink_frames);
                return with_backend!(backend, |be| generate_causal(
                    &be,
                    &g,
                    &transformer,
                    &tae,
                    &e,
                    seed,
                    &out,
                    ffn_chunk,
                    ccfg,
                    attn.weights,
                ));
            }
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
                attn.get(),
                attn.weights,
                upscale.as_deref()
            )?)
        }
    }
    Ok(())
}

fn check<B: Backend>(be: &B, kind: BackendKind) -> Result<()> {
    println!("{}", be.name());
    // bf16 activations on CUDA: ~3 significant digits per op.
    // f16 math on WebGPU: ~3 significant digits in the 16-term partial sums.
    let wgpu_f16 = kind == BackendKind::Wgpu && be.name().ends_with("+f16");
    let tol = if kind == BackendKind::Cuda {
        3e-2
    } else if wgpu_f16 {
        1e-2
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
    quant: fastwan_core::quant::WeightQuant,
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
        dit: WanDit::load_with(be, cfg.clone(), &src, quant)?,
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
    be.profile_reset();
    let latent = pipe.denoise(be, &embeds, &opt, &mut t)?;
    if let Some(r) = be.profile_report() {
        print!("denoise {r}");
    }
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
        be.profile_reset();
        let v = pipe.decode(be, &latent, g, opt.tae_chunk, &mut t);
        if let Some(r) = be.profile_report() {
            print!("decode {r}");
        }
        println!(
            "decode (TAEHV) {:.3}s -> {} frames {}x{}",
            t.decode_s, v.frames, v.width, v.height
        );
    }
    Ok(())
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
    quant: fastwan_core::quant::WeightQuant,
    upscale: Option<&Path>,
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
        dit: WanDit::load_with(be, cfg.clone(), &wsrc, quant)?,
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
    be.profile_reset();
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
    if let Some(r) = be.profile_report() {
        print!("all runs: {r}");
    }
    let mut video = last.expect("at least one run");
    if let Some(path) = upscale {
        use fastwan_model::upscale::{UpConfig, Upscaler, up_manifest};
        let ucfg = UpConfig::anime_video_v3();
        let usrc = SafetensorsDir::open(path)?;
        let miss = missing(&usrc, &up_manifest(&ucfg));
        if !miss.is_empty() {
            bail!(
                "{} is not realesr-animevideov3 (missing {:?})",
                path.display(),
                &miss[..miss.len().min(3)]
            );
        }
        let up = Upscaler::load(be, ucfg, &usrc)?;
        let (h, w, s) = (video.height, video.width, ucfg.scale);
        // Batches of a few frames: activations are 64 channels at full size.
        let batch = up.batch(h, w, 1 << 30).min(8);
        be.profile_reset();
        let t0 = std::time::Instant::now();
        let mut rgb = Vec::with_capacity(video.rgb.len() * s * s);
        // Pipelined like the decode: batch i+1 is queued before batch i is
        // read back and converted, so host work overlaps the GPU.
        let starts: Vec<usize> = (0..video.frames).step_by(batch).collect();
        let launch = |f0: usize| {
            let n = batch.min(video.frames - f0);
            up.launch(
                be,
                &video.rgb[f0 * h * w * 3..(f0 + n) * h * w * 3],
                n,
                h,
                w,
            )
        };
        let mut cur = launch(starts[0]);
        for i in 0..starts.len() {
            let next = starts.get(i + 1).map(|&f0| launch(f0));
            let rgba = be.download_rgba8(&cur.out);
            rgb.extend(fastwan_core::backend::rgba_to_rgb(&rgba));
            if let Some(n) = next {
                cur = n;
            }
        }
        println!(
            "upscale x{s} {:.2}s ({}x{} -> {}x{})",
            t0.elapsed().as_secs_f64(),
            w,
            h,
            w * s,
            h * s
        );
        if let Some(r) = be.profile_report() {
            print!("upscale: {r}");
        }
        video = Video {
            frames: video.frames,
            width: w * s,
            height: h * s,
            rgb,
        };
    }
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

/// Causal generation: blocks are denoised one after another and each block's
/// latent frames are decoded by the streaming TAEHV as soon as it is done, so
/// frames reach --out while later blocks are still being generated.
#[allow(clippy::too_many_arguments)]
fn generate_causal<B: Backend>(
    be: &B,
    g: &Geometry,
    transformer: &Path,
    tae: &Path,
    e: &[f32],
    seed: u64,
    out: &Path,
    ffn_chunk: Option<usize>,
    ccfg: fastwan_model::causal::CausalConfig,
    quant: fastwan_core::quant::WeightQuant,
) -> Result<()> {
    use fastwan_model::causal::{CausalRollout, frame_major};
    let cfg = WanConfig::fastwan_1_3b();
    let tcfg = TaeConfig::from_file_name(&tae.to_string_lossy()).unwrap_or_else(TaeConfig::taew2_1);
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
    let dit = WanDit::load_with(be, cfg.clone(), &wsrc, quant)?;
    let taehv = TaeHv::load(be, tcfg, &tsrc)?;
    be.sync();
    println!("load {:.1}s", t0.elapsed().as_secs_f64());
    let (lt, lh, lw) = g.latent();
    let bf = ccfg.block_frames;
    let blocks = lt.div_ceil(bf);
    println!(
        "causal: {blocks} blocks of {bf} latent frames, cache {} (sink {}, {:?}), timesteps {:?}",
        ccfg.cap_frames, ccfg.sink_frames, ccfg.memory, ccfg.timesteps
    );
    let t0 = std::time::Instant::now();
    let prompt = dit.prepare_prompt(be, e);
    let mut roll = CausalRollout::new(
        be,
        &dit,
        ccfg,
        (lh, lw),
        ffn_chunk.unwrap_or(usize::MAX),
        seed,
    )?;
    let mut st = taehv.state();
    let mut y4m: Option<Y4m> = None;
    let c = cfg.out_channels;
    let mut frames = 0;
    let mut first_frame = None;
    for b in 0..blocks {
        let tb = std::time::Instant::now();
        let latent = be.download(roll.next_block(be, &prompt)?);
        let den = tb.elapsed().as_secs_f64();
        let z = frame_major(&latent, c, bf, lh * lw);
        let chunk = taehv.launch_chunk(be, &mut st, &z, bf, lh, lw);
        let packed = chunk.pack(be);
        let raw = be.download_rgba8(&packed);
        let (n, h, w) = chunk.dims();
        let rgb = fastwan_core::backend::rgba_to_rgb(&raw[..n * h * w * 4]);
        let y = match y4m.as_mut() {
            Some(y) => y,
            None => y4m.insert(Y4m::create(out, w, h, 16)?),
        };
        y.frames(&rgb)?;
        frames += n;
        first_frame.get_or_insert(t0.elapsed().as_secs_f64());
        println!(
            "block {b}: denoise {den:.2}s  decode {:.2}s  {n} frames ({frames} total, {:.1} fps so far)",
            tb.elapsed().as_secs_f64() - den,
            frames as f64 / t0.elapsed().as_secs_f64()
        );
    }
    let total = t0.elapsed().as_secs_f64();
    println!(
        "causal: {frames} frames in {total:.2}s ({:.1} fps, first frames at {:.2}s)",
        frames as f64 / total,
        first_frame.unwrap_or(0.0)
    );
    println!("wrote {}", out.display());
    Ok(())
}

/// A YUV4MPEG2 file written frame by frame (see [`write_y4m`]).
struct Y4m {
    f: std::io::BufWriter<std::fs::File>,
    px: usize,
}

impl Y4m {
    fn create(path: &Path, w: usize, h: usize, fps: u32) -> Result<Self> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(
            f,
            "YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C444 XCOLORRANGE=FULL"
        )?;
        Ok(Self { f, px: w * h })
    }

    /// Append frames given as packed RGB.
    fn frames(&mut self, rgb: &[u8]) -> Result<()> {
        use std::io::Write;
        let px = self.px;
        let mut planes = vec![0u8; 3 * px];
        for fr in rgb.chunks_exact(3 * px) {
            for i in 0..px {
                let (r, g, b) = (fr[3 * i] as f32, fr[3 * i + 1] as f32, fr[3 * i + 2] as f32);
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
            self.f.write_all(b"FRAME\n")?;
            self.f.write_all(&planes)?;
        }
        self.f.flush()?;
        Ok(())
    }
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
