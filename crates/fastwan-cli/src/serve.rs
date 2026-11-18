//! `fastwan serve`: a local engine plus the prompt-to-clip demo page.
//!
//! One engine thread owns the GPU backend, the resident DiT and TAEHV, and the
//! UMT5 encoder; it takes jobs one at a time from a queue. The HTTP thread
//! only reads the shared job table, so polling never waits on the GPU.
//!
//! ```text
//! GET  /                         the demo page (web/demo.html)
//! GET  /api/info                 backend, model, limits, ready / error
//! POST /api/generate             {prompt, width, height, frames, seed?, upscale?} → {id}
//! GET  /api/jobs/<id>            status, per-stage timings, frames decoded so far
//! GET  /api/jobs/<id>/frames?from=N
//!                                binary: u32 LE width, height, first, count; then RGB8
//! ```
//!
//! It binds 127.0.0.1 by default and sends no CORS headers, so other web
//! pages cannot queue jobs on your GPU. `--allow-origin <url>` opens the API
//! to exactly one origin (the hosted demo page) and to nothing else.

use anyhow::{Context, Result, anyhow};
use fastwan_core::{
    Backend,
    config::{TaeConfig, WanConfig},
    geometry::Geometry,
    rng::Rng,
    schedule::DmdSchedule,
    weights::{SafetensorsDir, Synthetic, missing, tae_manifest, wan_manifest},
};
use fastwan_model::{
    dit::WanDit,
    pipeline::{Event, Options, Pipeline},
    taehv::TaeHv,
};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

const PAGE: &str = include_str!("../../../web/demo.html");
const KEEP_JOBS: usize = 8;
const MAX_QUEUE: usize = 3;
/// Finished jobs' frames kept in memory at most (an 832×480 clip upscaled ×4
/// is 1.55 GB of RGB): the oldest are dropped first.
const KEEP_BYTES: usize = 4 << 30;

pub enum Weights {
    /// Tiny random model: exercises the engine and the page without a download.
    /// With a gallery, only its prompts are accepted (embeddings stay random):
    /// gallery mode's page without the model.
    Synthetic { gallery: Option<PathBuf> },
    Real {
        transformer: PathBuf,
        tae: PathBuf,
        text: TextSource,
    },
}

pub enum TextSource {
    /// The native UMT5-XXL on the CPU: any prompt.
    Umt5 {
        text_encoder: PathBuf,
        tokenizer: PathBuf,
    },
    /// Precomputed embeddings (`fastwan encode`): only the gallery's prompts,
    /// no text encoder in memory.
    Gallery(PathBuf),
}

pub struct Settings {
    pub addr: String,
    pub weights: Weights,
    pub max_pixels: usize,
    pub max_frames: usize,
    pub tae_chunk: usize,
    /// One web origin (e.g. `https://fastwan.pages.dev`) allowed to call the
    /// API from the browser: CORS plus Chrome's private-network preflight,
    /// for that origin only. `None` = same-origin only (the default).
    pub allow_origin: Option<String>,
    pub attention: Option<fastwan_model::Attention>,
    pub quant: fastwan_core::quant::WeightQuant,
    /// `realesr-animevideov3.safetensors`: offer ×2 / ×4 upscaling.
    pub upscaler: Option<PathBuf>,
}

impl Settings {
    /// Upscale factors a request may ask for (the synthetic model's tiny
    /// upscaler is ×2).
    fn upscales(&self) -> Vec<usize> {
        match (&self.weights, &self.upscaler) {
            (Weights::Synthetic { .. }, _) => vec![2],
            (_, Some(_)) => vec![2, 4],
            _ => vec![],
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Queued,
    Encoding,
    Preparing,
    Denoising,
    Decoding,
    Done,
    Failed,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Queued => "queued",
            Stage::Encoding => "encoding",
            Stage::Preparing => "preparing",
            Stage::Denoising => "denoising",
            Stage::Decoding => "decoding",
            Stage::Done => "done",
            Stage::Failed => "failed",
        }
    }
}

struct Job {
    id: u64,
    prompt: String,
    geometry: Geometry,
    seed: u64,
    stage: Stage,
    tokens: usize,
    encode_s: f64,
    prepare_s: f64,
    steps: Vec<f64>,
    total_steps: usize,
    decode_s: f64,
    started: Option<Instant>,
    total_s: f64,
    width: usize,
    height: usize,
    frames: usize,
    rgb: Vec<u8>,
    error: Option<String>,
    /// 0 = off, else the output scale; frames are stored upscaled.
    upscale: usize,
    upscale_s: f64,
}

impl Job {
    fn status(&self) -> Value {
        json!({
            "id": self.id, "prompt": self.prompt, "stage": self.stage.name(), "seed": self.seed,
            "width": self.geometry.width, "height": self.geometry.height, "total_frames": self.geometry.frames,
            "tokens": self.tokens, "encode_s": self.encode_s, "prepare_s": self.prepare_s,
            "steps": self.steps, "total_steps": self.total_steps, "decode_s": self.decode_s,
            "elapsed_s": self.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0),
            "total_s": self.total_s, "frames": self.frames, "error": self.error,
            "upscale": self.upscale, "upscale_s": self.upscale_s,
        })
    }
}

pub struct Shared {
    jobs: Mutex<Vec<Job>>,
    info: Mutex<Value>,
    next: Mutex<u64>,
}

impl Shared {
    fn with<R>(&self, id: u64, f: impl FnOnce(&mut Job) -> R) -> Option<R> {
        self.jobs
            .lock()
            .unwrap()
            .iter_mut()
            .find(|j| j.id == id)
            .map(f)
    }
}

/// What the engine needs from the prompt side.
enum Text {
    Synthetic {
        len: usize,
        dim: usize,
        only: Option<crate::gallery::Gallery>,
    },
    Umt5(Box<fastwan_text::PromptEncoder>),
    Gallery(crate::gallery::Gallery, WanConfig),
}

impl Text {
    fn encode(&self, prompt: &str) -> Result<(Vec<f32>, usize)> {
        match self {
            Text::Synthetic { len, dim, only } => {
                if let Some(g) = only {
                    if g.find(prompt).is_none() {
                        return Err(anyhow!(
                            "This engine has no text encoder: pick one of its {} gallery prompts.",
                            g.entries.len()
                        ));
                    }
                }
                let seed = prompt.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
                });
                Ok((
                    Rng::new(seed).normal(len * dim),
                    prompt.split_whitespace().count(),
                ))
            }
            Text::Umt5(enc) => {
                let tokens = enc.tokenize(prompt)?.len();
                Ok((enc.encode(prompt)?, tokens))
            }
            Text::Gallery(g, cfg) => {
                let e = g.find(prompt).ok_or_else(|| {
                    anyhow!(
                        "This engine has no text encoder: pick one of its {} gallery prompts.",
                        g.entries.len()
                    )
                })?;
                Ok((g.embeds(e, cfg)?, e.tokens))
            }
        }
    }
}

/// Loads everything, then runs jobs forever. Runs on the engine thread.
pub fn engine<B: Backend>(
    be: B,
    s: &Settings,
    shared: Arc<Shared>,
    rx: mpsc::Receiver<u64>,
) -> Result<()> {
    let t0 = Instant::now();
    let stage = |what: &str| {
        eprintln!("engine: {what} ({:.1}s)", t0.elapsed().as_secs_f64());
        let mut info = shared.info.lock().unwrap();
        info["loading"] = json!(what);
        info["loading_s"] = json!(t0.elapsed().as_secs_f64());
    };
    stage(&format!("starting on {}", be.name()));
    let (pipe, text, model) = match &s.weights {
        Weights::Synthetic { gallery } => {
            // head_dim 32: small, but runs VSA on every backend.
            let (cfg, tcfg) = (WanConfig::tiny_vsa(), TaeConfig::tiny());
            let src = Synthetic { seed: 1 };
            let only = match gallery {
                Some(dir) => {
                    let g = crate::gallery::Gallery::open(dir)?;
                    shared.info.lock().unwrap()["gallery"] =
                        json!(g.entries.iter().map(|e| &e.prompt).collect::<Vec<_>>());
                    Some(g)
                }
                None => None,
            };
            let text = Text::Synthetic {
                len: cfg.text_len,
                dim: cfg.text_dim,
                only,
            };
            let pipe = Pipeline {
                dit: WanDit::load_with(&be, cfg, &src, s.quant)?,
                tae: TaeHv::load(&be, tcfg, &src)?,
                schedule: DmdSchedule::fastwan_1_3b(),
            };
            (
                pipe,
                text,
                "synthetic tiny model (random weights: output is noise)".to_string(),
            )
        }
        Weights::Real {
            transformer,
            tae,
            text,
        } => {
            let cfg = WanConfig::fastwan_1_3b();
            let tcfg = TaeConfig::from_file_name(&tae.to_string_lossy())
                .unwrap_or_else(TaeConfig::taew2_1);
            if tcfg.latent_channels != cfg.out_channels {
                return Err(anyhow!(
                    "{} is not a Wan 2.1 decoder; use taew2_1.safetensors",
                    tae.display()
                ));
            }
            stage("checking the checkpoint");
            let w = SafetensorsDir::open(transformer)?;
            let miss = missing(&w, &wan_manifest(&cfg));
            if !miss.is_empty() {
                return Err(anyhow!(
                    "{} DiT tensors missing, e.g. {:?}",
                    miss.len(),
                    &miss[..miss.len().min(3)]
                ));
            }
            let tw = SafetensorsDir::open(tae)?;
            let miss = missing(&tw, &tae_manifest(&tcfg));
            if !miss.is_empty() {
                return Err(anyhow!(
                    "{} TAEHV tensors missing, e.g. {:?}",
                    miss.len(),
                    &miss[..miss.len().min(3)]
                ));
            }
            let text = match text {
                TextSource::Umt5 {
                    text_encoder,
                    tokenizer,
                } => {
                    stage("opening the text encoder");
                    Text::Umt5(Box::new(fastwan_text::PromptEncoder::open(
                        tokenizer,
                        text_encoder,
                        fastwan_text::Umt5Config::xxl(),
                    )?))
                }
                TextSource::Gallery(dir) => {
                    stage("reading the prompt gallery");
                    let g = crate::gallery::Gallery::open(dir)?;
                    shared.info.lock().unwrap()["gallery"] =
                        json!(g.entries.iter().map(|e| &e.prompt).collect::<Vec<_>>());
                    Text::Gallery(g, cfg.clone())
                }
            };
            let stored = ["ffn.net.0.proj", "attn1.to_q"]
                .iter()
                .find_map(|k| {
                    let k = format!("blocks.0.{k}.weight");
                    ["q4", "q8"]
                        .into_iter()
                        .find(|q| w.contains_raw(&format!("{k}.{q}")))
                })
                .map(|q| if q == "q4" { " · 4-bit" } else { " · int8" })
                .unwrap_or("");
            stage("loading the transformer onto the GPU (1.4B parameters)");
            let dit = WanDit::load_with(&be, cfg, &w, s.quant)?;
            stage("loading the TAEHV decoder");
            let tae = TaeHv::load(&be, tcfg, &tw)?;
            let pipe = Pipeline {
                dit,
                tae,
                schedule: DmdSchedule::fastwan_1_3b(),
            };
            let model = format!(
                "FastWan2.1-T2V-1.3B{stored} · DMD 3 steps · TAEHV{}",
                if matches!(text, Text::Gallery(..)) {
                    " · prompt gallery"
                } else {
                    ""
                }
            );
            (pipe, text, model)
        }
    };
    let mut up = match (&s.weights, &s.upscaler) {
        (Weights::Synthetic { .. }, _) => Some(fastwan_model::upscale::Upscaler::load(
            &be,
            fastwan_model::upscale::UpConfig::tiny(),
            &Synthetic { seed: 2 },
        )?),
        (_, Some(p)) => {
            stage("loading the upscaler");
            Some(crate::upscale::load(&be, p, 4)?)
        }
        _ => None,
    };
    stage("waiting for the GPU");
    be.sync();
    {
        let mut info = shared.info.lock().unwrap();
        info["ready"] = json!(true);
        info["model"] = json!(model);
        info["backend"] = json!(be.name());
        info["upscale"] = json!(s.upscales());
        info["load_s"] = json!(t0.elapsed().as_secs_f64());
    }
    eprintln!(
        "engine ready on {} in {:.1}s",
        be.name(),
        t0.elapsed().as_secs_f64()
    );

    for id in rx {
        let Some((prompt, geometry, seed, upscale)) = shared.with(id, |j| {
            j.started = Some(Instant::now());
            j.stage = Stage::Encoding;
            (j.prompt.clone(), j.geometry, j.seed, j.upscale)
        }) else {
            continue;
        };
        if let (Some(u), true) = (up.as_mut(), upscale > 0) {
            u.set_output_scale(upscale)?;
        }
        let t = Instant::now();
        let run = || -> Result<()> {
            let (embeds, tokens) = text.encode(&prompt)?;
            shared.with(id, |j| {
                j.encode_s = t.elapsed().as_secs_f64();
                j.tokens = tokens;
                j.stage = Stage::Preparing;
            });
            let opt = Options {
                geometry,
                seed,
                ffn_chunk: usize::MAX,
                tae_chunk: s.tae_chunk,
                attention: s.attention,
            };
            pipe.generate_with(&be, &embeds, &opt, &mut |ev| match ev {
                Event::Prepared { secs } => {
                    shared.with(id, |j| {
                        j.prepare_s = secs;
                        j.total_steps = pipe.schedule.len();
                        j.stage = Stage::Denoising;
                    });
                }
                Event::Step { index, total, secs } => {
                    shared.with(id, |j| {
                        j.steps.push(secs);
                        if index + 1 == total {
                            j.stage = Stage::Decoding;
                        }
                    });
                }
                Event::Frames {
                    count,
                    width,
                    height,
                    rgb,
                    ..
                } => {
                    // Upscaled per decoded chunk, on the device, as it arrives.
                    let tu = Instant::now();
                    let upscaled = match (&up, upscale > 0) {
                        (Some(u), true) => {
                            let words: Vec<u32> = rgb
                                .chunks_exact(3)
                                .map(|p| p[0] as u32 | (p[1] as u32) << 8 | (p[2] as u32) << 16)
                                .collect();
                            let out = crate::upscale::frames(
                                &be,
                                u,
                                &be.upload_words(&words),
                                count,
                                height,
                                width,
                            );
                            Some((out, width * upscale, height * upscale))
                        }
                        _ => None,
                    };
                    let up_s = tu.elapsed().as_secs_f64();
                    shared.with(id, |j| {
                        match &upscaled {
                            Some((out, w, h)) => {
                                j.width = *w;
                                j.height = *h;
                                j.rgb.extend_from_slice(out);
                                j.upscale_s += up_s;
                            }
                            None => {
                                j.width = width;
                                j.height = height;
                                j.rgb.extend_from_slice(rgb);
                            }
                        }
                        j.frames += count;
                        j.decode_s = j.started.unwrap().elapsed().as_secs_f64()
                            - j.encode_s
                            - j.prepare_s
                            - j.steps.iter().sum::<f64>()
                            - j.upscale_s;
                    });
                }
            })?;
            Ok(())
        };
        let res = run();
        shared.with(id, |j| {
            j.total_s = t.elapsed().as_secs_f64();
            match res {
                Ok(()) => j.stage = Stage::Done,
                Err(e) => {
                    j.stage = Stage::Failed;
                    j.error = Some(format!("{e:#}"));
                }
            }
        });
    }
    Ok(())
}

fn hdr(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k, v).expect("valid header")
}

/// Send with the CORS headers for an allowed origin, if this request has one.
fn send<R: std::io::Read>(
    req: tiny_http::Request,
    resp: tiny_http::Response<R>,
    cors: &[tiny_http::Header],
) {
    let mut resp = resp;
    for h in cors {
        resp = resp.with_header(h.clone());
    }
    let _ = req.respond(resp);
}

fn respond_json(req: tiny_http::Request, code: u16, v: &Value, cors: &[tiny_http::Header]) {
    let r = tiny_http::Response::from_string(v.to_string())
        .with_status_code(code)
        .with_header(hdr("Content-Type", "application/json"));
    send(req, r, cors);
}

/// CORS headers when the request's `Origin` is the one allowed origin.
fn cors_for(req: &tiny_http::Request, allow: Option<&str>) -> Vec<tiny_http::Header> {
    let origin = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("Origin"))
        .map(|h| h.value.as_str().to_string());
    match (origin, allow) {
        (Some(o), Some(a)) if o.trim_end_matches('/') == a.trim_end_matches('/') => vec![
            hdr("Access-Control-Allow-Origin", &o),
            hdr("Vary", "Origin"),
            hdr("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            hdr("Access-Control-Allow-Headers", "Content-Type"),
            // Chrome's Private Network Access: a public https page calling 127.0.0.1.
            hdr("Access-Control-Allow-Private-Network", "true"),
            hdr("Access-Control-Max-Age", "600"),
        ],
        _ => Vec::new(),
    }
}

fn parse_request(body: &str, s: &Settings) -> Result<(String, Geometry, u64, usize), String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("body is not JSON: {e}"))?;
    let prompt = v["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return Err("Write a prompt first.".into());
    }
    if prompt.chars().count() > 2000 {
        return Err("Prompts are limited to 2000 characters.".into());
    }
    let num = |k: &str, d: u64| v[k].as_u64().unwrap_or(d) as usize;
    let g = Geometry {
        width: num("width", 832),
        height: num("height", 480),
        frames: num("frames", 81),
    };
    g.validate(&WanConfig::fastwan_1_3b())
        .map_err(|e| e.to_string())?;
    if g.width * g.height > s.max_pixels {
        return Err(format!(
            "{}×{} is above this engine's limit of {} pixels per frame.",
            g.width, g.height, s.max_pixels
        ));
    }
    if g.frames > s.max_frames {
        return Err(format!(
            "{} frames is above this engine's limit of {}.",
            g.frames, s.max_frames
        ));
    }
    let upscale = v["upscale"].as_u64().unwrap_or(0) as usize;
    if upscale != 0 && !s.upscales().contains(&upscale) {
        return Err(match s.upscales().as_slice() {
            [] => "This engine has no upscaler (start it with --upscaler).".to_string(),
            u => format!(
                "Upscale ×{upscale} is not offered here (only {}).",
                u.iter()
                    .map(|k| format!("×{k}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }
    let seed = v["seed"].as_u64().unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64 % 1_000_000_000)
            .unwrap_or(7)
    });
    Ok((prompt, g, seed, upscale))
}

/// Serve until killed. `start_engine` builds the backend on the engine thread.
pub fn serve(
    s: Settings,
    start_engine: impl FnOnce(&Settings, Arc<Shared>, mpsc::Receiver<u64>) -> Result<()>
    + Send
    + 'static,
) -> Result<()> {
    let shared = Arc::new(Shared {
        jobs: Mutex::new(Vec::new()),
        info: Mutex::new(json!({
            "ready": false, "error": null, "backend": null, "model": null,
            "synthetic": matches!(s.weights, Weights::Synthetic { .. }), "gallery": null,
            "max_pixels": s.max_pixels, "max_frames": s.max_frames, "fps": 16,
        })),
        next: Mutex::new(1),
    });
    let (tx, rx) = mpsc::channel::<u64>();
    let server = tiny_http::Server::http(&s.addr).map_err(|e| anyhow!("bind {}: {e}", s.addr))?;
    eprintln!("fastwan demo: http://{}/", s.addr);
    let s = Arc::new(s);
    {
        let (shared, s) = (shared.clone(), s.clone());
        std::thread::Builder::new()
            .name("engine".into())
            .spawn(move || {
                // A panic (GPU out of memory, a driver validation error, a
                // failed assertion) must reach the page: without this the
                // thread died silently and the page waited for ever.
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    start_engine(&s, shared.clone(), rx)
                }));
                let msg = match run {
                    Ok(Ok(())) => return,
                    Ok(Err(e)) => format!("{e:#}"),
                    Err(p) => format!(
                        "engine crashed: {}",
                        p.downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| p.downcast_ref::<&str>().copied())
                            .unwrap_or("panic")
                    ),
                };
                eprintln!("engine failed: {msg}");
                {
                    let mut info = shared.info.lock().unwrap();
                    info["ready"] = json!(false);
                    info["error"] = json!(msg.clone());
                }
                for j in shared.jobs.lock().unwrap().iter_mut() {
                    if !matches!(j.stage, Stage::Done | Stage::Failed) {
                        j.stage = Stage::Failed;
                        j.error = Some(msg.clone());
                    }
                }
            })
            .context("spawn engine")?;
    }

    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let method = req.method().clone();
        let cors = cors_for(&req, s.allow_origin.as_deref());
        match (method, path) {
            (tiny_http::Method::Options, _) => {
                let code = if cors.is_empty() { 403 } else { 204 };
                send(req, tiny_http::Response::empty(code), &cors);
            }
            (tiny_http::Method::Get, "/") | (tiny_http::Method::Get, "/index.html") => {
                let h = tiny_http::Header::from_bytes("Content-Type", "text/html; charset=utf-8")
                    .unwrap();
                send(
                    req,
                    tiny_http::Response::from_string(PAGE.replacen(
                        r#"<meta name="fastwan-served" content="">"#,
                        r#"<meta name="fastwan-served" content="engine">"#,
                        1,
                    ))
                    .with_header(h),
                    &cors,
                );
            }
            (tiny_http::Method::Get, "/api/info") => {
                let mut info = shared.info.lock().unwrap().clone();
                let jobs = shared.jobs.lock().unwrap();
                info["queued"] = json!(jobs.iter().filter(|j| j.stage == Stage::Queued).count());
                drop(jobs);
                respond_json(req, 200, &info, &cors);
            }
            (tiny_http::Method::Post, "/api/generate") => {
                // CSRF guard: a cross-site form or `text/plain` fetch skips the
                // CORS preflight, so check the origin and content type here.
                let header = |k: &str| {
                    req.headers()
                        .iter()
                        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(k))
                        .map(|h| h.value.as_str().to_string())
                };
                let json_body =
                    header("Content-Type").is_some_and(|c| c.starts_with("application/json"));
                let origin_ok = match header("Origin") {
                    None => true,
                    Some(o) => {
                        !cors.is_empty()
                            // Same origin, direct (http) or behind a TLS proxy / tunnel.
                            || header("Host").is_some_and(|h| {
                                o == format!("http://{h}") || o == format!("https://{h}")
                            })
                    }
                };
                if !json_body || !origin_ok {
                    respond_json(
                        req,
                        403,
                        &json!({"error": "Requests must be JSON and come from this engine's page or its --allow-origin site."}),
                        &cors,
                    );
                    continue;
                }
                let mut body = String::new();
                if req
                    .as_reader()
                    .take(64 * 1024)
                    .read_to_string(&mut body)
                    .is_err()
                {
                    respond_json(req, 400, &json!({"error": "unreadable body"}), &cors);
                    continue;
                }
                let info = shared.info.lock().unwrap().clone();
                if !info["ready"].as_bool().unwrap_or(false) {
                    let msg = info["error"]
                        .as_str()
                        .map(|e| format!("The engine failed to start: {e}"))
                        .unwrap_or_else(|| {
                            "The engine is still loading the model. Try again in a moment.".into()
                        });
                    respond_json(req, 503, &json!({"error": msg}), &cors);
                    continue;
                }
                // Gallery mode: refuse a prompt it cannot encode now, not
                // after it has waited in the queue.
                let parsed = parse_request(&body, &s).and_then(|r| {
                    let info = shared.info.lock().unwrap();
                    match info["gallery"].as_array() {
                        Some(g) if !g.iter().any(|p| {
                            p.as_str().is_some_and(|p| {
                                p.split_whitespace().eq(r.0.split_whitespace())
                            })
                        }) =>
                        {
                            Err(format!(
                                "This engine has no text encoder: pick one of its {} gallery prompts.",
                                g.len()
                            ))
                        }
                        _ => Ok(r),
                    }
                });
                match parsed {
                    Err(e) => respond_json(req, 400, &json!({"error": e}), &cors),
                    Ok((prompt, geometry, seed, upscale)) => {
                        let mut jobs = shared.jobs.lock().unwrap();
                        if jobs
                            .iter()
                            .filter(|j| matches!(j.stage, Stage::Queued))
                            .count()
                            >= MAX_QUEUE
                        {
                            drop(jobs);
                            respond_json(
                                req,
                                429,
                                &json!({"error": "Three clips are already waiting. Try again when one finishes."}),
                                &cors,
                            );
                            continue;
                        }
                        let id = {
                            let mut n = shared.next.lock().unwrap();
                            *n += 1;
                            *n - 1
                        };
                        jobs.push(Job {
                            id,
                            prompt,
                            geometry,
                            seed,
                            stage: Stage::Queued,
                            tokens: 0,
                            encode_s: 0.0,
                            prepare_s: 0.0,
                            steps: Vec::new(),
                            total_steps: 3,
                            decode_s: 0.0,
                            started: None,
                            total_s: 0.0,
                            width: 0,
                            height: 0,
                            frames: 0,
                            rgb: Vec::new(),
                            error: None,
                            upscale,
                            upscale_s: 0.0,
                        });
                        // Drop the oldest finished jobs (their frames are the memory).
                        while jobs.len() > KEEP_JOBS
                            || jobs.iter().map(|j| j.rgb.len()).sum::<usize>() > KEEP_BYTES
                        {
                            match jobs
                                .iter()
                                .position(|j| matches!(j.stage, Stage::Done | Stage::Failed))
                            {
                                Some(i) => {
                                    jobs.remove(i);
                                }
                                None => break,
                            }
                        }
                        drop(jobs);
                        let _ = tx.send(id);
                        respond_json(req, 200, &json!({"id": id, "seed": seed}), &cors);
                    }
                }
            }
            (tiny_http::Method::Get, p) if p.starts_with("/api/jobs/") => {
                let rest = &p["/api/jobs/".len()..];
                let (id_s, frames) = match rest.strip_suffix("/frames") {
                    Some(i) => (i, true),
                    None => (rest, false),
                };
                let Ok(id) = id_s.parse::<u64>() else {
                    respond_json(req, 404, &json!({"error": "no such job"}), &cors);
                    continue;
                };
                if frames {
                    let from = query
                        .split('&')
                        .find_map(|kv| kv.strip_prefix("from="))
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0);
                    let body = shared.with(id, |j| {
                        let px = j.width * j.height * 3;
                        let first = from.min(j.frames);
                        let count = j.frames - first;
                        let mut out = Vec::with_capacity(16 + count * px);
                        for v in [j.width, j.height, first, count] {
                            out.extend_from_slice(&(v as u32).to_le_bytes());
                        }
                        out.extend_from_slice(&j.rgb[first * px..(first + count) * px]);
                        out
                    });
                    match body {
                        Some(b) => {
                            let h = tiny_http::Header::from_bytes(
                                "Content-Type",
                                "application/octet-stream",
                            )
                            .unwrap();
                            send(req, tiny_http::Response::from_data(b).with_header(h), &cors);
                        }
                        None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                    }
                } else {
                    match shared.with(id, |j| j.status()) {
                        Some(v) => respond_json(req, 200, &v, &cors),
                        None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                    }
                }
            }
            _ => respond_json(req, 404, &json!({"error": "not found"}), &cors),
        }
    }
    Ok(())
}

use std::io::Read as _;

#[cfg(test)]
mod tests {
    /// The page decides "served by the engine" from this marker, which the
    /// GET / handler fills in; if the placeholder drifted, the engine's own
    /// page would offer browser mode and lose its engine.
    #[test]
    fn page_has_the_served_marker() {
        assert_eq!(
            super::PAGE
                .matches(r#"<meta name="fastwan-served" content="">"#)
                .count(),
            1
        );
    }
}
