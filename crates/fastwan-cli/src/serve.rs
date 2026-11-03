//! `fastwan serve`: a local engine plus the prompt-to-clip demo page.
//!
//! One engine thread owns the GPU backend, the resident DiT and TAEHV, and the
//! UMT5 encoder; it takes jobs one at a time from a queue. The HTTP thread
//! only reads the shared job table, so polling never waits on the GPU.
//!
//! ```text
//! GET  /                         the demo page (web/demo.html)
//! GET  /api/info                 backend, model, limits, ready / error
//! POST /api/generate             {prompt, width, height, frames, seed?} → {id}
//! GET  /api/jobs/<id>            status, per-stage timings, frames decoded so far
//! GET  /api/jobs/<id>/frames?from=N
//!                                binary: u32 LE width, height, first, count; then RGB8
//! ```
//!
//! It binds 127.0.0.1 by default and sends no CORS headers, so other web
//! pages cannot queue jobs on your GPU.

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

pub enum Weights {
    /// Tiny random model: exercises the engine and the page without a download.
    Synthetic,
    Real {
        transformer: PathBuf,
        tae: PathBuf,
        text_encoder: PathBuf,
        tokenizer: PathBuf,
    },
}

pub struct Settings {
    pub addr: String,
    pub weights: Weights,
    pub max_pixels: usize,
    pub max_frames: usize,
    pub tae_chunk: usize,
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
    Synthetic { len: usize, dim: usize },
    Umt5(Box<fastwan_text::PromptEncoder>),
}

impl Text {
    fn encode(&self, prompt: &str) -> Result<(Vec<f32>, usize)> {
        match self {
            Text::Synthetic { len, dim } => {
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
    let (pipe, text, model) = match &s.weights {
        Weights::Synthetic => {
            let (cfg, tcfg) = (WanConfig::tiny(), TaeConfig::tiny());
            let src = Synthetic { seed: 1 };
            let text = Text::Synthetic {
                len: cfg.text_len,
                dim: cfg.text_dim,
            };
            let pipe = Pipeline {
                dit: WanDit::load(&be, cfg, &src)?,
                tae: TaeHv::load(&be, tcfg, &src)?,
                schedule: DmdSchedule::fastwan_1_3b(),
            };
            (
                pipe,
                text,
                "synthetic tiny model (random weights: output is noise)",
            )
        }
        Weights::Real {
            transformer,
            tae,
            text_encoder,
            tokenizer,
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
            let enc = fastwan_text::PromptEncoder::open(
                tokenizer,
                text_encoder,
                fastwan_text::Umt5Config::xxl(),
            )?;
            let pipe = Pipeline {
                dit: WanDit::load(&be, cfg, &w)?,
                tae: TaeHv::load(&be, tcfg, &tw)?,
                schedule: DmdSchedule::fastwan_1_3b(),
            };
            (
                pipe,
                Text::Umt5(Box::new(enc)),
                "FastWan2.1-T2V-1.3B · DMD 3 steps · TAEHV",
            )
        }
    };
    be.sync();
    {
        let mut info = shared.info.lock().unwrap();
        info["ready"] = json!(true);
        info["model"] = json!(model);
        info["backend"] = json!(be.name());
        info["load_s"] = json!(t0.elapsed().as_secs_f64());
    }
    eprintln!(
        "engine ready on {} in {:.1}s",
        be.name(),
        t0.elapsed().as_secs_f64()
    );

    for id in rx {
        let Some((prompt, geometry, seed)) = shared.with(id, |j| {
            j.started = Some(Instant::now());
            j.stage = Stage::Encoding;
            (j.prompt.clone(), j.geometry, j.seed)
        }) else {
            continue;
        };
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
                    shared.with(id, |j| {
                        j.width = width;
                        j.height = height;
                        j.rgb.extend_from_slice(rgb);
                        j.frames += count;
                        j.decode_s = j.started.unwrap().elapsed().as_secs_f64()
                            - j.encode_s
                            - j.prepare_s
                            - j.steps.iter().sum::<f64>();
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

fn respond_json(req: tiny_http::Request, code: u16, v: &Value) {
    let h = tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap();
    let _ = req.respond(
        tiny_http::Response::from_string(v.to_string())
            .with_status_code(code)
            .with_header(h),
    );
}

fn parse_request(body: &str, s: &Settings) -> Result<(String, Geometry, u64), String> {
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
    let seed = v["seed"].as_u64().unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64 % 1_000_000_000)
            .unwrap_or(7)
    });
    Ok((prompt, g, seed))
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
            "synthetic": matches!(s.weights, Weights::Synthetic),
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
                if let Err(e) = start_engine(&s, shared.clone(), rx) {
                    eprintln!("engine failed: {e:#}");
                    shared.info.lock().unwrap()["error"] = json!(format!("{e:#}"));
                }
            })
            .context("spawn engine")?;
    }

    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let method = req.method().clone();
        match (method, path) {
            (tiny_http::Method::Get, "/") | (tiny_http::Method::Get, "/index.html") => {
                let h = tiny_http::Header::from_bytes("Content-Type", "text/html; charset=utf-8")
                    .unwrap();
                let _ = req.respond(tiny_http::Response::from_string(PAGE).with_header(h));
            }
            (tiny_http::Method::Get, "/api/info") => {
                let mut info = shared.info.lock().unwrap().clone();
                let jobs = shared.jobs.lock().unwrap();
                info["queued"] = json!(jobs.iter().filter(|j| j.stage == Stage::Queued).count());
                drop(jobs);
                respond_json(req, 200, &info);
            }
            (tiny_http::Method::Post, "/api/generate") => {
                let mut body = String::new();
                if req
                    .as_reader()
                    .take(64 * 1024)
                    .read_to_string(&mut body)
                    .is_err()
                {
                    respond_json(req, 400, &json!({"error": "unreadable body"}));
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
                    respond_json(req, 503, &json!({"error": msg}));
                    continue;
                }
                match parse_request(&body, &s) {
                    Err(e) => respond_json(req, 400, &json!({"error": e})),
                    Ok((prompt, geometry, seed)) => {
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
                        });
                        // Drop the oldest finished jobs (their frames are the memory).
                        while jobs.len() > KEEP_JOBS {
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
                        respond_json(req, 200, &json!({"id": id, "seed": seed}));
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
                    respond_json(req, 404, &json!({"error": "no such job"}));
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
                            let _ = req.respond(tiny_http::Response::from_data(b).with_header(h));
                        }
                        None => respond_json(req, 404, &json!({"error": "no such job"})),
                    }
                } else {
                    match shared.with(id, |j| j.status()) {
                        Some(v) => respond_json(req, 200, &v),
                        None => respond_json(req, 404, &json!({"error": "no such job"})),
                    }
                }
            }
            _ => respond_json(req, 404, &json!({"error": "not found"})),
        }
    }
    Ok(())
}

use std::io::Read as _;
