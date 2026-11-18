//! FastWan in the browser: the WebGPU backend, the DiT and TAEHV behind a
//! small wasm-bindgen API that Prompt Studio drives.
//!
//! ```js
//! const eng = await createEngine("fastwan");     // or "tiny" (tests)
//! eng.addBlock(blockBytes);  ...                 // 30×, one safetensors blob per DiT block
//! eng.finishDit(headBytes);                      // the tensors outside the blocks
//! eng.loadTae(taeBytes);
//! await eng.generate(embedsBytes, 480, 272, 33, seed, ev => ...);
//! ```
//!
//! The page fetches each block as one HTTP range of a block-ordered
//! checkpoint (`fastwan quantize`), so wasm memory holds one block at a time.
//! Events: `{type: "prepared"}`, `{type: "step", index, total, ms}`,
//! `{type: "frames", first, count, width, height, rgba: Uint8Array}` (packed
//! on the GPU, alpha 255: straight into `ImageData`).
//! There is no blocking readback in a browser: everything that leaves the
//! GPU goes through `Wgpu::try_download_async`.

#[cfg(target_arch = "wasm32")]
mod web {
    use fastwan_core::{
        Backend, DmdSchedule, Geometry,
        config::{TaeConfig, WanConfig},
        plan::{Device, plan},
        quant::WeightQuant,
        rng::Rng,
        weights::{SafetensorsDir, WeightSource},
    };
    use fastwan_model::{TaeHv, Upscaler, WanDit, dit::DitBuilder, upscale::UpConfig};
    use fastwan_wgpu::Wgpu;
    use js_sys::{Function, Object, Reflect, Uint8Array};
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;

    fn err(e: impl std::fmt::Display) -> JsValue {
        JsValue::from_str(&e.to_string())
    }

    struct Inner {
        be: Wgpu,
        cfg: WanConfig,
        tcfg: TaeConfig,
        builder: RefCell<Option<DitBuilder<Wgpu>>>,
        dit: RefCell<Option<WanDit<Wgpu>>>,
        tae: RefCell<Option<TaeHv<Wgpu>>>,
        up: RefCell<Option<Upscaler<Wgpu>>>,
        /// The canvas upscaled frames are drawn on, and the frames kept on
        /// the GPU: (packed buffer, frames in it), one entry per batch.
        view: RefCell<Option<fastwan_wgpu::CanvasView>>,
        kept: RefCell<Vec<(fastwan_wgpu::GBuf, usize)>>,
        kept_dims: std::cell::Cell<(u32, u32)>,
        /// Lower-memory settings for phones (see `create_engine`).
        mobile: bool,
        /// Set by `stop()`: a long video ends after the block in progress.
        stop: std::cell::Cell<bool>,
        /// Clip attention: `None` = the model's default (VSA from 81 frames).
        attention: std::cell::Cell<Option<fastwan_model::Attention>>,
        /// Long videos keep the K/V cache as 16-bit values (half its memory);
        /// on by default on phones.
        half_kv: std::cell::Cell<bool>,
    }

    #[wasm_bindgen]
    pub struct Engine {
        inner: Rc<Inner>,
    }

    /// `model`: "fastwan" (FastWan2.1-T2V-1.3B + taew2_1) or "tiny" (the
    /// synthetic test model).
    #[wasm_bindgen(js_name = createEngine)]
    pub async fn create_engine(
        model: String,
        kernels: Option<String>,
        mobile: Option<bool>,
        f16: Option<String>,
    ) -> Result<Engine, JsValue> {
        console_error_panic_hook::set_once();
        let (cfg, tcfg) = match model.as_str() {
            "fastwan" => (WanConfig::fastwan_1_3b(), TaeConfig::taew2_1()),
            "tiny" => (WanConfig::tiny_vsa(), TaeConfig::tiny()),
            m => return Err(err(format!("unknown model `{m}`"))),
        };
        // `kernels`: "LINEAR,FLASH" (e.g. "4,2") picks the GEMM / attention
        // kernels, for A/B runs from the page (?kernels=); empty = defaults.
        let mut opts = fastwan_wgpu::WgpuOptions::default();
        if let Some(k) = kernels.filter(|k| !k.is_empty()) {
            let mut it = k.split(',').map(|v| v.trim().parse::<u32>().ok());
            opts.linear = it.next().flatten();
            opts.flash = it.next().flatten();
        }
        // Phones: the tab's memory limit includes GPU memory, and iOS resets a
        // tab whose GPU work runs too long. Smaller submissions here; smaller
        // activation buffers in `run`.
        let mobile = mobile.unwrap_or(false);
        if mobile {
            opts.submit_every = Some(8);
        }
        // f16 math kernels where the adapter has `shader-f16`, fixed for the
        // device's lifetime: "1" = linear4h + flash3h, "attn" = flash3h only
        // (f32 GEMMs), anything else = off.
        let f16 = f16.unwrap_or_default();
        opts.f16 = f16 == "1" || f16 == "attn";
        opts.f16_attn_only = f16 == "attn";
        let be = Wgpu::with_options_async(opts).await.map_err(err)?;
        Ok(Engine {
            inner: Rc::new(Inner {
                builder: RefCell::new(Some(DitBuilder::new(cfg.clone(), WeightQuant::Native))),
                be,
                cfg,
                tcfg,
                dit: RefCell::new(None),
                tae: RefCell::new(None),
                up: RefCell::new(None),
                view: RefCell::new(None),
                kept: RefCell::new(Vec::new()),
                kept_dims: std::cell::Cell::new((0, 0)),
                mobile,
                stop: std::cell::Cell::new(false),
                attention: std::cell::Cell::new(None),
                half_kv: std::cell::Cell::new(mobile),
            }),
        })
    }

    fn set(o: &Object, k: &str, v: impl Into<JsValue>) {
        let _ = Reflect::set(o, &JsValue::from_str(k), &v.into());
    }

    fn now_ms() -> f64 {
        js_sys::Date::now()
    }

    #[wasm_bindgen]
    impl Engine {
        /// The WebGPU adapter, e.g. "wgpu WebGpu: …".
        pub fn adapter(&self) -> String {
            self.inner.be.name()
        }

        /// The adapter limits that decide which sizes fit.
        pub fn limits(&self) -> String {
            self.inner.be.limits_text()
        }

        /// Largest storage binding the device allows, bytes.
        #[wasm_bindgen(js_name = maxBinding)]
        pub fn max_binding(&self) -> f64 {
            self.inner.be.max_binding() as f64
        }

        /// DiT layers the model has (blocks to stream).
        pub fn layers(&self) -> usize {
            self.inner.cfg.layers
        }

        /// Upload the next DiT block from a safetensors blob holding its
        /// tensors (`blocks.N.*`, f16 or quantized as stored).
        #[wasm_bindgen(js_name = addBlock)]
        pub fn add_block(&self, blob: Vec<u8>) -> Result<usize, JsValue> {
            let src = SafetensorsDir::from_bytes(blob).map_err(err)?;
            let mut b = self.inner.builder.borrow_mut();
            let b = b
                .as_mut()
                .ok_or_else(|| err("the DiT is already complete"))?;
            b.add_block(&self.inner.be, &src).map_err(err)?;
            Ok(b.loaded())
        }

        /// The tensors outside the blocks; completes the DiT.
        #[wasm_bindgen(js_name = finishDit)]
        pub fn finish_dit(&self, blob: Vec<u8>) -> Result<(), JsValue> {
            let src = SafetensorsDir::from_bytes(blob).map_err(err)?;
            let b = self
                .inner
                .builder
                .borrow_mut()
                .take()
                .ok_or_else(|| err("the DiT is already complete"))?;
            *self.inner.dit.borrow_mut() = Some(b.finish(&self.inner.be, &src).map_err(err)?);
            Ok(())
        }

        /// The TAEHV decoder from its safetensors file.
        /// The ×4 upscaler (`realesr-animevideov3`, safetensors); the tiny
        /// test model takes the tiny ×2 one.
        #[wasm_bindgen(js_name = loadUpscaler)]
        pub fn load_upscaler(&self, blob: Vec<u8>) -> Result<usize, JsValue> {
            let src = SafetensorsDir::from_bytes(blob).map_err(err)?;
            let cfg = if self.inner.cfg.layers == WanConfig::fastwan_1_3b().layers {
                UpConfig::anime_video_v3()
            } else {
                UpConfig::tiny()
            };
            *self.inner.up.borrow_mut() =
                Some(Upscaler::load(&self.inner.be, cfg, &src).map_err(err)?);
            Ok(cfg.scale)
        }

        /// Frames per `upscale` call at `h × w` that fit one storage binding.
        #[wasm_bindgen(js_name = upscaleBatch)]
        pub fn upscale_batch(&self, h: usize, w: usize) -> Result<usize, JsValue> {
            let up = self.inner.up.borrow();
            let up = up
                .as_ref()
                .ok_or_else(|| err("the upscaler is not loaded"))?;
            Ok(up.batch(h, w, self.inner.be.max_binding()))
        }

        /// Upscale `frames` interleaved RGB frames (`[frames, h, w, 3]`);
        /// resolves to the `[frames, h·s, w·s, 4]` RGBA bytes (alpha 255),
        /// packed on the GPU.
        pub fn upscale(&self, rgb: Vec<u8>, frames: usize, h: usize, w: usize) -> js_sys::Promise {
            let inner = self.inner.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let chunk = {
                    let up = inner.up.borrow();
                    let up = up
                        .as_ref()
                        .ok_or_else(|| err("the upscaler is not loaded"))?;
                    if rgb.len() != frames * h * w * 3 {
                        return Err(err("rgb is not frames × h × w × 3"));
                    }
                    up.launch(&inner.be, &rgb, frames, h, w)
                };
                let raw = inner.be.try_download_async(&chunk.out).await.map_err(err)?;
                let px = chunk.rgba8(&raw);
                Ok(Uint8Array::from(&px[..]).into())
            })
        }

        /// The canvas upscaled frames are drawn on (it must not have a 2D
        /// context). Keep its width × height at the upscaled size.
        #[wasm_bindgen(js_name = attachCanvas)]
        pub fn attach_canvas(&self, canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
            *self.inner.view.borrow_mut() = Some(self.inner.be.canvas_view(canvas).map_err(err)?);
            Ok(())
        }

        /// Upscale `frames` RGB frames and keep the result on the GPU (no
        /// readback); resolves to the number of frames kept so far, once this
        /// batch has finished. Show them with `showUpscaled`.
        #[wasm_bindgen(js_name = upscaleKeep)]
        pub fn upscale_keep(
            &self,
            rgb: Vec<u8>,
            frames: usize,
            h: usize,
            w: usize,
        ) -> js_sys::Promise {
            let inner = self.inner.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let chunk = {
                    let up = inner.up.borrow();
                    let up = up
                        .as_ref()
                        .ok_or_else(|| err("the upscaler is not loaded"))?;
                    if rgb.len() != frames * h * w * 3 {
                        return Err(err("rgb is not frames × h × w × 3"));
                    }
                    up.launch(&inner.be, &rgb, frames, h, w)
                };
                let s = chunk.scale as u32;
                // Wait for the batch: progress for the page, and the
                // intermediates may only be dropped (destroyed) after it.
                let fence = inner.be.alloc(1);
                inner.be.try_download_async(&fence).await.map_err(err)?;
                let mut kept = inner.kept.borrow_mut();
                kept.push((chunk.into_out(), frames));
                inner.kept_dims.set((w as u32 * s, h as u32 * s));
                Ok(JsValue::from(kept.iter().map(|k| k.1).sum::<usize>() as u32))
            })
        }

        /// Upscaled frames kept on the GPU.
        #[wasm_bindgen(js_name = upscaledCount)]
        pub fn upscaled_count(&self) -> usize {
            self.inner.kept.borrow().iter().map(|k| k.1).sum()
        }

        /// Free the kept upscaled frames.
        #[wasm_bindgen(js_name = clearUpscaled)]
        pub fn clear_upscaled(&self) {
            self.inner.kept.borrow_mut().clear();
        }

        /// Draw kept upscaled frame `i` on the attached canvas.
        #[wasm_bindgen(js_name = showUpscaled)]
        pub fn show_upscaled(&self, i: usize) -> Result<(), JsValue> {
            let (buf, first) = self
                .locate(i)
                .ok_or_else(|| err("frame not upscaled yet"))?;
            let (w, h) = self.inner.kept_dims.get();
            let view = self.inner.view.borrow();
            let view = view.as_ref().ok_or_else(|| err("no canvas attached"))?;
            view.show(&self.inner.be, &buf, first, w, h).map_err(err)
        }

        /// Kept upscaled frame `i` as RGBA bytes (for saving one frame).
        #[wasm_bindgen(js_name = upscaledFrame)]
        pub fn upscaled_frame(&self, i: usize) -> js_sys::Promise {
            let found = self.locate(i);
            let (w, h) = self.inner.kept_dims.get();
            let inner = self.inner.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let (buf, first) = found.ok_or_else(|| err("frame not upscaled yet"))?;
                let n = (w * h) as usize;
                let one = inner.be.alloc_words(n);
                inner.be.copy(&buf, first, &one, 0, n);
                let raw = inner.be.try_download_async(&one).await.map_err(err)?;
                Ok(Uint8Array::from(&fastwan_core::backend::rgba8_bytes(&raw)[..]).into())
            })
        }

        #[wasm_bindgen(js_name = loadTae)]
        pub fn load_tae(&self, blob: Vec<u8>) -> Result<(), JsValue> {
            let src = SafetensorsDir::from_bytes(blob).map_err(err)?;
            *self.inner.tae.borrow_mut() =
                Some(TaeHv::load(&self.inner.be, self.inner.tcfg.clone(), &src).map_err(err)?);
            Ok(())
        }

        /// Generate a clip from `prompt_embeds` (`fastwan encode` gallery
        /// file), reporting progress and decoded frames to `on_event`.
        pub fn generate(
            &self,
            embeds: Vec<u8>,
            width: usize,
            height: usize,
            frames: usize,
            seed: f64,
            on_event: Function,
        ) -> js_sys::Promise {
            let inner = self.inner.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let r = run(
                    &inner,
                    embeds,
                    Geometry {
                        width,
                        height,
                        frames,
                    },
                    seed as u64,
                    &on_event,
                )
                .await;
                // Free the buffers kept for reuse: the next run may use other sizes.
                inner.be.trim_pool();
                r.map(|()| JsValue::UNDEFINED)
            })
        }

        /// A long video from a causal (Self-Forcing) checkpoint, streamed
        /// block by block: events `prepared`, then per block `block`
        /// `{index, total, ms, denoise_ms}` and `frames` (as in `generate`).
        /// `cache_frames` / `sink_frames` are latent frames (12 / 3).
        #[wasm_bindgen(js_name = generateLong)]
        #[allow(clippy::too_many_arguments)]
        pub fn generate_long(
            &self,
            embeds: Vec<u8>,
            width: usize,
            height: usize,
            frames: usize,
            seed: f64,
            cache_frames: usize,
            sink_frames: usize,
            on_event: Function,
        ) -> js_sys::Promise {
            let inner = self.inner.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let r = run_long(
                    &inner,
                    embeds,
                    Geometry {
                        width,
                        height,
                        frames,
                    },
                    seed as u64,
                    cache_frames,
                    sink_frames,
                    &on_event,
                )
                .await;
                // Free the buffers kept for reuse: the next run may use other sizes.
                inner.be.trim_pool();
                r.map(|()| JsValue::UNDEFINED)
            })
        }

        /// End a long video after the block in progress.
        pub fn stop(&self) {
            self.inner.stop.set(true);
        }

        /// Clip attention: "dense", "vsa", or "" / "auto" for the default
        /// (VSA from 81 frames on a VSA checkpoint, dense below).
        #[wasm_bindgen(js_name = setAttention)]
        pub fn set_attention(&self, mode: &str) -> Result<(), JsValue> {
            let a = match mode {
                "" | "auto" => None,
                "dense" => Some(fastwan_model::Attention::Dense),
                "vsa" => Some(fastwan_model::Attention::Vsa { sparsity: 0.8 }),
                m => return Err(err(format!("unknown attention `{m}`"))),
            };
            self.inner.attention.set(a);
            Ok(())
        }

        /// Every op check (`fastwan check`'s op suite plus VSA at the model's
        /// token grids) on this GPU against the CPU reference compiled into
        /// this module. The CPU side runs first, on the main thread, and takes
        /// a minute or two. Events: `{type: "check", index, total, name,
        /// rel, ok}`; resolves to the number of failures.
        #[wasm_bindgen(js_name = selfTest)]
        pub fn self_test(&self, on_event: Function) -> js_sys::Promise {
            let inner = self.inner.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let be = &inner.be;
                let tol = if be.name().ends_with("+f16") {
                    1e-2
                } else {
                    2e-3
                };
                let d = fastwan_model::check::ops_deferred(be, tol);
                let total = d.pending.len();
                let mut bad = 0u32;
                for (i, p) in d.pending.iter().enumerate() {
                    let got = be.try_download_async(&p.buf).await.map_err(err)?;
                    let r = p.finish(&got, tol);
                    bad += (!r.ok) as u32;
                    let o = Object::new();
                    set(&o, "type", "check");
                    set(&o, "index", i as u32);
                    set(&o, "total", total as u32);
                    set(&o, "name", r.name.as_str());
                    set(&o, "rel", r.rel_err as f64);
                    set(&o, "ok", r.ok);
                    let _ = on_event.call1(&JsValue::NULL, &o);
                }
                // VSA stage by stage (the op checks above only see the
                // result): tile means, coarse output, selected tiles.
                for &(grid, heads, d) in &[
                    ((5usize, 6usize, 9usize), 2usize, 128usize),
                    ((21, 17, 30), 2, 128),
                ] {
                    use fastwan_core::Backend as _;
                    let spec = fastwan_core::vsa::VsaSpec::new(grid, heads, d, 0.8);
                    let n = heads * spec.plan.seq * d;
                    let mut rng = fastwan_core::rng::Rng::new(7);
                    let (q, k, v, g) = (rng.normal(n), rng.normal(n), rng.normal(n), rng.normal(n));
                    let (gq, gk, gv, gg, go) = (
                        be.upload(&q),
                        be.upload(&k),
                        be.upload(&v),
                        be.upload(&g),
                        be.alloc(n),
                    );
                    let st = be.vsa_prepare(&spec);
                    be.vsa_attention(&gq, &gk, &gv, &gg, &st, &go);
                    let (want_p, want_c, want_s) =
                        fastwan_core::vsa::coarse_reference(&spec, &q, &k, &v);
                    let (bp, bc, bs) = st.stages();
                    let got_p = be.try_download_async(bp).await.map_err(err)?;
                    let got_c = be.try_download_async(bc).await.map_err(err)?;
                    let got_s = be.try_download_async(bs).await.map_err(err)?;
                    let relf = |a: &[f32], b: &[f32]| {
                        let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())) + 1e-6;
                        a.iter().zip(b).fold(0f32, |m, (x, y)| {
                            let e = (x - y).abs();
                            if e.is_nan() { f32::INFINITY } else { m.max(e) }
                        }) / scale
                    };
                    // Selection: tiles the GPU picked that the host did not, as a set per query tile.
                    let topk = spec.topk;
                    let got_s: Vec<u32> = got_s.iter().map(|x| x.to_bits()).collect();
                    let mut wrong = 0usize;
                    for (gr, wr) in got_s.chunks(topk).zip(want_s.chunks(topk)) {
                        wrong += gr.iter().filter(|t| !wr.contains(t)).count();
                    }
                    let total_sel = want_s.len();
                    let tag = format!(
                        "vsa stages {grid:?} h{heads} d{d} topk {topk}/{}",
                        spec.plan.num_tiles()
                    );
                    let rows = [
                        (
                            format!("{tag}: pool (tile means)"),
                            relf(&got_p, &want_p),
                            None,
                        ),
                        (
                            format!("{tag}: coarse (tile attention)"),
                            relf(&got_c, &want_c),
                            None,
                        ),
                        (
                            format!("{tag}: select ({wrong} of {total_sel} picks differ)"),
                            wrong as f32 / total_sel as f32,
                            Some(wrong),
                        ),
                    ];
                    for (name, rel, wrong) in rows {
                        // A near-tie can flip a pick between float orders: allow 1%.
                        let ok = match wrong {
                            Some(w) => w * 100 <= total_sel,
                            None => rel <= tol,
                        };
                        bad += (!ok) as u32;
                        let o = Object::new();
                        set(&o, "type", "check");
                        set(&o, "name", name.as_str());
                        set(&o, "rel", rel as f64);
                        set(&o, "ok", ok);
                        let _ = on_event.call1(&JsValue::NULL, &o);
                    }
                    if got_s.iter().take(topk * 4).all(|&x| x == 0) {
                        let o = Object::new();
                        set(&o, "type", "check");
                        set(
                            &o,
                            "name",
                            "note: the first query tiles selected only tile 0 (sel never written?)",
                        );
                        set(&o, "rel", 0.0);
                        set(&o, "ok", false);
                        let _ = on_event.call1(&JsValue::NULL, &o);
                    }
                }
                Ok(JsValue::from(bad))
            })
        }

        /// Long videos: keep the K/V cache as 16-bit values (half the cache
        /// memory; on by default on phones).
        #[wasm_bindgen(js_name = setHalfKv)]
        pub fn set_half_kv(&self, on: bool) {
            self.inner.half_kv.set(on);
        }

        /// Whether the f16 math kernels are in use (asked for at
        /// `createEngine` and supported by the adapter).
        #[wasm_bindgen(js_name = isF16)]
        pub fn is_f16(&self) -> bool {
            self.inner.be.f16()
        }

        /// Time the model's main kernels on this GPU with synthetic data: the
        /// FFN GEMMs (f16 weights), dense attention and attention over a
        /// 16-bit K/V cache, each at a model shape. Events:
        /// `{type: "bench", name, ms, gflops}`, one per kernel (mean of `reps`
        /// calls after a warm-up, fenced by a readback).
        pub fn bench(&self, reps: u32, on_event: Function) -> js_sys::Promise {
            let inner = self.inner.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                use fastwan_core::Backend as _;
                let be = &inner.be;
                let reps = reps.max(1);
                let mut rng = Rng::new(3);
                let fence = be.alloc(1);
                let emit = |name: &str, ms: f64, flop: f64| {
                    let o = Object::new();
                    set(&o, "type", "bench");
                    set(&o, "name", name);
                    set(&o, "ms", ms);
                    set(&o, "gflops", flop / ms / 1e6);
                    set(&o, "tflops", flop / ms / 1e9);
                    let _ = on_event.call1(&JsValue::NULL, &o);
                };
                // Rows: 3 latent frames at 480×272 (one long-video block) ≈
                // 1530 tokens; 4096 keeps every kernel on its large-tile path.
                let (m, d, f) = (4096usize, 1536usize, 8960usize);
                let x = be.upload(&rng.normal(m * f));
                let wu = be.weight(&rng.uniform(f * d, 0.05), f, d);
                let wd = be.weight(&rng.uniform(d * f, 0.05), d, f);
                let h = be.alloc(m * f);
                let y = be.alloc(m * d);
                let mut results: Vec<(String, f64, f64)> = Vec::new();
                // In f64: usize is 32 bits on wasm and these products overflow it.
                let ffn = 2.0 * m as f64 * d as f64 * f as f64;
                let ms = time_it(be, &fence, reps, &|| {
                    be.linear(
                        &x,
                        0,
                        m,
                        &wu,
                        None,
                        fastwan_core::backend::Act::GeluTanh,
                        &h,
                        0,
                    )
                })
                .await?;
                results.push(("GEMM 4096×1536→8960 (FFN up, GELU)".into(), ffn, ms));
                let ms = time_it(be, &fence, reps, &|| {
                    be.linear(&h, 0, m, &wd, None, fastwan_core::backend::Act::None, &y, 0)
                })
                .await?;
                results.push(("GEMM 4096×8960→1536 (FFN down)".into(), ffn, ms));
                let (heads, hd, s) = (12usize, 128usize, 4096usize);
                let q = be.upload(&rng.normal(heads * s * hd));
                let k = be.upload(&rng.normal(heads * s * hd));
                let v = be.upload(&rng.normal(heads * s * hd));
                let o = be.alloc(s * heads * hd);
                let ms = time_it(be, &fence, reps, &|| {
                    be.attention(&q, &k, &v, heads, s, s, hd, &o)
                })
                .await?;
                results.push((
                    "attention 12×4096×4096 d128".into(),
                    4.0 * heads as f64 * s as f64 * s as f64 * hd as f64,
                    ms,
                ));
                if be.kv_caps(hd).1 {
                    // One long-video block (1530 queries) over a full 12-frame
                    // cache at 480×272 (6120 keys), 16-bit.
                    let (sq, sk) = (1530usize, 6120usize);
                    let kc = be.alloc_kv(heads * sk * hd, true);
                    let vc = be.alloc_kv(heads * sk * hd, true);
                    let src = be.upload(&rng.normal(heads * sk * hd));
                    be.kv_store(&src, sk, 0, sk, heads, hd, &kc, 0, sk, true);
                    be.kv_store(&src, sk, 0, sk, heads, hd, &vc, 0, sk, true);
                    let ms = time_it(be, &fence, reps, &|| {
                        be.attention_kv(&q, &kc, &vc, heads, sq, sk, sk, hd, true, &o)
                    })
                    .await?;
                    results.push((
                        "attention over 16-bit cache 1530×6120".into(),
                        4.0 * heads as f64 * sq as f64 * sk as f64 * hd as f64,
                        ms,
                    ));
                }
                for (n, fl, ms) in &results {
                    emit(n, *ms, *fl);
                }
                Ok(JsValue::from(results.len() as u32))
            })
        }

        /// Diagnostics: turn off the device keep-alive during readbacks.
        #[wasm_bindgen(js_name = setKeepAlive)]
        pub fn set_keep_alive(&self, on: bool) {
            self.inner.be.set_keepalive(on);
        }
    }

    /// `prompt_embeds` from a gallery file, zero-padded to `[text_len, text_dim]`.
    fn read_embeds(cfg: &WanConfig, embeds: Vec<u8>) -> Result<Vec<f32>, JsValue> {
        let src = SafetensorsDir::from_bytes(embeds).map_err(err)?;
        let name = src
            .names()
            .find(|n| n.contains("embed"))
            .map(str::to_string)
            .ok_or_else(|| err("no prompt_embeds tensor"))?;
        let (_, shape, _) = src.raw(&name).map_err(err)?;
        let mut e = src.get(&name, &shape).map_err(err)?;
        if e.len() % cfg.text_dim != 0 || e.len() > cfg.text_len * cfg.text_dim {
            return Err(err(format!(
                "prompt_embeds {shape:?} is not [L, {}]",
                cfg.text_dim
            )));
        }
        e.resize(cfg.text_len * cfg.text_dim, 0.0);
        Ok(e)
    }

    /// A long video from a causal (Self-Forcing) checkpoint: blocks of three
    /// latent frames over a rolling K/V cache, each block decoded and sent as
    /// soon as it is denoised. Stops early on `stop()`.
    async fn run_long(
        inner: &Inner,
        embeds: Vec<u8>,
        g: Geometry,
        seed: u64,
        cache_frames: usize,
        sink_frames: usize,
        on: &Function,
    ) -> Result<(), JsValue> {
        use fastwan_model::causal::{CausalConfig, CausalRollout};
        let emit = |o: Object| {
            let _ = on.call1(&JsValue::NULL, &o);
        };
        let (be, cfg) = (&inner.be, &inner.cfg);
        let dit = inner.dit.borrow();
        let dit = dit.as_ref().ok_or_else(|| err("the model is not loaded"))?;
        let tae = inner.tae.borrow();
        let tae = tae
            .as_ref()
            .ok_or_else(|| err("the decoder is not loaded"))?;
        g.validate(cfg).map_err(err)?;
        let e = read_embeds(cfg, embeds)?;
        let mut ccfg = CausalConfig::self_forcing();
        ccfg.cache_frames(cache_frames, sink_frames);
        ccfg.half_kv = inner.half_kv.get();
        let kv_bytes = if ccfg.half_kv && be.kv_caps(cfg.head_dim).1 {
            2
        } else {
            4
        };
        let bf = ccfg.block_frames;
        // One block's activations: plan a clip of one block's length.
        let block = Geometry {
            frames: (bf - 1) * 4 + 1,
            ..g
        };
        let p = plan(
            cfg,
            &inner.tcfg,
            &block,
            &Device {
                budget_bytes: u64::MAX,
                max_binding_bytes: be.max_binding(),
                weight_bytes: 2,
                act_bytes: 4,
            },
        );
        let (lt, lh, lw) = g.latent();
        let tpf = (lh / cfg.patch) * (lw / cfg.patch);
        let cache_layer = (cache_frames * tpf * cfg.dim() * kv_bytes) as u64;
        let largest = p.largest_binding.max(cache_layer);
        if largest > be.max_binding() {
            return Err(err(format!(
                "{}×{} needs a {} MiB GPU buffer; this browser's GPU allows {} ({}). Pick a smaller size.",
                g.width,
                g.height,
                largest >> 20,
                be.max_binding() >> 20,
                be.limits_text()
            )));
        }
        let ffn_chunk = if inner.mobile {
            p.ffn_chunk.min(1024)
        } else {
            p.ffn_chunk
        };
        inner.stop.set(false);
        let prompt = dit.prepare_prompt(be, &e);
        let mut roll = CausalRollout::new(be, dit, ccfg, (lh, lw), ffn_chunk, seed).map_err(err)?;
        let fence = be.alloc(1);
        be.try_download_async(&fence).await.map_err(err)?;
        emit(Object::new().tap(|o| set(o, "type", "prepared")));
        let blocks = lt.div_ceil(bf);
        let mut st = tae.state();
        // Pipelined: block b is queued (denoise, decode from the device
        // latent, pack, readback copy) before the page waits for block b − 1's
        // frames, so the GPU always has the next block while the browser
        // delivers the previous readback. No latent leaves the GPU.
        let mut done = 0usize;
        let mut pending: Option<(
            fastwan_wgpu::Readback,
            fastwan_model::taehv::Chunk<Wgpu>,
            usize,
        )> = None;
        let mut queued = 0usize;
        let mut t_last = now_ms();
        loop {
            let next = if queued < blocks && !inner.stop.get() {
                let x = roll.next_block(be, &prompt).map_err(err)?;
                let ck = tae.launch_chunk_dev(be, &mut st, x, bf, lh, lw);
                let packed = ck.pack(be);
                let rb = be
                    .readback_begin(&packed)
                    .map_err(|e| err(format!("decode failed: {e} ({})", be.limits_text())))?;
                queued += 1;
                Some((rb, ck, queued - 1))
            } else {
                None
            };
            if let Some((rb, ck, b)) = pending.take() {
                let raw = rb
                    .wait(be)
                    .await
                    .map_err(|e| err(format!("decode failed: {e} ({})", be.limits_text())))?;
                let (count, h, w, rgba) = ck.rgba8(&raw);
                let now = now_ms();
                let o = Object::new();
                set(&o, "type", "block");
                set(&o, "index", b as u32);
                set(&o, "total", blocks as u32);
                set(&o, "ms", now - t_last);
                set(&o, "denoise_ms", now - t_last);
                let m = fastwan_wgpu::mem_stats();
                set(&o, "buffers_created", m.created as f64);
                set(&o, "buffers_reused", m.reused as f64);
                set(&o, "mb_created", m.created_bytes as f64 / 1e6);
                set(&o, "mb_held", m.held_bytes as f64 / 1e6);
                emit(o);
                t_last = now;
                let o = Object::new();
                set(&o, "type", "frames");
                set(&o, "first", done as u32);
                set(&o, "count", count as u32);
                set(&o, "width", w as u32);
                set(&o, "height", h as u32);
                set(&o, "rgba", Uint8Array::from(&rgba[..]));
                emit(o);
                done += count;
            }
            match next {
                Some(n) => pending = Some(n),
                None => break,
            }
        }
        Ok(())
    }

    async fn run(
        inner: &Inner,
        embeds: Vec<u8>,
        g: Geometry,
        seed: u64,
        on: &Function,
    ) -> Result<(), JsValue> {
        let emit = |o: Object| {
            let _ = on.call1(&JsValue::NULL, &o);
        };
        let (be, cfg) = (&inner.be, &inner.cfg);
        let dit = inner.dit.borrow();
        let dit = dit.as_ref().ok_or_else(|| err("the model is not loaded"))?;
        let tae = inner.tae.borrow();
        let tae = tae
            .as_ref()
            .ok_or_else(|| err("the decoder is not loaded"))?;
        g.validate(cfg).map_err(err)?;

        let e = read_embeds(cfg, embeds)?;

        let p = plan(
            cfg,
            &inner.tcfg,
            &g,
            &Device {
                budget_bytes: u64::MAX,
                max_binding_bytes: be.max_binding(),
                weight_bytes: 2,
                act_bytes: 4,
            },
        );
        // Refuse a size this GPU cannot bind before spending the denoise on it.
        if p.largest_binding > be.max_binding() {
            return Err(err(format!(
                "{}×{} needs a {} MiB GPU buffer; this browser's GPU allows {} ({}). Pick a smaller size.",
                g.width,
                g.height,
                p.largest_binding >> 20,
                be.max_binding() >> 20,
                be.limits_text()
            )));
        }
        let schedule = DmdSchedule::fastwan_1_3b();
        let prompt = dit.prepare_prompt(be, &e);
        let mods: Vec<_> = schedule
            .timesteps
            .iter()
            .map(|&ts| dit.step_mods(be, ts as f32))
            .collect();
        // On phones the FFN hidden state runs in chunks of 1024 tokens (36 MiB
        // at f32) instead of all at once (384 MiB at 480×272×81).
        let ffn_chunk = if inner.mobile {
            p.ffn_chunk.min(1024)
        } else {
            p.ffn_chunk
        };
        let ws = dit.workspace(be, g.latent(), ffn_chunk, inner.attention.get());
        let fence = be.alloc(1);
        be.try_download_async(&fence).await.map_err(err)?;
        emit(Object::new().tap(|o| set(o, "type", "prepared")));

        let n = g.latent_len(cfg);
        let mut rng = Rng::new(seed);
        let x = be.upload(&rng.normal(n));
        let v = be.alloc(n);
        let noise = be.alloc(n);
        for (i, m) in mods.iter().enumerate() {
            let t0 = now_ms();
            dit.forward(be, &ws, &x, m, &prompt, &v);
            let (st, next) = schedule.coeffs(i);
            if next.is_some() {
                be.write(&noise, &rng.normal(n));
            }
            be.dmd_step(&x, &v, next.map(|_| &noise), st, next.unwrap_or(0.0), n);
            be.try_download_async(&fence).await.map_err(err)?; // wait for the step: honest timing
            let o = Object::new();
            set(&o, "type", "step");
            set(&o, "index", i as u32);
            set(&o, "total", mods.len() as u32);
            set(&o, "ms", now_ms() - t0);
            emit(o);
        }
        let latent = be.try_download_async(&x).await.map_err(err)?;
        drop((ws, prompt, mods, v, noise));

        // TAEHV, chunk by chunk.
        let (lt, lh, lw) = g.latent();
        let c = tae.cfg.latent_channels;
        let plane = lh * lw;
        // The plan's chunk respects the binding cap. 2 is faster, but above
        // 480×272 one latent frame at a time halves the decoder's peak memory
        // (818 → 409 MiB for its largest buffer at 832×480): browsers run on
        // laptop GPUs too. Chunking does not change the frames.
        let chunk = if inner.mobile || g.width * g.height > 480 * 272 {
            1
        } else {
            p.tae_chunk.clamp(1, 2)
        };
        let decode_err = |e: String| err(format!("decode failed: {e} ({})", be.limits_text()));
        let mut st = tae.state();
        let gather = |f0: usize, nfr: usize| {
            let mut z = Vec::with_capacity(nfr * c * plane);
            for f in f0..f0 + nfr {
                for ch in 0..c {
                    z.extend_from_slice(&latent[(ch * lt + f) * plane..][..plane]);
                }
            }
            z
        };
        // The device packs each chunk to 8-bit RGBA (`Chunk::pack`); the
        // next chunk is queued before this one's readback is awaited, so the
        // GPU never idles on the copy.
        let starts: Vec<usize> = (0..lt).step_by(chunk).collect();
        let mut launch = |f0: usize| {
            let nfr = chunk.min(lt - f0);
            let ck = tae.launch_chunk(be, &mut st, &gather(f0, nfr), nfr, lh, lw);
            let packed = ck.pack(be);
            (ck, packed)
        };
        let mut done = 0usize;
        let mut cur = launch(starts[0]);
        for i in 0..starts.len() {
            let next = (i + 1 < starts.len()).then(|| launch(starts[i + 1]));
            let raw = be.try_download_async(&cur.1).await.map_err(decode_err)?;
            let (count, h, w, rgba) = cur.0.rgba8(&raw);
            let o = Object::new();
            set(&o, "type", "frames");
            set(&o, "first", done as u32);
            set(&o, "count", count as u32);
            set(&o, "width", w as u32);
            set(&o, "height", h as u32);
            set(&o, "rgba", Uint8Array::from(&rgba[..]));
            emit(o);
            done += count;
            if let Some(n) = next {
                cur = n;
            }
        }
        Ok(())
    }

    impl Engine {
        fn locate(&self, i: usize) -> Option<(fastwan_wgpu::GBuf, usize)> {
            let (w, h) = self.inner.kept_dims.get();
            let mut i = i;
            for (buf, n) in self.inner.kept.borrow().iter() {
                if i < *n {
                    return Some((buf.clone(), i * (w * h) as usize));
                }
                i -= n;
            }
            None
        }
    }

    /// Mean time of `run` over `reps` calls after one warm-up call, fenced
    /// by readbacks so queued work is counted.
    async fn time_it(
        be: &Wgpu,
        fence: &fastwan_wgpu::GBuf,
        reps: u32,
        run: &dyn Fn(),
    ) -> Result<f64, JsValue> {
        run();
        be.try_download_async(fence).await.map_err(err)?;
        let t0 = now_ms();
        for _ in 0..reps {
            run();
        }
        be.try_download_async(fence).await.map_err(err)?;
        Ok((now_ms() - t0) / reps as f64)
    }

    trait Tap: Sized {
        fn tap(self, f: impl FnOnce(&Self)) -> Self {
            f(&self);
            self
        }
    }
    impl Tap for Object {}
}
