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
                run(
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
                .await
                .map(|()| JsValue::UNDEFINED)
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
                run_long(
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
                .await
                .map(|()| JsValue::UNDEFINED)
            })
        }

        /// End a long video after the block in progress.
        pub fn stop(&self) {
            self.inner.stop.set(true);
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
        use fastwan_model::causal::{CausalConfig, CausalRollout, frame_major};
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
        let cache_layer = (cache_frames * tpf * cfg.dim() * 4) as u64;
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
        let c = tae.cfg.latent_channels;
        let mut st = tae.state();
        let mut done = 0usize;
        for b in 0..blocks {
            if inner.stop.get() {
                break;
            }
            let t0 = now_ms();
            let x = roll.next_block(be, &prompt).map_err(err)?;
            let latent = be.try_download_async(x).await.map_err(err)?;
            let denoise_ms = now_ms() - t0;
            let z = frame_major(&latent, c, bf, lh * lw);
            let ck = tae.launch_chunk(be, &mut st, &z, bf, lh, lw);
            let packed = ck.pack(be);
            let raw = be
                .try_download_async(&packed)
                .await
                .map_err(|e| err(format!("decode failed: {e} ({})", be.limits_text())))?;
            let (count, h, w, rgba) = ck.rgba8(&raw);
            let o = Object::new();
            set(&o, "type", "block");
            set(&o, "index", b as u32);
            set(&o, "total", blocks as u32);
            set(&o, "ms", now_ms() - t0);
            set(&o, "denoise_ms", denoise_ms);
            emit(o);
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
        let ws = dit.workspace(be, g.latent(), ffn_chunk, None);
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

    trait Tap: Sized {
        fn tap(self, f: impl FnOnce(&Self)) -> Self {
            f(&self);
            self
        }
    }
    impl Tap for Object {}
}
