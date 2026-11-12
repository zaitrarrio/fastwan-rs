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
//! `{type: "frames", first, count, width, height, rgb: Uint8Array}`.
//! There is no blocking readback in a browser: everything that leaves the
//! GPU goes through `Wgpu::download_async`.

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
    use fastwan_model::{
        TaeHv, Upscaler, WanDit,
        dit::DitBuilder,
        upscale::{UpConfig, rgb8},
    };
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
    }

    #[wasm_bindgen]
    pub struct Engine {
        inner: Rc<Inner>,
    }

    /// `model`: "fastwan" (FastWan2.1-T2V-1.3B + taew2_1) or "tiny" (the
    /// synthetic test model).
    #[wasm_bindgen(js_name = createEngine)]
    pub async fn create_engine(model: String) -> Result<Engine, JsValue> {
        console_error_panic_hook::set_once();
        let (cfg, tcfg) = match model.as_str() {
            "fastwan" => (WanConfig::fastwan_1_3b(), TaeConfig::taew2_1()),
            "tiny" => (WanConfig::tiny_vsa(), TaeConfig::tiny()),
            m => return Err(err(format!("unknown model `{m}`"))),
        };
        let be = Wgpu::new_async().await.map_err(err)?;
        Ok(Engine {
            inner: Rc::new(Inner {
                builder: RefCell::new(Some(DitBuilder::new(cfg.clone(), WeightQuant::Native))),
                be,
                cfg,
                tcfg,
                dit: RefCell::new(None),
                tae: RefCell::new(None),
                up: RefCell::new(None),
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
        /// resolves to the `[frames, h·s, w·s, 3]` bytes.
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
                let raw = inner.be.download_async(&chunk.out).await;
                let px = rgb8(&raw, &rgb, frames, h, w, chunk.scale);
                Ok(Uint8Array::from(&px[..]).into())
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

        // Prompt embeddings: [L ≤ text_len, text_dim], zero-padded.
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
        let schedule = DmdSchedule::fastwan_1_3b();
        let prompt = dit.prepare_prompt(be, &e);
        let mods: Vec<_> = schedule
            .timesteps
            .iter()
            .map(|&ts| dit.step_mods(be, ts as f32))
            .collect();
        let ws = dit.workspace(be, g.latent(), p.ffn_chunk, None);
        let fence = be.alloc(1);
        be.download_async(&fence).await;
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
            be.download_async(&fence).await; // wait for the step: honest timing
            let o = Object::new();
            set(&o, "type", "step");
            set(&o, "index", i as u32);
            set(&o, "total", mods.len() as u32);
            set(&o, "ms", now_ms() - t0);
            emit(o);
        }
        let latent = be.download_async(&x).await;
        drop((ws, prompt, mods, v, noise));

        // TAEHV, chunk by chunk.
        let (lt, lh, lw) = g.latent();
        let c = tae.cfg.latent_channels;
        let plane = lh * lw;
        let chunk = p.tae_chunk.clamp(1, 2);
        let mut st = tae.state();
        let mut done = 0usize;
        let mut f0 = 0;
        while f0 < lt {
            let nfr = chunk.min(lt - f0);
            let mut z = Vec::with_capacity(nfr * c * plane);
            for f in f0..f0 + nfr {
                for ch in 0..c {
                    z.extend_from_slice(&latent[(ch * lt + f) * plane..][..plane]);
                }
            }
            let ck = tae.launch_chunk(be, &mut st, &z, nfr, lh, lw);
            let raw = be.download_async(&ck.rgb).await;
            let (count, h, w, rgb) = ck.rgb8(&raw);
            let o = Object::new();
            set(&o, "type", "frames");
            set(&o, "first", done as u32);
            set(&o, "count", count as u32);
            set(&o, "width", w as u32);
            set(&o, "height", h as u32);
            set(&o, "rgb", Uint8Array::from(&rgb[..]));
            emit(o);
            done += count;
            f0 += nfr;
        }
        Ok(())
    }

    trait Tap: Sized {
        fn tap(self, f: impl FnOnce(&Self)) -> Self {
            f(&self);
            self
        }
    }
    impl Tap for Object {}
}
