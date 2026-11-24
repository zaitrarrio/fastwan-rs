//! Backend conformance: every op, and then the whole tiny model, against the
//! CPU reference on the same random inputs. `fastwan check --backend …` runs
//! this on real hardware; the WebGPU backend runs it in CI on lavapipe.
//!
//! Shapes are chosen to hit tile edges (rows and columns that are not
//! multiples of 64, odd head counts, sequence lengths below one tile).

use crate::{
    dit::WanDit,
    pipeline::{Options, Pipeline, Timings},
    taehv::TaeHv,
};
use fastwan_core::{
    backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind, rgba8_bytes},
    config::{TaeConfig, WanConfig},
    geometry::Geometry,
    quant::WeightQuant,
    rng::Rng,
    schedule::DmdSchedule,
    weights::{Synthetic, WeightSource},
};
use fastwan_cpu::Cpu;

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    /// `max |a − b| / (max |b| + 1e-6)` over the output.
    pub rel_err: f32,
    pub ok: bool,
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "output lengths differ");
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
    let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| {
        let d = (x - y).abs();
        if d.is_nan() { f32::INFINITY } else { m.max(d) }
    });
    worst / scale
}

/// Runs `f` on both backends with the same host inputs.
struct Pair<'a, B: Backend> {
    be: &'a B,
    tol: f32,
    out: Vec<CheckResult>,
    /// Browser: keep GPU outputs on the device and compare them after an
    /// async readback ([`ops_deferred`]); a blocking download is impossible there.
    defer: bool,
    pending: Vec<Pending<B>>,
}

/// A backend output not read yet.
pub enum Got<B: Backend> {
    Now(Vec<f32>),
    Dev(B::Buf),
}

trait IntoGot<B: Backend> {
    fn into_got(self) -> Got<B>;
}
impl<B: Backend> IntoGot<B> for Vec<f32> {
    fn into_got(self) -> Got<B> {
        Got::Now(self)
    }
}
impl<B: Backend> IntoGot<B> for Got<B> {
    fn into_got(self) -> Got<B> {
        self
    }
}

/// One check whose backend output is still on the device.
pub struct Pending<B: Backend> {
    pub name: String,
    /// Read this back (e.g. `Wgpu::try_download_async`) and pass it to [`Pending::finish`].
    pub buf: B::Buf,
    want: Vec<f32>,
    rgba: bool,
}

impl<B: Backend> Pending<B> {
    /// Compare the read-back output (`f32` words; packed RGBA for pack checks).
    pub fn finish(&self, got: &[f32], tol: f32) -> CheckResult {
        let got = &got[..self.want.len().min(got.len())];
        if self.rgba {
            result_rgba8(self.name.clone(), got, &self.want)
        } else {
            result(self.name.clone(), got, &self.want, tol)
        }
    }
}

fn result(name: String, got: &[f32], want: &[f32], tol: f32) -> CheckResult {
    let e = rel(got, want);
    CheckResult {
        name,
        rel_err: e,
        ok: e <= tol,
    }
}

/// Packed RGBA words: ok when no byte differs by more than one 8-bit level.
fn result_rgba8(name: String, got: &[f32], want: &[f32]) -> CheckResult {
    let (g, w) = (rgba8_bytes(got), rgba8_bytes(want));
    let max = g
        .iter()
        .zip(&w)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    let lv = |v: &[u8]| v.iter().map(|&b| b as f32).collect::<Vec<_>>();
    CheckResult {
        name,
        rel_err: rel(&lv(&g), &lv(&w)),
        ok: g.len() == w.len() && max <= 1,
    }
}

/// Ops whose GPU outputs are left on the device: compare after reading back.
pub struct Deferred<B: Backend> {
    /// Checks that needed no readback (none today).
    pub done: Vec<CheckResult>,
    pub pending: Vec<Pending<B>>,
}

impl<'a, B: Backend> Pair<'a, B> {
    /// The backend's copy of `b`: downloaded now, or left on the device.
    fn dl(&self, b: B::Buf) -> Got<B> {
        if self.defer {
            Got::Dev(b)
        } else {
            Got::Now(self.be.download(&b))
        }
    }

    /// Packed RGBA words (`pack_rgba8`): ok when no byte differs by more than
    /// one 8-bit level (a ~1e-4 difference upstream may flip a rounding);
    /// the reported error is the relative error of the levels.
    fn record_rgba8(&mut self, name: impl Into<String>, got: impl IntoGot<B>, want: Vec<f32>) {
        match got.into_got() {
            Got::Now(g) => self.record_rgba8_now(name, g, want),
            Got::Dev(buf) => self.pending.push(Pending {
                name: name.into(),
                buf,
                want,
                rgba: true,
            }),
        }
    }

    fn record_rgba8_now(&mut self, name: impl Into<String>, got: Vec<f32>, want: Vec<f32>) {
        self.out.push(result_rgba8(name.into(), &got, &want));
    }

    fn record(&mut self, name: impl Into<String>, got: impl IntoGot<B>, want: Vec<f32>) {
        match got.into_got() {
            Got::Now(g) => self.record_now(name, g, want),
            Got::Dev(buf) => self.pending.push(Pending {
                name: name.into(),
                buf,
                want,
                rgba: false,
            }),
        }
    }

    fn record_now(&mut self, name: impl Into<String>, got: Vec<f32>, want: Vec<f32>) {
        self.out.push(result(name.into(), &got, &want, self.tol));
    }
}

pub fn run_all<B: Backend>(be: &B, tol: f32) -> Vec<CheckResult> {
    let mut p = Pair {
        be,
        tol,
        out: Vec::new(),
        defer: false,
        pending: Vec::new(),
    };
    let mut rng = Rng::new(42);
    ops(&mut p, &mut rng);
    vsa_real(&mut p, &mut rng);
    model(&mut p);
    upscaler(&mut p);
    causal(&mut p);
    causal_memrope(&mut p);
    pack(&mut p, &mut rng);
    unpack(&mut p, &mut rng);
    model_lora(&mut p);
    model_vsa(&mut p, WeightQuant::Native);
    model_vsa(&mut p, WeightQuant::Int8);
    model_vsa(&mut p, WeightQuant::Fp8);
    model_vsa(
        &mut p,
        WeightQuant::Int4 {
            group: 32,
            hqq: true,
            ffn_only: false,
        },
    );
    p.out
}

/// The op checks with the backend's outputs left on the device, for the
/// browser (Prompt Studio `?selftest=1`): read each `pending` buffer back
/// asynchronously and call [`Pending::finish`]. The model-level checks need
/// blocking readbacks and are not included; the clip itself is the browser's
/// model-level check.
pub fn ops_deferred<B: Backend>(be: &B, tol: f32) -> Deferred<B> {
    let mut p = Pair {
        be,
        tol,
        out: Vec::new(),
        defer: true,
        pending: Vec::new(),
    };
    let mut rng = Rng::new(42);
    ops(&mut p, &mut rng);
    vsa_real(&mut p, &mut rng);
    pack(&mut p, &mut rng);
    unpack(&mut p, &mut rng);
    Deferred {
        done: p.out,
        pending: p.pending,
    }
}

/// VSA at the token grids the browser runs (480×272×81 and 832×480×81 after
/// the 2×2 patch), d = 128 like the model, few heads to keep the CPU
/// reference cheap. The small cases in `ops` never exceed 256 key tiles.
/// `x` rounded to bf16 (nearest even on the top 16 bits, finite inputs):
/// the values CUDA's uploads hold, so a VSA case's top-k sees the same
/// scores on both sides.
fn bf(x: Vec<f32>) -> Vec<f32> {
    x.into_iter()
        .map(|v| {
            let b = v.to_bits();
            f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
        })
        .collect()
}

fn vsa_real<B: Backend>(p: &mut Pair<'_, B>, rng: &mut Rng) {
    let (be, cpu) = (p.be, &Cpu);
    for &(grid, heads) in &[((21, 17, 30), 2), ((21, 30, 52), 1)] {
        let (d, sparsity) = (128, 0.8);
        let spec = fastwan_core::vsa::VsaSpec::new(grid, heads, d, sparsity);
        let s = spec.plan.seq;
        // bf16-representable inputs: CUDA rounds uploads to bf16, and with
        // random q / k the pooled tile scores sit so close together that the
        // rounding alone flips top-k picks (the CPU oracle on bf16-rounded
        // inputs moves 0.41 / 0.21 from its f32 self), which the backend
        // would be charged for. Both sides get the same rounded values.
        let (q, k, v, g) = (
            bf(rng.normal(heads * s * d)),
            bf(rng.normal(heads * s * d)),
            bf(rng.normal(heads * s * d)),
            bf(rng.normal(heads * s * d)),
        );
        let (gq, gk, gv, gg, go) = (
            be.upload(&q),
            be.upload(&k),
            be.upload(&v),
            be.upload(&g),
            be.alloc(s * heads * d),
        );
        let st = be.vsa_prepare(&spec);
        be.vsa_attention(&gq, &gk, &gv, &gg, &st, &go);
        let (cq, ck, cv, cg, co) = (
            cpu.upload(&q),
            cpu.upload(&k),
            cpu.upload(&v),
            cpu.upload(&g),
            cpu.alloc(s * heads * d),
        );
        let cs = cpu.vsa_prepare(&spec);
        cpu.vsa_attention(&cq, &ck, &cv, &cg, &cs, &co);
        let want = cpu.download(&co);
        p.record(
            format!(
                "vsa {grid:?} h{heads} d{d} topk {}/{} (model grid)",
                spec.topk,
                spec.plan.num_tiles()
            ),
            p.dl(go),
            want,
        );
    }
}

fn ops<B: Backend>(p: &mut Pair<'_, B>, rng: &mut Rng) {
    let (be, cpu) = (p.be, &Cpu);
    // linear: odd M/N/K, row offsets, both activations.
    for &(m, k, n, act) in &[
        (1, 8, 4, Act::None),
        (37, 70, 33, Act::None),
        (130, 96, 200, Act::GeluTanh),
        (65, 256, 64, Act::None),
        (300, 96, 260, Act::GeluTanh),
        (129, 1536, 130, Act::None),
        (1100, 1536, 200, Act::GeluTanh),
    ] {
        let x = rng.normal((m + 3) * k);
        let w = rng.uniform(n * k, 0.2);
        let b = rng.normal(n);
        let run = |x0: &[f32]| -> (B::Buf, Vec<f32>) {
            let gx = be.upload(x0);
            let gw = be.weight(&w, n, k);
            let gb = be.upload(&b);
            let gy = be.alloc((m + 2) * n);
            be.linear(&gx, 3, m, &gw, Some(&gb), act, &gy, 2);
            let cx = cpu.upload(x0);
            let cw = cpu.weight(&w, n, k);
            let cb = cpu.upload(&b);
            let cy = cpu.alloc((m + 2) * n);
            cpu.linear(&cx, 3, m, &cw, Some(&cb), act, &cy, 2);
            (gy, cpu.download(&cy))
        };
        let (g, c) = run(&x);
        p.record(format!("linear {m}x{k}x{n} {act:?}"), p.dl(g), c);
        if k % 2 == 0 && m > 1 {
            // int8 weight-only: grouped when k is a multiple of 128, per row
            // otherwise; the CPU dequantizes, so this checks the kernels'
            // unpacking and scaling, not the quantization error.
            let q = fastwan_core::quant::Q8::quantize(&w, n, k);
            let gx = be.upload(&x);
            let gb = be.upload(&b);
            let gy = be.alloc((m + 2) * n);
            be.linear(&gx, 3, m, &be.weight_q8(&q), Some(&gb), act, &gy, 2);
            let cx = cpu.upload(&x);
            let cb = cpu.upload(&b);
            let cy = cpu.alloc((m + 2) * n);
            cpu.linear(&cx, 3, m, &cpu.weight_q8(&q), Some(&cb), act, &cy, 2);
            p.record(
                format!("linear int8/{} {m}x{k}x{n} {act:?}", q.group),
                p.dl(gy),
                cpu.download(&cy),
            );
            // 4-bit: group 64 (per row when k is not a multiple), HQQ on the
            // odd-K case so a fractional zero point is exercised.
            let q = fastwan_core::quant::Q4::quantize(&w, n, k, 64, k % 64 != 0);
            let gy = be.alloc((m + 2) * n);
            be.linear(&gx, 3, m, &be.weight_q4(&q), Some(&gb), act, &gy, 2);
            let cy = cpu.alloc((m + 2) * n);
            cpu.linear(&cx, 3, m, &cpu.weight_q4(&q), Some(&cb), act, &cy, 2);
            p.record(
                format!("linear int4/{} {m}x{k}x{n} {act:?}", q.group),
                p.dl(gy),
                cpu.download(&cy),
            );
            // FP8 W8A8: E4M3 weights per row and the input rows per token; the
            // CPU applies the same rounding (fq8_rows), so this checks the
            // kernels' quantize, decode and scaling, and on CUDA sm_89+ the
            // FP8 tensor-core GEMM.
            let q = fastwan_core::quant::F8::quantize(&w, n, k);
            let gy = be.alloc((m + 2) * n);
            be.linear(&gx, 3, m, &be.weight_fp8(&q), Some(&gb), act, &gy, 2);
            let cy = cpu.alloc((m + 2) * n);
            cpu.linear(&cx, 3, m, &cpu.weight_fp8(&q), Some(&cb), act, &cy, 2);
            p.record(
                format!("linear fp8 {m}x{k}x{n} {act:?}"),
                p.dl(gy),
                cpu.download(&cy),
            );
        }
    }

    // Norms: AdaLN, residual+affine, residual+mod, gated residual.
    let (rows, dim) = (19, 96);
    let h = rng.normal(rows * dim);
    let a = rng.normal(rows * dim);
    let mods = rng.uniform(6 * dim, 0.5);
    let nw = rng
        .uniform(dim, 0.5)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let nb = rng.uniform(dim, 0.5);
    {
        let (gh, gm, go) = (be.upload(&h), be.upload(&mods), be.alloc(rows * dim));
        be.layer_norm(
            &gh,
            rows,
            dim,
            NormKind::Mod { scale: 1, shift: 0 },
            &gm,
            1e-6,
            &go,
        );
        let (ch, cm, co) = (cpu.upload(&h), cpu.upload(&mods), cpu.alloc(rows * dim));
        cpu.layer_norm(
            &ch,
            rows,
            dim,
            NormKind::Mod { scale: 1, shift: 0 },
            &cm,
            1e-6,
            &co,
        );
        let (g, c) = (p.dl(go), cpu.download(&co));
        p.record("layer_norm mod", g, c);
    }
    for gated in [true, false] {
        let gate = gated.then_some(2);
        let (gh, ga, gm, gw, gb, go) = (
            be.upload(&h),
            be.upload(&a),
            be.upload(&mods),
            be.upload(&nw),
            be.upload(&nb),
            be.alloc(rows * dim),
        );
        let (ch, ca, cm, cw, cb, co) = (
            cpu.upload(&h),
            cpu.upload(&a),
            cpu.upload(&mods),
            cpu.upload(&nw),
            cpu.upload(&nb),
            cpu.alloc(rows * dim),
        );
        if gated {
            be.residual_norm(
                &gh,
                &ga,
                rows,
                dim,
                &gm,
                gate,
                NormKind::Affine { w: &gw, b: &gb },
                1e-6,
                &go,
            );
            cpu.residual_norm(
                &ch,
                &ca,
                rows,
                dim,
                &cm,
                gate,
                NormKind::Affine { w: &cw, b: &cb },
                1e-6,
                &co,
            );
        } else {
            be.residual_norm(
                &gh,
                &ga,
                rows,
                dim,
                &gm,
                gate,
                NormKind::Mod { scale: 4, shift: 3 },
                1e-6,
                &go,
            );
            cpu.residual_norm(
                &ch,
                &ca,
                rows,
                dim,
                &cm,
                gate,
                NormKind::Mod { scale: 4, shift: 3 },
                1e-6,
                &co,
            );
        }
        let name = if gated {
            "residual_norm gate+affine"
        } else {
            "residual_norm add+mod"
        };
        let (g1, c1, g2, c2) = (p.dl(gh), cpu.download(&ch), p.dl(go), cpu.download(&co));
        p.record(format!("{name} (h)"), g1, c1);
        p.record(format!("{name} (normed)"), g2, c2);
    }
    {
        let (gh, ga, gm) = (be.upload(&h), be.upload(&a), be.upload(&mods));
        be.gate_residual(&gh, &ga, rows, dim, &gm, 5);
        let (ch, ca, cm) = (cpu.upload(&h), cpu.upload(&a), cpu.upload(&mods));
        cpu.gate_residual(&ch, &ca, rows, dim, &cm, 5);
        let (g, c) = (p.dl(gh), cpu.download(&ch));
        p.record("gate_residual", g, c);
    }

    // head_split with RMS + RoPE, and plain.
    let (heads, hd, s) = (3, 12, 23);
    let stride = 3 * heads * hd + 5;
    let src = rng.normal(s * stride);
    let w = rng
        .uniform(heads * hd, 0.3)
        .iter()
        .map(|v| v + 1.0)
        .collect::<Vec<_>>();
    let ang = rng.uniform(s * hd / 2, 3.0);
    let (cs, sn): (Vec<f32>, Vec<f32>) = ang.iter().map(|a| (a.cos(), a.sin())).unzip();
    for (name, norm, rope) in [
        ("head_split rms+rope", true, true),
        ("head_split plain", false, false),
    ] {
        let (gs, gw, gc, gn, go) = (
            be.upload(&src),
            be.upload(&w),
            be.upload_table(&cs),
            be.upload_table(&sn),
            be.alloc(s * heads * hd),
        );
        let (xs, xw, xc, xn, xo) = (
            cpu.upload(&src),
            cpu.upload(&w),
            cpu.upload(&cs),
            cpu.upload(&sn),
            cpu.alloc(s * heads * hd),
        );
        be.head_split(
            &gs,
            s,
            stride,
            7,
            heads,
            hd,
            norm.then_some(&gw),
            rope.then_some((&gc, &gn)),
            1e-6,
            &go,
        );
        cpu.head_split(
            &xs,
            s,
            stride,
            7,
            heads,
            hd,
            norm.then_some(&xw),
            rope.then_some((&xc, &xn)),
            1e-6,
            &xo,
        );
        let (g, c) = (p.dl(go), cpu.download(&xo));
        p.record(name, g, c);
    }

    // attention: self (sq = sk), cross (sk ≠ sq), short sequences, d = 128.
    for &(heads, sq, sk, d) in &[
        (2, 5, 5, 16),
        (3, 70, 70, 64),
        (2, 100, 13, 128),
        (1, 33, 129, 128),
        (12, 200, 200, 128),
        (3, 130, 512, 128),
    ] {
        let q = rng.normal(heads * sq * d);
        let k = rng.normal(heads * sk * d);
        let v = rng.normal(heads * sk * d);
        let (gq, gk, gv, go) = (
            be.upload(&q),
            be.upload(&k),
            be.upload(&v),
            be.alloc(sq * heads * d),
        );
        be.attention(&gq, &gk, &gv, heads, sq, sk, d, &go);
        let (cq, ck, cv, co) = (
            cpu.upload(&q),
            cpu.upload(&k),
            cpu.upload(&v),
            cpu.alloc(sq * heads * d),
        );
        cpu.attention(&cq, &ck, &cv, heads, sq, sk, d, &co);
        let (g, c) = (p.dl(go), cpu.download(&co));
        p.record(format!("attention h{heads} {sq}x{sk} d{d}"), g, c);
    }

    // K/V cache: stores in two runs (one landing mid-cache), attention over
    // the valid prefix of a larger cache read in place, f32 and 16-bit.
    for &(heads, sq, valid, cap, d, half) in &[
        (3, 70, 130, 200, 128, false),
        (3, 70, 130, 200, 128, true),
        (2, 33, 64, 96, 64, true),
        (12, 100, 300, 300, 128, true),
    ] {
        let (strided, can_half) = be.kv_caps(d);
        let half = half && can_half;
        let cap = if strided { cap } else { valid };
        let n = valid + 7; // source rows; the store takes rows 3 .. 3 + valid
        let (q, k, v) = (
            rng.normal(heads * sq * d),
            rng.normal(heads * n * d),
            rng.normal(heads * n * d),
        );
        let a = valid / 3;
        let (gq, gk, gv, gkc, gvc, go) = (
            be.upload(&q),
            be.upload(&k),
            be.upload(&v),
            be.alloc_kv(heads * cap * d, half),
            be.alloc_kv(heads * cap * d, half),
            be.alloc(sq * heads * d),
        );
        // rows 3 .. 3 + a to cache rows 0 .. a; the rest after them.
        be.kv_store(&gk, n, 3, a, heads, d, &gkc, 0, cap, half);
        be.kv_store(&gk, n, 3 + a, valid - a, heads, d, &gkc, a, cap, half);
        be.kv_store(&gv, n, 3, a, heads, d, &gvc, 0, cap, half);
        be.kv_store(&gv, n, 3 + a, valid - a, heads, d, &gvc, a, cap, half);
        be.attention_kv(&gq, &gkc, &gvc, heads, sq, valid, cap, d, half, &go);
        let (cq, ck, cv, ckc, cvc, co) = (
            cpu.upload(&q),
            cpu.upload(&k),
            cpu.upload(&v),
            cpu.alloc_kv(heads * cap * d, half),
            cpu.alloc_kv(heads * cap * d, half),
            cpu.alloc(sq * heads * d),
        );
        cpu.kv_store(&ck, n, 3, a, heads, d, &ckc, 0, cap, half);
        cpu.kv_store(&ck, n, 3 + a, valid - a, heads, d, &ckc, a, cap, half);
        cpu.kv_store(&cv, n, 3, a, heads, d, &cvc, 0, cap, half);
        cpu.kv_store(&cv, n, 3 + a, valid - a, heads, d, &cvc, a, cap, half);
        cpu.attention_kv(&cq, &ckc, &cvc, heads, sq, valid, cap, d, half, &co);
        let want = cpu.download(&co);
        p.record(
            format!(
                "kv cache h{heads} {sq}x{valid}/{cap} d{d}{}",
                if half { " f16" } else { "" }
            ),
            p.dl(go),
            want,
        );
    }

    // VSA: partial tiles in every axis, two head dims, two sparsities. The
    // inputs are bf16-representable like the model-grid cases': bf16 upload
    // rounding alone moved (5, 6, 9) by 0.29 from its f32 oracle once the
    // RNG draw changed (a top-k near-tie), which CUDA was charged for.
    for &(grid, heads, d, sparsity) in &[
        ((5, 6, 9), 2, 128, 0.75),
        ((4, 8, 8), 3, 16, 0.5),
        ((9, 5, 7), 1, 64, 0.8),
    ] {
        let spec = fastwan_core::vsa::VsaSpec::new(grid, heads, d, sparsity);
        let s = spec.plan.seq;
        let (q, k, v, g) = (
            bf(rng.normal(heads * s * d)),
            bf(rng.normal(heads * s * d)),
            bf(rng.normal(heads * s * d)),
            bf(rng.normal(heads * s * d)),
        );
        let (gq, gk, gv, gg, go) = (
            be.upload(&q),
            be.upload(&k),
            be.upload(&v),
            be.upload(&g),
            be.alloc(s * heads * d),
        );
        let st = be.vsa_prepare(&spec);
        be.vsa_attention(&gq, &gk, &gv, &gg, &st, &go);
        let (cq, ck, cv, cg, co) = (
            cpu.upload(&q),
            cpu.upload(&k),
            cpu.upload(&v),
            cpu.upload(&g),
            cpu.alloc(s * heads * d),
        );
        let cs = cpu.vsa_prepare(&spec);
        cpu.vsa_attention(&cq, &ck, &cv, &cg, &cs, &co);
        let (gout, cout) = (p.dl(go), cpu.download(&co));
        p.record(
            format!(
                "vsa {grid:?} h{heads} d{d} topk {}/{}",
                spec.topk,
                spec.plan.num_tiles()
            ),
            gout,
            cout,
        );
    }

    // patchify / unpatchify / dmd_step / tanh_clamp.
    let (c, t, hh, ww, pp) = (3, 2, 6, 10, 2);
    let lat = rng.normal(c * t * hh * ww);
    {
        let (gl, go) = (be.upload(&lat), be.alloc(lat.len()));
        be.patchify(&gl, c, t, hh, ww, pp, &go);
        let (cl, co) = (cpu.upload(&lat), cpu.alloc(lat.len()));
        cpu.patchify(&cl, c, t, hh, ww, pp, &co);
        let (g, cc) = (p.dl(go), cpu.download(&co));
        p.record("patchify", g, cc);
        let (gu, cu) = (be.alloc(lat.len()), cpu.alloc(lat.len()));
        be.unpatchify(&gl, c, t, hh, ww, pp, &gu);
        cpu.unpatchify(&cl, c, t, hh, ww, pp, &cu);
        let (g, cc) = (p.dl(gu), cpu.download(&cu));
        p.record("unpatchify", g, cc);
    }
    {
        let n = 1001;
        let (x, v, e) = (rng.normal(n), rng.normal(n), rng.normal(n));
        let (gx, gv, ge) = (be.upload(&x), be.upload(&v), be.upload(&e));
        be.dmd_step(&gx, &gv, Some(&ge), 0.757, 0.522, n);
        be.tanh_clamp(&gx, n, 3.0);
        let (cx, cv, ce) = (cpu.upload(&x), cpu.upload(&v), cpu.upload(&e));
        cpu.dmd_step(&cx, &cv, Some(&ce), 0.757, 0.522, n);
        cpu.tanh_clamp(&cx, n, 3.0);
        let (g, c) = (p.dl(gx), cpu.download(&cx));
        p.record("dmd_step + tanh_clamp", g, c);
    }
    // add_act (a LoRA branch's delta before the linear's activation), at
    // offsets into both buffers; its own RNG so the cases after it keep
    // their draws.
    {
        let mut r = Rng::new(77);
        let (ny, nd, n, yo, dof) = (1500, 1300, 1001, 211, 97);
        let (y, d) = (r.normal(ny), r.normal(nd));
        for act in [Act::None, Act::GeluTanh] {
            let (gy, gd) = (be.upload(&y), be.upload(&d));
            be.add_act(&gy, yo, &gd, dof, n, act);
            let (cy, cd) = (cpu.upload(&y), cpu.upload(&d));
            cpu.add_act(&cy, yo, &cd, dof, n, act);
            let (g, c) = (p.dl(gy), cpu.download(&cy));
            p.record(format!("add_act {act:?} at offsets"), g, c);
        }
    }

    // MemRoPE cache ops: rope_heads between strided head layouts, kv_ema
    // (initialising and blending) on a [heads, rows, d] cache.
    {
        let (heads, rows, d, stride) = (3, 37, 24, 50 * 24);
        let src = rng.normal(heads * stride);
        let half = d / 2;
        let ang: Vec<f32> = (0..rows * half)
            .map(|i| (i as f32 * 0.37).sin() * 3.0)
            .collect();
        let (cs, sn): (Vec<f32>, Vec<f32>) = ang.iter().map(|a| (a.cos(), a.sin())).unzip();
        let n_out = heads * rows * d + 5;
        let gs = be.upload(&src);
        let go = be.alloc(n_out);
        let (gc, gn) = (be.upload_table(&cs), be.upload_table(&sn));
        be.rope_heads(
            &gs,
            2 * d,
            stride,
            &go,
            5,
            rows * d,
            heads,
            rows,
            d,
            (&gc, &gn),
        );
        let cs_ = cpu.upload(&src);
        let co = cpu.alloc(n_out);
        let (cc, cn) = (cpu.upload_table(&cs), cpu.upload_table(&sn));
        cpu.rope_heads(
            &cs_,
            2 * d,
            stride,
            &co,
            5,
            rows * d,
            heads,
            rows,
            d,
            (&cc, &cn),
        );
        let (g, c) = (p.dl(go), cpu.download(&co));
        p.record("rope_heads 3x37x24 (strided)", g, c);

        let frame = 6;
        let buf = rng.normal(heads * stride);
        let gb = be.upload(&buf);
        let cb = cpu.upload(&buf);
        // First eviction: both memory frames become the mean of 3 frames;
        // then a blend with the long / short rates.
        be.kv_ema(
            &gb,
            heads,
            stride,
            d,
            5 * frame,
            3 * frame,
            frame,
            frame,
            [1.0, 1.0],
        );
        cpu.kv_ema(
            &cb,
            heads,
            stride,
            d,
            5 * frame,
            3 * frame,
            frame,
            frame,
            [1.0, 1.0],
        );
        be.kv_ema(
            &gb,
            heads,
            stride,
            d,
            3 * frame + 1,
            2 * frame + 3,
            frame,
            frame,
            [0.01, 0.1],
        );
        cpu.kv_ema(
            &cb,
            heads,
            stride,
            d,
            3 * frame + 1,
            2 * frame + 3,
            frame,
            frame,
            [0.01, 0.1],
        );
        let (g, c) = (p.dl(gb), cpu.download(&cb));
        p.record("kv_ema init + blend 3x50x24", g, c);
    }

    // conv2d: 3×3 with concat + upsample + epilogues, 1×1.
    let cases = [
        (
            ConvShape {
                frames: 2,
                cin_a: 5,
                cin_b: 3,
                cout: 7,
                h: 9,
                w: 11,
                upsample: false,
                k: 3,
            },
            0u8,
        ),
        (
            ConvShape {
                frames: 3,
                cin_a: 4,
                cin_b: 0,
                cout: 6,
                h: 5,
                w: 7,
                upsample: true,
                k: 3,
            },
            1,
        ),
        (
            ConvShape {
                frames: 1,
                cin_a: 6,
                cin_b: 0,
                cout: 6,
                h: 8,
                w: 8,
                upsample: false,
                k: 3,
            },
            2,
        ),
        (
            ConvShape {
                frames: 2,
                cin_a: 8,
                cin_b: 0,
                cout: 16,
                h: 6,
                w: 5,
                upsample: false,
                k: 1,
            },
            0,
        ),
        // Tensor-core tiling on CUDA: many K tiles, two N tiles + tail,
        // concat + upsample, M tail.
        (
            ConvShape {
                frames: 2,
                cin_a: 40,
                cin_b: 24,
                cout: 72,
                h: 13,
                w: 17,
                upsample: true,
                k: 3,
            },
            2,
        ),
        (
            ConvShape {
                frames: 3,
                cin_a: 130,
                cin_b: 0,
                cout: 3,
                h: 11,
                w: 9,
                upsample: false,
                k: 1,
            },
            1,
        ),
        // PReLU (the upscaler): slopes indexed past the first N tile.
        (
            ConvShape {
                frames: 2,
                cin_a: 64,
                cin_b: 0,
                cout: 72,
                h: 9,
                w: 10,
                upsample: false,
                k: 3,
            },
            3,
        ),
    ];
    for (s, ep) in cases {
        let a = rng.normal(s.frames * s.cin_a * s.h * s.w);
        let b = rng.normal(s.frames * s.cin_b * s.h * s.w);
        let w = rng.uniform(s.cout * s.k_len(), 0.3);
        let bias = rng.normal(s.cout);
        let r = rng.normal(s.out_len());
        let (ga, gb, gw, gbias, gr, go) = (
            be.upload(&a),
            be.upload(&b),
            be.weight(&w, s.cout, s.k_len()),
            be.upload(&bias),
            be.upload(&r),
            be.alloc(s.out_len()),
        );
        let (ca, cb, cw, cbias, cr, co) = (
            cpu.upload(&a),
            cpu.upload(&b),
            cpu.weight(&w, s.cout, s.k_len()),
            cpu.upload(&bias),
            cpu.upload(&r),
            cpu.alloc(s.out_len()),
        );
        let (gep, cep) = match ep {
            1 => (ConvEpilogue::Relu, ConvEpilogue::Relu),
            2 => (ConvEpilogue::AddRelu(&gr), ConvEpilogue::AddRelu(&cr)),
            3 => (ConvEpilogue::Prelu(&gr), ConvEpilogue::Prelu(&cr)),
            _ => (ConvEpilogue::None, ConvEpilogue::None),
        };
        let has_b = s.cin_b > 0;
        be.conv2d(&ga, has_b.then_some(&gb), &s, &gw, Some(&gbias), gep, &go);
        cpu.conv2d(&ca, has_b.then_some(&cb), &s, &cw, Some(&cbias), cep, &co);
        let (g, c) = (p.dl(go), cpu.download(&co));
        p.record(
            format!(
                "conv2d k{} cin {}+{} up={} ep={ep}",
                s.k, s.cin_a, s.cin_b, s.upsample
            ),
            g,
            c,
        );
    }
}

/// The tiny DiT for 3 DMD steps and the tiny TAEHV, synthetic weights, both backends.
fn model<B: Backend>(p: &mut Pair<'_, B>) {
    let (be, cpu) = (p.be, &Cpu);
    let cfg = WanConfig::tiny();
    let tae = TaeConfig::tiny();
    let src = Synthetic { seed: 3 };
    let g = Geometry {
        width: 64,
        height: 32,
        frames: 9,
    };
    let embeds = Rng::new(5).normal(cfg.text_len * cfg.text_dim);
    let opt = Options {
        geometry: g,
        seed: 11,
        ffn_chunk: 7,
        tae_chunk: 2,
        attention: Some(crate::dit::Attention::Dense),
    };
    let gp = Pipeline {
        dit: WanDit::load(be, cfg.clone(), &src).unwrap(),
        tae: TaeHv::load(be, tae.clone(), &src).unwrap(),
        schedule: DmdSchedule::fastwan_1_3b(),
    };
    let cp = Pipeline {
        dit: WanDit::load(cpu, cfg, &src).unwrap(),
        tae: TaeHv::load(cpu, tae, &src).unwrap(),
        schedule: DmdSchedule::fastwan_1_3b(),
    };
    let (mut t1, mut t2) = (Timings::default(), Timings::default());
    let gl = gp.denoise(be, &embeds, &opt, &mut t1).unwrap();
    let cl = cp.denoise(cpu, &embeds, &opt, &mut t2).unwrap();
    p.record("tiny DiT, 3 DMD steps (latent)", gl.clone(), cl.clone());
    // Decode the *same* latent on both (float frames, not 8-bit: a one-level
    // rounding flip is not an error), so decoder error is not compounded.
    let (lt, lh, lw) = g.latent();
    let ch = gp.tae.cfg.latent_channels;
    let plane = lh * lw;
    let mut z = Vec::with_capacity(cl.len());
    for f in 0..lt {
        for c in 0..ch {
            z.extend_from_slice(&cl[(c * lt + f) * plane..][..plane]);
        }
    }
    let gf = gp
        .tae
        .decode_chunk(be, &mut gp.tae.state(), &z, lt, lh, lw)
        .data;
    let cf = cp
        .tae
        .decode_chunk(cpu, &mut cp.tae.state(), &z, lt, lh, lw)
        .data;
    p.record("tiny TAEHV decode (frames)", gf, cf);
}

/// The tiny DiT's block linears with every module name an adapter may use:
/// one adapter in Wan's names (LongLive-Plug's few-step file), one in PEFT /
/// Diffusers names with a per-module alpha (its CFG file), written once per
/// process.
fn tiny_adapters(cfg: &WanConfig) -> Vec<fastwan_core::lora::Adapter> {
    use fastwan_core::export::{f32_bytes, write};
    let shapes: std::collections::HashMap<String, Vec<usize>> =
        fastwan_core::weights::wan_manifest(cfg)
            .into_iter()
            .collect();
    let dir = std::env::temp_dir().join(format!("fw-check-lora-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wan = [
        ("self_attn.q", "attn1.to_q"),
        ("self_attn.k", "attn1.to_k"),
        ("self_attn.v", "attn1.to_v"),
        ("self_attn.o", "attn1.to_out.0"),
        ("cross_attn.q", "attn2.to_q"),
        ("cross_attn.k", "attn2.to_k"),
        ("cross_attn.v", "attn2.to_v"),
        ("cross_attn.o", "attn2.to_out.0"),
        ("ffn.0", "ffn.net.0.proj"),
        ("ffn.2", "ffn.net.2"),
    ];
    let mut out = Vec::new();
    for (k, (rank, strength, wan_names)) in [(3usize, 1.0f32, true), (2, 0.5, false)]
        .into_iter()
        .enumerate()
    {
        let path = dir.join(format!("a{k}.safetensors"));
        if !path.is_file() {
            let mut r = Rng::new(100 + k as u64);
            let mut t = Vec::new();
            for l in 0..cfg.layers {
                for (w, d) in wan {
                    let sh = &shapes[&format!("blocks.{l}.{d}.weight")];
                    let (o, i) = (sh[0], sh[1]);
                    let name = if wan_names {
                        format!("blocks.{l}.{w}")
                    } else {
                        format!("base_model.model.blocks.{l}.{d}")
                    };
                    t.push((
                        format!("{name}.lora_A.weight"),
                        fastwan_core::export::Dtype::F32,
                        vec![rank, i],
                        f32_bytes(&r.uniform(rank * i, 0.3)),
                    ));
                    t.push((
                        format!("{name}.lora_B.weight"),
                        fastwan_core::export::Dtype::F32,
                        vec![o, rank],
                        f32_bytes(&r.uniform(o * rank, 0.3)),
                    ));
                    if !wan_names {
                        t.push((
                            format!("{name}.alpha"),
                            fastwan_core::export::Dtype::F32,
                            vec![1],
                            f32_bytes(&[4.0]),
                        ));
                    }
                }
            }
            write(&path, &t, Default::default()).unwrap();
        }
        out.push(fastwan_core::lora::Adapter::open(&path, strength).unwrap());
    }
    out
}

/// A weight source with the adapters merged into the block linears in f32:
/// what the unmerged branch means.
struct Merged<'a> {
    inner: &'a dyn WeightSource,
    adapters: Vec<fastwan_core::lora::Adapter>,
}

impl WeightSource for Merged<'_> {
    fn get(&self, name: &str, shape: &[usize]) -> fastwan_core::Result<Vec<f32>> {
        let mut w = self.inner.get(name, shape)?;
        if let Some(m) = name.strip_suffix(".weight") {
            let (o, i) = (shape[0], shape[1..].iter().product::<usize>());
            for a in &self.adapters {
                if let Some((fa, fb, r)) = a.factors(m, o, i)? {
                    for row in 0..o {
                        for col in 0..i {
                            w[row * i + col] += (0..r)
                                .map(|k| fb[row * r + k] * fa[k * i + col])
                                .sum::<f32>();
                        }
                    }
                }
            }
        }
        Ok(w)
    }

    fn contains(&self, name: &str) -> bool {
        self.inner.contains(name)
    }
}

/// Unmerged LoRA branches on the tiny DiT (two stacked adapters, a fused QKV
/// and KV, the FFN's GELU after the delta): the backend against the CPU, and
/// on the CPU the branch against merging both in f32.
fn model_lora<B: Backend>(p: &mut Pair<'_, B>) {
    let (be, cpu) = (p.be, &Cpu);
    let cfg = WanConfig::tiny();
    let src = Synthetic { seed: 3 };
    let g = Geometry {
        width: 64,
        height: 32,
        frames: 9,
    };
    let embeds = Rng::new(5).normal(cfg.text_len * cfg.text_dim);
    let opt = Options {
        geometry: g,
        seed: 11,
        ffn_chunk: 7,
        tae_chunk: 2,
        attention: Some(crate::dit::Attention::Dense),
    };
    let tae = TaeConfig::tiny();
    let run = |dit: WanDit<Cpu>| {
        let pl = Pipeline {
            dit,
            tae: TaeHv::load(cpu, tae.clone(), &src).unwrap(),
            schedule: DmdSchedule::fastwan_1_3b(),
        };
        pl.denoise(cpu, &embeds, &opt, &mut Timings::default())
            .unwrap()
    };
    let gp = Pipeline {
        dit: WanDit::load_with_adapters(
            be,
            cfg.clone(),
            &src,
            WeightQuant::Native,
            tiny_adapters(&cfg),
        )
        .unwrap(),
        tae: TaeHv::load(be, tae.clone(), &src).unwrap(),
        schedule: DmdSchedule::fastwan_1_3b(),
    };
    let got = gp
        .denoise(be, &embeds, &opt, &mut Timings::default())
        .unwrap();
    let branch = run(WanDit::load_with_adapters(
        cpu,
        cfg.clone(),
        &src,
        WeightQuant::Native,
        tiny_adapters(&cfg),
    )
    .unwrap());
    p.record(
        "tiny DiT + two stacked LoRA branches, 3 DMD steps (latent)",
        got,
        branch.clone(),
    );
    let merged_src = Merged {
        inner: &src,
        adapters: tiny_adapters(&cfg),
    };
    let merged = run(WanDit::load(cpu, cfg.clone(), &merged_src).unwrap());
    let plain = run(WanDit::load(cpu, cfg.clone(), &src).unwrap());
    let e = rel(&branch, &merged);
    p.out.push(CheckResult {
        name: "tiny DiT LoRA branches == merged weights (CPU, f32)".into(),
        rel_err: e,
        ok: e <= 1e-4,
    });
    let moved = rel(&branch, &plain);
    p.out.push(CheckResult {
        name: "tiny DiT LoRA changes the latent (CPU)".into(),
        rel_err: moved,
        ok: moved > 1e-2,
    });
}

/// `unpack_rgba8` (8-bit RGBA words → planar RGB in [0, 1], the upscaler's
/// input): exact up to the backend's activation precision.
fn unpack<B: Backend>(p: &mut Pair<'_, B>, rng: &mut Rng) {
    let (be, cpu) = (p.be, &Cpu);
    let (frames, plane) = (3, 7 * 11);
    let words: Vec<u32> = rng
        .uniform(frames * plane * 3, 1.0)
        .chunks_exact(3)
        .map(|c| {
            let q = |v: f32| ((v + 1.0) * 127.5) as u32;
            q(c[0]) | q(c[1]) << 8 | q(c[2]) << 16 | 0xff00_0000
        })
        .collect();
    let (gs, cs) = (be.upload_words(&words), cpu.upload_words(&words));
    let (go, co) = (be.alloc(frames * 3 * plane), cpu.alloc(frames * 3 * plane));
    be.unpack_rgba8(&gs, frames, plane, &go);
    cpu.unpack_rgba8(&cs, frames, plane, &co);
    p.record("unpack_rgba8 3 frames", p.dl(go), cpu.download(&co));
    // A frame range of packed words copied out (the upscaler's batches).
    let (gw, cw) = (be.alloc_words(2 * plane), cpu.alloc_words(2 * plane));
    be.copy(&gs, plane, &gw, 0, 2 * plane);
    cpu.copy(&cs, plane, &cw, 0, 2 * plane);
    p.record_rgba8(
        "copy of packed words (frames 1..3)",
        p.dl(gw),
        cpu.download(&cw),
    );
}

/// `pack_rgba8` (decoder / upscaler output → 8-bit RGBA), with and without
/// the residual, values straddling [0, 1]; compared as 8-bit levels.
fn pack<B: Backend>(p: &mut Pair<'_, B>, rng: &mut Rng) {
    use fastwan_core::backend::PackShape;
    let (be, cpu) = (p.be, &Cpu);
    for (s, with_base) in [
        (
            PackShape {
                frames: 3,
                skip: 1,
                h: 5,
                w: 7,
                p: 2,
                down: 1,
            },
            false,
        ),
        (
            PackShape {
                frames: 2,
                skip: 0,
                h: 6,
                w: 9,
                p: 4,
                down: 1,
            },
            true,
        ),
        (
            PackShape {
                frames: 2,
                skip: 1,
                h: 5,
                w: 6,
                p: 4,
                down: 2,
            },
            true,
        ),
    ] {
        let total = s.skip + s.frames;
        let src: Vec<f32> = rng
            .uniform(total * 3 * s.p * s.p * s.h * s.w, 0.7)
            .into_iter()
            .map(|v| v + 0.4)
            .collect();
        let base: Vec<u32> = rng
            .uniform(total * s.h * s.w * 3, 1.0)
            .chunks_exact(3)
            .map(|c| {
                let q = |v: f32| ((v + 1.0) * 127.5) as u32;
                q(c[0]) | q(c[1]) << 8 | q(c[2]) << 16
            })
            .collect();
        let (gs, gb, go) = (
            be.upload(&src),
            be.upload_words(&base),
            be.alloc_words(s.pixels()),
        );
        let (cs, cb, co) = (
            cpu.upload(&src),
            cpu.upload_words(&base),
            cpu.alloc_words(s.pixels()),
        );
        be.pack_rgba8(&gs, with_base.then_some(&gb), &s, &go);
        cpu.pack_rgba8(&cs, with_base.then_some(&cb), &s, &co);
        p.record_rgba8(
            format!(
                "pack_rgba8 p{} down {} skip {} base {with_base} (≤1 level)",
                s.p, s.down, s.skip
            ),
            p.dl(go),
            cpu.download(&co),
        );
    }
}

/// The causal rollout end to end on the tiny model: four 2-frame blocks
/// through a 5-frame cache with a 1-frame sink (the third and fourth blocks
/// evict), four DMD steps each plus the clean rewrite; the GPU's latents vs
/// the CPU's, block by block (the cache carries any error forward).
fn causal<B: Backend>(p: &mut Pair<'_, B>) {
    causal_with(p, WanConfig::tiny(), false);
    // head_dim 32: the cache is read in place by the flash kernel (where the
    // backend can) and stored as 16-bit values.
    causal_with(p, WanConfig::tiny_vsa(), true);
}

fn causal_with<B: Backend>(p: &mut Pair<'_, B>, cfg: WanConfig, half_kv: bool) {
    use crate::causal::{CausalConfig, CausalRollout};
    let (be, cpu) = (p.be, &Cpu);
    let src = Synthetic { seed: 3 };
    let embeds = Rng::new(5).normal(cfg.text_len * cfg.text_dim);
    let ccfg = CausalConfig {
        block_frames: 2,
        cap_frames: 5,
        sink_frames: 1,
        timesteps: CausalConfig::self_forcing().timesteps,
        half_kv,
        memory: crate::causal::Memory::Window,
    };
    let d = cfg.head_dim;
    let gd = WanDit::load(be, cfg.clone(), &src).unwrap();
    let cd = WanDit::load(cpu, cfg, &src).unwrap();
    let (gpr, cpr) = (
        gd.prepare_prompt(be, &embeds),
        cd.prepare_prompt(cpu, &embeds),
    );
    let mut g = CausalRollout::new(be, &gd, ccfg.clone(), (4, 6), 7, 21).unwrap();
    let mut c = CausalRollout::new(cpu, &cd, ccfg, (4, 6), 7, 21).unwrap();
    let (mut gl, mut cl) = (Vec::new(), Vec::new());
    for _ in 0..4 {
        let x = g.next_block(be, &gpr).unwrap();
        gl.extend(be.download(x));
        let x = c.next_block(cpu, &cpr).unwrap();
        cl.extend(cpu.download(x));
    }
    p.record(
        format!(
            "tiny causal rollout d{d}{}, 4 blocks, rolling cache (latent)",
            if half_kv { " f16 cache" } else { "" }
        ),
        gl,
        cl,
    );
}

/// The MemRoPE rollout (History Forcing's memory) on the tiny model: six
/// 2-frame blocks through a 7-frame cache (1 sink, 2 EMA, 2 recent, the
/// block): the fourth block evicts one frame and initialises the memory,
/// the fifth and sixth evict two and blend it. GPU latents vs the CPU's.
fn causal_memrope<B: Backend>(p: &mut Pair<'_, B>) {
    causal_memrope_with(p, WanConfig::tiny(), false);
    // head_dim 32: attention reads the strided cache in place where the
    // backend can; a 16-bit cache request is ignored with MemRoPE.
    causal_memrope_with(p, WanConfig::tiny_vsa(), true);
}

fn causal_memrope_with<B: Backend>(p: &mut Pair<'_, B>, cfg: WanConfig, half_kv: bool) {
    use crate::causal::{CausalConfig, CausalRollout, Memory};
    let (be, cpu) = (p.be, &Cpu);
    let d = cfg.head_dim;
    let src = Synthetic { seed: 3 };
    let embeds = Rng::new(5).normal(cfg.text_len * cfg.text_dim);
    let ccfg = CausalConfig {
        block_frames: 2,
        cap_frames: 7,
        sink_frames: 1,
        timesteps: CausalConfig::self_forcing().timesteps,
        half_kv,
        memory: Memory::MemRope {
            alpha_long: 0.01,
            alpha_short: 0.1,
        },
    };
    let gd = WanDit::load(be, cfg.clone(), &src).unwrap();
    let cd = WanDit::load(cpu, cfg, &src).unwrap();
    let (gpr, cpr) = (
        gd.prepare_prompt(be, &embeds),
        cd.prepare_prompt(cpu, &embeds),
    );
    let mut g = CausalRollout::new(be, &gd, ccfg.clone(), (4, 6), 7, 21).unwrap();
    let mut c = CausalRollout::new(cpu, &cd, ccfg, (4, 6), 7, 21).unwrap();
    let (mut gl, mut cl) = (Vec::new(), Vec::new());
    for _ in 0..6 {
        let x = g.next_block(be, &gpr).unwrap();
        gl.extend(be.download(x));
        let x = c.next_block(cpu, &cpr).unwrap();
        cl.extend(cpu.download(x));
    }
    p.record(
        format!("tiny MemRoPE rollout d{d}, 6 blocks, EMA memory (latent)"),
        gl,
        cl,
    );
}

/// The tiny ×2 upscaler on two noise frames (raw network output, before the
/// host-side shuffle), both backends.
fn upscaler<B: Backend>(p: &mut Pair<'_, B>) {
    use crate::upscale::{UpConfig, Upscaler};
    let (be, cpu) = (p.be, &Cpu);
    let cfg = UpConfig::tiny();
    let src = Synthetic { seed: 11 };
    let (frames, h, w) = (2, 9, 13);
    let mut rng = Rng::new(4);
    let rgb: Vec<u8> = rng
        .uniform(frames * h * w * 3, 1.0)
        .into_iter()
        .map(|v| ((v + 1.0) * 127.5) as u8)
        .collect();
    let g = Upscaler::load(be, cfg, &src).expect("synthetic upscaler");
    let c = Upscaler::load(cpu, cfg, &src).expect("synthetic upscaler");
    let gc = g.launch(be, &rgb, frames, h, w);
    let cc = c.launch(cpu, &rgb, frames, h, w);
    // `out` is packed RGBA: within one 8-bit level (the convs alone are the
    // `conv2d … ep=3` case at f32 precision).
    p.record_rgba8(
        "tiny upscaler, packed RGBA (≤1 level)",
        be.download(&gc.out),
        cpu.download(&cc.out),
    );
    // Output scale below the network's: the box filter in pack_rgba8.
    let (mut g, mut c) = (g, c);
    g.set_output_scale(1).expect("÷2");
    c.set_output_scale(1).expect("÷2");
    let gc = g.launch(be, &rgb, frames, h, w);
    let cc = c.launch(cpu, &rgb, frames, h, w);
    p.record_rgba8(
        "tiny upscaler box-filtered to ×1, packed RGBA (≤1 level)",
        be.download(&gc.out),
        cpu.download(&cc.out),
    );
}

/// The VSA path end to end: a tiny head_dim-32 DiT, 3 DMD steps, 4 tiles at
/// sparsity 0.5, on both backends.
fn model_vsa<B: Backend>(p: &mut Pair<'_, B>, quant: WeightQuant) {
    let (be, cpu) = (p.be, &Cpu);
    let cfg = WanConfig::tiny_vsa();
    let src = Synthetic { seed: 6 };
    let g = Geometry {
        width: 128,
        height: 64,
        frames: 17,
    };
    let embeds = Rng::new(8).normal(cfg.text_len * cfg.text_dim);
    let opt = Options {
        geometry: g,
        seed: 12,
        ffn_chunk: usize::MAX,
        tae_chunk: 2,
        attention: Some(crate::dit::Attention::Vsa { sparsity: 0.5 }),
    };
    let mk = |_: &WanConfig| DmdSchedule::fastwan_1_3b();
    let gp = Pipeline {
        dit: WanDit::load_with(be, cfg.clone(), &src, quant).unwrap(),
        tae: TaeHv::load(be, TaeConfig::tiny(), &src).unwrap(),
        schedule: mk(&cfg),
    };
    let cp = Pipeline {
        dit: WanDit::load_with(cpu, cfg.clone(), &src, quant).unwrap(),
        tae: TaeHv::load(cpu, TaeConfig::tiny(), &src).unwrap(),
        schedule: mk(&cfg),
    };
    let (mut t1, mut t2) = (Timings::default(), Timings::default());
    let gl = gp.denoise(be, &embeds, &opt, &mut t1).unwrap();
    let cl = cp.denoise(cpu, &embeds, &opt, &mut t2).unwrap();
    let tag = match quant {
        WeightQuant::Native => String::new(),
        q => format!(", {q}"),
    };
    p.record(
        format!("tiny DiT, VSA 4 tiles, 3 DMD steps{tag} (latent)"),
        gl,
        cl,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_matches_itself() {
        for r in run_all(&Cpu, 0.0) {
            assert!(r.ok, "{r:?}");
        }
    }

    /// The pipelined decode (`launch_chunk` → `rgb8`) gives exactly the
    /// bytes of the chunk-by-chunk f32 path, trimming and chunking included.
    #[test]
    fn pipelined_rgb8_decode_equals_f32_path() {
        let cpu = Cpu;
        let src = Synthetic { seed: 9 };
        let cfg = WanConfig::tiny();
        let pipe = Pipeline {
            dit: WanDit::load(&cpu, cfg.clone(), &src).unwrap(),
            tae: TaeHv::load(&cpu, TaeConfig::tiny(), &src).unwrap(),
            schedule: DmdSchedule::fastwan_1_3b(),
        };
        let g = Geometry {
            width: 64,
            height: 32,
            frames: 17,
        };
        let (lt, lh, lw) = g.latent();
        let c = pipe.tae.cfg.latent_channels;
        let latent = Rng::new(4).normal(c * lt * lh * lw);
        let mut t = Timings::default();
        let video = pipe.decode(&cpu, &latent, &g, 2, &mut t);
        // Reference: decode_chunk per chunk, then the old f32 → u8 loop.
        let mut st = pipe.tae.state();
        let mut want = Vec::new();
        let plane = lh * lw;
        for f0 in (0..lt).step_by(2) {
            let nfr = 2.min(lt - f0);
            let mut z = Vec::new();
            for f in f0..f0 + nfr {
                for ch in 0..c {
                    z.extend_from_slice(&latent[(ch * lt + f) * plane..][..plane]);
                }
            }
            let fr = pipe.tae.decode_chunk(&cpu, &mut st, &z, nfr, lh, lw);
            let px = fr.width * fr.height;
            for f in 0..fr.frames {
                for i in 0..px {
                    for ch in 0..3 {
                        let v = fr.data[(f * 3 + ch) * px + i];
                        want.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                    }
                }
            }
        }
        assert_eq!(video.frames, g.frames);
        assert_eq!(video.rgb, want);
    }

    #[test]
    fn chunked_decode_equals_one_shot() {
        let tae = TaeConfig::tiny();
        let src = Synthetic { seed: 9 };
        let cpu = Cpu;
        let t = TaeHv::load(&cpu, tae.clone(), &src).unwrap();
        let (lt, h, w) = (5, 3, 4);
        let z = Rng::new(1).normal(lt * tae.latent_channels * h * w);
        let mut s1 = t.state();
        let whole = t.decode_chunk(&cpu, &mut s1, &z, lt, h, w);
        let mut s2 = t.state();
        let per = tae.latent_channels * h * w;
        let mut parts = Vec::new();
        for (f0, n) in [(0, 2), (2, 1), (3, 2)] {
            parts.extend(
                t.decode_chunk(&cpu, &mut s2, &z[f0 * per..(f0 + n) * per], n, h, w)
                    .data,
            );
        }
        assert_eq!(whole.frames, 4 * lt - 3);
        assert!(rel(&parts, &whole.data) < 1e-6);
    }

    /// A quantized checkpoint written by `fastwan quantize` loads to the same
    /// model as `--weights` on the original (up to the f16 storage of the
    /// unquantized tensors).
    #[test]
    fn stored_quantized_checkpoint_matches_load_time_quantization() {
        use fastwan_core::{
            export,
            quant::WeightQuant,
            weights::{SafetensorsDir, WeightSource, wan_manifest},
        };
        let cfg = WanConfig::tiny_vsa();
        let syn = Synthetic { seed: 4 };
        let mut names = wan_manifest(&cfg);
        for l in 0..cfg.layers {
            let k = format!("blocks.{l}.to_gate_compress");
            names.push((format!("{k}.weight"), vec![cfg.dim(), cfg.dim()]));
            names.push((format!("{k}.bias"), vec![cfg.dim()]));
        }
        let t: Vec<export::Tensor> = names
            .iter()
            .map(|(n, s)| {
                let v = syn.get(n, s).unwrap();
                (
                    n.clone(),
                    fastwan_core::export::Dtype::F32,
                    s.clone(),
                    export::f32_bytes(&v),
                )
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("fastwan-qload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (fsrc, fq) = (dir.join("src.safetensors"), dir.join("q.safetensors"));
        export::write(&fsrc, &t, Default::default()).unwrap();
        let src = SafetensorsDir::open(&fsrc).unwrap();
        let quant: WeightQuant = "int4-hqq-g16-ffn".parse().unwrap();
        export::write(
            &fq,
            &export::quantize_dit(&src, &cfg, quant).unwrap(),
            export::metadata(quant),
        )
        .unwrap();
        let stored = SafetensorsDir::open(&fq).unwrap();
        let cpu = Cpu;
        let run = |dit: &WanDit<Cpu>| {
            let prompt = dit.prepare_prompt(&cpu, &Rng::new(2).normal(cfg.text_len * cfg.text_dim));
            let mods = dit.step_mods(&cpu, 757.0);
            let lat = (2, 4, 6);
            let x = cpu.upload(&Rng::new(3).normal(cfg.in_channels * 2 * 4 * 6));
            let ws = dit.workspace(&cpu, lat, usize::MAX, None);
            let out = cpu.alloc(cfg.in_channels * 2 * 4 * 6);
            dit.forward(&cpu, &ws, &x, &mods, &prompt, &out);
            cpu.download(&out)
        };
        let a = run(&WanDit::load_with(&cpu, cfg.clone(), &src, quant).unwrap());
        // The stored file decides its own format: `Native` here is ignored
        // for the block linears.
        let b = run(&WanDit::load_with(&cpu, cfg.clone(), &stored, WeightQuant::Native).unwrap());
        let native = run(&WanDit::load(&cpu, cfg.clone(), &src).unwrap());
        std::fs::remove_dir_all(&dir).ok();
        let e = rel(&b, &a);
        assert!(e < 1e-3, "stored vs load-time: {e}");
        assert!(
            rel(&a, &native) > 10.0 * e,
            "quantization should dominate f16 storage"
        );
    }

    #[test]
    fn ffn_chunking_does_not_change_the_output() {
        let cfg = WanConfig::tiny();
        let src = Synthetic { seed: 4 };
        let cpu = Cpu;
        let dit = WanDit::load(&cpu, cfg.clone(), &src).unwrap();
        let prompt = dit.prepare_prompt(&cpu, &Rng::new(2).normal(cfg.text_len * cfg.text_dim));
        let mods = dit.step_mods(&cpu, 757.0);
        let lat = (2, 4, 6);
        let x = cpu.upload(&Rng::new(3).normal(cfg.in_channels * 2 * 4 * 6));
        let run = |chunk| {
            let ws = dit.workspace(&cpu, lat, chunk, None);
            let out = cpu.alloc(cfg.in_channels * 2 * 4 * 6);
            dit.forward(&cpu, &ws, &x, &mods, &prompt, &out);
            cpu.download(&out)
        };
        assert!(rel(&run(5), &run(usize::MAX)) < 1e-6);
    }
}
