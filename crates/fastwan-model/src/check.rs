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
    weights::Synthetic,
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
    pack(&mut p, &mut rng);
    model_vsa(&mut p, WeightQuant::Native);
    model_vsa(&mut p, WeightQuant::Int8);
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
    Deferred {
        done: p.out,
        pending: p.pending,
    }
}

/// VSA at the token grids the browser runs (480×272×81 and 832×480×81 after
/// the 2×2 patch), d = 128 like the model, few heads to keep the CPU
/// reference cheap. The small cases in `ops` never exceed 256 key tiles.
fn vsa_real<B: Backend>(p: &mut Pair<'_, B>, rng: &mut Rng) {
    let (be, cpu) = (p.be, &Cpu);
    for &(grid, heads) in &[((21, 17, 30), 2), ((21, 30, 52), 1)] {
        let (d, sparsity) = (128, 0.8);
        let spec = fastwan_core::vsa::VsaSpec::new(grid, heads, d, sparsity);
        let s = spec.plan.seq;
        let (q, k, v, g) = (
            rng.normal(heads * s * d),
            rng.normal(heads * s * d),
            rng.normal(heads * s * d),
            rng.normal(heads * s * d),
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

    // VSA: partial tiles in every axis, two head dims, two sparsities.
    for &(grid, heads, d, sparsity) in &[
        ((5, 6, 9), 2, 128, 0.75),
        ((4, 8, 8), 3, 16, 0.5),
        ((9, 5, 7), 1, 64, 0.8),
    ] {
        let spec = fastwan_core::vsa::VsaSpec::new(grid, heads, d, sparsity);
        let s = spec.plan.seq;
        let (q, k, v, g) = (
            rng.normal(heads * s * d),
            rng.normal(heads * s * d),
            rng.normal(heads * s * d),
            rng.normal(heads * s * d),
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
                "pack_rgba8 p{} skip {} base {with_base} (≤1 level)",
                s.p, s.skip
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
    use crate::causal::{CausalConfig, CausalRollout};
    let (be, cpu) = (p.be, &Cpu);
    let cfg = WanConfig::tiny();
    let src = Synthetic { seed: 3 };
    let embeds = Rng::new(5).normal(cfg.text_len * cfg.text_dim);
    let ccfg = CausalConfig {
        block_frames: 2,
        cap_frames: 5,
        sink_frames: 1,
        timesteps: CausalConfig::self_forcing().timesteps,
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
    for _ in 0..4 {
        let x = g.next_block(be, &gpr).unwrap();
        gl.extend(be.download(x));
        let x = c.next_block(cpu, &cpr).unwrap();
        cl.extend(cpu.download(x));
    }
    p.record(
        "tiny causal rollout, 4 blocks, rolling cache (latent)",
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
                    safetensors::Dtype::F32,
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
