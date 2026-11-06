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
    backend::{Act, Backend, ConvEpilogue, ConvShape, NormKind},
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
}

impl<'a, B: Backend> Pair<'a, B> {
    fn record(&mut self, name: impl Into<String>, got: Vec<f32>, want: Vec<f32>) {
        let e = rel(&got, &want);
        self.out.push(CheckResult {
            name: name.into(),
            rel_err: e,
            ok: e <= self.tol,
        });
    }
}

pub fn run_all<B: Backend>(be: &B, tol: f32) -> Vec<CheckResult> {
    let mut p = Pair {
        be,
        tol,
        out: Vec::new(),
    };
    let mut rng = Rng::new(42);
    ops(&mut p, &mut rng);
    model(&mut p);
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
        let run = |x0: &[f32]| -> (Vec<f32>, Vec<f32>) {
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
            (be.download(&gy), cpu.download(&cy))
        };
        let (g, c) = run(&x);
        p.record(format!("linear {m}x{k}x{n} {act:?}"), g, c);
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
                be.download(&gy),
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
                be.download(&gy),
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
        let (g, c) = (be.download(&go), cpu.download(&co));
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
        let (g1, c1, g2, c2) = (
            be.download(&gh),
            cpu.download(&ch),
            be.download(&go),
            cpu.download(&co),
        );
        p.record(format!("{name} (h)"), g1, c1);
        p.record(format!("{name} (normed)"), g2, c2);
    }
    {
        let (gh, ga, gm) = (be.upload(&h), be.upload(&a), be.upload(&mods));
        be.gate_residual(&gh, &ga, rows, dim, &gm, 5);
        let (ch, ca, cm) = (cpu.upload(&h), cpu.upload(&a), cpu.upload(&mods));
        cpu.gate_residual(&ch, &ca, rows, dim, &cm, 5);
        let (g, c) = (be.download(&gh), cpu.download(&ch));
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
        let (g, c) = (be.download(&go), cpu.download(&xo));
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
        let (g, c) = (be.download(&go), cpu.download(&co));
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
        let (gout, cout) = (be.download(&go), cpu.download(&co));
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
        let (g, cc) = (be.download(&go), cpu.download(&co));
        p.record("patchify", g, cc);
        let (gu, cu) = (be.alloc(lat.len()), cpu.alloc(lat.len()));
        be.unpatchify(&gl, c, t, hh, ww, pp, &gu);
        cpu.unpatchify(&cl, c, t, hh, ww, pp, &cu);
        let (g, cc) = (be.download(&gu), cpu.download(&cu));
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
        let (g, c) = (be.download(&gx), cpu.download(&cx));
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
            _ => (ConvEpilogue::None, ConvEpilogue::None),
        };
        let has_b = s.cin_b > 0;
        be.conv2d(&ga, has_b.then_some(&gb), &s, &gw, Some(&gbias), gep, &go);
        cpu.conv2d(&ca, has_b.then_some(&cb), &s, &cw, Some(&cbias), cep, &co);
        let (g, c) = (be.download(&go), cpu.download(&co));
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
