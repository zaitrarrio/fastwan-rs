// Small elementwise kernels, one thread per output element, 256 per group.
//   op 0 gate_residual: x[i] += y[i] · aux[gate·dim + i % dim]
//   op 1 dmd_step:      x0 = x − σt·y; x = has_aux ? (1−σn)·x0 + σn·aux : x0
//   op 2 tanh_clamp:    x = tanh(x / s) · s
//   op 3 patchify:      x[tok, f] = y[latent]       (gather)
//   op 4 unpatchify:    x[latent] = y[tok, f]       (gather)
struct P { n: u32, op: u32, dim: u32, gate: u32, has_aux: u32, c: u32, t: u32, hh: u32, ww: u32, pp: u32, s0: f32, s1: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> aux: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    switch p.op {
        case 0u: { x[i] = x[i] + y[i] * aux[p.gate * p.dim + i % p.dim]; }
        case 1u: {
            let x0 = x[i] - p.s0 * y[i];
            if (p.has_aux != 0u) { x[i] = (1.0 - p.s1) * x0 + p.s1 * aux[i]; } else { x[i] = x0; }
        }
        case 2u: { x[i] = tanh_safe(x[i] / p.s0) * p.s0; }
        case 3u: {
            // i = tok·F + f, f = c'·p² + kh·p + kw
            let f = p.c * p.pp * p.pp;
            let tok = i / f;
            let fi = i % f;
            let cc = fi / (p.pp * p.pp);
            let kh = (fi / p.pp) % p.pp;
            let kw = fi % p.pp;
            let gw = p.ww / p.pp;
            let gh = p.hh / p.pp;
            let tt = tok / (gh * gw);
            let yy = (tok / gw) % gh;
            let xg = tok % gw;
            x[i] = y[((cc * p.t + tt) * p.hh + yy * p.pp + kh) * p.ww + xg * p.pp + kw];
        }
        default: {
            // i = ((c'·t + tt)·hh + y)·ww + xx
            let xx = i % p.ww;
            let yy = (i / p.ww) % p.hh;
            let tt = (i / (p.ww * p.hh)) % p.t;
            let cc = i / (p.ww * p.hh * p.t);
            let gw = p.ww / p.pp;
            let gh = p.hh / p.pp;
            let tok = (tt * gh + yy / p.pp) * gw + xx / p.pp;
            let f = p.c * p.pp * p.pp;
            x[i] = y[tok * f + ((yy % p.pp) * p.pp + xx % p.pp) * p.c + cc];
        }
    }
    if (p.op == 9999u) { x[0] = aux[0]; }
}
