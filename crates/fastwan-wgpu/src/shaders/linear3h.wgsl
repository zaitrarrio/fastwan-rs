// f16 math variant of linear3.wgsl (needs `shader-f16`; f16 weights only).
// Shared tiles are f16 (half the shared-memory traffic); W comes from its packed f16 pairs. Each K-tile of 16 is summed in
// vec4<f16> (packed f16 FMA: 2× the f32 rate on AMD / Intel / Apple GPUs,
// equal on NVIDIA GeForce) and then flushed into f32 accumulators, so f16
// rounding never compounds past 16 terms.
//
// Original linear3 notes:
// y[y_row + m, n] = act(Σ_k x[x_row + m, k] · W[n, k] + b[n]), third version:
// the 64×64 / 4×4-per-thread kernel of `linear.wgsl` with the register block
// written out by hand — four named vec4 accumulators, no arrays, no loops over
// them. Tests whether the array accumulators were being spilled to local
// memory by the driver's compiler (both 64×64 and 128×128 tiles measured
// ~2 TFLOP/s on an RTX 4090).
struct P { m: u32, n: u32, k: u32, x_row: u32, y_row: u32, has_bias: u32, act: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

// Exact: the packed pair already holds f16 values. (`bitcast<vec2<f16>>` would
// skip the round trip, but naga does not accept it yet.)
fn load_w(i: u32) -> f16 {
    let v = unpack2x16float(w[i >> 1u]);
    return f16(select(v.x, v.y, (i & 1u) == 1u));
}

var<workgroup> xs: array<f16, 1024>;   // [16 k][64 rows]
var<workgroup> ws: array<f16, 1024>;   // [16 k][64 cols]

fn store(m: u32, n: u32, v: f32) {
    if (m >= p.m || n >= p.n) { return; }
    var o = v;
    if (p.has_bias != 0u) { o = o + bias[n]; }
    if (p.act == 1u) { o = gelu_tanh(o); }
    y[(p.y_row + m) * p.n + n] = o;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    var c0 = vec4<f32>(0.0);   // row ty,      cols tx + 16j
    var c1 = vec4<f32>(0.0);   // row ty + 16
    var c2 = vec4<f32>(0.0);   // row ty + 32
    var c3 = vec4<f32>(0.0);   // row ty + 48
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
        for (var t = 0u; t < 4u; t = t + 1u) {
            let e = tid + t * 256u;
            let r = e / 16u;
            let kk = e % 16u;
            let gk = k0 + kk;
            var xv = 0.0h;
            if (m0 + r < p.m && gk < p.k) { xv = f16(x[(p.x_row + m0 + r) * p.k + gk]); }
            xs[kk * 64u + r] = xv;
            var wv = 0.0h;
            if (n0 + r < p.n && gk < p.k) { wv = load_w((n0 + r) * p.k + gk); }
            ws[kk * 64u + r] = wv;
        }
        workgroupBarrier();
        var h0 = vec4<f16>(0.0h);
        var h1 = vec4<f16>(0.0h);
        var h2 = vec4<f16>(0.0h);
        var h3 = vec4<f16>(0.0h);
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            let b = vec4<f16>(ws[o + tx], ws[o + tx + 16u], ws[o + tx + 32u], ws[o + tx + 48u]);
            h0 = fma(vec4<f16>(xs[o + ty]), b, h0);
            h1 = fma(vec4<f16>(xs[o + ty + 16u]), b, h1);
            h2 = fma(vec4<f16>(xs[o + ty + 32u]), b, h2);
            h3 = fma(vec4<f16>(xs[o + ty + 48u]), b, h3);
        }
        c0 = c0 + vec4<f32>(h0);
        c1 = c1 + vec4<f32>(h1);
        c2 = c2 + vec4<f32>(h2);
        c3 = c3 + vec4<f32>(h3);
        workgroupBarrier();
    }
    let rows = array<vec4<f32>, 4>(c0, c1, c2, c3);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + ty + 16u * i;
        let v = rows[i];
        store(m, n0 + tx, v.x);
        store(m, n0 + tx + 16u, v.y);
        store(m, n0 + tx + 32u, v.z);
        store(m, n0 + tx + 48u, v.w);
    }
}
