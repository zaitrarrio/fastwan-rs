// y[y_row + m, n] = act(Σ_k x[x_row + m, k] · W[n, k] + b[n]), second version:
// 128×128 output tile per workgroup, 16×16 threads, an 8×8 block per thread
// (rows ty + 16i, cols tx + 16j: conflict-free shared reads, coalesced
// stores), K step 16. 16 shared reads per 64 FMAs, against 8 per 16 in the
// 64×64 kernel (`linear.wgsl`).
struct P { m: u32, n: u32, k: u32, x_row: u32, y_row: u32, has_bias: u32, act: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<{W_TYPE}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

fn load_w(i: u32) -> f32 { {W_LOAD} }

var<workgroup> xs: array<f32, 2048>;   // [16 k][128 rows]
var<workgroup> ws: array<f32, 2048>;   // [16 k][128 cols]

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = tid % 16u;
    let ty = tid / 16u;
    let m0 = wid.y * 128u;
    let n0 = wid.x * 128u;
    var acc: array<array<f32, 8>, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        for (var j = 0u; j < 8u; j = j + 1u) { acc[i][j] = 0.0; }
    }
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
        // 2048 values per tile, 8 per thread; e → (row e / 16, k e % 16):
        // consecutive threads read consecutive k of one row (coalesced).
        for (var t = 0u; t < 8u; t = t + 1u) {
            let e = tid + t * 256u;
            let r = e / 16u;
            let kk = e % 16u;
            let gk = k0 + kk;
            var xv = 0.0;
            if (m0 + r < p.m && gk < p.k) { xv = x[(p.x_row + m0 + r) * p.k + gk]; }
            xs[kk * 128u + r] = xv;
            var wv = 0.0;
            if (n0 + r < p.n && gk < p.k) { wv = load_w((n0 + r) * p.k + gk); }
            ws[kk * 128u + r] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            var a: array<f32, 8>;
            var b: array<f32, 8>;
            for (var i = 0u; i < 8u; i = i + 1u) {
                a[i] = xs[kk * 128u + ty + 16u * i];
                b[i] = ws[kk * 128u + tx + 16u * i];
            }
            for (var i = 0u; i < 8u; i = i + 1u) {
                for (var j = 0u; j < 8u; j = j + 1u) { acc[i][j] = fma(a[i], b[j], acc[i][j]); }
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        let m = m0 + ty + 16u * i;
        if (m >= p.m) { continue; }
        for (var j = 0u; j < 8u; j = j + 1u) {
            let n = n0 + tx + 16u * j;
            if (n >= p.n) { continue; }
            var v = acc[i][j];
            if (p.has_bias != 0u) { v = v + bias[n]; }
            if (p.act == 1u) { v = gelu_tanh(v); }
            y[(p.y_row + m) * p.n + n] = v;
        }
    }
}
