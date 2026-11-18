// f16 math variant of linear4.wgsl (needs `shader-f16`; f16 weights only):
// 128×128 tiles, 256 threads, 8×8 outputs per thread. Shared tiles hold
// vec4<f16> (half linear4's shared memory and traffic); each K tile of 16 is
// summed in vec4<f16> (packed f16 FMA: 2× the f32 rate on AMD, Intel and Apple
// GPUs, equal on NVIDIA GeForce) and then flushed into the f32 accumulators,
// so f16 rounding never compounds past 16 terms (as linear3h).
struct P { m: u32, n: u32, k: u32, x_row: u32, y_row: u32, has_bias: u32, act: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

// Exact: the packed pair already holds f16 values.
fn load_w(i: u32) -> f32 {
    let v = unpack2x16float(w[i >> 1u]);
    return select(v.x, v.y, (i & 1u) == 1u);
}

var<workgroup> xs: array<vec4<f16>, 512>;   // [16 k][128 rows / 4]
var<workgroup> ws: array<vec4<f16>, 512>;   // [16 k][128 cols / 4]

fn out1(m: u32, n: u32, v: f32) {
    if (n >= p.n) { return; }
    var o = v;
    if (p.has_bias != 0u) { o = o + bias[n]; }
    if (p.act == 1u) { o = gelu_tanh(o); }
    y[(p.y_row + m) * p.n + n] = o;
}

fn store4(m: u32, n: u32, v: vec4<f32>) {
    if (m >= p.m) { return; }
    out1(m, n, v.x);
    out1(m, n + 1u, v.y);
    out1(m, n + 2u, v.z);
    out1(m, n + 3u, v.w);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m0 = wid.y * 128u;
    let n0 = wid.x * 128u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    // Loads: each thread writes whole vec4s (a component store into a shared
    // vec4 may compile to read-modify-write, racing with its neighbours):
    // rows 4q .. 4q+3 at k0 + kk, kk fastest across lanes for coalescing.
    var c00 = vec4<f32>(0.0);
    var c01 = vec4<f32>(0.0);
    var c10 = vec4<f32>(0.0);
    var c11 = vec4<f32>(0.0);
    var c20 = vec4<f32>(0.0);
    var c21 = vec4<f32>(0.0);
    var c30 = vec4<f32>(0.0);
    var c31 = vec4<f32>(0.0);
    var c40 = vec4<f32>(0.0);
    var c41 = vec4<f32>(0.0);
    var c50 = vec4<f32>(0.0);
    var c51 = vec4<f32>(0.0);
    var c60 = vec4<f32>(0.0);
    var c61 = vec4<f32>(0.0);
    var c70 = vec4<f32>(0.0);
    var c71 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
        for (var t = 0u; t < 2u; t = t + 1u) {
            let e = tid + t * 256u;
            let kk = e % 16u;
            let q4 = e / 16u;            // 0 .. 31: rows 4·q4 .. 4·q4 + 3
            let gk = k0 + kk;
            var xv = vec4<f32>(0.0);
            var wv = vec4<f32>(0.0);
            if (gk < p.k) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let r = 4u * q4 + j;
                    if (m0 + r < p.m) { xv[j] = x[(p.x_row + m0 + r) * p.k + gk]; }
                    if (n0 + r < p.n) { wv[j] = load_w((n0 + r) * p.k + gk); }
                }
            }
            xs[kk * 32u + q4] = vec4<f16>(xv);
            ws[kk * 32u + q4] = vec4<f16>(wv);
        }
        workgroupBarrier();
        var h00 = vec4<f16>(0.0h);
        var h01 = vec4<f16>(0.0h);
        var h10 = vec4<f16>(0.0h);
        var h11 = vec4<f16>(0.0h);
        var h20 = vec4<f16>(0.0h);
        var h21 = vec4<f16>(0.0h);
        var h30 = vec4<f16>(0.0h);
        var h31 = vec4<f16>(0.0h);
        var h40 = vec4<f16>(0.0h);
        var h41 = vec4<f16>(0.0h);
        var h50 = vec4<f16>(0.0h);
        var h51 = vec4<f16>(0.0h);
        var h60 = vec4<f16>(0.0h);
        var h61 = vec4<f16>(0.0h);
        var h70 = vec4<f16>(0.0h);
        var h71 = vec4<f16>(0.0h);
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 32u;
            let a0 = xs[o + ty];
            let a1 = xs[o + 16u + ty];
            let b0 = ws[o + tx];
            let b1 = ws[o + 16u + tx];
            h00 = fma(vec4<f16>(a0.x), b0, h00);
            h01 = fma(vec4<f16>(a0.x), b1, h01);
            h10 = fma(vec4<f16>(a0.y), b0, h10);
            h11 = fma(vec4<f16>(a0.y), b1, h11);
            h20 = fma(vec4<f16>(a0.z), b0, h20);
            h21 = fma(vec4<f16>(a0.z), b1, h21);
            h30 = fma(vec4<f16>(a0.w), b0, h30);
            h31 = fma(vec4<f16>(a0.w), b1, h31);
            h40 = fma(vec4<f16>(a1.x), b0, h40);
            h41 = fma(vec4<f16>(a1.x), b1, h41);
            h50 = fma(vec4<f16>(a1.y), b0, h50);
            h51 = fma(vec4<f16>(a1.y), b1, h51);
            h60 = fma(vec4<f16>(a1.z), b0, h60);
            h61 = fma(vec4<f16>(a1.z), b1, h61);
            h70 = fma(vec4<f16>(a1.w), b0, h70);
            h71 = fma(vec4<f16>(a1.w), b1, h71);
        }
        c00 = c00 + vec4<f32>(h00);
        c01 = c01 + vec4<f32>(h01);
        c10 = c10 + vec4<f32>(h10);
        c11 = c11 + vec4<f32>(h11);
        c20 = c20 + vec4<f32>(h20);
        c21 = c21 + vec4<f32>(h21);
        c30 = c30 + vec4<f32>(h30);
        c31 = c31 + vec4<f32>(h31);
        c40 = c40 + vec4<f32>(h40);
        c41 = c41 + vec4<f32>(h41);
        c50 = c50 + vec4<f32>(h50);
        c51 = c51 + vec4<f32>(h51);
        c60 = c60 + vec4<f32>(h60);
        c61 = c61 + vec4<f32>(h61);
        c70 = c70 + vec4<f32>(h70);
        c71 = c71 + vec4<f32>(h71);
        workgroupBarrier();
    }
    store4(m0 + 4u * ty + 0u, n0 + 4u * tx, c00);
    store4(m0 + 4u * ty + 0u, n0 + 64u + 4u * tx, c01);
    store4(m0 + 4u * ty + 1u, n0 + 4u * tx, c10);
    store4(m0 + 4u * ty + 1u, n0 + 64u + 4u * tx, c11);
    store4(m0 + 4u * ty + 2u, n0 + 4u * tx, c20);
    store4(m0 + 4u * ty + 2u, n0 + 64u + 4u * tx, c21);
    store4(m0 + 4u * ty + 3u, n0 + 4u * tx, c30);
    store4(m0 + 4u * ty + 3u, n0 + 64u + 4u * tx, c31);
    store4(m0 + 64u + 4u * ty + 0u, n0 + 4u * tx, c40);
    store4(m0 + 64u + 4u * ty + 0u, n0 + 64u + 4u * tx, c41);
    store4(m0 + 64u + 4u * ty + 1u, n0 + 4u * tx, c50);
    store4(m0 + 64u + 4u * ty + 1u, n0 + 64u + 4u * tx, c51);
    store4(m0 + 64u + 4u * ty + 2u, n0 + 4u * tx, c60);
    store4(m0 + 64u + 4u * ty + 2u, n0 + 64u + 4u * tx, c61);
    store4(m0 + 64u + 4u * ty + 3u, n0 + 4u * tx, c70);
    store4(m0 + 64u + 4u * ty + 3u, n0 + 64u + 4u * tx, c71);
}
