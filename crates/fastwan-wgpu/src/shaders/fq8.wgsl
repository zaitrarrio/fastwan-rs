// FP8 W8A8, activation side (fastwan_core::quant::fq8_rows): each input row
// replaced by its E4M3 rounding at the row's own scale, amax / 448. One
// workgroup per row; the row's amax by a tree reduction in workgroup memory
// (no atomics). out[r, c] = e4m3(x[x_row + r, c] · inv) · s.
struct P { rows: u32, k: u32, x_row: u32, _p: u32 }
@group(0) @binding(0) var<uniform> pp: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;

var<workgroup> red: array<f32, 256>;

// The value of e4m3_encode(v): round to nearest, ties to even (WGSL `round`),
// 3 mantissa bits, subnormals below 2^-6 in steps of 2^-9, saturated at 448.
fn e4m3_round(v: f32) -> f32 {
    let a = min(abs(v), 448.0);
    var r: f32;
    if (a < 0.015625) {
        r = round(a * 512.0) / 512.0;
    } else {
        let e = i32((bitcast<u32>(a) >> 23u) & 255u) - 127;
        let st = exp2(f32(e - 3));
        r = min(round(a / st) * st, 448.0);
    }
    return select(r, -r, v < 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wg_index(wid, nwg);
    if (r >= pp.rows) { return; }
    let base = (pp.x_row + r) * pp.k;
    var m = 0.0;
    for (var c = lid.x; c < pp.k; c = c + 256u) { m = max(m, abs(x[base + c])); }
    red[lid.x] = m;
    workgroupBarrier();
    for (var o = 128u; o > 0u; o = o >> 1u) {
        if (lid.x < o) { red[lid.x] = max(red[lid.x], red[lid.x + o]); }
        workgroupBarrier();
    }
    let amax = red[0];
    var s = 0.0;
    var inv = 0.0;
    if (amax > 0.0) { s = amax / 448.0; inv = 448.0 / amax; }
    for (var c = lid.x; c < pp.k; c = c + 256u) {
        out[r * pp.k + c] = e4m3_round(x[base + c] * inv) * s;
    }
}
