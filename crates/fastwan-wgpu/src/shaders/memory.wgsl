// MemRoPE cache kernels (History Forcing's long-video memory).
//   op 0 rope_heads: dst[h, r] = rope(src[h, r], row r of cos/sin), one thread
//        per interleaved pair of `heads · rows · hd / 2`.
//   op 1 kv_ema: one 256-thread group per (head, 32 channels): 8 lanes sum the
//        source rows, then every row of the two memory frames becomes
//        a_i · mean + (1 − a_i) · x. In place on `dst`.
struct P {
    op: u32, heads: u32, rows: u32, hd: u32,
    src_off: u32, src_stride: u32, dst_off: u32, dst_stride: u32,
    src_row: u32, src_rows: u32, dst_row: u32, frame_rows: u32,
    a0: f32, a1: f32, _a: u32, _b: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;
@group(0) @binding(3) var<storage, read> sn: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = wg_index(wid, nwg);
    if (p.op == 0u) {
        let half = p.hd / 2u;
        let i = g * 256u + lid.x;
        if (i >= p.heads * p.rows * half) { return; }
        let c = i % half;
        let r = (i / half) % p.rows;
        let h = i / (half * p.rows);
        let s = p.src_off + h * p.src_stride + r * p.hd + 2u * c;
        let o = p.dst_off + h * p.dst_stride + r * p.hd + 2u * c;
        let co = cs[r * half + c];
        let si = sn[r * half + c];
        let x1 = src[s];
        let x2 = src[s + 1u];
        dst[o] = x1 * co - x2 * si;
        dst[o + 1u] = x1 * si + x2 * co;
        return;
    }
    // op 1: groups = heads · ceil(hd / 32).
    let cpb = (p.hd + 31u) / 32u;
    let h = g / cpb;
    let ch = (g % cpb) * 32u + lid.x % 32u;
    let lane = lid.x / 32u;
    if (h >= p.heads) { return; }
    let base = h * p.dst_stride;
    var acc = 0.0;
    if (ch < p.hd) {
        for (var r = lane; r < p.src_rows; r = r + 8u) {
            acc = acc + dst[base + (p.src_row + r) * p.hd + ch];
        }
    }
    red[lid.x] = acc;
    workgroupBarrier();
    if (lane == 0u) {
        var s = 0.0;
        for (var l = 0u; l < 8u; l = l + 1u) { s = s + red[l * 32u + lid.x]; }
        red[lid.x] = s / f32(p.src_rows);
    }
    workgroupBarrier();
    let m = red[lid.x % 32u];
    if (ch < p.hd) {
        for (var r = lane; r < 2u * p.frame_rows; r = r + 8u) {
            var a = p.a0;
            if (r >= p.frame_rows) { a = p.a1; }
            let o = base + (p.dst_row + r) * p.hd + ch;
            dst[o] = a * m + (1.0 - a) * dst[o];
        }
    }
    // Every declared binding must be statically used (auto layouts drop
    // the rest).
    if (p.op == 9999u) { dst[0] = src[0] + cs[0] + sn[0]; }
}
