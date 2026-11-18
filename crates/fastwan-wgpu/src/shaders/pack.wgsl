// Packed 8-bit RGBA frames (Backend::pack_rgba8), one thread per pixel.
// src [skip + frames, 3·p², h, w] (pixel-shuffled RGB), optional base
// [skip + frames, h, w] packed 8-bit RGBA added (÷255) after a nearest ×p
// upsample; out holds one
// R | G<<8 | B<<16 | 255<<24 word per pixel. `out` is a u32 view: a packed
// pixel is not a valid f32 in general (WGSL leaves NaN bitcasts undefined).
// `down` (divides p) averages each down×down block of the shuffled image
// after the residual and clamp, before rounding (×2 = ×4 network + 2×2 box).
struct P { n: u32, skip: u32, h: u32, w: u32, p: u32, has_base: u32, down: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> pp: P;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<u32>;

fn q8(v: f32) -> u32 {
    return u32(clamp(v, 0.0, 1.0) * 255.0 + 0.5);
}

// One sample of the shuffled image at (y, x) in frame f, residual added, clamped.
fn sample(f: u32, y: u32, x: u32) -> vec3<f32> {
    let p = pp.p;
    let plane = pp.h * pp.w;
    let at = (y / p) * pp.w + x / p;
    let s0 = f * 3u * p * p + (y % p) * p + x % p;
    var v = vec3<f32>(src[s0 * plane + at], src[(s0 + p * p) * plane + at], src[(s0 + 2u * p * p) * plane + at]);
    if (pp.has_base != 0u) {
        let wd = base[f * plane + at];
        v = v + vec3<f32>(f32(wd & 255u), f32((wd >> 8u) & 255u), f32((wd >> 16u) & 255u)) / 255.0;
    }
    return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= pp.n) { return; }
    let d = pp.down;
    let ow = pp.w * pp.p / d;
    let oh = pp.h * pp.p / d;
    let f = i / (oh * ow) + pp.skip;
    let y = ((i / ow) % oh) * d;
    let x = (i % ow) * d;
    var acc = vec3<f32>(0.0);
    for (var dy = 0u; dy < d; dy = dy + 1u) {
        for (var dx = 0u; dx < d; dx = dx + 1u) {
            acc = acc + sample(f, y + dy, x + dx);
        }
    }
    let c = acc / f32(d * d);
    out[i] = q8(c.x) | (q8(c.y) << 8u) | (q8(c.z) << 16u) | 0xff000000u;
    // Keep every binding statically used (auto layouts drop unused ones).
    if (pp.n == 0xffffffffu) { out[0] = base[0]; }
}
