// Packed 8-bit RGBA frames (Backend::pack_rgba8), one thread per pixel.
// src [skip + frames, 3·p², h, w] (pixel-shuffled RGB), optional base
// [skip + frames, h, w] packed 8-bit RGBA added (÷255) after a nearest ×p
// upsample; out holds one
// R | G<<8 | B<<16 | 255<<24 word per pixel. `out` is a u32 view: a packed
// pixel is not a valid f32 in general (WGSL leaves NaN bitcasts undefined).
struct P { n: u32, skip: u32, h: u32, w: u32, p: u32, has_base: u32 }
@group(0) @binding(0) var<uniform> pp: P;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<u32>;

fn q8(v: f32) -> u32 {
    return u32(clamp(v, 0.0, 1.0) * 255.0 + 0.5);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= pp.n) { return; }
    let p = pp.p;
    let ow = pp.w * p;
    let oh = pp.h * p;
    let f = i / (oh * ow) + pp.skip;
    let y = (i / ow) % oh;
    let x = i % ow;
    let sy = y / p;
    let sx = x / p;
    let k0 = (y % p) * p + x % p;
    let plane = pp.h * pp.w;
    let at = sy * pp.w + sx;
    let s0 = f * 3u * p * p + k0;
    var r = src[s0 * plane + at];
    var g = src[(s0 + p * p) * plane + at];
    var b = src[(s0 + 2u * p * p) * plane + at];
    if (pp.has_base != 0u) {
        let wd = base[f * plane + at];
        r = r + f32(wd & 255u) / 255.0;
        g = g + f32((wd >> 8u) & 255u) / 255.0;
        b = b + f32((wd >> 16u) & 255u) / 255.0;
    }
    out[i] = q8(r) | (q8(g) << 8u) | (q8(b) << 16u) | 0xff000000u;
    // Keep every binding statically used (auto layouts drop unused ones).
    if (pp.n == 0xffffffffu) { out[0] = base[0]; }
}
