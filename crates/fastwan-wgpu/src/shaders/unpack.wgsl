// 8-bit RGBA words [frames, plane] -> planar RGB [frames, 3, plane] as
// byte / 255 (Backend::unpack_rgba8), one thread per output value.
struct P { n: u32, plane: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> pp: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= pp.n) { return; }
    let p = i % pp.plane;
    let fc = i / pp.plane;
    let f = fc / 3u;
    let c = fc % 3u;
    out[i] = f32((src[f * pp.plane + p] >> (8u * c)) & 255u) / 255.0;
}
