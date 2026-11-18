// K/V cache store: rows row0 .. row0 + len of every head of src ([heads, n, d])
// into the cache at row dst_row of each head, heads `stride` rows apart. One
// dispatch per run of slots for all heads (it replaced 2 · heads
// copy_buffer_to_buffer calls per run and layer). half = 1: the cache holds
// f16 pairs packed in u32 (d and the offsets even), one thread per pair;
// half = 0: f32 bits, one thread per value.
struct P { n: u32, row0: u32, len: u32, heads: u32, d: u32, dst_row: u32, stride: u32, half: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    let per = select(1u, 2u, p.half != 0u);
    let row_items = p.d / per;
    if (i >= p.heads * p.len * row_items) { return; }
    let c = (i % row_items) * per;
    let r = (i / row_items) % p.len;
    let h = i / (row_items * p.len);
    let s = (h * p.n + p.row0 + r) * p.d + c;
    let o = (h * p.stride + p.dst_row + r) * p.d + c;
    if (p.half != 0u) {
        dst[o / 2u] = pack2x16float(vec2<f32>(src[s], src[s + 1u]));
    } else {
        dst[o] = bitcast<u32>(src[s]);
    }
}
