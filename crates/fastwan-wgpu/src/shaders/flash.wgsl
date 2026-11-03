// Register-tiled flash attention, for dense attention and the VSA fine stage.
//
// One workgroup = 64 query rows of one head, 256 threads as 16×16. Each key
// chunk of 64:
//   S = Q·Kᵀ    like the linear kernel: d in steps of 16 through shared memory,
//               each thread a 4×4 block of S in registers (rows ty+16i, cols tx+16j);
//   softmax     online, base 2, one thread per row over the 64×64 tile in shared memory;
//   O += P·V    each thread owns 4 rows × d/16 columns of O in registers.
// Nothing lives in per-thread arrays larger than 32 floats, which is what the
// one-row-per-thread kernel got wrong (its q/acc arrays of 128 spill).
//
// mode 0 dense: query rows wid.x·64 + r, key chunks of consecutive rows.
// mode 1 VSA:   query tile wid.x (slots via `slot`), key chunk c = selected tile
//               sel[h][qt][c]; out = O/l + coarse[h][qt] · gate[h][token].
struct P { heads: u32, sq: u32, sk: u32, d: u32, scale_log2: f32, mode: u32, nb: u32, topk: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> slot: array<u32>;
@group(0) @binding(6) var<storage, read> sel: array<u32>;
@group(0) @binding(7) var<storage, read> coarse: array<f32>;
@group(0) @binding(8) var<storage, read> gate: array<f32>;

const PAD: u32 = 0xffffffffu;
const SS: u32 = 65u;   // padded row stride of the S/P tile

var<workgroup> buf: array<f32, 2048>;     // Q|K panels (2×16×64) or a V panel (16×128)
var<workgroup> ss: array<f32, 4160>;      // 64 × 65
var<workgroup> rowm: array<f32, 64>;
var<workgroup> rowl: array<f32, 64>;
var<workgroup> rowc: array<f32, 64>;
var<workgroup> qrows: array<u32, 64>;     // token of each query row (PAD if none)
var<workgroup> krows: array<u32, 64>;     // token of each key row in this chunk

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = tid % 16u;
    let ty = tid / 16u;
    let h = wid.y;
    let qt = wid.x;
    let d = p.d;
    let nd = d / 16u;
    if (tid < 64u) {
        var tok = PAD;
        if (p.mode == 0u) {
            let r = qt * 64u + tid;
            if (r < p.sq) { tok = r; }
        } else {
            tok = slot[qt * 64u + tid];
        }
        qrows[tid] = tok;
        rowm[tid] = -1e30;
        rowl[tid] = 0.0;
    }
    // O block: rows ty + 16i, cols tx + 16j (j < 8) as two vec4 per row.
    var o0a = vec4<f32>(0.0); var o0b = vec4<f32>(0.0);
    var o1a = vec4<f32>(0.0); var o1b = vec4<f32>(0.0);
    var o2a = vec4<f32>(0.0); var o2b = vec4<f32>(0.0);
    var o3a = vec4<f32>(0.0); var o3b = vec4<f32>(0.0);
    var nchunks = (p.sk + 63u) / 64u;
    if (p.mode == 1u) { nchunks = p.topk; }
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        if (tid < 64u) {
            var tok = PAD;
            if (p.mode == 0u) {
                let r = c * 64u + tid;
                if (r < p.sk) { tok = r; }
            } else {
                let tile = sel[(h * p.nb + qt) * p.topk + c];
                tok = slot[tile * 64u + tid];
            }
            krows[tid] = tok;
        }
        workgroupBarrier();
        // S = Q·Kᵀ over d in panels of 16. The 4×4 block is four named vec4
        // (rows ty + 16i, cols tx + 16j): arrays here were kept in local
        // memory by NVIDIA's Vulkan compiler (5.75× on the GEMM when unrolled).
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < d; d0 = d0 + 16u) {
            for (var e = tid; e < 1024u; e = e + 256u) {
                let r = e / 16u;
                let kk = e % 16u;
                let qtok = qrows[r];
                var qv = 0.0;
                if (qtok != PAD) { qv = q[(h * p.sq + qtok) * d + d0 + kk]; }
                buf[kk * 64u + r] = qv;
                let ktok = krows[r];
                var kv = 0.0;
                if (ktok != PAD) { kv = k[(h * p.sk + ktok) * d + d0 + kk]; }
                buf[1024u + kk * 64u + r] = kv;
            }
            workgroupBarrier();
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                let o = kk * 64u;
                let b = vec4<f32>(buf[1024u + o + tx], buf[1024u + o + tx + 16u], buf[1024u + o + tx + 32u], buf[1024u + o + tx + 48u]);
                s0 = fma(vec4<f32>(buf[o + ty]), b, s0);
                s1 = fma(vec4<f32>(buf[o + ty + 16u]), b, s1);
                s2 = fma(vec4<f32>(buf[o + ty + 32u]), b, s2);
                s3 = fma(vec4<f32>(buf[o + ty + 48u]), b, s3);
            }
            workgroupBarrier();
        }
        let sv = array<vec4<f32>, 4>(s0, s1, s2, s3);
        for (var i = 0u; i < 4u; i = i + 1u) {
            let row = sv[i];
            for (var j = 0u; j < 4u; j = j + 1u) {
                let col = tx + 16u * j;
                var val = -1e30;
                if (krows[col] != PAD) { val = row[j] * p.scale_log2; }
                ss[(ty + 16u * i) * SS + col] = val;
            }
        }
        workgroupBarrier();
        // Online softmax, one thread per row.
        if (tid < 64u) {
            let base = tid * SS;
            var mx = rowm[tid];
            for (var j = 0u; j < 64u; j = j + 1u) { mx = max(mx, ss[base + j]); }
            let corr = exp2(rowm[tid] - mx);
            var sum = 0.0;
            for (var j = 0u; j < 64u; j = j + 1u) {
                let e = exp2(ss[base + j] - mx);
                ss[base + j] = e;
                sum = sum + e;
            }
            rowl[tid] = rowl[tid] * corr + sum;
            rowm[tid] = mx;
            rowc[tid] = corr;
        }
        workgroupBarrier();
        let c0 = rowc[ty]; let c1 = rowc[ty + 16u]; let c2 = rowc[ty + 32u]; let c3 = rowc[ty + 48u];
        o0a = o0a * c0; o0b = o0b * c0;
        o1a = o1a * c1; o1b = o1b * c1;
        o2a = o2a * c2; o2b = o2b * c2;
        o3a = o3a * c3; o3b = o3b * c3;
        // O += P·V, keys in panels of 16.
        for (var k0 = 0u; k0 < 64u; k0 = k0 + 16u) {
            for (var e = tid; e < 16u * d; e = e + 256u) {
                let kk = e / d;
                let col = e % d;
                let ktok = krows[k0 + kk];
                var vv = 0.0;
                if (ktok != PAD) { vv = v[(h * p.sk + ktok) * d + col]; }
                buf[kk * 128u + col] = vv;
            }
            workgroupBarrier();
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                // V columns beyond d hold stale values; they only feed O
                // columns that are never stored.
                let vo = kk * 128u + tx;
                let va = vec4<f32>(buf[vo], buf[vo + 16u], buf[vo + 32u], buf[vo + 48u]);
                let vb = vec4<f32>(buf[vo + 64u], buf[vo + 80u], buf[vo + 96u], buf[vo + 112u]);
                let pc = k0 + kk;
                let p0 = vec4<f32>(ss[ty * SS + pc]);
                let p1 = vec4<f32>(ss[(ty + 16u) * SS + pc]);
                let p2 = vec4<f32>(ss[(ty + 32u) * SS + pc]);
                let p3 = vec4<f32>(ss[(ty + 48u) * SS + pc]);
                o0a = fma(p0, va, o0a); o0b = fma(p0, vb, o0b);
                o1a = fma(p1, va, o1a); o1b = fma(p1, vb, o1b);
                o2a = fma(p2, va, o2a); o2b = fma(p2, vb, o2b);
                o3a = fma(p3, va, o3a); o3b = fma(p3, vb, o3b);
            }
            workgroupBarrier();
        }
    }
    let oa = array<vec4<f32>, 4>(o0a, o1a, o2a, o3a);
    let ob = array<vec4<f32>, 4>(o0b, o1b, o2b, o3b);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let r = ty + 16u * i;
        let tok = qrows[r];
        if (tok == PAD) { continue; }
        let inv = 1.0 / rowl[r];
        for (var j = 0u; j < nd; j = j + 1u) {
            let col = tx + 16u * j;
            var o: f32;
            if (j < 4u) { o = oa[i][j]; } else { o = ob[i][j - 4u]; }
            o = o * inv;
            if (p.mode == 1u) {
                o = o + coarse[(h * p.nb + qt) * d + col] * gate[(h * p.sq + tok) * d + col];
            }
            out[(tok * p.heads + h) * d + col] = o;
        }
    }
}
