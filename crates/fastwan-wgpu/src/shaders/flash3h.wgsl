// f16 math variant of flash3.wgsl (needs `shader-f16`), same contract,
// bindings and layout. Q/K and V panels are vec4<f16> in workgroup memory
// (half of flash3's panel traffic); Q·Kᵀ and P·V are summed per panel of 16 in
// vec4<f16> (packed f16 FMA: 2× the f32 rate on AMD, Intel and Apple GPUs)
// and flushed into f32. The S/P tile, softmax state and O stay f32. As
// flash_h did for flash.wgsl; flash3's structure notes:
// Flash attention for dense attention and the VSA fine stage (same contract as
// flash.wgsl; the default since 2026-11-14). One workgroup = 64 query rows of
// one head, 256 threads; per key chunk of 64:
//   S = Q·Kᵀ   each thread owns rows 4·ty .. 4·ty+3, cols 4·tx .. 4·tx+3, and
//              reads Q and K out of shared memory as vec4 (two loads per 16
//              FMAs; flash.wgsl did eight scalar loads);
//   softmax    online, base 2, four threads per row (16 scores each), maxima
//              and sums combined through shared memory: all 256 threads work
//              (flash.wgsl: 64 threads, one row each, while 192 waited);
//   O += P·V   rows 4·ty .. 4·ty+3, cols 4·tx .. and 64 + 4·tx .. as vec4.
// No dynamically indexed vectors anywhere (loads, S stores and outputs are
// unrolled): a first version that wrote `v[j]` in small loops ran 1.7×
// slower than flash.wgsl natively, because NVIDIA's Vulkan compiler moved
// those vectors to local memory. Shared vec4s are only ever stored whole: a
// component store may compile to read-modify-write and race with the
// neighbouring thread. RTX 4090: 69.9 → 46.7 ms per call natively, 1.9× the
// whole step in Chrome with linear4 (docs/runs/2026-11-14-vast-4090-webgpu-kernels).
// d ≤ 128, d % 16 == 0.
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

// Q|K panels: [16 d][64 rows / 4] each (256 + 256 vec4), or a V panel
// [16 keys][128 cols / 4] (512 vec4).
var<workgroup> pan: array<vec4<f16>, 512>;
var<workgroup> ss: array<f32, 4160>;      // 64 × 65
var<workgroup> redm: array<f32, 256>;     // per-thread partial row max
var<workgroup> reds: array<f32, 256>;     // per-thread partial row sum
var<workgroup> rowm: array<f32, 64>;
var<workgroup> rowl: array<f32, 64>;
var<workgroup> rowc: array<f32, 64>;
var<workgroup> qrows: array<u32, 64>;
var<workgroup> krows: array<u32, 64>;

fn put_out(r: u32, col: u32, val: f32, h: u32, qt: u32) {
    let tok = qrows[r];
    if (tok == PAD || col >= p.d) { return; }
    var o = val / rowl[r];
    if (p.mode == 1u) {
        o = o + coarse[(h * p.nb + qt) * p.d + col] * gate[(h * p.sq + tok) * p.d + col];
    }
    out[(tok * p.heads + h) * p.d + col] = o;
}

fn sval(c: u32, x: f32) -> f32 {
    if (krows[c] == PAD) { return -1e30; }
    return x * p.scale_log2;
}

fn put_s(r: u32, c0: u32, s: vec4<f32>) {
    let o = r * SS + c0;
    ss[o] = sval(c0, s.x);
    ss[o + 1u] = sval(c0 + 1u, s.y);
    ss[o + 2u] = sval(c0 + 2u, s.z);
    ss[o + 3u] = sval(c0 + 3u, s.w);
}

fn put4(r: u32, c0: u32, o: vec4<f32>, h: u32, qt: u32) {
    put_out(r, c0, o.x, h, qt);
    put_out(r, c0 + 1u, o.y, h, qt);
    put_out(r, c0 + 2u, o.z, h, qt);
    put_out(r, c0 + 3u, o.w, h, qt);
}

fn qload(r: u32, col: u32, h: u32) -> f32 {
    let tok = qrows[r];
    if (tok == PAD) { return 0.0; }
    return q[(h * p.sq + tok) * p.d + col];
}

fn kload(r: u32, col: u32, h: u32) -> f32 {
    let tok = krows[r];
    if (tok == PAD) { return 0.0; }
    return k[(h * p.sk + tok) * p.d + col];
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = tid % 16u;
    let ty = tid / 16u;
    let h = wid.y;
    let qt = wid.x;
    let d = p.d;
    let r0 = 4u * ty;
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
    // O: rows r0 + i, cols 4·tx .. (a) and 64 + 4·tx .. (b).
    var o0a = vec4<f32>(0.0); var o0b = vec4<f32>(0.0);
    var o1a = vec4<f32>(0.0); var o1b = vec4<f32>(0.0);
    var o2a = vec4<f32>(0.0); var o2b = vec4<f32>(0.0);
    var o3a = vec4<f32>(0.0); var o3b = vec4<f32>(0.0);
    var nchunks = (p.sk + 63u) / 64u;
    if (p.mode == 1u) { nchunks = p.topk; }
    // Softmax roles: row sr, columns 16·sp .. 16·sp + 15.
    let sr = tid / 4u;
    let sp = tid % 4u;
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
        // S = Q·Kᵀ, d in panels of 16.
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < d; d0 = d0 + 16u) {
            // Whole-vec4 stores (a component store into a shared vec4 may
            // compile to read-modify-write and race): thread = (rows 4·q4 ..
            // 4·q4 + 3, d0 + kk), kk fastest across lanes.
            {
                let kk = tid % 16u;
                let q4 = tid / 16u;
                let r = 4u * q4;
                let col = d0 + kk;
                pan[kk * 16u + q4] = vec4<f16>(vec4<f32>(qload(r, col, h), qload(r + 1u, col, h), qload(r + 2u, col, h), qload(r + 3u, col, h)));
                pan[256u + kk * 16u + q4] = vec4<f16>(vec4<f32>(kload(r, col, h), kload(r + 1u, col, h), kload(r + 2u, col, h), kload(r + 3u, col, h)));
            }
            workgroupBarrier();
            var hs0 = vec4<f16>(0.0h);
            var hs1 = vec4<f16>(0.0h);
            var hs2 = vec4<f16>(0.0h);
            var hs3 = vec4<f16>(0.0h);
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                let a = pan[kk * 16u + ty];
                let b = pan[256u + kk * 16u + tx];
                hs0 = fma(vec4<f16>(a.x), b, hs0);
                hs1 = fma(vec4<f16>(a.y), b, hs1);
                hs2 = fma(vec4<f16>(a.z), b, hs2);
                hs3 = fma(vec4<f16>(a.w), b, hs3);
            }
            s0 = s0 + vec4<f32>(hs0);
            s1 = s1 + vec4<f32>(hs1);
            s2 = s2 + vec4<f32>(hs2);
            s3 = s3 + vec4<f32>(hs3);
            workgroupBarrier();
        }
        put_s(r0, 4u * tx, s0);
        put_s(r0 + 1u, 4u * tx, s1);
        put_s(r0 + 2u, 4u * tx, s2);
        put_s(r0 + 3u, 4u * tx, s3);
        workgroupBarrier();
        // Online softmax, four threads per row.
        let base = sr * SS + 16u * sp;
        var pm = -1e30;
        for (var j = 0u; j < 16u; j = j + 1u) { pm = max(pm, ss[base + j]); }
        redm[tid] = pm;
        workgroupBarrier();
        let old = rowm[sr];
        let g = sr * 4u;
        let mx = max(max(max(old, redm[g]), max(redm[g + 1u], redm[g + 2u])), redm[g + 3u]);
        var ps = 0.0;
        for (var j = 0u; j < 16u; j = j + 1u) {
            let e = exp2(ss[base + j] - mx);
            ss[base + j] = e;
            ps = ps + e;
        }
        reds[tid] = ps;
        workgroupBarrier();
        if (sp == 0u) {
            let corr = exp2(old - mx);
            rowl[sr] = rowl[sr] * corr + reds[g] + reds[g + 1u] + reds[g + 2u] + reds[g + 3u];
            rowm[sr] = mx;
            rowc[sr] = corr;
        }
        workgroupBarrier();
        let c0 = rowc[r0]; let c1 = rowc[r0 + 1u]; let c2 = rowc[r0 + 2u]; let c3 = rowc[r0 + 3u];
        o0a = o0a * c0; o0b = o0b * c0;
        o1a = o1a * c1; o1b = o1b * c1;
        o2a = o2a * c2; o2b = o2b * c2;
        o3a = o3a * c3; o3b = o3b * c3;
        // O += P·V, keys in panels of 16.
        for (var k0 = 0u; k0 < 64u; k0 = k0 + 16u) {
            for (var t = 0u; t < 2u; t = t + 1u) {
                let e = tid + t * 256u;
                let kk = e / 32u;
                let cq = e % 32u;            // columns 4·cq .. 4·cq + 3
                let ktok = krows[k0 + kk];
                var vv = vec4<f32>(0.0);
                if (ktok != PAD && 4u * cq < d) {
                    let b = (h * p.sk + ktok) * d + 4u * cq;
                    vv = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]);
                }
                pan[kk * 32u + cq] = vec4<f16>(vv);
            }
            workgroupBarrier();
            var h0a = vec4<f16>(0.0h); var h0b = vec4<f16>(0.0h);
            var h1a = vec4<f16>(0.0h); var h1b = vec4<f16>(0.0h);
            var h2a = vec4<f16>(0.0h); var h2b = vec4<f16>(0.0h);
            var h3a = vec4<f16>(0.0h); var h3b = vec4<f16>(0.0h);
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                // Columns ≥ d hold stale values; they only feed O columns
                // that are never stored.
                let va = pan[kk * 32u + tx];
                let vb = pan[kk * 32u + 16u + tx];
                let pc = k0 + kk;
                let p0 = vec4<f16>(f16(ss[r0 * SS + pc]));
                let p1 = vec4<f16>(f16(ss[(r0 + 1u) * SS + pc]));
                let p2 = vec4<f16>(f16(ss[(r0 + 2u) * SS + pc]));
                let p3 = vec4<f16>(f16(ss[(r0 + 3u) * SS + pc]));
                h0a = fma(p0, va, h0a); h0b = fma(p0, vb, h0b);
                h1a = fma(p1, va, h1a); h1b = fma(p1, vb, h1b);
                h2a = fma(p2, va, h2a); h2b = fma(p2, vb, h2b);
                h3a = fma(p3, va, h3a); h3b = fma(p3, vb, h3b);
            }
            o0a = o0a + vec4<f32>(h0a); o0b = o0b + vec4<f32>(h0b);
            o1a = o1a + vec4<f32>(h1a); o1b = o1b + vec4<f32>(h1b);
            o2a = o2a + vec4<f32>(h2a); o2b = o2b + vec4<f32>(h2b);
            o3a = o3a + vec4<f32>(h3a); o3b = o3b + vec4<f32>(h3b);
            workgroupBarrier();
        }
    }
    let ca = 4u * tx;
    let cb = 64u + 4u * tx;
    put4(r0, ca, o0a, h, qt);      put4(r0, cb, o0b, h, qt);
    put4(r0 + 1u, ca, o1a, h, qt); put4(r0 + 1u, cb, o1b, h, qt);
    put4(r0 + 2u, ca, o2a, h, qt); put4(r0 + 2u, cb, o2b, h, qt);
    put4(r0 + 3u, ca, o3a, h, qt); put4(r0 + 3u, cb, o3b, h, qt);
}
