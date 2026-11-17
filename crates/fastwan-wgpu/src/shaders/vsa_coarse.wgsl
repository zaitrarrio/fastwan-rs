// VSA coarse stage: one workgroup per (query tile, head), 256 threads.
//   scores    s[kt] = mean q · mean k · scale over all key tiles (≤ 2048);
//   select    the top-k key tiles: every tile scoring above the k-th largest,
//             then ties at it by lower index (= sorting by (score desc, index
//             asc)), written in index order;
//   coarse    coarse[h][qt] = Σ softmax(s)[kt] · mean v[kt].
//
// Top-k is a radix select over the scores' order-preserving u32 keys, 4 bits
// per pass (8 passes): each thread counts its own keys into 16 digit counters
// packed 4 per u32 (≤ 8 keys per thread, so a byte holds a count) in a column
// only it writes; the 16 totals come from a two-level sum. No atomics: the
// workgroup-atomic version of this search picked the wrong tiles on an AMD
// RDNA1 Mac (Chrome → Metal) while Apple, NVIDIA and SwiftShader were exact
// (2026-11-17 browser self-test), and a plain atomic-free bit search needed
// ~288 barriers per workgroup where this needs ~50. The selected tiles are
// written by all threads at once after a prefix scan of per-thread counts
// over contiguous index slices (the old code had thread 0 write them alone).
// exp(s − max) is computed once per tile, after the selection, not once per
// tile per output column. The scores read this query tile's mean q from
// workgroup memory (once, not once per key tile) with four independent
// accumulators: on an RTX 4090 (Vulkan) 3.25 ms per call at 832×480 with 12
// heads vs 9.38 without that and 5.96 for the atomic version
// (docs/runs/2026-11-17-vast-4090-vsa-coarse). Workgroup memory:
// 8 + 1 + 4 + 1 + 0.5 KiB (+64 B), under WebGPU's 16 KiB default.
struct P { heads: u32, d: u32, nb: u32, topk: u32, scale: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pooled: array<f32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<f32>;
@group(0) @binding(3) var<storage, read_write> sel: array<u32>;

var<workgroup> sc: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;
// Digit counters: word w, thread t at hist[w * 256 + t]; digit g in byte g % 4 of word g / 4.
var<workgroup> hist: array<u32, 1024>;
// Partial digit sums (bucket · 16 + chunk), then the prefix scan.
var<workgroup> part: array<u32, 256>;
var<workgroup> tot: array<u32, 16>;
// This query tile's mean q (d ≤ 128), read once instead of once per key tile.
var<workgroup> qm: array<f32, 128>;

// Float → u32 with the same order.
fn okey(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let qt = wid.x;
    let h = wid.y;
    let nb = p.nb;
    let plane = p.heads * nb * p.d;
    let qb = (h * nb + qt) * p.d;
    if (tid < p.d) { qm[tid] = pooled[qb + tid]; }
    workgroupBarrier();
    var mx = -3.0e38;
    for (var kt = tid; kt < nb; kt = kt + 256u) {
        let kb = plane + (h * nb + kt) * p.d;
        // Four independent accumulators (d % 4 == 0): the loads overlap.
        var a0 = 0.0;
        var a1 = 0.0;
        var a2 = 0.0;
        var a3 = 0.0;
        for (var c = 0u; c < p.d; c = c + 4u) {
            a0 = fma(qm[c], pooled[kb + c], a0);
            a1 = fma(qm[c + 1u], pooled[kb + c + 1u], a1);
            a2 = fma(qm[c + 2u], pooled[kb + c + 2u], a2);
            a3 = fma(qm[c + 3u], pooled[kb + c + 3u], a3);
        }
        let s = ((a0 + a1) + (a2 + a3)) * p.scale;
        sc[kt] = s;
        mx = max(mx, s);
    }
    red[tid] = mx;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) { red[tid] = max(red[tid], red[tid + st]); }
        workgroupBarrier();
    }
    let m = red[0];

    // Each thread's contiguous slice of key tiles (≤ 8 of them).
    let per = (nb + 255u) / 256u;
    let lo = min(tid * per, nb);
    let hi = min(lo + per, nb);

    // Radix select: the k-th largest key, 4 bits per pass from the top.
    var prefix = 0u;
    var krem = p.topk;
    for (var pi = 0u; pi < 8u; pi = pi + 1u) {
        let shift = 28u - 4u * pi;
        var w0 = 0u; var w1 = 0u; var w2 = 0u; var w3 = 0u;
        for (var kt = lo; kt < hi; kt = kt + 1u) {
            let key = okey(sc[kt]);
            if (pi == 0u || (key >> (shift + 4u)) == prefix) {
                let g = (key >> shift) & 15u;
                let inc = 1u << ((g & 3u) * 8u);
                let wsel = g >> 2u;
                w0 = w0 + select(0u, inc, wsel == 0u);
                w1 = w1 + select(0u, inc, wsel == 1u);
                w2 = w2 + select(0u, inc, wsel == 2u);
                w3 = w3 + select(0u, inc, wsel == 3u);
            }
        }
        hist[tid] = w0;
        hist[256u + tid] = w1;
        hist[512u + tid] = w2;
        hist[768u + tid] = w3;
        workgroupBarrier();
        {
            // Bucket b's count over threads 16·c .. 16·c + 15.
            let b = tid / 16u;
            let c = tid % 16u;
            let base = (b >> 2u) * 256u + c * 16u;
            let sh = (b & 3u) * 8u;
            var s = 0u;
            for (var i = 0u; i < 16u; i = i + 1u) { s = s + ((hist[base + i] >> sh) & 255u); }
            part[b * 16u + c] = s;
        }
        workgroupBarrier();
        if (tid < 16u) {
            var s = 0u;
            for (var i = 0u; i < 16u; i = i + 1u) { s = s + part[tid * 16u + i]; }
            tot[tid] = s;
        }
        workgroupBarrier();
        // Every thread walks the totals from the top digit (same answer everywhere).
        var acc = 0u;
        var digit = 0u;
        var found = false;
        for (var j = 0u; j < 16u; j = j + 1u) {
            let g = 15u - j;
            let n = tot[g];
            if (!found) {
                if (acc + n >= krem) {
                    digit = g;
                    krem = krem - acc;
                    found = true;
                } else {
                    acc = acc + n;
                }
            }
        }
        prefix = (prefix << 4u) | digit;
        workgroupBarrier();
    }
    let t = prefix;

    // Write the selection: keys above t, then ties at t up to top-k, each in
    // index order. Exclusive prefix of (above, tie) counts packed 16 + 16 bits.
    var na = 0u;
    var ne = 0u;
    for (var kt = lo; kt < hi; kt = kt + 1u) {
        let key = okey(sc[kt]);
        na = na + select(0u, 1u, key > t);
        ne = ne + select(0u, 1u, key == t);
    }
    let mine = na | (ne << 16u);
    part[tid] = mine;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off << 1u) {
        var v = part[tid];
        if (tid >= off) { v = v + part[tid - off]; }
        workgroupBarrier();
        part[tid] = v;
        workgroupBarrier();
    }
    let excl = part[tid] - mine;
    let above = part[255] & 0xffffu;
    let out0 = (h * nb + qt) * p.topk;
    var pa = excl & 0xffffu;
    var pe = above + (excl >> 16u);
    for (var kt = lo; kt < hi; kt = kt + 1u) {
        let key = okey(sc[kt]);
        if (key > t) {
            sel[out0 + pa] = kt;
            pa = pa + 1u;
        } else if (key == t) {
            if (pe < p.topk) { sel[out0 + pe] = kt; }
            pe = pe + 1u;
        }
    }
    workgroupBarrier();

    // Softmax weights once per tile, then the coarse output.
    var z = 0.0;
    for (var kt = tid; kt < nb; kt = kt + 256u) {
        let e = exp(sc[kt] - m);
        sc[kt] = e;
        z = z + e;
    }
    red[tid] = z;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
        workgroupBarrier();
    }
    let zt = red[0];
    if (tid < p.d) {
        var o = 0.0;
        for (var kt = 0u; kt < nb; kt = kt + 1u) {
            o = fma(sc[kt], pooled[2u * plane + (h * nb + kt) * p.d + tid], o);
        }
        coarse[qb + tid] = o / zt;
    }
}
