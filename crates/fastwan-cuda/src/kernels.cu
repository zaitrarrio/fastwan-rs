// fastwan CUDA kernels (NVRTC, no headers). Activations and weights are bf16
// stored as `unsigned short`; all arithmetic is f32. Each kernel is the CUDA
// twin of the WGSL kernel of the same name in fastwan-wgpu, which is verified
// against the CPU reference — keep the two in step.

typedef unsigned short bf16_t;
typedef unsigned long long u64;

__device__ __forceinline__ float ld(const bf16_t* p, long i) {
    return __uint_as_float(((unsigned)p[i]) << 16);
}
__device__ __forceinline__ bf16_t bf(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16_t)0x7fc0;  // NaN
    u += 0x7fffu + ((u >> 16) & 1u);                               // round to nearest even
    return (bf16_t)(u >> 16);
}
__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float block_sum256(float v, float* red) {
    const int t = threadIdx.x;
    // warp reduce, then across the 8 warps
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((t & 31) == 0) red[t >> 5] = v;
    __syncthreads();
    float r = 0.0f;
    if (t < 8) r = red[t];
    if (t < 32) for (int o = 4; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    if (t == 0) red[8] = r;
    __syncthreads();
    r = red[8];
    __syncthreads();
    return r;
}

// y[rows, n] = act(y + bias) in place (the epilogue of the cuBLAS GEMM).
extern "C" __global__ void fw_bias_act(u64 y_, const u64 b_, long total, int n, int has_bias, int act) {
    bf16_t* y = (bf16_t*)y_;
    const bf16_t* b = (const bf16_t*)b_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = ld(y, i);
    if (has_bias) v += ld(b, i % n);
    if (act == 1) v = gelu_tanh(v);
    y[i] = bf(v);
}

// mode 0: out = norm(h); mode 1: h += a·gate (or a); out = norm(h).
extern "C" __global__ void fw_norm(u64 h_, u64 a_, u64 mods_, u64 w_, u64 b_, u64 out_,
                                int rows, int dim, int mode, int gate, int has_gate,
                                int affine, int scale, int shift, float eps) {
    __shared__ float red[9];
    bf16_t* h = (bf16_t*)h_;
    const bf16_t *a = (const bf16_t*)a_, *mods = (const bf16_t*)mods_, *nw = (const bf16_t*)w_, *nb = (const bf16_t*)b_;
    bf16_t* out = (bf16_t*)out_;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const long base = (long)row * dim;
    float s = 0.0f;
    for (int j = threadIdx.x; j < dim; j += 256) {
        float v = ld(h, base + j);
        if (mode == 1) {
            float g = has_gate ? ld(mods, (long)gate * dim + j) : 1.0f;
            v += ld(a, base + j) * g;
            h[base + j] = bf(v);
        }
        s += v;
    }
    // Statistics over the stored (rounded) residual, re-read below.
    const float mean = block_sum256(s, red) / dim;
    float q = 0.0f;
    for (int j = threadIdx.x; j < dim; j += 256) { float d = ld(h, base + j) - mean; q += d * d; }
    const float inv = rsqrtf(block_sum256(q, red) / dim + eps);
    for (int j = threadIdx.x; j < dim; j += 256) {
        const float n = (ld(h, base + j) - mean) * inv;
        const float o = affine ? n * ld(nw, j) + ld(nb, j)
                               : n * (1.0f + ld(mods, (long)scale * dim + j)) + ld(mods, (long)shift * dim + j);
        out[base + j] = bf(o);
    }
}

extern "C" __global__ void fw_head_split(u64 src_, u64 w_, u64 cos_, u64 sin_, u64 out_,
                                      int rows, int stride, int col_off, int heads, int hd,
                                      int has_norm, int has_rope, float eps) {
    __shared__ float red[9];
    const bf16_t *src = (const bf16_t*)src_, *nw = (const bf16_t*)w_;
    const float *cs = (const float*)cos_, *sn = (const float*)sin_;
    bf16_t* out = (bf16_t*)out_;
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int width = heads * hd;
    const long base = (long)r * stride + col_off;
    float inv = 1.0f;
    if (has_norm) {
        float s = 0.0f;
        for (int j = threadIdx.x; j < width; j += 256) { float v = ld(src, base + j); s += v * v; }
        inv = rsqrtf(block_sum256(s, red) / width + eps);
    }
    const int half = hd / 2;
    for (int pi = threadIdx.x; pi < width / 2; pi += 256) {
        const int j = 2 * pi, hh = j / hd, c = j % hd;
        float x1 = ld(src, base + j), x2 = ld(src, base + j + 1);
        if (has_norm) { x1 = x1 * inv * ld(nw, j); x2 = x2 * inv * ld(nw, j + 1); }
        if (has_rope) {
            const long ti = (long)r * half + c / 2;
            const float co = cs[ti], si = sn[ti];
            const float y1 = x1 * co - x2 * si, y2 = x1 * si + x2 * co;
            x1 = y1; x2 = y2;
        }
        const long o = ((long)hh * rows + r) * hd + c;
        out[o] = bf(x1);
        out[o + 1] = bf(x2);
    }
}

// Flash attention, scalar f32 math: 64 query rows (one per thread) of one
// head per block; K/V tiles of BK keys in shared memory. Tensor-core (mma)
// attention is the next step: see docs/design.md.
#define BK 32
extern "C" __global__ void fw_attention(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int sq, int sk, int d, float scale) {
    __shared__ float ks[BK * 128];
    __shared__ float vs[BK * 128];
    const bf16_t *q = (const bf16_t*)q_, *k = (const bf16_t*)k_, *v = (const bf16_t*)v_;
    bf16_t* out = (bf16_t*)out_;
    const int h = blockIdx.y, i = blockIdx.x * 64 + threadIdx.x;
    const bool live = i < sq;
    float qr[128], acc[128], s[BK];
    for (int c = 0; c < d; ++c) { qr[c] = live ? ld(q, ((long)h * sq + i) * d + c) * scale : 0.0f; acc[c] = 0.0f; }
    float m = -3.0e38f, l = 0.0f;
    for (int j0 = 0; j0 < sk; j0 += BK) {
        const int nk = min(BK, sk - j0);
        for (int e = threadIdx.x; e < BK * d; e += 64) {
            const int jj = e / d;
            float kv = 0.0f, vv = 0.0f;
            if (jj < nk) { const long g = ((long)h * sk + j0 + jj) * d + (e % d); kv = ld(k, g); vv = ld(v, g); }
            ks[e] = kv; vs[e] = vv;
        }
        __syncthreads();
        if (live) {
            float mt = m;
            for (int jj = 0; jj < nk; ++jj) {
                float dot = 0.0f;
                for (int c = 0; c < d; ++c) dot = fmaf(qr[c], ks[jj * d + c], dot);
                s[jj] = dot; mt = fmaxf(mt, dot);
            }
            const float corr = __expf(m - mt);
            l *= corr;
            for (int c = 0; c < d; ++c) acc[c] *= corr;
            for (int jj = 0; jj < nk; ++jj) {
                const float p = __expf(s[jj] - mt);
                l += p;
                for (int c = 0; c < d; ++c) acc[c] = fmaf(p, vs[jj * d + c], acc[c]);
            }
            m = mt;
        }
        __syncthreads();
    }
    if (live) {
        const float invl = 1.0f / l;
        for (int c = 0; c < d; ++c) out[((long)i * heads + h) * d + c] = bf(acc[c] * invl);
    }
}

extern "C" __global__ void fw_gate_residual(u64 h_, u64 a_, u64 mods_, long n, int dim, int gate) {
    bf16_t* h = (bf16_t*)h_;
    const bf16_t *a = (const bf16_t*)a_, *mods = (const bf16_t*)mods_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = bf(ld(h, i) + ld(a, i) * ld(mods, (long)gate * dim + i % dim));
}

extern "C" __global__ void fw_dmd_step(u64 x_, u64 v_, u64 e_, long n, int has_noise, float st, float sn) {
    bf16_t* x = (bf16_t*)x_;
    const bf16_t *v = (const bf16_t*)v_, *e = (const bf16_t*)e_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x0 = ld(x, i) - st * ld(v, i);
    x[i] = bf(has_noise ? (1.0f - sn) * x0 + sn * ld(e, i) : x0);
}

extern "C" __global__ void fw_tanh_clamp(u64 x_, long n, float s) {
    bf16_t* x = (bf16_t*)x_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = bf(tanhf(ld(x, i) / s) * s);
}

// dir 0: patchify (tokens ← latent), dir 1: unpatchify (latent ← tokens).
extern "C" __global__ void fw_patch(u64 dst_, u64 src_, long n, int c, int t, int hh, int ww, int p, int dir) {
    bf16_t* dst = (bf16_t*)dst_;
    const bf16_t* src = (const bf16_t*)src_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int gw = ww / p, gh = hh / p, f = c * p * p;
    if (dir == 0) {
        const long tok = i / f; const int fi = i % f;
        const int cc = fi / (p * p), kh = (fi / p) % p, kw = fi % p;
        const long tt = tok / (gh * gw); const int yy = (tok / gw) % gh, xg = tok % gw;
        dst[i] = src[((cc * (long)t + tt) * hh + yy * p + kh) * ww + xg * p + kw];
    } else {
        const int xx = i % ww, yy = (i / ww) % hh, tt = (i / ((long)ww * hh)) % t;
        const long cc = i / ((long)ww * hh * t);
        const long tok = ((long)tt * gh + yy / p) * gw + xx / p;
        dst[i] = src[tok * f + ((yy % p) * p + xx % p) * c + cc];
    }
}

// Implicit-GEMM conv2d (see fastwan-wgpu conv.wgsl): 64×64 tile, 16×16
// threads, 4×4 outputs per thread at stride 16, K step 16.
extern "C" __global__ void fw_conv2d(u64 a_, u64 b_, u64 w_, u64 bias_, u64 res_, u64 out_,
                                  int frames, int cin_a, int cin_b, int cout, int h, int w,
                                  int up, int k, int has_bias, int ep) {
    __shared__ float xs[16][64];
    __shared__ float wsm[16][64];
    const bf16_t *a = (const bf16_t*)a_, *b = (const bf16_t*)b_, *wt = (const bf16_t*)w_;
    const bf16_t *bias = (const bf16_t*)bias_, *res = (const bf16_t*)res_;
    bf16_t* out = (bf16_t*)out_;
    const int oh = h * up, ow = w * up, kk2 = k * k, kt = (cin_a + cin_b) * kk2, pad = k / 2;
    const long plane = (long)oh * ow, mt = frames * plane;
    const long m0 = ((long)blockIdx.y + (long)blockIdx.z * gridDim.y) * 64;
    const int n0 = blockIdx.x * 64, tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < kt; k0 += 16) {
        for (int i = 0; i < 4; ++i) {
            const int e = tid + i * 256, r = e / 16, kk = e % 16, gk = k0 + kk;
            float xv = 0.0f;
            const long m = m0 + r;
            if (m < mt && gk < kt) {
                const long f = m / plane, rr = m % plane;
                const int oy = rr / ow, ox = rr % ow;
                const int ci = gk / kk2, q = gk % kk2, ky = q / k, kx = q % k;
                const int yy = oy + ky - pad, xx = ox + kx - pad;
                if (yy >= 0 && xx >= 0 && yy < oh && xx < ow) {
                    const int sy = yy / up, sx = xx / up;
                    xv = ci < cin_a ? ld(a, ((f * cin_a + ci) * h + sy) * w + sx)
                                    : ld(b, ((f * cin_b + ci - cin_a) * h + sy) * w + sx);
                }
            }
            xs[kk][r] = xv;
            wsm[kk][r] = (n0 + r < cout && gk < kt) ? ld(wt, (long)(n0 + r) * kt + gk) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float av[4], bv[4];
            for (int i = 0; i < 4; ++i) { av[i] = xs[kk][ty + 16 * i]; bv[i] = wsm[kk][tx + 16 * i]; }
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const long m = m0 + ty + 16 * i;
        if (m >= mt) continue;
        const long f = m / plane, px = m % plane;
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + 16 * j;
            if (n >= cout) continue;
            float v = acc[i][j] + (has_bias ? ld(bias, n) : 0.0f);
            const long o = (f * cout + n) * plane + px;
            if (ep == 1) v = fmaxf(v, 0.0f);
            if (ep == 2) v = fmaxf(v + ld(res, o), 0.0f);
            if (ep == 3 && v < 0.0f) v *= ld(res, n);
            out[o] = bf(v);
        }
    }
}

// Tensor-core flash attention for head_dim 128 (sm_80+): FlashAttention-2
// structure on bf16 mma.sync.m16n8k16 with f32 accumulation. One block = 4
// warps = 64 query rows of one head (16 per warp); K and Vᵀ tiles of 64 keys
// in shared memory; online softmax in base 2. The S accumulator fragments
// are reused directly as the A operand of P·V (the C and A layouts of
// m16n8k16 line up), so P never touches shared memory.
#define MMA_BK 64
#define MMA_D 128
#define KS_STRIDE (MMA_D + 8)
#define VT_STRIDE (MMA_BK + 8)

__device__ __forceinline__ void mma_bf16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    return (unsigned)bf(lo) | ((unsigned)bf(hi) << 16);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

// Body shared by dense attention (mode 0) and the VSA fine stage (mode 1).
// Mode 1: the block is query tile blockIdx.x (rows via `slot`), key chunk c is
// selected tile sel[h][qt][c], and the epilogue adds coarse[h][qt] · gate.
//
// FlashAttention-2 structure: 4 warps × 16 query rows, 64-key chunks. K and V
// chunks arrive row-major by 16-byte cp.async (invalid keys zero-filled) and
// are pipelined: chunk c+1's K loads during softmax and P·V of chunk c, and
// its V during the next Q·Kᵀ. V fragments come from ldmatrix.trans, so V is
// never transposed in shared memory.
__device__ __forceinline__ void attn_mma_body(int mode, const bf16_t* q, const bf16_t* k, const bf16_t* v, bf16_t* out,
        int heads, int sq, int sk, float scale, const unsigned* slot, const unsigned* sel,
        const float* coarse, const bf16_t* gate, int nb, int topk,
        bf16_t* ks, bf16_t* vs, unsigned char* kvalid) {
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int qt = blockIdx.x;
    const int lr0 = warp * 16 + g, lr1 = lr0 + 8;
    int tok0, tok1;
    if (mode == 0) {
        tok0 = qt * 64 + lr0 < sq ? qt * 64 + lr0 : -1;
        tok1 = qt * 64 + lr1 < sq ? qt * 64 + lr1 : -1;
    } else {
        const unsigned a = slot[qt * 64 + lr0], b = slot[qt * 64 + lr1];
        tok0 = a == 0xffffffffu ? -1 : (int)a;
        tok1 = b == 0xffffffffu ? -1 : (int)b;
    }
    const float sl2 = scale * 1.4426950408889634f;

    unsigned qa[8][4];
    const unsigned* q0 = (const unsigned*)(q + ((long)h * sq + (tok0 < 0 ? 0 : tok0)) * MMA_D);
    const unsigned* q1 = (const unsigned*)(q + ((long)h * sq + (tok1 < 0 ? 0 : tok1)) * MMA_D);
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const int c = (kk * 16 + t * 2) >> 1;
        qa[kk][0] = tok0 >= 0 ? q0[c] : 0u;
        qa[kk][1] = tok1 >= 0 ? q1[c] : 0u;
        qa[kk][2] = tok0 >= 0 ? q0[c + 4] : 0u;
        qa[kk][3] = tok1 >= 0 ? q1[c + 4] : 0u;
    }
    float o[16][4];
    #pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;

    const int nchunks = mode == 0 ? (sk + MMA_BK - 1) / MMA_BK : topk;
    const bf16_t* kh = k + (long)h * sk * MMA_D;
    const bf16_t* vh = v + (long)h * sk * MMA_D;
    // Loader: thread → 16-byte column ch of keys (tid>>4) + 8j, j < 8.
    const int ch = tid & 15, krow = tid >> 4;
    auto key_tok = [&](int c, int key) -> int {
        if (mode == 0) return c * MMA_BK + key < sk ? c * MMA_BK + key : -1;
        const unsigned s2 = slot[(long)sel[((long)h * nb + qt) * topk + c] * 64 + key];
        return s2 == 0xffffffffu ? -1 : (int)s2;
    };
    auto load = [&](bf16_t* dst, const bf16_t* src, int c, unsigned char* valid) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int key = krow + 8 * j, tk = key_tok(c, key);
            cp_async16(dst + key * KS_STRIDE + ch * 8, src + (long)(tk < 0 ? 0 : tk) * MMA_D + ch * 8, tk >= 0);
            if (valid && ch == 0) valid[key] = tk >= 0;
        }
    };
    load(ks, kh, 0, kvalid);
    cp_async_commit();
    load(vs, vh, 0, 0);
    cp_async_commit();

    for (int c = 0; c < nchunks; ++c) {
        const unsigned char* kv = kvalid + (c & 1) * MMA_BK;
        cp_async_wait1();          // K_c landed (V_c may still be in flight)
        __syncthreads();
        float s[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            const bf16_t* kr = ks + (nt * 8 + g) * KS_STRIDE;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const unsigned b0 = *(const unsigned*)(kr + kk * 16 + t * 2);
                const unsigned b1 = *(const unsigned*)(kr + kk * 16 + 8 + t * 2);
                mma_bf16(s[nt], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
            }
        }
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const bool ok = kv[nt * 8 + t * 2 + cc];
                s[nt][cc] = ok ? s[nt][cc] * sl2 : -1e30f;
                s[nt][2 + cc] = ok ? s[nt][2 + cc] * sl2 : -1e30f;
                mx0 = fmaxf(mx0, s[nt][cc]);
                mx1 = fmaxf(mx1, s[nt][2 + cc]);
            }
        }
        __syncthreads();           // every warp is done with ks
        if (c + 1 < nchunks) load(ks, kh, c + 1, kvalid + ((c + 1) & 1) * MMA_BK);
        cp_async_commit();
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float c0 = exp2f(m0 - mx0), c1 = exp2f(m1 - mx1);
        m0 = mx0; m1 = mx1;
        l0 *= c0; l1 *= c1;
        #pragma unroll
        for (int i = 0; i < 16; ++i) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
        cp_async_wait1();          // V_c landed (K_{c+1} may still be in flight)
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            float p[8];
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                p[cc] = exp2f(s[2 * kk][cc] - (cc < 2 ? m0 : m1));
                p[4 + cc] = exp2f(s[2 * kk + 1][cc] - (cc < 2 ? m0 : m1));
            }
            l0 += p[0] + p[1] + p[4] + p[5];
            l1 += p[2] + p[3] + p[6] + p[7];
            const unsigned a0 = pack_bf16(p[0], p[1]), a1 = pack_bf16(p[2], p[3]);
            const unsigned a2 = pack_bf16(p[4], p[5]), a3 = pack_bf16(p[6], p[7]);
            // ldmatrix.x4.trans: matrices (keys +0..7 | +8..15) × (dims dt·8 | dt·8 + 8).
            const int vkey = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            #pragma unroll
            for (int dt = 0; dt < 16; dt += 2) {
                const unsigned addr = (unsigned)__cvta_generic_to_shared(vs + vkey * KS_STRIDE + dt * 8 + (lane >> 4) * 8);
                unsigned b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                mma_bf16(o[dt], a0, a1, a2, a3, b0, b1);
                mma_bf16(o[dt + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        __syncthreads();           // every warp is done with vs
        if (c + 1 < nchunks) load(vs, vh, c + 1, 0);
        cp_async_commit();
    }
    #pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        const int c = dt * 8 + t * 2;
        float a0 = o[dt][0] * i0, a1 = o[dt][1] * i0, b0 = o[dt][2] * i1, b1 = o[dt][3] * i1;
        if (mode == 1) {
            const float* cr = coarse + ((long)h * nb + qt) * MMA_D + c;
            if (tok0 >= 0) { const bf16_t* gr = gate + ((long)h * sq + tok0) * MMA_D + c; a0 += cr[0] * ld(gr, 0); a1 += cr[1] * ld(gr, 1); }
            if (tok1 >= 0) { const bf16_t* gr = gate + ((long)h * sq + tok1) * MMA_D + c; b0 += cr[0] * ld(gr, 0); b1 += cr[1] * ld(gr, 1); }
        }
        if (tok0 >= 0) *(unsigned*)(out + ((long)tok0 * heads + h) * MMA_D + c) = pack_bf16(a0, a1);
        if (tok1 >= 0) *(unsigned*)(out + ((long)tok1 * heads + h) * MMA_D + c) = pack_bf16(b0, b1);
    }
}

extern "C" __global__ void __launch_bounds__(128) fw_attention_mma(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int sq, int sk, float scale) {
    __shared__ __align__(16) bf16_t ks[MMA_BK * KS_STRIDE];
    __shared__ __align__(16) bf16_t vs[MMA_BK * KS_STRIDE];
    __shared__ unsigned char kvalid[2 * MMA_BK];
    attn_mma_body(0, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, sq, sk, scale,
                  0, 0, 0, 0, 0, 0, ks, vs, kvalid);
}

extern "C" __global__ void __launch_bounds__(128) fw_vsa_fine_mma(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int seq, float scale,
        u64 slot_, u64 sel_, u64 coarse_, u64 gate_, int nb, int topk) {
    __shared__ __align__(16) bf16_t ks[MMA_BK * KS_STRIDE];
    __shared__ __align__(16) bf16_t vs[MMA_BK * KS_STRIDE];
    __shared__ unsigned char kvalid[2 * MMA_BK];
    attn_mma_body(1, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, seq, seq, scale,
                  (const unsigned*)slot_, (const unsigned*)sel_, (const float*)coarse_, (const bf16_t*)gate_, nb, topk, ks, vs, kvalid);
}

// Variants of attn_mma_body for A/B (selected on the host by
// FASTWAN_CUDA_ATTN_VARIANT; v0 above stays the default until measured):
//   V = 1: two barriers per key chunk instead of four (V_c is issued after
//          the top barrier, K_{c+1} after the V-ready barrier), and the
//          output is rescaled only when some row's running max moved.
//   V = 2: V = 1 plus unpadded shared tiles with an XOR swizzle of the
//          16-byte column chunks (bank-conflict-free for the 32-bit K reads
//          and ldmatrix.trans V reads) and __launch_bounds__(128, 3), for
//          three blocks (12 warps) per SM.
template <int V>
__device__ __forceinline__ int att_sidx(int row, int col) {
    if (V == 2) return row * MMA_D + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7);
    return row * KS_STRIDE + col;
}

template <int V>
__device__ __forceinline__ void attn_mma_v(int mode, const bf16_t* q, const bf16_t* k, const bf16_t* v, bf16_t* out,
        int heads, int sq, int sk, float scale, const unsigned* slot, const unsigned* sel,
        const float* coarse, const bf16_t* gate, int nb, int topk,
        bf16_t* ks, bf16_t* vs, unsigned char* kvalid) {
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int qt = blockIdx.x;
    const int lr0 = warp * 16 + g, lr1 = lr0 + 8;
    int tok0, tok1;
    if (mode == 0) {
        tok0 = qt * 64 + lr0 < sq ? qt * 64 + lr0 : -1;
        tok1 = qt * 64 + lr1 < sq ? qt * 64 + lr1 : -1;
    } else {
        const unsigned a = slot[qt * 64 + lr0], b = slot[qt * 64 + lr1];
        tok0 = a == 0xffffffffu ? -1 : (int)a;
        tok1 = b == 0xffffffffu ? -1 : (int)b;
    }
    const float sl2 = scale * 1.4426950408889634f;
    unsigned qa[8][4];
    const unsigned* q0 = (const unsigned*)(q + ((long)h * sq + (tok0 < 0 ? 0 : tok0)) * MMA_D);
    const unsigned* q1 = (const unsigned*)(q + ((long)h * sq + (tok1 < 0 ? 0 : tok1)) * MMA_D);
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const int c = (kk * 16 + t * 2) >> 1;
        qa[kk][0] = tok0 >= 0 ? q0[c] : 0u;
        qa[kk][1] = tok1 >= 0 ? q1[c] : 0u;
        qa[kk][2] = tok0 >= 0 ? q0[c + 4] : 0u;
        qa[kk][3] = tok1 >= 0 ? q1[c + 4] : 0u;
    }
    float o[16][4];
    #pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;

    const int nchunks = mode == 0 ? (sk + MMA_BK - 1) / MMA_BK : topk;
    const bf16_t* kh = k + (long)h * sk * MMA_D;
    const bf16_t* vh = v + (long)h * sk * MMA_D;
    const int ch = tid & 15, krow = tid >> 4;
    auto key_tok = [&](int c, int key) -> int {
        if (mode == 0) return c * MMA_BK + key < sk ? c * MMA_BK + key : -1;
        const unsigned s2 = slot[(long)sel[((long)h * nb + qt) * topk + c] * 64 + key];
        return s2 == 0xffffffffu ? -1 : (int)s2;
    };
    auto load = [&](bf16_t* dst, const bf16_t* src, int c, unsigned char* valid) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int key = krow + 8 * j, tk = key_tok(c, key);
            cp_async16(dst + att_sidx<V>(key, ch * 8), src + (long)(tk < 0 ? 0 : tk) * MMA_D + ch * 8, tk >= 0);
            if (valid && ch == 0) valid[key] = tk >= 0;
        }
    };
    load(ks, kh, 0, kvalid);
    cp_async_commit();

    for (int c = 0; c < nchunks; ++c) {
        const unsigned char* kvb = kvalid + (c & 1) * MMA_BK;
        cp_async_wait0();          // K_c landed
        __syncthreads();           // ... for everyone; and every warp is past P·V(c-1): vs is free
        load(vs, vh, c, 0);        // V_c streams in during Q·Kᵀ and softmax
        cp_async_commit();
        float s[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const unsigned b0 = *(const unsigned*)(ks + att_sidx<V>(nt * 8 + g, kk * 16 + t * 2));
                const unsigned b1 = *(const unsigned*)(ks + att_sidx<V>(nt * 8 + g, kk * 16 + 8 + t * 2));
                mma_bf16(s[nt], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
            }
        }
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const bool ok = kvb[nt * 8 + t * 2 + cc];
                s[nt][cc] = ok ? s[nt][cc] * sl2 : -1e30f;
                s[nt][2 + cc] = ok ? s[nt][2 + cc] * sl2 : -1e30f;
                mx0 = fmaxf(mx0, s[nt][cc]);
                mx1 = fmaxf(mx1, s[nt][2 + cc]);
            }
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        // Rescale only when some row of this warp has a new max.
        if (__any_sync(0xffffffffu, mx0 != m0 || mx1 != m1)) {
            const float c0 = exp2f(m0 - mx0), c1 = exp2f(m1 - mx1);
            l0 *= c0; l1 *= c1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
            m0 = mx0; m1 = mx1;
        }
        cp_async_wait0();          // V_c landed
        __syncthreads();           // ... for everyone; and every warp is past Q·Kᵀ(c): ks is free
        if (c + 1 < nchunks) load(ks, kh, c + 1, kvalid + ((c + 1) & 1) * MMA_BK);
        cp_async_commit();         // K_{c+1} streams in during P·V(c)
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            float p[8];
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                p[cc] = exp2f(s[2 * kk][cc] - (cc < 2 ? m0 : m1));
                p[4 + cc] = exp2f(s[2 * kk + 1][cc] - (cc < 2 ? m0 : m1));
            }
            l0 += p[0] + p[1] + p[4] + p[5];
            l1 += p[2] + p[3] + p[6] + p[7];
            const unsigned a0 = pack_bf16(p[0], p[1]), a1 = pack_bf16(p[2], p[3]);
            const unsigned a2 = pack_bf16(p[4], p[5]), a3 = pack_bf16(p[6], p[7]);
            const int vkey = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            #pragma unroll
            for (int dt = 0; dt < 16; dt += 2) {
                const unsigned addr = (unsigned)__cvta_generic_to_shared(vs + att_sidx<V>(vkey, dt * 8 + (lane >> 4) * 8));
                unsigned b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                mma_bf16(o[dt], a0, a1, a2, a3, b0, b1);
                mma_bf16(o[dt + 1], a0, a1, a2, a3, b2, b3);
            }
        }
    }
    #pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        const int c = dt * 8 + t * 2;
        float a0 = o[dt][0] * i0, a1 = o[dt][1] * i0, b0 = o[dt][2] * i1, b1 = o[dt][3] * i1;
        if (mode == 1) {
            const float* cr = coarse + ((long)h * nb + qt) * MMA_D + c;
            if (tok0 >= 0) { const bf16_t* gr = gate + ((long)h * sq + tok0) * MMA_D + c; a0 += cr[0] * ld(gr, 0); a1 += cr[1] * ld(gr, 1); }
            if (tok1 >= 0) { const bf16_t* gr = gate + ((long)h * sq + tok1) * MMA_D + c; b0 += cr[0] * ld(gr, 0); b1 += cr[1] * ld(gr, 1); }
        }
        if (tok0 >= 0) *(unsigned*)(out + ((long)tok0 * heads + h) * MMA_D + c) = pack_bf16(a0, a1);
        if (tok1 >= 0) *(unsigned*)(out + ((long)tok1 * heads + h) * MMA_D + c) = pack_bf16(b0, b1);
    }
}

#define ATT_V1_SMEM (MMA_BK * KS_STRIDE)
#define ATT_V2_SMEM (MMA_BK * MMA_D)
extern "C" __global__ void __launch_bounds__(128) fw_attention_mma_v1(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int sq, int sk, float scale) {
    __shared__ __align__(16) bf16_t ks[ATT_V1_SMEM];
    __shared__ __align__(16) bf16_t vs[ATT_V1_SMEM];
    __shared__ unsigned char kvalid[2 * MMA_BK];
    attn_mma_v<1>(0, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, sq, sk, scale,
                  0, 0, 0, 0, 0, 0, ks, vs, kvalid);
}
extern "C" __global__ void __launch_bounds__(128) fw_vsa_fine_mma_v1(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int seq, float scale,
        u64 slot_, u64 sel_, u64 coarse_, u64 gate_, int nb, int topk) {
    __shared__ __align__(16) bf16_t ks[ATT_V1_SMEM];
    __shared__ __align__(16) bf16_t vs[ATT_V1_SMEM];
    __shared__ unsigned char kvalid[2 * MMA_BK];
    attn_mma_v<1>(1, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, seq, seq, scale,
                  (const unsigned*)slot_, (const unsigned*)sel_, (const float*)coarse_, (const bf16_t*)gate_, nb, topk, ks, vs, kvalid);
}
extern "C" __global__ void __launch_bounds__(128, 3) fw_attention_mma_v2(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int sq, int sk, float scale) {
    __shared__ __align__(16) bf16_t ks[ATT_V2_SMEM];
    __shared__ __align__(16) bf16_t vs[ATT_V2_SMEM];
    __shared__ unsigned char kvalid[2 * MMA_BK];
    attn_mma_v<2>(0, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, sq, sk, scale,
                  0, 0, 0, 0, 0, 0, ks, vs, kvalid);
}
extern "C" __global__ void __launch_bounds__(128, 3) fw_vsa_fine_mma_v2(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int seq, float scale,
        u64 slot_, u64 sel_, u64 coarse_, u64 gate_, int nb, int topk) {
    __shared__ __align__(16) bf16_t ks[ATT_V2_SMEM];
    __shared__ __align__(16) bf16_t vs[ATT_V2_SMEM];
    __shared__ unsigned char kvalid[2 * MMA_BK];
    attn_mma_v<2>(1, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, seq, seq, scale,
                  (const unsigned*)slot_, (const unsigned*)sel_, (const float*)coarse_, (const bf16_t*)gate_, nb, topk, ks, vs, kvalid);
}

// VSA tile means (f32 out): block (tile, head), thread c < d.
extern "C" __global__ void fw_vsa_pool(u64 q_, u64 k_, u64 v_, u64 slot_, u64 bsz_, u64 pooled_, int heads, int seq, int d, int nb) {
    const bf16_t *q = (const bf16_t*)q_, *k = (const bf16_t*)k_, *v = (const bf16_t*)v_;
    const unsigned *slot = (const unsigned*)slot_, *bsz = (const unsigned*)bsz_;
    float* pooled = (float*)pooled_;
    const int t = blockIdx.x, h = blockIdx.y, c = threadIdx.x;
    if (c >= d) return;
    const unsigned n = bsz[t];
    float sq = 0.f, sk = 0.f, sv = 0.f;
    for (unsigned i = 0; i < n; ++i) {
        const long idx = ((long)h * seq + slot[t * 64 + i]) * d + c;
        sq += ld(q, idx); sk += ld(k, idx); sv += ld(v, idx);
    }
    const float inv = 1.0f / (float)(n ? n : 1);
    const long o = ((long)h * nb + t) * d + c, plane = (long)heads * nb * d;
    pooled[o] = sq * inv; pooled[plane + o] = sk * inv; pooled[2 * plane + o] = sv * inv;
}

__device__ __forceinline__ unsigned okey(float x) {
    const unsigned b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// VSA coarse stage: block (query tile, head), 256 threads; nb <= 2048,
// d <= 128 and d % 4 == 0. Scores: one warp per key tile (float4 rows,
// shuffle reduce). Top-k: radix select on order-preserving keys, 8 bits per
// pass (4 passes of a 256-bin histogram), then a ballot / prefix-sum
// compaction that writes the selected tiles in ascending order (ties at the
// threshold taken lowest index first, as the CPU reference does).
extern "C" __global__ void __launch_bounds__(256) fw_vsa_coarse(u64 pooled_, u64 coarse_, u64 sel_, int heads, int d, int nb, int topk, float scale) {
    __shared__ float sc[2048];
    __shared__ float qrow[128];
    __shared__ float part[128];
    __shared__ unsigned hist[256];
    __shared__ unsigned wsum[8];
    __shared__ float red[9];
    __shared__ unsigned sh_prefix, sh_need;
    const float* pooled = (const float*)pooled_;
    float* coarse = (float*)coarse_;
    unsigned* sel = (unsigned*)sel_;
    const int qt = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long plane = (long)heads * nb * d, qb = ((long)h * nb + qt) * d;
    if (tid < d) qrow[tid] = pooled[qb + tid];
    __syncthreads();

    // 1. scores
    const int d4 = d >> 2;
    for (int kt = warp; kt < nb; kt += 8) {
        const float4* kr = (const float4*)(pooled + plane + ((long)h * nb + kt) * d);
        float dot = 0.f;
        for (int c = lane; c < d4; c += 32) {
            const float4 kv = kr[c];
            dot += qrow[4 * c] * kv.x + qrow[4 * c + 1] * kv.y + qrow[4 * c + 2] * kv.z + qrow[4 * c + 3] * kv.w;
        }
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) sc[kt] = dot * scale;
    }
    __syncthreads();

    // 2. radix select of the topk-th largest key
    unsigned prefix = 0u, mask = 0u, need = (unsigned)topk;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        hist[tid] = 0u;
        __syncthreads();
        for (int kt = tid; kt < nb; kt += 256) {
            const unsigned key = okey(sc[kt]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (warp == 0) {
            // lane l owns bins 255-8l .. 248-8l (descending).
            unsigned c[8], s = 0u;
            #pragma unroll
            for (int b = 0; b < 8; ++b) { c[b] = hist[255 - 8 * lane - b]; s += c[b]; }
            unsigned incl = s;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned y = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += y;
            }
            const unsigned excl = incl - s;
            const unsigned hit = __ballot_sync(0xffffffffu, incl >= need && excl < need);
            if (lane == __ffs(hit) - 1) {
                unsigned above = excl;
                int b = 0;
                for (; b < 7 && above + c[b] < need; ++b) above += c[b];
                const unsigned bin = 255u - 8u * lane - b;
                sh_prefix = prefix | (bin << shift);
                sh_need = need - above;
            }
        }
        __syncthreads();
        prefix = sh_prefix; need = sh_need; mask |= 255u << shift;
        __syncthreads();
    }
    // prefix = threshold key T; take every key > T and the first `need` keys == T.

    // 3. stable compaction in ascending kt
    unsigned* o = sel + ((long)h * nb + qt) * topk;
    const unsigned lt = (1u << lane) - 1u;
    unsigned base_eq = 0u, base_sel = 0u;
    for (int c0 = 0; c0 < nb; c0 += 256) {
        const int kt = c0 + tid;
        const unsigned key = kt < nb ? okey(sc[kt]) : 0u;
        const bool eq = kt < nb && key == prefix, gt = kt < nb && key > prefix;
        unsigned b = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) wsum[warp] = __popc(b);
        __syncthreads();
        unsigned woff = 0u, tot = 0u;
        for (int w = 0; w < 8; ++w) { if (w < warp) woff += wsum[w]; tot += wsum[w]; }
        const unsigned tie_rank = base_eq + woff + __popc(b & lt);
        base_eq += tot;
        __syncthreads();
        const bool take = gt || (eq && tie_rank < need);
        b = __ballot_sync(0xffffffffu, take);
        if (lane == 0) wsum[warp] = __popc(b);
        __syncthreads();
        woff = 0u; tot = 0u;
        for (int w = 0; w < 8; ++w) { if (w < warp) woff += wsum[w]; tot += wsum[w]; }
        if (take) o[base_sel + woff + __popc(b & lt)] = (unsigned)kt;
        base_sel += tot;
        __syncthreads();
    }

    // 4. coarse output: softmax over the tile means, all 256 threads.
    float mx = -3.0e38f;
    for (int kt = tid; kt < nb; kt += 256) mx = fmaxf(mx, sc[kt]);
    for (int s2 = 16; s2 > 0; s2 >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s2));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    float m = red[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float z = 0.f;
    for (int kt = tid; kt < nb; kt += 256) { const float e = __expf(sc[kt] - m); sc[kt] = e; z += e; }
    const float zt = block_sum256(z, red);   // syncs, so sc[] holds the weights
    const int c = tid & 127, half = tid >> 7, mid = (nb + 1) >> 1;
    float acc = 0.f;
    if (c < d) {
        const float* vp = pooled + 2 * plane + (long)h * nb * d + c;
        for (int kt = half ? mid : 0; kt < (half ? nb : mid); ++kt) acc = fmaf(sc[kt], vp[(long)kt * d], acc);
        if (half) part[c] = acc;
    }
    __syncthreads();
    if (!half && c < d) coarse[qb + c] = (acc + part[c]) / zt;
}

// Scalar VSA fine stage (any d <= 128, any SM): 64 threads = the 64 slots of
// query tile blockIdx.x; selected tiles streamed through shared memory in
// halves of 32 keys.
extern "C" __global__ void fw_vsa_fine(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int seq, int d, float scale,
        u64 slot_, u64 sel_, u64 coarse_, u64 gate_, int nb, int topk) {
    __shared__ float kss[32 * 128];
    __shared__ float vss[32 * 128];
    __shared__ int ktok[32];
    const bf16_t *q = (const bf16_t*)q_, *k = (const bf16_t*)k_, *v = (const bf16_t*)v_, *gate = (const bf16_t*)gate_;
    const unsigned *slot = (const unsigned*)slot_, *sel = (const unsigned*)sel_;
    const float* coarse = (const float*)coarse_;
    bf16_t* out = (bf16_t*)out_;
    const int qt = blockIdx.x, h = blockIdx.y;
    const unsigned qs = slot[qt * 64 + threadIdx.x];
    const bool live = qs != 0xffffffffu;
    float qr[128], acc[128];
    for (int c = 0; c < d; ++c) { qr[c] = live ? ld(q, ((long)h * seq + qs) * d + c) * scale : 0.f; acc[c] = 0.f; }
    float m = -1e30f, l = 0.f;
    for (int ci = 0; ci < topk; ++ci) {
        const unsigned tile = sel[((long)h * nb + qt) * topk + ci];
        for (int half = 0; half < 2; ++half) {
            __syncthreads();
            if (threadIdx.x < 32) { const unsigned s = slot[tile * 64 + half * 32 + threadIdx.x]; ktok[threadIdx.x] = s == 0xffffffffu ? -1 : (int)s; }
            __syncthreads();
            for (int e = threadIdx.x; e < 32 * d; e += 64) {
                const int j = e / d, c = e % d;
                const int kt = ktok[j];
                kss[e] = kt >= 0 ? ld(k, ((long)h * seq + kt) * d + c) : 0.f;
                vss[e] = kt >= 0 ? ld(v, ((long)h * seq + kt) * d + c) : 0.f;
            }
            __syncthreads();
            if (live) {
                for (int j = 0; j < 32; ++j) {
                    if (ktok[j] < 0) continue;
                    float dot = 0.f;
                    for (int c = 0; c < d; ++c) dot = fmaf(qr[c], kss[j * d + c], dot);
                    const float mn = fmaxf(m, dot), corr = __expf(m - mn), p = __expf(dot - mn);
                    l = l * corr + p;
                    for (int c = 0; c < d; ++c) acc[c] = acc[c] * corr + p * vss[j * d + c];
                    m = mn;
                }
            }
        }
    }
    if (live) {
        const float* cr = coarse + ((long)h * nb + qt) * d;
        for (int c = 0; c < d; ++c)
            out[((long)qs * heads + h) * d + c] = bf(acc[c] / l + cr[c] * ld(gate, ((long)h * seq + qs) * d + c));
    }
}

// int8 weight-only (fastwan_core::quant::Q8): codes packed 2 per u16 word,
// one f32 scale per `group` columns. Expands to bf16 `q · s` for cuBLAS — the
// twin of `load_w` in the WGSL `_q8` GEMMs.
extern "C" __global__ void fw_dequant_q8(u64 out_, u64 q_, u64 s_, long n, int group) {
    bf16_t* out = (bf16_t*)out_;
    const signed char* q = (const signed char*)q_;
    const float* s = (const float*)s_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf((float)q[i] * s[i / group]);
}

// 4-bit weight-only (fastwan_core::quant::Q4): codes packed 2 per byte (low
// nibble first), (scale, zero) f32 pairs per `group` columns. Expands to
// bf16 `(q − zero) · scale` — the twin of the WGSL `_q4` GEMMs' `load_w`.
extern "C" __global__ void fw_dequant_q4(u64 out_, u64 q_, u64 sz_, long n, int group) {
    bf16_t* out = (bf16_t*)out_;
    const unsigned char* q = (const unsigned char*)q_;
    const float* sz = (const float*)sz_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = (float)((q[i >> 1] >> ((i & 1) * 4)) & 15);
    long g = i / group;
    out[i] = bf((c - sz[2 * g + 1]) * sz[2 * g]);
}

// Tensor-core implicit-GEMM conv (sm_80+), same contract as fw_conv2d:
// M = output pixels (frames·oh·ow), N = cout, K = (cin_a + cin_b)·k² in
// (ci, ky, kx) order, bf16 mma.sync.m16n8k16 with f32 accumulation.
// Block tile 128 (M) × 64 (N) × 32 (K), 4 warps as 2 × 2, 64 × 32 per warp.
// Each thread owns one pixel row of the A tile: its (f, oy, ox) is computed
// once, and (ci, ky, kx) advance by counting, so the gather has no divisions.
// The next K tile is fetched into registers while the current one is used.
#define CV_BM 128
#define CV_BN 64
#define CV_BK 32
#define CV_LD 20   // u32 words per smem row: 32 bf16 + 8 pad; conflict-free fragment reads

extern "C" __global__ void __launch_bounds__(128) fw_conv2d_mma(
        u64 a_, u64 b_, u64 w_, u64 bias_, u64 res_, u64 out_,
        int frames, int cin_a, int cin_b, int cout, int h, int w,
        int up, int k, int has_bias, int ep) {
    __shared__ unsigned As[CV_BM * CV_LD];
    __shared__ unsigned Bs[CV_BN * CV_LD];
    const unsigned short *a = (const unsigned short*)a_, *b = (const unsigned short*)b_;
    const unsigned short *wt = (const unsigned short*)w_;
    const bf16_t *bias = (const bf16_t*)bias_, *res = (const bf16_t*)res_;
    bf16_t* out = (bf16_t*)out_;
    const int oh = h * up, ow = w * up, kk2 = k * k, kt = (cin_a + cin_b) * kk2, pad = k / 2;
    const long plane = (long)oh * ow, mt = (long)frames * plane;
    const long m0 = ((long)blockIdx.y + (long)blockIdx.z * gridDim.y) * CV_BM;
    const int n0 = blockIdx.x * CV_BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1, g = lane >> 2, t = lane & 3;

    // This thread's A row: one output pixel.
    const long m = m0 + tid;
    const bool mok = m < mt;
    long fa = 0, fb = 0;
    int oy = 0, ox = 0;
    if (mok) {
        const long f = m / plane, rr = m - f * plane;
        oy = (int)(rr / ow); ox = (int)(rr - (long)oy * ow);
        fa = f * cin_a * (long)h * w;
        fb = f * (long)cin_b * h * w;
    }
    const long hw = (long)h * w;
    // B: 64 rows × 32 k; thread → row tid/2, k half (tid&1)·16.
    const int bn = tid >> 1, bh = (tid & 1) * 16;
    const bool bok = n0 + bn < cout;
    const unsigned short* wrow = wt + (long)(n0 + bn) * kt;

    unsigned ra[16], rb[8];
    auto fetch = [&](int k0) {
        // A: 32 consecutive k of this pixel, (ci, ky, kx) counted from k0.
        int ci = k0 / kk2, q = k0 - ci * kk2, ky = q / k, kx = q - ky * k;
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            unsigned v[2];
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int gk = k0 + 2 * p + e;
                unsigned short x = 0;
                if (mok && gk < kt) {
                    const int yy = oy + ky - pad, xx = ox + kx - pad;
                    if (yy >= 0 && xx >= 0 && yy < oh && xx < ow) {
                        const int sy = up == 2 ? yy >> 1 : yy, sx = up == 2 ? xx >> 1 : xx;
                        x = ci < cin_a ? a[fa + ci * hw + (long)sy * w + sx]
                                       : b[fb + (ci - cin_a) * hw + (long)sy * w + sx];
                    }
                }
                v[e] = x;
                if (++kx == k) { kx = 0; if (++ky == k) { ky = 0; ++ci; } }
            }
            ra[p] = v[0] | (v[1] << 16);
        }
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int gk = k0 + bh + 2 * p;
            const unsigned lo = (bok && gk < kt) ? wrow[gk] : 0u;
            const unsigned hi = (bok && gk + 1 < kt) ? wrow[gk + 1] : 0u;
            rb[p] = lo | (hi << 16);
        }
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

    fetch(0);
    for (int k0 = 0; k0 < kt; k0 += CV_BK) {
        #pragma unroll
        for (int p = 0; p < 16; ++p) As[tid * CV_LD + p] = ra[p];
        #pragma unroll
        for (int p = 0; p < 8; ++p) Bs[bn * CV_LD + (bh >> 1) + p] = rb[p];
        __syncthreads();
        if (k0 + CV_BK < kt) fetch(k0 + CV_BK);
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int kw = ks * 8 + t;   // u32 column of k = ks·16 + 2t
            unsigned bf[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = wn * 32 + j * 8 + g;
                bf[j][0] = Bs[n * CV_LD + kw];
                bf[j][1] = Bs[n * CV_LD + kw + 4];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r0 = wm * 64 + i * 16 + g;
                const unsigned a0 = As[r0 * CV_LD + kw], a1 = As[(r0 + 8) * CV_LD + kw];
                const unsigned a2 = As[r0 * CV_LD + kw + 4], a3 = As[(r0 + 8) * CV_LD + kw + 4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a0, a1, a2, a3, bf[j][0], bf[j][1]);
            }
        }
        __syncthreads();
    }

    // Epilogue: c0,c1 = (row g, cols 2t, 2t+1); c2,c3 = row g + 8.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int hrow = 0; hrow < 2; ++hrow) {
            const long mm = m0 + wm * 64 + i * 16 + g + hrow * 8;
            if (mm >= mt) continue;
            const long f = mm / plane, px = mm - f * plane;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int n = n0 + wn * 32 + j * 8 + 2 * t + c;
                    if (n >= cout) continue;
                    float v = acc[i][j][hrow * 2 + c] + (has_bias ? ld(bias, n) : 0.0f);
                    const long o = (f * cout + n) * plane + px;
                    if (ep == 1) v = fmaxf(v, 0.0f);
                    if (ep == 2) v = fmaxf(v + ld(res, o), 0.0f);
                    if (ep == 3 && v < 0.0f) v *= ld(res, n);
                    out[o] = bf(v);
                }
            }
        }
    }
}
