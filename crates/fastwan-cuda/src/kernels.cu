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
            out[o] = bf(v);
        }
    }
}
