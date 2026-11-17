"""Reference fixture for MemRoPE, History Forcing's long-video memory.

    pip install torch diffusers safetensors      (fixture made with torch 2.15.0+cpu, diffusers 0.43.0)
    python scripts/reference/make_memrope_fixture.py

The cached self-attention below transcribes MemRoPE's `CausalWanSelfAttention`
and `_apply_cache_updates` as History Forcing ships them
(github.com/ziplab/History-Forcing at 9d69c1e,
dmd/long_video/third_party/memrope/attention.py, configured by
long_video/memrope.py with compression "ema", ema_adaptive false,
use_block_rope true) onto the KV-cached block forward of
make_causal_fixture.py:

  * the cache holds keys *without* RoPE; every forward rotates the cache at
    its slot index (0, 1, …) and the queries at the block's slots;
  * a new block that does not fit (`current_end > global_end` and
    `local_end + n > capacity`) evicts `n + local_end − capacity` tokens from
    the front of the recent window (after the sink and the two memory
    frames), averages them over all their tokens into one vector per head
    and channel, and sets both memory frames to it (first time) or blends
    them, `α·mean + (1 − α)·old` with α 0.01 (long) and 0.1 (short); the
    rest of the recent window moves up and the block is appended;
  * otherwise the block is written at `local_end + current_end − global_end
    − n` (a repeated forward of the same block rewrites its own slots;
    recomputes never overwrite the sink);
  * attention covers every valid slot (the window is the whole cache).

It runs on a tiny random diffusers `WanTransformer3DModel` (the wan_tiny
config) for six 2-frame blocks with a 7-frame cache (1 sink + 2 memory + 2
recent + the block), three forwards per block like the pipeline (denoise at
t = 757 and 500, then the clean context write at t = 0, all with the same
`current_start`): the fourth block evicts one frame and initialises the
memory, the fifth and sixth evict two and blend it.

Writes crates/fastwan-model/tests/fixtures/wan_memrope_tiny.safetensors: the
weights (f32, Diffusers names), `input.context`, `input.x.{b}.{i}` and
`expected.v.{b}.{i}` for block b, forward i.
"""
import pathlib

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel
from safetensors.torch import save_file

root = pathlib.Path(__file__).resolve().parents[2]
torch.manual_seed(2)

TIMESTEPS = [757.0, 500.0, 0.0]
BLOCKS, BLOCK = 6, 2
SINK, RECENT = 1, 2
CACHE = SINK + 2 + RECENT + BLOCK  # frames
ALPHA_LONG, ALPHA_SHORT = 0.01, 0.1


def rope_at(rope, frames, height, width):
    """cos / sin for latent frames at the given temporal positions."""
    _, p_h, p_w = rope.patch_size
    pph, ppw = height // p_h, width // p_w
    split = [rope.t_dim, rope.h_dim, rope.w_dim]
    idx = torch.tensor(frames)
    nf = len(frames)

    def build(freqs):
        fr = freqs.split(split, dim=1)
        f = fr[0][idx].view(nf, 1, 1, -1).expand(nf, pph, ppw, -1)
        h = fr[1][:pph].view(1, pph, 1, -1).expand(nf, pph, ppw, -1)
        w = fr[2][:ppw].view(1, 1, ppw, -1).expand(nf, pph, ppw, -1)
        return torch.cat([f, h, w], dim=-1).reshape(1, nf * pph * ppw, 1, -1)

    return build(rope.freqs_cos), build(rope.freqs_sin)


def apply_rope(x, cos, sin):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    c, s = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out.type_as(x)


def memrope_attn(attn, hs, rope, grid, cache, current_start):
    """MemRoPE's cached self-attention plus its cache commit."""
    q = attn.norm_q(attn.to_q(hs)).unflatten(2, (attn.heads, -1))
    k = attn.norm_k(attn.to_k(hs)).unflatten(2, (attn.heads, -1))
    v = attn.to_v(hs).unflatten(2, (attn.heads, -1))
    nf, h, w = grid
    fs = (h // 2) * (w // 2)
    n = k.shape[1]
    new_frames = n // fs
    cap = cache["k"].shape[1]
    sink = SINK * fs
    current_end = current_start + n
    g_end, l_end = cache["global_end"], cache["local_end"]
    is_recompute = current_end <= g_end and current_start > 0
    tk, tv = cache["k"].clone(), cache["v"].clone()
    if current_end > g_end and n + l_end > cap:
        # Rolling: EMA memory of the evicted front of the recent window.
        evicted = n + l_end - cap
        long_, short_ = slice(sink, sink + fs), slice(sink + fs, sink + 2 * fs)
        recent = sink + 2 * fs
        mk = cache["k"][:, recent:recent + evicted].mean(dim=1, keepdim=True).expand(-1, fs, -1, -1)
        mv = cache["v"][:, recent:recent + evicted].mean(dim=1, keepdim=True).expand(-1, fs, -1, -1)
        if not cache["ema"]:
            tk[:, long_], tv[:, long_], tk[:, short_], tv[:, short_] = mk, mv, mk, mv
        else:
            for sl, a in ((long_, ALPHA_LONG), (short_, ALPHA_SHORT)):
                tk[:, sl] = a * mk + (1 - a) * cache["k"][:, sl]
                tv[:, sl] = a * mv + (1 - a) * cache["v"][:, sl]
        kept_k = cache["k"][:, recent + evicted:l_end].clone()
        kept_v = cache["v"][:, recent + evicted:l_end].clone()
        tk[:, recent:recent + kept_k.shape[1]] = kept_k
        tv[:, recent:recent + kept_v.shape[1]] = kept_v
        start = recent + kept_k.shape[1]
        new_end = start + n
        tk[:, start:new_end], tv[:, start:new_end] = k, v
        q_frames = list(range(CACHE - new_frames, CACHE))
        cache["ema"] = True
    else:
        # Direct insert (a new block in the filling cache, or the same block
        # again).
        new_end = l_end + current_end - g_end
        start = new_end - n
        ws = max(start, sink) if is_recompute else start
        off = ws - start
        tk[:, ws:new_end], tv[:, ws:new_end] = k[:, off:], v[:, off:]
        q_frames = list(range(start // fs, start // fs + new_frames))
    cache["k"], cache["v"] = tk, tv
    if not is_recompute:
        cache["global_end"] = current_end
    cache["local_end"] = new_end
    qc, qs = rope_at(rope, q_frames, h, w)
    kc, ks = rope_at(rope, list(range(new_end // fs)), h, w)
    qr = apply_rope(q, qc, qs)
    kr = apply_rope(tk[:, :new_end], kc, ks)
    out = F.scaled_dot_product_attention(qr.transpose(1, 2), kr.transpose(1, 2),
                                         tv[:, :new_end].transpose(1, 2)).transpose(1, 2)
    return attn.to_out[1](attn.to_out[0](out.flatten(2, 3).type_as(q)))


def cross_attn(attn, hs, enc):
    q = attn.norm_q(attn.to_q(hs)).unflatten(2, (attn.heads, -1))
    k = attn.norm_k(attn.to_k(enc)).unflatten(2, (attn.heads, -1))
    v = attn.to_v(enc).unflatten(2, (attn.heads, -1))
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    return attn.to_out[1](attn.to_out[0](out.flatten(2, 3).type_as(q)))


def causal_forward(tr, x, t, enc, caches, current_start):
    b, c, nf, h, w = x.shape
    _, p_h, p_w = tr.config.patch_size
    tpf = (h // p_h) * (w // p_w)
    hs = tr.patch_embedding(x).flatten(2).transpose(1, 2)
    ts = torch.full((1, nf), t).repeat_interleave(tpf, dim=1).flatten()
    temb, tproj, enc_, _ = tr.condition_embedder(ts, enc, None, timestep_seq_len=nf * tpf)
    tproj = tproj.unflatten(2, (6, -1))
    for blk, cache in zip(tr.blocks, caches):
        sm, scm, gm, cs, csc, cg = (blk.scale_shift_table.unsqueeze(0) + tproj.float()).chunk(6, dim=2)
        sm, scm, gm, cs, csc, cg = (z.squeeze(2) for z in (sm, scm, gm, cs, csc, cg))
        norm = (blk.norm1(hs.float()) * (1 + scm) + sm).type_as(hs)
        att = memrope_attn(blk.attn1, norm, tr.rope, (nf, h, w), cache, current_start)
        hs = (hs.float() + att * gm).type_as(hs)
        hs = hs + cross_attn(blk.attn2, blk.norm2(hs.float()).type_as(hs), enc_)
        norm = (blk.norm3(hs.float()) * (1 + csc) + cs).type_as(hs)
        hs = (hs.float() + blk.ffn(norm).float() * cg).type_as(hs)
    shift, scale = (tr.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
    hs = (tr.norm_out(hs.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(hs)
    hs = tr.proj_out(hs)
    pp = (nf, h // p_h, w // p_w)
    hs = hs.reshape(b, *pp, 1, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def jitter(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))


dit = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                            in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=40,
                            num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                            eps=1e-6, rope_max_seq_len=64).eval()
jitter(dit)
ctx = torch.randn(1, 8, 16)
H, W = 8, 12
tpf = (H // 2) * (W // 2)
heads, hd = 2, 12
caches = [{"k": torch.zeros(1, CACHE * tpf, heads, hd), "v": torch.zeros(1, CACHE * tpf, heads, hd),
           "global_end": 0, "local_end": 0, "ema": False} for _ in dit.blocks]
extra = {"input.context": ctx[0]}
with torch.no_grad():
    for blk in range(BLOCKS):
        for i, t in enumerate(TIMESTEPS):
            x = torch.randn(1, 4, BLOCK, H, W)
            v = causal_forward(dit, x, t, ctx, caches, current_start=blk * BLOCK * tpf)
            extra[f"input.x.{blk}.{i}"] = x[0]
            extra[f"expected.v.{blk}.{i}"] = v[0]
sd = {k: v.detach().float().contiguous().clone() for k, v in dit.state_dict().items()}
sd.update({k: v.detach().float().contiguous() for k, v in extra.items()})
out = root / "crates/fastwan-model/tests/fixtures/wan_memrope_tiny.safetensors"
save_file(sd, out)
print("wrote", out, len(extra), "inputs/outputs")
