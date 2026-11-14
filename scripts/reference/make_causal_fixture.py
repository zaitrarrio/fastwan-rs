"""Reference fixture for the causal (Self-Forcing) block forward.

    pip install torch diffusers safetensors      (fixture made with diffusers 0.43.0, torch 2.15.0)
    python scripts/reference/make_causal_fixture.py

The KV-cached block forward below is strobe's `causal_wan.causal_forward`
(github.com/zaitrarrio/strobe, src/strobe/sources/causal_wan.py; validated
there against stock diffusers to 1e-4), trimmed to what the fixture needs:
per-block RoPE offset by the block's start frame, per-layer K/V cache with
rolling eviction of the oldest non-sink tokens, full attention within a
block. It runs on a tiny random diffusers `WanTransformer3DModel` (the same
config as wan_tiny) for four one-frame blocks with a 3-frame cache and a
1-frame sink, three forwards per block like the real rollout (denoise at
t=757, denoise at t=500, clean context write at t=0, all on the same slots),
so the fourth block evicts the second.

Writes crates/fastwan-model/tests/fixtures/wan_causal_tiny.safetensors: the
weights (f32, diffusers names), `input.context`, `input.x.{b}.{i}` and
`expected.v.{b}.{i}` for block b, forward i.
"""
import pathlib

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel
from safetensors.torch import save_file

root = pathlib.Path(__file__).resolve().parents[2]
torch.manual_seed(1)

TIMESTEPS = [757.0, 500.0, 0.0]  # two denoise forwards, then the clean write
BLOCKS = 4
CAP_FRAMES, SINK_FRAMES = 3, 1


def offset_rope(rope, num_frames, height, width, start_frame):
    p_t, p_h, p_w = rope.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    split = [rope.t_dim, rope.h_dim, rope.w_dim]
    fc = rope.freqs_cos.split(split, dim=1)
    fs = rope.freqs_sin.split(split, dim=1)

    def build(freqs):
        f = freqs[0][start_frame:start_frame + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        return torch.cat([f, h, w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

    return build(fc), build(fs)


def apply_rope(x, cos, sin):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    c, s = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out.type_as(x)


def evict(cache, need):
    free = cache["capacity"] - cache["end"]
    if need <= free:
        return
    sink, end = cache["sink_tokens"], cache["end"]
    to_evict = need - free
    new_end = end - to_evict
    if new_end > sink:
        cache["k"][:, sink:new_end] = cache["k"][:, sink + to_evict:end].clone()
        cache["v"][:, sink:new_end] = cache["v"][:, sink + to_evict:end].clone()
    cache["end"] = new_end


def self_attn_cached(attn, hs, rope, cache, start_tok):
    q = attn.norm_q(attn.to_q(hs)).unflatten(2, (attn.heads, -1))
    k = attn.norm_k(attn.to_k(hs)).unflatten(2, (attn.heads, -1))
    v = attn.to_v(hs).unflatten(2, (attn.heads, -1))
    q, k = apply_rope(q, *rope), apply_rope(k, *rope)
    n = k.shape[1]
    if cache.get("last_start_tok") == start_tok:
        at = cache["block_start"]
    else:
        evict(cache, n)
        at = cache["end"]
        cache["block_start"], cache["last_start_tok"] = at, start_tok
        cache["end"] = at + n
    cache["k"][:, at:at + n] = k
    cache["v"][:, at:at + n] = v
    kk, vv = cache["k"][:, :cache["end"]], cache["v"][:, :cache["end"]]
    out = F.scaled_dot_product_attention(q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2)).transpose(1, 2)
    return attn.to_out[1](attn.to_out[0](out.flatten(2, 3).type_as(q)))


def cross_attn(attn, hs, enc):
    q = attn.norm_q(attn.to_q(hs)).unflatten(2, (attn.heads, -1))
    k = attn.norm_k(attn.to_k(enc)).unflatten(2, (attn.heads, -1))
    v = attn.to_v(enc).unflatten(2, (attn.heads, -1))
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    return attn.to_out[1](attn.to_out[0](out.flatten(2, 3).type_as(q)))


def causal_forward(tr, x, t, enc, caches, start_frame):
    b, c, nf, h, w = x.shape
    _, p_h, p_w = tr.config.patch_size
    tpf = (h // p_h) * (w // p_w)
    rope = offset_rope(tr.rope, nf, h, w, start_frame)
    hs = tr.patch_embedding(x).flatten(2).transpose(1, 2)
    ts = torch.full((1, nf), t).repeat_interleave(tpf, dim=1).flatten()
    temb, tproj, enc_, _ = tr.condition_embedder(ts, enc, None, timestep_seq_len=nf * tpf)
    tproj = tproj.unflatten(2, (6, -1))
    start_tok = start_frame * tpf
    for blk, cache in zip(tr.blocks, caches):
        sm, scm, gm, cs, csc, cg = (blk.scale_shift_table.unsqueeze(0) + tproj.float()).chunk(6, dim=2)
        sm, scm, gm, cs, csc, cg = (z.squeeze(2) for z in (sm, scm, gm, cs, csc, cg))
        norm = (blk.norm1(hs.float()) * (1 + scm) + sm).type_as(hs)
        hs = (hs.float() + self_attn_cached(blk.attn1, norm, rope, cache, start_tok) * gm).type_as(hs)
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
caches = [{"k": torch.zeros(1, CAP_FRAMES * tpf, heads, hd), "v": torch.zeros(1, CAP_FRAMES * tpf, heads, hd),
           "end": 0, "capacity": CAP_FRAMES * tpf, "sink_tokens": SINK_FRAMES * tpf} for _ in dit.blocks]
extra = {"input.context": ctx[0]}
with torch.no_grad():
    for blk in range(BLOCKS):
        for i, t in enumerate(TIMESTEPS):
            x = torch.randn(1, 4, 1, H, W)
            v = causal_forward(dit, x, t, ctx, caches, start_frame=blk)
            extra[f"input.x.{blk}.{i}"] = x[0]
            extra[f"expected.v.{blk}.{i}"] = v[0]
sd = {k: v.detach().float().contiguous().clone() for k, v in dit.state_dict().items()}
sd.update({k: v.detach().float().contiguous() for k, v in extra.items()})
out = root / "crates/fastwan-model/tests/fixtures/wan_causal_tiny.safetensors"
save_file(sd, out)
print("wrote", out, len(extra), "inputs/outputs")
