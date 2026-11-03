"""Reference fixtures from the real implementations, tiny random weights.

    pip install torch transformers diffusers safetensors
    python scripts/reference/make_fixtures.py

Writes small safetensors files that the Rust tests compare against:
  crates/fastwan-text/tests/fixtures/umt5_tiny.safetensors   transformers UMT5EncoderModel
  crates/fastwan-model/tests/fixtures/wan_tiny.safetensors    diffusers WanTransformer3DModel
Each holds the model's weights (f32, their real key names) plus `input.*` and
`expected.*` tensors. Regenerate only when the reference library changes.
"""
import pathlib
import torch
from safetensors.torch import save_file

root = pathlib.Path(__file__).resolve().parents[2]
torch.manual_seed(0)


def jitter(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))


def tensors(model, extra):
    sd = {k: v.detach().float().contiguous().clone() for k, v in model.state_dict().items()}
    sd.update({k: v.detach().float().contiguous() for k, v in extra.items()})
    return sd


# ---- UMT5 ---------------------------------------------------------------------
from transformers import UMT5Config, UMT5EncoderModel

cfg = UMT5Config(vocab_size=64, d_model=32, d_kv=8, num_heads=4, d_ff=48, num_layers=2,
                 relative_attention_num_buckets=32, relative_attention_max_distance=128,
                 feed_forward_proj="gated-gelu", dropout_rate=0.0)
enc = UMT5EncoderModel(cfg).eval()
jitter(enc)
ids = torch.tensor([[5, 17, 3, 44, 9, 1, 60, 22, 8, 31, 2, 50, 1]])
pad = torch.cat([ids, torch.zeros(1, 7, dtype=torch.long)], 1)
mask = torch.cat([torch.ones(1, 13, dtype=torch.long), torch.zeros(1, 7, dtype=torch.long)], 1)
with torch.no_grad():
    # Padded and masked, as WanPipeline calls it; the Rust encoder sees only the 13 real ids.
    out = enc(pad, mask).last_hidden_state[0, :13]
save_file(tensors(enc, {"input.ids": ids[0].float(), "expected.hidden": out}),
          root / "crates/fastwan-text/tests/fixtures/umt5_tiny.safetensors")
print("umt5", out.shape)

# ---- Wan DiT --------------------------------------------------------------------
from diffusers import WanTransformer3DModel

dit = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                            in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=40,
                            num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                            eps=1e-6, rope_max_seq_len=64).eval()
jitter(dit)
x = torch.randn(1, 4, 3, 8, 12)
ctx = torch.randn(1, 8, 16)
t = torch.tensor([757.0])
with torch.no_grad():
    v = dit(hidden_states=x, timestep=t, encoder_hidden_states=ctx, return_dict=False)[0]
save_file(tensors(dit, {"input.latent": x[0], "input.context": ctx[0], "input.timestep": t, "expected.velocity": v[0]}),
          root / "crates/fastwan-model/tests/fixtures/wan_tiny.safetensors")
print("wan", v.shape)
