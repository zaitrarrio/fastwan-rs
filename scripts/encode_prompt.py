"""UMT5-XXL prompt embeddings for `fastwan generate --embeds`.

fastwan-rs does not port the 5.7B-parameter text encoder yet (docs/design.md,
M1), so prompts are encoded once with transformers and saved:

    pip install torch transformers safetensors sentencepiece
    python scripts/encode_prompt.py "a fox running through snow" fox.safetensors

Matches Diffusers `WanPipeline._get_t5_prompt_embeds`: 512 tokens, the
embeddings past the prompt's real length zeroed.
"""
import sys
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, UMT5EncoderModel

MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"

prompt, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
enc = UMT5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.bfloat16)
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc.to(dev).eval()
t = tok([prompt], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_tensors="pt")
with torch.no_grad():
    h = enc(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[0].float()
n = int(t.attention_mask.sum())
h[n:] = 0
save_file({"prompt_embeds": h.cpu().contiguous()}, out)
print(f"{out}: [512, {h.shape[1]}], {n} prompt tokens")
