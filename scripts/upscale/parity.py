"""Parity of fastwan's upscaler against an independent numpy SRVGGNetCompact.

    python3 -I scripts/upscale/parity.py WEIGHTS.safetensors FRAME.png OUT_DIR

Crops a 64×48 patch of FRAME, upscales it with numpy (straight from the
Real-ESRGAN definition: conv3×3+PReLU ×17, conv → pixel_shuffle(4) + nearest
×4 of the input) and with `cargo run --example upscale_raw` on the CPU
backend, and reports the 8-bit difference. Writes both PNGs to OUT_DIR.
"""
import json, os, struct, subprocess, sys

import numpy as np
from PIL import Image


def load(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8 : 8 + n])
    h.pop("__metadata__", None)
    base = 8 + n
    return {k: np.frombuffer(b[base + v["data_offsets"][0] : base + v["data_offsets"][1]], "<f4").reshape(v["shape"]) for k, v in h.items()}


def conv3(x, w, b):  # x [C,H,W], w [O,C,3,3]
    c, hh, ww = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.stack([p[:, i : i + hh, j : j + ww] for i in range(3) for j in range(3)], 1)  # C,9,H,W
    return np.einsum("ckhw,ock->ohw", cols, w.reshape(w.shape[0], c, 9), optimize=True) + b[:, None, None]


def srvgg(sd, x, scale=4):
    y, i = x, 0
    while f"body.{i}.weight" in sd and sd[f"body.{i}.weight"].ndim == 4:
        y = conv3(y, sd[f"body.{i}.weight"], sd[f"body.{i}.bias"])
        a = sd.get(f"body.{i + 1}.weight")
        if a is not None and a.ndim == 1:
            y = np.where(y >= 0, y, y * a[:, None, None])
            i += 2
        else:
            i += 1
    c, h, w = 3, x.shape[1], x.shape[2]
    y = y.reshape(c, scale, scale, h, w).transpose(0, 3, 1, 4, 2).reshape(c, h * scale, w * scale)
    return y + x.repeat(scale, 1).repeat(scale, 2)


wts, frame, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
im = np.asarray(Image.open(frame).convert("RGB"))[200:248, 400:464].copy()  # H=48, W=64
h, w = im.shape[:2]
ref = srvgg(load(wts), im.transpose(2, 0, 1).astype(np.float32) / 255.0)
ref8 = (np.clip(ref, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)
im.tofile(f"{out}/in.rgb")
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["cargo", "run", "-q", "--release", "-p", "fastwan-model", "--example", "upscale_raw", "--",
                wts, f"{out}/in.rgb", "1", str(h), str(w), f"{out}/rs.rgb"], cwd=root, check=True)
got = np.fromfile(f"{out}/rs.rgb", np.uint8).reshape(h * 4, w * 4, 3)
d = np.abs(got.astype(int) - ref8.astype(int))
print(f"numpy vs fastwan CPU, {w}x{h} -> {w*4}x{h*4}: max {d.max()} levels, mean {d.mean():.4f}, exact {100*(d==0).mean():.2f}%")
Image.fromarray(ref8).save(f"{out}/numpy.png")
Image.fromarray(got).save(f"{out}/fastwan.png")
Image.fromarray(im).resize((w * 4, h * 4), Image.NEAREST).save(f"{out}/input_nearest.png")
