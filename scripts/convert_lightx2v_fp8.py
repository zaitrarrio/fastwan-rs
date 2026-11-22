"""LightX2V's FP8 Self-Forcing (lightx2v/Self-Forcing-FP8, MIT) -> Diffusers-named bf16 safetensors.

    python3 -I scripts/convert_lightx2v_fp8.py self_forcing_fp8.safetensors transformer.safetensors
    fastwan quantize --weights native|int8 ...          (then as for scripts/convert_self_forcing.py)

The file holds the Self-Forcing generator in the original Wan key names behind
`model.`, the block linears as E4M3 codes (`.weight`, F8_E4M3 `[out, in]`)
with one f32 scale per output row (`.weight_scale`, `[out, 1]`), everything
else bf16. This decodes `code · scale` exactly (E4M3 through a 256-entry
table, the scale applied in f32) and writes bf16, the renames being
convert_self_forcing.py's. bf16 holds an E4M3 code times a per-row scale to
within one bf16 rounding of the scale product, so `fastwan quantize
--weights native` keeps what LightX2V's W8 checkpoint means; the FP8 rounding
itself is part of the weights (compare against the bf16 Self-Forcing release
before trusting a clip). Needs numpy; no torch.
"""
import json
import struct
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from convert_self_forcing import RENAMES  # noqa: E402


def e4m3_table():
    """float32 value of every E4M3 (OCP, finite-only, NaN at 0x7f / 0xff) byte."""
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


def bf16_bits(x):
    """Round-to-nearest-even f32 -> bf16 bits."""
    u = x.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def main(src, dst):
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        header.pop("__metadata__", None)

        def read(name):
            h = header[name]
            a, b = h["data_offsets"]
            f.seek(base + a)
            return f.read(b - a), h["dtype"], h["shape"]

        lut = e4m3_table()
        out = {}
        for name in sorted(header):
            if name.endswith(".weight_scale"):
                continue
            raw, dt, shape = read(name)
            if dt == "F8_E4M3":
                scale, sdt, sshape = read(name + "_scale")
                if sdt != "F32" or sshape != [shape[0], 1]:
                    raise ValueError(f"{name}_scale is {sdt} {sshape}, expected F32 [{shape[0]}, 1]")
                w = lut[np.frombuffer(raw, dtype=np.uint8)].reshape(shape)
                w = w * np.frombuffer(scale, dtype=np.float32).reshape(shape[0], 1)
                if not np.isfinite(w).all():
                    raise ValueError(f"{name}: non-finite values after decoding")
                data, dt = bf16_bits(w).tobytes(), "BF16"
            elif dt in ("BF16", "F16", "F32"):
                data = raw
            else:
                raise ValueError(f"{name}: unexpected dtype {dt}")
            nk = name[len("model."):] if name.startswith("model.") else name
            for old, new in RENAMES:
                if old in nk:
                    nk = nk.replace(old, new)
            out[nk] = (dt, shape, data)

    hdr, off = {"__metadata__": {"source": src.rsplit("/", 1)[-1], "license": "MIT (lightx2v/Self-Forcing-FP8)"}}, 0
    names = sorted(out)
    for k in names:
        dt, shape, data = out[k]
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(dst, "wb") as g:
        g.write(struct.pack("<Q", len(h)) + h)
        for k in names:
            g.write(out[k][2])
    print(f"wrote {dst}: {len(names)} tensors, {off / 1e9:.2f} GB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
