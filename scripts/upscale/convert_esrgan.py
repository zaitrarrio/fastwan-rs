"""Real-ESRGAN compact checkpoint (.pth) -> safetensors, without torch.

    python3 -I scripts/upscale/convert_esrgan.py realesr-animevideov3.pth out.safetensors

The .pth is a torch zip; its pickle is read with an unpickler that only
resolves the handful of globals a plain state dict needs (tensor rebuild,
storage types, OrderedDict) and refuses everything else, so loading it runs
no code from the file. Keys are kept as stored (`body.N.weight` / `.bias`;
under `params` / `params_ema` when present). Weights stay f32 (2.4 MB).
"""
import collections, json, pickle, struct, sys, zipfile

import numpy as np

DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}


class Storage:
    def __init__(self, kind, key):
        self.kind, self.key = kind, key


def rebuild(storage, offset, size, stride, *_):
    a = storage.data[offset:]
    n = int(np.prod(size)) if size else 1
    # Contiguous only (what torch.save writes for module parameters).
    exp, acc = [], 1
    for d in reversed(size):
        exp.insert(0, acc)
        acc *= d
    assert list(stride) == exp or n <= 1, f"non-contiguous tensor {size} {stride}"
    return a[:n].reshape(size).astype(np.float32)


class Unpickler(pickle.Unpickler):
    def __init__(self, f, z, prefix):
        super().__init__(f)
        self.z, self.prefix = z, prefix

    def find_class(self, mod, name):
        if (mod, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild
        if mod == "torch" and name in DT:
            return name
        raise pickle.UnpicklingError(f"refusing global {mod}.{name}")

    def persistent_load(self, pid):
        kind, cls, key, _loc, _n = pid
        assert kind == "storage" and cls in DT, pid
        s = Storage(cls, key)
        raw = self.z.read(f"{self.prefix}/data/{key}")
        s.data = np.frombuffer(raw, dtype=DT[cls])
        return s


def load(path):
    z = zipfile.ZipFile(path)
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    prefix = pkl[: -len("/data.pkl")]
    import io
    sd = Unpickler(io.BytesIO(z.read(pkl)), z, prefix).load()
    for k in ("params_ema", "params"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    return sd


def save(path, tensors, meta):
    header, off, blobs = {"__metadata__": meta}, 0, []
    for k, v in tensors.items():
        b = np.ascontiguousarray(v, dtype="<f4").tobytes()
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    sd = load(src)
    for k, v in sd.items():
        print(k, tuple(v.shape))
    save(dst, sd, {"source": src.rsplit("/", 1)[-1], "arch": "SRVGGNetCompact", "license": "BSD-3-Clause (xinntao/Real-ESRGAN)"})
    print("wrote", dst, sum(v.size for v in sd.values()), "params")
