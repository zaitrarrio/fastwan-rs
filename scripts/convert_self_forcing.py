"""Self-Forcing-style checkpoint (.pt) -> Diffusers-named safetensors, no torch.
Also reads History Forcing's dmd/model.pt (tensors saved with attributes),
which runs with `generate --causal --memory memrope`.

    python3 -I scripts/convert_self_forcing.py self_forcing_dmd.pt transformer.safetensors

`self_forcing_dmd.pt` (huggingface.co/gdhe17/Self-Forcing, Apache-2.0) is a
training checkpoint: the distilled Wan2.1-T2V-1.3B generator under
`generator_ema` (and the raw `generator`), in the original Wan key names behind
a `model.` wrapper. This keeps `generator_ema`, strips the wrapper, renames
the keys to Diffusers' `WanTransformer3DModel` names (the mapping strobe's
selfforcing.py uses, after diffusers' own converter) and writes them unchanged
(bf16). The pickle is read with an unpickler that resolves only tensor
rebuilds and storage types and refuses everything else, and tensors are read
lazily from the zip, so memory stays small.
"""
import collections, io, json, pickle, struct, sys, zipfile

DTYPES = {"BFloat16Storage": ("BF16", 2), "HalfStorage": ("F16", 2), "FloatStorage": ("F32", 4)}

# Original Wan -> Diffusers, applied in order as substring replacements.
RENAMES = [
    ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
    ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
    ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
    ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
    ("time_projection.1", "condition_embedder.time_proj"),
    ("head.modulation", "scale_shift_table"),
    ("head.head", "proj_out"),
    ("modulation", "scale_shift_table"),
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
    # norm2 (FFN) and norm3 (cross-attention) swap names in Diffusers.
    ("norm2", "norm__placeholder"),
    ("norm3", "norm2"),
    ("norm__placeholder", "norm3"),
    ("self_attn.q", "attn1.to_q"),
    ("self_attn.k", "attn1.to_k"),
    ("self_attn.v", "attn1.to_v"),
    ("self_attn.o", "attn1.to_out.0"),
    ("self_attn.norm_q", "attn1.norm_q"),
    ("self_attn.norm_k", "attn1.norm_k"),
    ("cross_attn.q", "attn2.to_q"),
    ("cross_attn.k", "attn2.to_k"),
    ("cross_attn.v", "attn2.to_v"),
    ("cross_attn.o", "attn2.to_out.0"),
    ("cross_attn.norm_q", "attn2.norm_q"),
    ("cross_attn.norm_k", "attn2.norm_k"),
]
PREFIXES = ["model._fsdp_wrapped_module.", "generator.model.", "generator.", "_fsdp_wrapped_module.",
            "_orig_mod.", "module.", "diffusion_model.", "model.", "dit."]


class Lazy:
    """A tensor described by its storage; bytes are read on demand."""

    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def rebuild(storage, offset, size, stride, *_):
    exp, acc = [], 1
    for d in reversed(size):
        exp.insert(0, acc)
        acc *= d
    if list(stride) != exp and any(s > 1 for s in size):
        raise ValueError(f"non-contiguous tensor {size} {stride}")
    return Lazy(storage, offset, size, stride)


TENSOR = object()  # stands in for the torch.Tensor class


def rebuild_from_type(func, new_type, args, state):
    # A tensor saved with Python attributes (History Forcing's checkpoints):
    # the plain tensor, attributes dropped. Only plain Tensor is accepted.
    if func is not rebuild or new_type is not TENSOR:
        raise pickle.UnpicklingError("unexpected tensor subclass")
    return rebuild(*args)


def rebuild_parameter(data, requires_grad, hooks, *_):
    # nn.Parameter: the tensor it wraps.
    if not isinstance(data, Lazy):
        raise pickle.UnpicklingError("parameter without a tensor")
    return data


class Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild
        if (mod, name) == ("torch._tensor", "_rebuild_from_type_v2"):
            return rebuild_from_type
        if (mod, name) == ("torch._utils", "_rebuild_parameter"):
            return rebuild_parameter
        if (mod, name) == ("torch", "Tensor"):
            return TENSOR
        if mod == "torch" and name in DTYPES:
            return name
        raise pickle.UnpicklingError(f"refusing global {mod}.{name}")

    def persistent_load(self, pid):
        kind, cls, key, _loc, _n = pid
        if kind != "storage" or cls not in DTYPES:
            raise pickle.UnpicklingError(f"unexpected storage {pid}")
        return (cls, key)


def main(src, dst):
    z = zipfile.ZipFile(src)
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    prefix = pkl[: -len("/data.pkl")]
    obj = Unpickler(io.BytesIO(z.read(pkl))).load()
    print("top-level keys:", list(obj)[:10])
    for key in ("generator_ema", "generator", "model", "state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            print("using", key)
            obj = obj[key]
            break
    state = {k: v for k, v in obj.items() if isinstance(v, Lazy)}
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if state and all(k.startswith(p) for k in state):
                state = {k[len(p):]: v for k, v in state.items()}
                changed = True
                break
    out = {}
    for k, v in state.items():
        nk = k
        for old, new in RENAMES:
            if old in nk:
                nk = nk.replace(old, new)
        out[nk] = v
    header, off = {"__metadata__": {"source": src.rsplit("/", 1)[-1], "license": "Apache-2.0 (gdhe17/Self-Forcing)"}}, 0
    names = sorted(out)
    for k in names:
        t = out[k]
        dt, nb = DTYPES[t.storage[0]]
        n = 1
        for d in t.size:
            n *= d
        header[k] = {"dtype": dt, "shape": list(t.size), "data_offsets": [off, off + n * nb]}
        off += n * nb
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for k in names:
            t = out[k]
            cls, key = t.storage
            nb = DTYPES[cls][1]
            a, b = header[k]["data_offsets"]
            with z.open(f"{prefix}/data/{key}") as s:
                s.seek(t.offset * nb)
                f.write(s.read(b - a))
    print(f"wrote {dst}: {len(names)} tensors, {off / 1e9:.2f} GB")
    for k in names[:3] + names[-3:]:
        print(" ", k, header[k]["shape"], header[k]["dtype"])


if __name__ == "__main__":
    main(*sys.argv[1:3])
