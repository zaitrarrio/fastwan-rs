"""Upscale a 480x272 fastwan clip with several models; time them and score
temporal detail change ("flicker") against bicubic. Usage:
    python upscale.py IN.y4m OUTDIR WEIGHTS_DIR"""
import json, subprocess, sys, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

src, outdir, wdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
dev = "cuda"

def probe(path):
    s = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=width,height", "-of", "csv=p=0", path]).decode().strip()
    w, h = map(int, s.split(","))
    return w, h

def read(path):
    w, h = probe(path)
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)

def write_mp4(frames, path):
    t, h, w, _ = frames.shape
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", "16", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path],
                         stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes()); p.stdin.close(); p.wait()

lr = read(src)                                   # [t, h, w, 3] uint8
t, h, w, _ = lr.shape
x = torch.from_numpy(lr).to(dev).permute(0, 3, 1, 2).float() / 255.0   # [t, 3, h, w]

# ---- RealBasicVSR (mmedit architecture, written out) ----------------------
class RB(nn.Module):   # ResidualBlockNoBN
    def __init__(self, c):
        super().__init__(); self.conv1 = nn.Conv2d(c, c, 3, 1, 1); self.conv2 = nn.Conv2d(c, c, 3, 1, 1); self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return x + self.conv2(self.relu(self.conv1(x)))
class RBWithInput(nn.Module):  # ResidualBlocksWithInputConv
    def __init__(self, cin, c, n):
        super().__init__(); self.main = nn.Sequential(nn.Conv2d(cin, c, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True), nn.Sequential(*[RB(c) for _ in range(n)]))
    def forward(self, x): return self.main(x)
class ConvModule(nn.Module):
    def __init__(self, cin, cout, act=True):
        super().__init__(); self.conv = nn.Conv2d(cin, cout, 7, 1, 3); self.act = act
    def forward(self, x): x = self.conv(x); return F.relu(x) if self.act else x
class SPyBasic(nn.Module):
    def __init__(self):
        super().__init__(); self.basic_module = nn.Sequential(ConvModule(8, 32), ConvModule(32, 64), ConvModule(64, 32), ConvModule(32, 16), ConvModule(16, 2, act=False))
    def forward(self, x): return self.basic_module(x)
def flow_warp(x, flow, padding_mode="zeros"):
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    grid = torch.stack((gx, gy), 2).type_as(x)
    g = grid + flow
    gxn = 2.0 * g[..., 0] / max(w - 1, 1) - 1.0; gyn = 2.0 * g[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((gxn, gyn), 3), mode="bilinear", padding_mode=padding_mode, align_corners=True)
class SPyNet(nn.Module):
    def __init__(self):
        super().__init__(); self.basic_module = nn.ModuleList([SPyBasic() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)); self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
    def compute_flow(self, ref, supp):
        n, _, h, w = ref.shape
        ref = [(ref - self.mean) / self.std]; supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False)); supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for lv in range(len(ref)):
            up = flow if lv == 0 else F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            flow = up + self.basic_module[lv](torch.cat([ref[lv], flow_warp(supp[lv], up.permute(0, 2, 3, 1), "border"), up], 1))
        return flow
    def forward(self, ref, supp):
        h, w = ref.shape[2:]
        hu, wu = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        ref = F.interpolate(ref, size=(hu, wu), mode="bilinear", align_corners=False); supp = F.interpolate(supp, size=(hu, wu), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / wu; flow[:, 1] *= h / hu
        return flow
class PSP(nn.Module):  # PixelShufflePack
    def __init__(self, cin, cout):
        super().__init__(); self.upsample_conv = nn.Conv2d(cin, cout * 4, 3, 1, 1)
    def forward(self, x): return F.pixel_shuffle(self.upsample_conv(x), 2)
class BasicVSR(nn.Module):
    def __init__(self, c=64, n=20):
        super().__init__(); self.spynet = SPyNet()
        self.backward_resblocks = RBWithInput(c + 3, c, n); self.forward_resblocks = RBWithInput(c + 3, c, n)
        self.fusion = nn.Conv2d(2 * c, c, 1); self.upsample1 = PSP(c, c); self.upsample2 = PSP(c, 64)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1); self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
    def forward(self, lrs):
        n, t, c, h, w = lrs.shape
        l1 = lrs[:, :-1].reshape(-1, c, h, w); l2 = lrs[:, 1:].reshape(-1, c, h, w)
        fb = self.spynet(l1, l2).view(n, t - 1, 2, h, w); ff = self.spynet(l2, l1).view(n, t - 1, 2, h, w)
        lrelu = lambda z: F.leaky_relu(z, 0.1)
        outs = []; fp = lrs.new_zeros(n, 64, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1: fp = flow_warp(fp, fb[:, i].permute(0, 2, 3, 1))
            fp = self.backward_resblocks(torch.cat([lrs[:, i], fp], 1)); outs.append(fp)
        outs = outs[::-1]; fp = torch.zeros_like(fp); res = []
        for i in range(t):
            cur = lrs[:, i]
            if i > 0: fp = flow_warp(fp, ff[:, i - 1].permute(0, 2, 3, 1))
            fp = self.forward_resblocks(torch.cat([cur, fp], 1))
            o = lrelu(self.fusion(torch.cat([outs[i], fp], 1))); o = lrelu(self.upsample1(o)); o = lrelu(self.upsample2(o)); o = lrelu(self.conv_hr(o))
            res.append(self.conv_last(o) + F.interpolate(cur, scale_factor=4, mode="bilinear", align_corners=False))
        return torch.stack(res, 1)
class RealBasicVSR(nn.Module):
    def __init__(self):
        super().__init__(); self.image_cleaning = nn.Sequential(RBWithInput(3, 64, 20), nn.Conv2d(64, 3, 3, 1, 1)); self.basicvsr = BasicVSR(); self.thres = 5 / 255.0
    def forward(self, lqs):
        n, t, c, h, w = lqs.shape
        for _ in range(3):
            flat = lqs.reshape(-1, c, h, w); r = self.image_cleaning(flat); lqs = (flat + r).view(n, t, c, h, w)
            if torch.mean(torch.abs(r)) < self.thres: break
        return self.basicvsr(lqs)

def load_rbvsr(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)  # tensors only: a pickle cannot run code
    sd = sd.get("state_dict", sd)
    pre = "generator_ema." if any(k.startswith("generator_ema.") for k in sd) else "generator."
    sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    m = RealBasicVSR(); missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"RealBasicVSR keys: missing {missing[:5]} unexpected {unexpected[:5]} ({len(missing)}/{len(unexpected)})")
    return m

# ---- runners ----------------------------------------------------------------
def to_u8(y): return (y.clamp(0, 1) * 255 + 0.5).byte().permute(0, 2, 3, 1).cpu().numpy()
def timed(fn):
    torch.cuda.synchronize(); fn(); torch.cuda.synchronize()       # warm-up
    t0 = time.time(); y = fn(); torch.cuda.synchronize(); return y, time.time() - t0

results = {}
def detail(fr):  # high-pass luma, float
    yv = torch.from_numpy(fr).float().to(dev).mean(-1, keepdim=True).permute(0, 3, 1, 2)
    return yv - F.avg_pool2d(F.pad(yv, (2, 2, 2, 2), mode="replicate"), 5, 1)
def flicker(fr):  # mean |Δ detail| between consecutive frames
    d = detail(fr); return float((d[1:] - d[:-1]).abs().mean())

def run(name, scale, fn):
    try:
        with torch.inference_mode():
            y, dt = timed(fn)
        fr = to_u8(y)
        results[name] = {"scale": scale, "seconds": round(dt, 3), "fps": round(t / dt, 1),
                         "size": f"{fr.shape[2]}x{fr.shape[1]}", "flicker": round(flicker(fr), 3)}
        write_mp4(fr, f"{outdir}/{name}.mp4")
        np.save(f"{outdir}/{name}_f40.npy", fr[40]); np.save(f"{outdir}/{name}_f41.npy", fr[41])
        print(name, results[name], flush=True)
    except Exception as e:
        results[name] = {"error": str(e)[:300]}; print(name, "FAILED", e, flush=True)
    torch.cuda.empty_cache()

run("bicubic_x2", 2, lambda: F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False))
run("bicubic_x4", 4, lambda: F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False))

import spandrel
def sp(file):
    m = spandrel.ModelLoader().load_from_file(f"{wdir}/{file}").model.to(dev).half().eval()
    def fn():
        outs = [m(x[i:i + 8].half()).float() for i in range(0, t, 8)]
        return torch.cat(outs)
    return fn
for name, file, sc in [("esrgan_general_x4v3", "realesr-general-x4v3.pth", 4),
                       ("esrgan_animevideov3_x4", "realesr-animevideov3.pth", 4),
                       ("esrgan_x4plus", "RealESRGAN_x4plus.pth", 4),
                       ("esrgan_x2plus", "RealESRGAN_x2plus.pth", 2)]:
    try:
        fn = sp(file)
    except Exception as e:
        results[name] = {"error": f"load: {e}"[:300]}; print(name, "LOAD FAILED", e); continue
    run(name, sc, fn)

try:
    rb = load_rbvsr(f"{wdir}/RealBasicVSR_x4.pth").to(dev).half().eval()
    def rbfn():
        # 81 frames at once fits on 24 GB in fp16; chunks of 27 with 2-frame overlap keep it safe.
        outs, step, ov = [], 27, 2
        for s in range(0, t, step):
            a, b = max(0, s - ov), min(t, s + step + ov)
            y = rb(x[a:b].half().unsqueeze(0))[0].float()
            outs.append(y[s - a: s - a + min(step, t - s)])
        return torch.cat(outs)
    run("realbasicvsr_x4", 4, rbfn)
except SystemExit as e:
    results["realbasicvsr_x4"] = {"error": str(e)}; print("realbasicvsr LOAD FAILED", e)

results["_input"] = {"frames": t, "size": f"{w}x{h}", "flicker": round(flicker(lr), 3)}
json.dump(results, open(f"{outdir}/results.json", "w"), indent=1)
print(json.dumps(results, indent=1))
