"""Centre-crop grid at frame 40: every method's crop of the same LR region,
shown at 4x (2x outputs are nearest-doubled so pixels are honest)."""
import json, sys
import numpy as np
from PIL import Image, ImageDraw

d = sys.argv[1]
res = json.load(open(f"{d}/results.json"))
order = ["bicubic_x4", "esrgan_general_x4v3", "esrgan_animevideov3_x4", "esrgan_x4plus",
         "realbasicvsr_x4", "bicubic_x2", "esrgan_x2plus"]
names = [n for n in order if n in res and "error" not in res[n]]
cx, cy, cw, ch = 160, 90, 160, 90          # LR crop (x, y, w, h) in 480x272
tiles = []
for n in names:
    f = np.load(f"{d}/{n}_f40.npy")
    s = res[n]["scale"]
    c = f[cy * s:(cy + ch) * s, cx * s:(cx + cw) * s]
    if s == 2:
        c = c.repeat(2, 0).repeat(2, 1)
    im = Image.fromarray(np.ascontiguousarray(c))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 330, 26], fill=(0, 0, 0))
    dr.text((6, 6), n, fill=(255, 255, 255))
    tiles.append(im)
cols = 4
rows = (len(tiles) + cols - 1) // cols
W, H = tiles[0].size
g = Image.new("RGB", (cols * W, rows * H), (30, 30, 30))
for i, im in enumerate(tiles):
    g.paste(im, ((i % cols) * W, (i // cols) * H))
g.save(f"{d}/crops_f40.png")
print("grid", g.size, names)
