"""Inline crates/fastwan-wgpu/src/shaders/*.wgsl into the WebGPU test page.

Writes web/fastwan-webgpu-bench.html (standalone, open over http://localhost)
and web/dist/page.html (the same content without a document skeleton, for
publishing as an Artifact). Re-run after any shader change; the Rust test
`web_page_embeds_current_shaders` fails if the page is stale.
"""
import json, pathlib
root = pathlib.Path(__file__).resolve().parent
sh = root.parent / "crates/fastwan-wgpu/src/shaders"
names = {"common": "common", "linear": "linear", "conv": "conv", "attention": "attention",
         "norm": "norm", "head_split": "head_split", "elementwise": "elementwise",
         "flash": "flash", "vsa_pool": "vsa_pool", "vsa_coarse": "vsa_coarse",
         "linear3": "linear3"}
shaders = {k: (sh / f"{v}.wgsl").read_text() for k, v in names.items()}
page = (root / "test-page.template.html").read_text().replace("/*SHADERS*/{}", json.dumps(shaders, indent=1, ensure_ascii=False))
(root / "dist").mkdir(exist_ok=True)
(root / "dist/page.html").write_text(page)
(root / "fastwan-webgpu-bench.html").write_text(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
print("wrote", root / "fastwan-webgpu-bench.html")

# Static site for Cloudflare Pages: web/site/{index, bench/, demo/}.
import shutil
site = root / "site"
shutil.rmtree(site, ignore_errors=True)
(site / "bench").mkdir(parents=True)
(site / "demo").mkdir()
shutil.copy(root / "landing.html", site / "index.html")
shutil.copy(root / "fastwan-webgpu-bench.html", site / "bench/index.html")
shutil.copy(root / "demo.html", site / "demo/index.html")
(site / "media").mkdir()
shutil.copy(root.parent / "docs/runs/2026-11-03-vast-4090/fox-832x480.mp4", site / "media/fox-832x480.mp4")
# The in-browser engine (scripts/build-web.sh), served at /pkg/.
if (root / "pkg/fastwan_web_bg.wasm").exists():
    shutil.copytree(root / "pkg", site / "pkg")
else:
    print("note: web/pkg missing; run scripts/build-web.sh for the in-browser mode")
(site / "_headers").write_text("/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n")
print("wrote", site)
