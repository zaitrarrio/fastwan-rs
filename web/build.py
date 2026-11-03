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
         "norm": "norm", "head_split": "head_split", "elementwise": "elementwise"}
shaders = {k: (sh / f"{v}.wgsl").read_text() for k, v in names.items()}
page = (root / "test-page.template.html").read_text().replace("/*SHADERS*/{}", json.dumps(shaders, indent=1, ensure_ascii=False))
(root / "dist").mkdir(exist_ok=True)
(root / "dist/page.html").write_text(page)
(root / "fastwan-webgpu-bench.html").write_text(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
print("wrote", root / "fastwan-webgpu-bench.html")
