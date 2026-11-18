"""Upscaling in Prompt Studio's browser mode on a real GPU. argv: BASE_URL ARM.

ARM = <variant>-<WxH>-<frames>-x<k>, e.g. f16-832x480-81-x4 (a clip) or
sf16-480x272-321-x2 (a long video). Prints the summary, the upscaler's
ms/frame, and the playback rate over 10 s once generation is done (with
upscaling on, playback waits for each upscaled frame).
"""
import sys, time
from playwright.sync_api import sync_playwright

base, arm = sys.argv[1], sys.argv[2]
variant, size, frames, k = arm.split("-")
args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,VulkanFromANGLE", "--use-angle=vulkan",
        "--use-vulkan=native", "--ignore-gpu-blocklist", "--disable-vulkan-surface", "--enable-gpu"]
with sync_playwright() as p:
    b = p.chromium.launch(args=args, channel="chromium")
    pg = b.new_page(viewport={"width": 1280, "height": 1000})
    pg.on("pageerror", lambda e: print("pageerror:", e, flush=True))
    pg.goto(base)
    pg.click('#modes button[data-mode="browser"]')
    pg.wait_for_function("document.querySelectorAll('#bw-variant option').length > 0", timeout=60000)
    pg.select_option("#bw-variant", variant); pg.click("#bw-load")
    pg.wait_for_function("/ready|failed|error/i.test(document.getElementById('bw-status').textContent)", timeout=900000)
    print("load:", pg.text_content("#bw-status"), flush=True)
    pg.evaluate("(() => { const c = [...document.querySelectorAll('#examples .chip')].find(c => (gallery[c.dataset.i]||'').startsWith('A red fox')); c?.click(); })()")
    pg.select_option("#size", size); pg.select_option("#length", frames); pg.fill("#seed", "7")
    pg.select_option("#bw-up", k[1:])
    print("hint:", pg.text_content("#bw-up-hint"), flush=True)
    t0 = time.time(); pg.click("#go")
    pg.wait_for_function("document.getElementById('summary').textContent.includes('total') || !document.getElementById('err').hidden", timeout=1800000)
    gen = time.time() - t0
    err = pg.evaluate("document.getElementById('err').hidden ? '' : document.getElementById('err').textContent")
    print(f"{arm}: generated in {gen:.1f}s |", pg.text_content("#summary"), "| err:", err, flush=True)
    if pg.evaluate("!!clip.blockMs"):
        print(f"{arm} block s:", " ".join(f"{x:.1f}" for x in pg.evaluate("clip.blockMs")), flush=True)
    # Playback with upscaling: frames advanced in 10 s, and how many are upscaled now.
    pg.evaluate("() => { frame = 0; playing = true; }")
    f0 = pg.evaluate("frame"); w0 = pg.evaluate("clip.up ? [clip.up.done, clip.up.ms] : [0, 0]")
    pg.wait_for_timeout(10000)
    f1 = pg.evaluate("frame"); w1 = pg.evaluate("clip.up ? [clip.up.done, clip.up.ms] : [0, 0]")
    st = pg.evaluate("clip.up ? `batch ${clip.up.batch}/${clip.up.max}, kept ${bw.eng.upscaledCount()} of ${clip.frames.length}, canvas ${document.getElementById('canvas-up').width}x${document.getElementById('canvas-up').height}` : 'no upscale'")
    d = w1[0] - w0[0]
    print(f"{arm} playback: {(f1 - f0) / 10:.1f} fps over 10 s (frames {f0}->{f1}); upscaled {d} frames in that time"
          f" ({(w1[1] - w0[1]) / max(d, 1):.0f} ms/frame); total {w1[0]} frames at {w1[1] / max(w1[0], 1):.0f} ms/frame; {st}", flush=True)
    pg.evaluate("() => { playing = false; }"); pg.wait_for_timeout(1500)
    pg.locator("#screen").screenshot(path=f"/w/shots/up-{arm}.png")
    b.close()
