"""Per-block times of a long video in Chrome (20 s, 480x272). argv: BASE_URL ARM
(plain|old|oldf16|halfkv|f16); old = pre-pool engine 7b20eb2."""
import sys, time
from playwright.sync_api import sync_playwright
base, arm = sys.argv[1], sys.argv[2]
args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,VulkanFromANGLE", "--use-angle=vulkan",
        "--use-vulkan=native", "--ignore-gpu-blocklist", "--disable-vulkan-surface", "--enable-gpu"]
if arm == "gc":
    args.append("--js-flags=--expose-gc")
with sync_playwright() as p:
    b = p.chromium.launch(args=args, channel="chromium")
    pg = b.new_page(viewport={"width": 1280, "height": 1000})
    pg.on("pageerror", lambda e: print("pageerror:", e, flush=True))
    pg.goto(base + {"old2": "?engine=7b20eb2", "old": "?engine=7b20eb2", "oldf16": "?engine=7b20eb2&f16=1", "halfkv": "?halfkv=1", "f16": "?f16=1"}.get(arm, ""))
    print(arm, "gc available:", pg.evaluate("typeof gc"), flush=True)
    pg.click('#modes button[data-mode="browser"]')
    pg.wait_for_function("document.querySelectorAll('#bw-variant option').length > 0", timeout=60000)
    print("variants:", pg.eval_on_selector_all("#bw-variant option", "o => o.map(x => x.value)"), flush=True)
    pg.select_option("#bw-variant", "sf16"); pg.click("#bw-load")
    pg.wait_for_function("/ready|failed|error/i.test(document.getElementById('bw-status').textContent)", timeout=900000)
    print("load:", pg.text_content("#bw-status"), "| sizes", pg.eval_on_selector_all("#size option", "o => o.map(x => x.value)"), flush=True)
    pg.evaluate("(() => { const c = [...document.querySelectorAll('#examples .chip')].find(c => (gallery[c.dataset.i]||'').startsWith('A red fox')); c?.click(); })()")
    pg.select_option("#size", "480x272"); pg.select_option("#length", "321"); pg.fill("#seed", "7")
    t0 = time.time(); pg.click("#go")
    pg.wait_for_function("document.getElementById('summary').textContent.includes('total') || !document.getElementById('err').hidden", timeout=1200000)
    print(f"{arm}: {time.time()-t0:.1f}s |", pg.text_content("#summary"), flush=True)
    print(f"{arm} block s:", " ".join(f"{x:.1f}" for x in pg.evaluate("clip.blockMs")), flush=True)
    print(f"{arm} denoise+readback s:", " ".join(f"{x:.1f}" for x in pg.evaluate("clip.denoiseMs")), flush=True)
    prev = {"n": 0, "created": 0.0, "reused": 0}
    for i, m in enumerate(pg.evaluate("clip.blockMem || []")):
        m = {k: (m.get(k) or 0) for k in ("n", "created", "reused", "held")}
        print(f"{arm} mem {i}: created +{m['n'] - prev['n']} ({m['created'] - prev['created']:.1f} MB) reused +{m['reused'] - prev['reused']} held {m['held']:.1f} MB", flush=True)
        prev = m
    mem = pg.evaluate("performance.memory ? Math.round(performance.memory.usedJSHeapSize/1e6) + ' MB JS heap' : ''")
    print(f"{arm} {mem}", flush=True)
    pg.evaluate("() => { frame = Math.floor(clip.frames.length / 2); playing = false; draw(); }")
    pg.wait_for_timeout(300)
    pg.locator("#canvas").screenshot(path=f"/w/shots/long-{arm}-mid.png")
    b.close()
