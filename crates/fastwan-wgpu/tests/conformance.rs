//! Every op and the tiny model against the CPU reference. Runs on whatever
//! adapter wgpu finds (lavapipe in CI); skipped when there is none.

#[test]
fn wgpu_matches_cpu_reference() {
    let be = match fastwan_wgpu::Wgpu::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    eprintln!("{}", fastwan_core::Backend::name(&be));
    let results = fastwan_model::check::run_all(&be, 2e-3);
    let mut bad = Vec::new();
    for r in &results {
        eprintln!(
            "{:<48} rel {:.2e} {}",
            r.name,
            r.rel_err,
            if r.ok { "ok" } else { "FAIL" }
        );
        if !r.ok {
            bad.push(r.name.clone());
        }
    }
    assert!(bad.is_empty(), "failed: {bad:?}");
}

/// The f16 math kernels (where the adapter has `shader-f16`) against the CPU
/// reference, at f16 tolerance.
#[test]
fn wgpu_f16_matches_cpu_reference() {
    let be = match fastwan_wgpu::Wgpu::with_options(fastwan_wgpu::WgpuOptions { f16: true }) {
        Ok(b) if b.f16() => b,
        Ok(_) => {
            eprintln!("skipping: adapter has no shader-f16");
            return;
        }
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    eprintln!("{}", fastwan_core::Backend::name(&be));
    let results = fastwan_model::check::run_all(&be, 1e-2);
    let mut bad = Vec::new();
    for r in &results {
        eprintln!(
            "f16 {:<48} rel {:.2e} {}",
            r.name,
            r.rel_err,
            if r.ok { "ok" } else { "FAIL" }
        );
        if !r.ok {
            bad.push(r.name.clone());
        }
    }
    assert!(bad.is_empty(), "failed: {bad:?}");
}

/// web/fastwan-webgpu-bench.html inlines these shaders (web/build.py); a
/// stale page would test kernels the crate no longer ships.
#[test]
fn web_page_embeds_current_shaders() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let page =
        std::fs::read_to_string(root.join("../../web/fastwan-webgpu-bench.html")).expect("page");
    for name in [
        "common",
        "linear",
        "conv",
        "attention",
        "norm",
        "head_split",
        "elementwise",
    ] {
        let src = std::fs::read_to_string(root.join(format!("src/shaders/{name}.wgsl"))).unwrap();
        for line in src.lines().filter(|l| !l.trim().is_empty()) {
            let esc = line.replace('\\', "\\\\").replace('"', "\\\"");
            assert!(
                page.contains(&esc),
                "{name}.wgsl changed; re-run `python3 web/build.py`: {line}"
            );
        }
    }
}
