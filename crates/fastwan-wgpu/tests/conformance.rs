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
        eprintln!("{:<48} rel {:.2e} {}", r.name, r.rel_err, if r.ok { "ok" } else { "FAIL" });
        if !r.ok {
            bad.push(r.name.clone());
        }
    }
    assert!(bad.is_empty(), "failed: {bad:?}");
}
