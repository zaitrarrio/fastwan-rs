//! List adapters and the features fastwan cares about.
fn main() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    for a in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
        let i = a.get_info();
        let f = a.features();
        println!(
            "{} ({:?}, driver {} {}): shader-f16={} timestamp={} subgroup={}",
            i.name,
            i.backend,
            i.driver,
            i.driver_info,
            f.contains(wgpu::Features::SHADER_F16),
            f.contains(wgpu::Features::TIMESTAMP_QUERY),
            f.contains(wgpu::Features::SUBGROUP)
        );
    }
}
