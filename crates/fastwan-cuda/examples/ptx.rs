//! Print the kernels' PTX for one architecture, as NVRTC compiles them at
//! start-up: `cargo run -p fastwan-cuda --example ptx -- compute_89 > k.ptx`,
//! then `ptxas -arch=sm_89 -v k.ptx` for registers, spills and shared memory
//! per kernel. Needs libnvrtc only, no GPU.
fn main() {
    let arch: &'static str = Box::leak(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "compute_89".into())
            .into_boxed_str(),
    );
    let ptx = fastwan_cuda::compile_kernels(arch).expect("nvrtc");
    print!("{}", ptx.to_src());
}
