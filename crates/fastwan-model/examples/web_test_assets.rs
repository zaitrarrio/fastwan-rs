//! Assets for testing the browser engine without the real model:
//! `cargo run --release -p fastwan-model --example web_test_assets -- OUT_DIR`
//! writes the synthetic tiny DiT block-ordered as `tiny-f16.safetensors` and
//! `tiny-f4.safetensors` (the formats Prompt Studio streams), the tiny TAEHV,
//! one prompt-embeddings file, and the CPU reference clip for each DiT file
//! (`tiny-<fmt>.ref.rgb`, 8-bit RGB frames) at 128×64×17, seed 7.
use fastwan_core::{
    Geometry,
    config::{TaeConfig, WanConfig},
    export,
    quant::WeightQuant,
    rng::Rng,
    weights::{SafetensorsDir, Synthetic, WeightSource, tae_manifest, wan_manifest},
};
use fastwan_cpu::Cpu;
use fastwan_model::{Options, Pipeline, TaeHv, WanDit};
use safetensors::Dtype;
use std::path::Path;

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: web_test_assets OUT_DIR");
    let out = Path::new(&out);
    std::fs::create_dir_all(out).unwrap();
    let (cfg, tcfg) = (WanConfig::tiny_vsa(), TaeConfig::tiny());
    let syn = Synthetic { seed: 3 };
    let f32_tensors = |names: Vec<(String, Vec<usize>)>| -> Vec<export::Tensor> {
        names
            .into_iter()
            .map(|(n, s)| {
                let v = syn.get(&n, &s).unwrap();
                (n, Dtype::F32, s, export::f32_bytes(&v))
            })
            .collect()
    };
    let mut names = wan_manifest(&cfg);
    for l in 0..cfg.layers {
        let k = format!("blocks.{l}.to_gate_compress");
        names.push((format!("{k}.weight"), vec![cfg.dim(), cfg.dim()]));
        names.push((format!("{k}.bias"), vec![cfg.dim()]));
    }
    let src_path = out.join("tiny-src-f32.safetensors");
    export::write(&src_path, &f32_tensors(names), Default::default()).unwrap();
    let src = SafetensorsDir::open(&src_path).unwrap();
    export::write(
        &out.join("tiny-tae.safetensors"),
        &f32_tensors(tae_manifest(&tcfg)),
        Default::default(),
    )
    .unwrap();
    let embeds = Rng::new(5).normal(5 * cfg.text_dim); // 5 real tokens (text_len is 8)
    export::write(
        &out.join("tiny-prompt.safetensors"),
        &[(
            "prompt_embeds".into(),
            Dtype::F16,
            vec![5, cfg.text_dim],
            export::f16_bytes(&embeds),
        )],
        Default::default(),
    )
    .unwrap();
    let mut padded: Vec<f32> = embeds
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect();
    padded.resize(cfg.text_len * cfg.text_dim, 0.0);

    let g = Geometry {
        width: 128,
        height: 64,
        frames: 17,
    };
    let tsrc = SafetensorsDir::open(&out.join("tiny-tae.safetensors")).unwrap();
    for (tag, quant) in [
        ("f16", WeightQuant::Native),
        ("f4", "int4-hqq-g16".parse::<WeightQuant>().unwrap()),
    ] {
        let path = out.join(format!("tiny-{tag}.safetensors"));
        let t = export::quantize_dit(&src, &cfg, quant).unwrap();
        export::write(&path, &t, export::metadata(quant)).unwrap();
        let file = SafetensorsDir::open(&path).unwrap();
        let cpu = Cpu;
        let pipe = Pipeline {
            dit: WanDit::load_with(&cpu, cfg.clone(), &file, WeightQuant::Native).unwrap(),
            tae: TaeHv::load(&cpu, tcfg.clone(), &tsrc).unwrap(),
            schedule: fastwan_core::DmdSchedule::fastwan_1_3b(),
        };
        let opt = Options {
            geometry: g,
            seed: 7,
            ffn_chunk: usize::MAX,
            tae_chunk: 2,
            attention: None,
        };
        let (video, _) = pipe.generate(&cpu, &padded, &opt).unwrap();
        std::fs::write(out.join(format!("tiny-{tag}.ref.rgb")), &video.rgb).unwrap();
        println!(
            "tiny-{tag}: {} bytes, reference {} frames {}x{}",
            std::fs::metadata(&path).unwrap().len(),
            video.frames,
            video.width,
            video.height
        );
    }
    std::fs::remove_file(&src_path).ok();

    // The tiny ×2 upscaler, and its CPU output on the f16 reference clip.
    use fastwan_model::upscale::{UpConfig, Upscaler, up_manifest};
    let ucfg = UpConfig::tiny();
    let up_path = out.join("tiny-up.safetensors");
    export::write(
        &up_path,
        &f32_tensors(up_manifest(&ucfg)),
        Default::default(),
    )
    .unwrap();
    let up = Upscaler::load(&Cpu, ucfg, &SafetensorsDir::open(&up_path).unwrap()).unwrap();
    let clip = std::fs::read(out.join("tiny-f16.ref.rgb")).unwrap();
    let c = up.launch(&Cpu, &clip, g.frames, g.height, g.width);
    use fastwan_core::Backend;
    let raw = Cpu.download(&c.out);
    let big = fastwan_core::backend::rgba_to_rgb(&c.rgba8(&raw));
    std::fs::write(out.join("tiny-f16.up.rgb"), &big).unwrap();
    println!("tiny-up: x{} reference {} bytes", c.scale, big.len());
}
