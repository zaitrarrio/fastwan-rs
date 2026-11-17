//! Draw packed RGBA frames (`pack_rgba8` output) straight onto an HTML canvas,
//! so frames that only need to be seen never leave the GPU: no readback, no
//! per-pixel host loop, no `ImageData`. A fullscreen triangle whose fragment
//! shader reads the word for its pixel from the frame buffer.

use crate::{GBuf, Wgpu};
use std::sync::Mutex;
use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct P { w: u32, h: u32, off: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> px: array<u32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = min(u32(pos.x), p.w - 1u);
    let y = min(u32(pos.y), p.h - 1u);
    let wd = px[p.off + y * p.w + x];
    return vec4<f32>(f32(wd & 255u), f32((wd >> 8u) & 255u), f32((wd >> 16u) & 255u), 255.0) / 255.0;
}
"#;

/// A canvas the engine draws frames on.
pub struct CanvasView {
    surface: wgpu::Surface<'static>,
    pipeline: wgpu::RenderPipeline,
    format: wgpu::TextureFormat,
    alpha: wgpu::CompositeAlphaMode,
    size: Mutex<(u32, u32)>,
}

impl Wgpu {
    /// Attach an HTML canvas (it must not already have a 2D context).
    pub fn canvas_view(&self, canvas: web_sys::HtmlCanvasElement) -> Result<CanvasView, String> {
        let surface = self
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("canvas surface: {e}"))?;
        let caps = surface.get_capabilities(&self.adapter);
        // The packed bytes are already sRGB-encoded: write them to a linear
        // format unchanged (an *Srgb format would encode them twice).
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or("the canvas offers no texture format")?;
        let alpha = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("canvas"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("canvas"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        Ok(CanvasView {
            surface,
            pipeline,
            format,
            alpha,
            size: Mutex::new((0, 0)),
        })
    }
}

impl CanvasView {
    /// Draw the `w × h` frame starting at word `first` of `frames` (the
    /// canvas element should be `w × h` pixels; CSS scales it).
    pub fn show(
        &self,
        be: &Wgpu,
        frames: &GBuf,
        first: usize,
        w: u32,
        h: u32,
    ) -> Result<(), String> {
        if (first + (w * h) as usize) > frames.len {
            return Err("frame out of range".into());
        }
        {
            let mut size = self.size.lock().unwrap();
            if *size != (w, h) {
                self.surface.configure(
                    &be.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: self.format,
                        color_space: Default::default(),
                        width: w,
                        height: h,
                        desired_maximum_frame_latency: 2,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: self.alpha,
                        view_formats: vec![],
                    },
                );
                *size = (w, h);
            }
        }
        let tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => return Err(format!("canvas texture unavailable: {other:?}")),
        };
        let view = tex.texture.create_view(&Default::default());
        let params = be
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("canvas params"),
                contents: bytemuck::cast_slice(&[w, h, first as u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = be.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frames.buf.as_entire_binding(),
                },
            ],
        });
        // Compute work recorded so far must reach the queue first.
        be.flush();
        let mut enc = be.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("canvas"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        be.queue.submit([enc.finish()]);
        be.queue.present(tex);
        Ok(())
    }
}
