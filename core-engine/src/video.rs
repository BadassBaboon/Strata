//! Video-wallpaper decoding interface.
//!
//! The concrete decoder is platform-specific (Windows: Media Foundation, implemented in
//! the desktop shell), so `core-engine` stays OS-agnostic and just defines the abstract
//! frame + decoder the renderer consumes. A `type = "video"` wallpaper owns a boxed
//! `VideoDecoder` and uploads its latest frame into a `wgpu` texture each render tick,
//! the same way the audio engine feeds its FFT texture.

/// One decoded video frame in **NV12** - the format hardware decoders output natively,
/// so there is no CPU colour conversion (the renderer converts YUV->RGB on the GPU) and
/// each frame is only `width*height*3/2` bytes instead of `*4`.
///
/// * `y`  - full-resolution luma plane, `width * height` bytes, tightly packed.
/// * `uv` - half-resolution interleaved chroma plane, `width * (height/2)` bytes
///          (`U,V,U,V…`), tightly packed.
///
/// A decoder on another platform converts its native output to NV12 (which VA-API and
/// VideoToolbox also produce natively), so the renderer never has to special-case the OS.
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
}

/// A looping video source that feeds the wallpaper renderer one frame at a time.
///
/// Implementations decode on their own worker thread (hardware-accelerated where the OS
/// supports it) and keep only the most recent frame; `next_frame` hands it over when a
/// newer one has arrived since the last call, so the render thread never blocks on
/// decoding and an idle/occluded wallpaper can simply stop pulling frames. The decoder
/// loops the clip on end-of-stream.
pub trait VideoDecoder: Send {
    /// Native video dimensions in pixels (width, height).
    fn dimensions(&self) -> (u32, u32);

    /// The most recent decoded frame if a new one is available since the last call,
    /// else `None` (caller keeps showing the previous frame).
    fn next_frame(&mut self) -> Option<VideoFrame>;

    /// Pause/resume decoding. While paused the decoder does NO work (CPU ~0). Used to
    /// stop a video wallpaper that's covered by a fullscreen app (game, maximized window).
    fn set_paused(&self, paused: bool);
}

// NV12 -> RGB on the GPU. Samples the luma (Y) and half-res interleaved chroma (UV)
// planes and applies BT.709 limited-range conversion. A fullscreen triangle stretches
// the frame across the target (aspect-correct fitting is a later polish pass).
const VIDEO_BLIT_WGSL: &str = r#"
@group(0) @binding(0) var ytex: texture_2d<f32>;
@group(0) @binding(1) var uvtex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
    let xy = p[i];
    var o: VsOut;
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
    return o;
}
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let yv = textureSample(ytex, samp, in.uv).r;
    let c = textureSample(uvtex, samp, in.uv).rg;
    let y = 1.164383 * (yv - 0.0627451);
    let u = c.x - 0.501961;
    let v = c.y - 0.501961;
    let r = y + 1.792741 * v;
    let g = y - 0.213249 * u - 0.532909 * v;
    let b = y + 2.112402 * u;
    return vec4<f32>(r, g, b, 1.0);
}
"#;

/// GPU-side video wallpaper: owns the decoder plus the NV12 textures and the YUV->RGB
/// blit pipeline. Driven by the `Renderer` each frame.
pub struct VideoLayer {
    decoder: Box<dyn VideoDecoder>,
    y_tex: wgpu::Texture,
    uv_tex: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    // False until the first frame is uploaded; we clear to black (not the green that
    // empty NV12 textures convert to) during the brief gap before decoding starts.
    ready: bool,
}

impl VideoLayer {
    /// `target_format` must match the surface/texture this layer blits into.
    pub fn new(device: &wgpu::Device, decoder: Box<dyn VideoDecoder>, target_format: wgpu::TextureFormat) -> Self {
        let (width, height) = decoder.dimensions();
        let mk = |label: &str, w: u32, h: u32, fmt: wgpu::TextureFormat| device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let y_tex = mk("video Y", width, height, wgpu::TextureFormat::R8Unorm);
        let uv_tex = mk("video UV", width / 2, height / 2, wgpu::TextureFormat::Rg8Unorm);
        let y_view = y_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let uv_view = uv_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("video sampler"),
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("video blit"), source: wgpu::ShaderSource::Wgsl(VIDEO_BLIT_WGSL.into()),
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("video bgl"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video bind group"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("video pl"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("video pipeline"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format: target_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None, multisample: wgpu::MultisampleState::default(),
            multiview_mask: None, cache: None,
        });
        Self { decoder, y_tex, uv_tex, bind_group, pipeline, ready: false }
    }

    /// Poll the decoder; if a NEW frame is available, upload it and return `true`. The
    /// `Renderer` uses this to present only at the video's frame rate (not the monitor's
    /// refresh rate), so a 30 fps clip on a 170 Hz display doesn't re-present 170x/s.
    pub fn poll(&mut self, queue: &wgpu::Queue) -> bool {
        if let Some(frame) = self.decoder.next_frame() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.y_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &frame.y,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(frame.width), rows_per_image: Some(frame.height) },
                wgpu::Extent3d { width: frame.width, height: frame.height, depth_or_array_layers: 1 },
            );
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.uv_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &frame.uv,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(frame.width), rows_per_image: Some(frame.height / 2) },
                wgpu::Extent3d { width: frame.width / 2, height: frame.height / 2, depth_or_array_layers: 1 },
            );
            self.ready = true;
            true
        } else {
            false
        }
    }

    /// True once at least one frame has been uploaded.
    pub fn ready(&self) -> bool {
        self.ready
    }

    /// Pause/resume the underlying decoder (covered-by-fullscreen-app handling).
    pub fn set_paused(&self, paused: bool) {
        self.decoder.set_paused(paused);
    }

    /// Blit the current frame to `view`.
    pub fn blit(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("video blit pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view, depth_slice: None, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        });
        // Until the first frame lands, leave the black clear (avoids the green flash).
        if self.ready {
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}
