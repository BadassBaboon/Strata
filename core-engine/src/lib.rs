pub mod manifest;
pub mod preprocessor;
pub mod uniform;
pub mod pipeline;
pub mod audio;
pub mod thumbnail;
pub mod depth;
pub mod inpaint;
pub mod segment;
pub mod upscale;
pub mod parallax;

pub use wgpu;
pub use manifest::WallpaperConfig;
pub use pipeline::WallpaperPipeline;
pub use uniform::{ShaderUniforms, UniformState};

use std::sync::Arc;
use winit::window::Window;

pub struct GraphicsContext {
    pub instance: Arc<wgpu::Instance>,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    // System-audio capture + FFT, shared by every monitor's render thread to
    // drive audio-reactive shaders. None only if explicitly disabled.
    pub audio: Option<Arc<audio::AudioEngine>>,
    // Set true if the GPU device is lost (driver TDR/reset, GPU hang, driver
    // update). The shell polls this and recovers. Once true, the device is dead
    // and cannot be revived — a fresh device/context is required.
    pub device_lost: Arc<std::sync::atomic::AtomicBool>,
}

impl GraphicsContext {
    /// Full context for the live engine (with system-audio capture).
    pub async fn new() -> Result<Self, String> {
        Self::create(true).await
    }

    /// Lightweight context for offscreen work like thumbnail generation — no
    /// audio engine. Create it, do the work, then DROP it: destroying the device
    /// returns all the driver's shader-compilation memory to the OS.
    pub async fn new_render_only() -> Result<Self, String> {
        Self::create(false).await
    }

    /// Rough GPU class from the active adapter — used to pick a depth-model
    /// precision (see `depth::ModelTier::choose`).
    pub fn gpu_class(&self) -> depth::GpuClass {
        match self.adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => depth::GpuClass::Discrete,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => depth::GpuClass::Integrated,
            _ => depth::GpuClass::Cpu,
        }
    }

    async fn create(with_audio: bool) -> Result<Self, String> {
        // Prefer each platform's most mature naga backend. On Windows that's DX12
        // (HLSL): naga's SPIR-V/Vulkan backend *panics* during codegen on some
        // complex shadertoy shaders (e.g. neonwave-sunrise — "Expression is not
        // cached") that the HLSL backend compiles cleanly. We try the preferred
        // backend first and fall back to all backends if it has no adapter.
        // (Metal on macOS/iOS and the platform default elsewhere are likewise the
        // well-trodden paths.)
        let preference: &[wgpu::Backends] = if cfg!(target_os = "windows") {
            &[wgpu::Backends::DX12, wgpu::Backends::all()]
        } else {
            &[wgpu::Backends::all()]
        };
        let mut chosen: Option<(wgpu::Instance, wgpu::Adapter)> = None;
        for &backends in preference {
            let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                flags: wgpu::InstanceFlags::default(),
                memory_budget_thresholds: Default::default(),
                backend_options: Default::default(),
                display: None,
            });
            if let Ok(adapter) = inst.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await {
                log::info!("GPU backend: {:?}", adapter.get_info().backend);
                chosen = Some((inst, adapter));
                break;
            }
        }
        let (instance, adapter) = chosen.ok_or_else(|| "Failed to find a GPU adapter".to_string())?;
        let instance = Arc::new(instance);

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Strata GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            },
        ).await.map_err(|e| format!("Failed to request logical device: {}", e))?;

        // By default wgpu treats uncaptured errors as FATAL (it logs
        // "Handling wgpu errors as fatal by default" and aborts). A single bad
        // shadertoy shader could thus take down the whole app / freeze the
        // desktop. Install our own handler that just logs the error so the
        // render thread survives — the offending layer renders wrong at worst.
        //
        // A broken shader can emit the SAME validation error every frame, so we
        // throttle: an identical message is logged at most once every 10s, and a
        // changed message logs immediately. This keeps a bad shader from spamming
        // the log (and spiking CPU/IO) without hiding genuinely new errors.
        let last_err: std::sync::Mutex<Option<(String, std::time::Instant)>> =
            std::sync::Mutex::new(None);
        device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
            let msg = err.to_string();
            // Skip cascade noise: once a resource fails, wgpu emits follow-on
            // "[Invalid X] is invalid." errors for everything that used it. They
            // carry no new info — only the original error matters.
            if msg.contains("] is invalid") {
                return;
            }
            let now = std::time::Instant::now();
            let mut guard = last_err.lock().unwrap();
            let should_log = match &*guard {
                Some((prev, when)) =>
                    *prev != msg || now.duration_since(*when) >= std::time::Duration::from_secs(10),
                None => true,
            };
            if should_log {
                log::error!("wgpu error (non-fatal): {}", msg);
                *guard = Some((msg, now));
            }
        }));

        // Device-loss (TDR / driver reset / GPU hang) is unrecoverable in place —
        // raise a flag the shell polls to rebuild a fresh device.
        let device_lost = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let flag = device_lost.clone();
            device.set_device_lost_callback(move |reason, msg| {
                // `Destroyed` fires on our own normal teardown — ignore it, only a
                // real loss (`Unknown`: driver TDR/reset/errors) triggers recovery.
                if reason == wgpu::DeviceLostReason::Unknown {
                    log::error!("GPU device lost ({:?}): {}", reason, msg);
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                } else {
                    log::info!("GPU device closed ({:?})", reason);
                }
            });
        }

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            audio: if with_audio { Some(audio::AudioEngine::new()) } else { None },
            device_lost,
        })
    }
}

pub struct Renderer {
    pub context: Arc<GraphicsContext>,
    // None for a headless renderer (offscreen tests). Production always has one.
    pub surface: Option<wgpu::Surface<'static>>,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    
    pub uniform_state: uniform::UniformState,
    pub pipelines: Vec<LayerPipeline>,
    // Offset relative to global desktop space
    pub monitor_offset: (f32, f32),
    pub global_resolution: (f32, f32),
    // Ping-pong parity (0/1), flipped each frame. Drives double-buffered render
    // targets deterministically so a self-feeding pass (e.g. a Shadertoy BufferA)
    // always writes the slot opposite the one it samples — no texture is ever
    // both a color target and a bound resource in the same pass.
    pub frame_parity: usize,
    // When Some (headless thumbnail capture), these 512×2 RGBA bytes are uploaded
    // to audio-channel textures instead of the live AudioEngine — so an audio
    // visualizer renders a representative frame even with no sound playing.
    pub headless_audio: Option<Vec<u8>>,
    // Global render-quality scale (0.25–1.0). Below 1.0, each layer's final image
    // pass renders into a reduced-resolution offscreen "scene" target that is then
    // cheaply upscaled to the surface — the single biggest perf lever for heavy
    // shaders. 1.0 renders straight to the surface (no overhead).
    quality_scale: f32,
    scene: Option<SceneUpscale>,
}

/// Reduced-resolution composite target + a fullscreen blit pipeline used to
/// upscale it to the surface when `quality_scale < 1.0`.
struct SceneUpscale {
    size: (u32, u32),
    format: wgpu::TextureFormat,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

pub struct LayerPipeline {
    pub pipeline: pipeline::WallpaperPipeline,
    pub opacity: f32,
    pub resolution_scale: f32,
    pub positioning: String,
    pub transform: [f32; 4],
    pub blend_mode: String,
}

/// Fullscreen-triangle blit that upscales the reduced-resolution scene target to
/// the surface with linear filtering (the `quality_scale < 1.0` path).
const UPSCALE_WGSL: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let xy = p[i];
    var o: VsOut;
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.uv = vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
    return o;
}
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;

/// Map a blend-mode name to the shader's iBlendMode int (0=normal,1=add,2=mul).
fn blend_mode_to_int(mode: &str) -> i32 {
    match mode {
        "additive" => 1,
        "multiply" => 2,
        "screen" => 3, // shader uses the premultiplied (non-multiply) path
        _ => 0,
    }
}

impl Renderer {
    pub fn new(
        context: Arc<GraphicsContext>,
        window: Arc<Window>,
        surface: wgpu::Surface<'static>,
        // Physical pixel size passed explicitly from the UI thread so we never
        // race window.inner_size() before Win32 has committed the correct layout.
        explicit_size: winit::dpi::PhysicalSize<u32>,
    ) -> Result<Self, String> {
        // Prefer the caller-supplied size; fall back to inner_size only if the
        // explicit value is degenerate (zero width or height means it was not set).
        let size = if explicit_size.width > 0 && explicit_size.height > 0 {
            explicit_size
        } else {
            window.inner_size()
        };
        let surface_caps = surface.get_capabilities(&context.adapter);
        // Prefer a NON-sRGB (linear UNORM) surface. Shadertoy displays fragColor
        // raw — it never applies an sRGB conversion — so shader authors bake
        // their own gamma/tonemapping into the output (e.g. `pow(c, 1.0/2.2)` or
        // an explicit `sRGB()`). If we used an sRGB surface the GPU would encode
        // that already-display-ready output a SECOND time, lifting blacks to grey
        // and washing out colors. A linear surface presents the bytes unchanged,
        // matching Shadertoy exactly.
        let screen_format = surface_caps.formats.iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: screen_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&context.device, &config);

        let uniform_state = uniform::UniformState::new(&context.device, size.width as f32, size.height as f32);

        Ok(Self {
            context,
            surface: Some(surface),
            config,
            size,
            uniform_state,
            pipelines: Vec::new(),
            monitor_offset: (0.0, 0.0),
            global_resolution: (size.width as f32, size.height as f32),
            frame_parity: 0,
            headless_audio: None,
            quality_scale: 1.0,
            scene: None,
        })
    }

    /// Build a surfaceless renderer for offscreen rendering (headless tests).
    /// `render()` is a no-op; use `encode_frame(&view)` against your own texture.
    pub fn new_headless(context: Arc<GraphicsContext>, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let size = winit::dpi::PhysicalSize::new(width.max(1), height.max(1));
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        let uniform_state = uniform::UniformState::new(&context.device, size.width as f32, size.height as f32);
        Self {
            context,
            surface: None,
            config,
            size,
            uniform_state,
            pipelines: Vec::new(),
            monitor_offset: (0.0, 0.0),
            global_resolution: (size.width as f32, size.height as f32),
            frame_parity: 0,
            headless_audio: None,
            quality_scale: 1.0,
            scene: None,
        }
    }

    pub fn set_vsync(&mut self, mode: wgpu::PresentMode) {
        self.config.present_mode = mode;
        if let Some(ref surface) = self.surface {
            surface.configure(&self.context.device, &self.config);
        }
    }

    /// Effective internal render resolution = surface size × quality scale. The
    /// whole shader pipeline (offscreen buffers + the image pass's scene target)
    /// renders at this size; a final blit upscales to the full-size surface.
    fn effective_size(&self) -> (u32, u32) {
        let w = (self.size.width as f32 * self.quality_scale).max(1.0) as u32;
        let h = (self.size.height as f32 * self.quality_scale).max(1.0) as u32;
        (w, h)
    }

    /// Set the global render-quality scale (0.25–1.0). 1.0 = native resolution.
    /// On change, resizes every layer's offscreen targets to the new effective
    /// resolution and drops the scene target (rebuilt next frame) — so both the
    /// GPU workload and the reported VRAM track the setting. No-op when unchanged.
    pub fn set_quality(&mut self, q: f32) {
        let q = q.clamp(0.25, 1.0);
        if (q - self.quality_scale).abs() > f32::EPSILON {
            self.quality_scale = q;
            let (ew, eh) = self.effective_size();
            for layer in &mut self.pipelines {
                layer.pipeline.resize_targets(&self.context.device, ew, eh);
            }
            self.scene = None;
        }
    }

    /// (Re)build the reduced-resolution scene target + upscale pipeline at `size`.
    /// Idempotent: a no-op if a matching target already exists.
    fn ensure_scene(&mut self, width: u32, height: u32) {
        let format = self.config.format;
        if let Some(s) = &self.scene {
            if s.size == (width, height) && s.format == format {
                return;
            }
        }
        let device = &self.context.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("quality scene target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("quality upscale sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upscale blit"),
            source: wgpu::ShaderSource::Wgsl(UPSCALE_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("upscale bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upscale bind group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("upscale pl"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("upscale pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        self.scene = Some(SceneUpscale { size: (width, height), format, _texture: texture, view, bind_group, pipeline });
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            // When global_resolution mirrors the surface size we're in the
            // independent (non-span) case — keep them locked so the shader's
            // iResolution tracks the new size.  In span mode global_resolution
            // is the whole-canvas size and must NOT be clobbered by a single
            // monitor's resize.
            let locked = self.global_resolution
                == (self.size.width as f32, self.size.height as f32);

            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            if let Some(ref surface) = self.surface {
                surface.configure(&self.context.device, &self.config);
            }

            self.uniform_state.resize(new_size.width as f32, new_size.height as f32);

            if locked {
                self.global_resolution = (new_size.width as f32, new_size.height as f32);
                self.uniform_state
                    .set_global_resolution(new_size.width as f32, new_size.height as f32);
            }

            let (ew, eh) = self.effective_size();
            for layer in &mut self.pipelines {
                layer.pipeline.resize_targets(&self.context.device, ew, eh);
            }
        }
    }

    pub fn set_global_info(&mut self, offset: (f32, f32), total_res: (f32, f32)) {
        self.monitor_offset = offset;
        self.global_resolution = total_res;
        self.uniform_state.set_monitor_offset(offset.0, offset.1);
        self.uniform_state.set_global_resolution(total_res.0, total_res.1);
    }

    pub fn add_layer(&mut self, path: &std::path::Path, opacity: f32, scale: f32, positioning: String, transform: [f32; 4], blend_mode: String) -> Result<(), String> {
        let config = manifest::WallpaperConfig::load_from_dir(path)?;
        // Build offscreen targets at the current effective (quality-scaled) size.
        let (ew, eh) = self.effective_size();
        let pipeline = pipeline::WallpaperPipeline::new(
            &self.context.device,
            &self.context.queue,
            config,
            path,
            self.config.format,
            ew,
            eh,
            &self.uniform_state.bind_group_layout,
            &blend_mode,
        )?;

        self.pipelines.push(LayerPipeline {
            pipeline,
            opacity,
            resolution_scale: scale,
            positioning,
            transform,
            blend_mode,
        });
        Ok(())
    }

    pub fn clear_layers(&mut self) {
        self.pipelines.clear();
    }

    /// Live-update a layer's spatial rect (no pipeline rebuild) — for the
    /// Compositor's drag preview. Marks the layer "Custom" so render() viewports it.
    pub fn set_layer_transform(&mut self, index: usize, transform: [f32; 4]) {
        if let Some(layer) = self.pipelines.get_mut(index) {
            layer.transform = transform;
            layer.positioning = "Custom".to_string();
        }
    }

    /// Live-update a layer's opacity (just the per-frame uniform — no rebuild).
    pub fn set_layer_opacity(&mut self, index: usize, opacity: f32) {
        if let Some(layer) = self.pipelines.get_mut(index) {
            layer.opacity = opacity;
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::CurrentSurfaceTexture> {
        let surface = match &self.surface {
            Some(s) => s,
            None => return Ok(()), // headless: nothing to present
        };
        let surface_texture = surface.get_current_texture();
        let output = match surface_texture {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => return Err(other),
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.encode_frame(&view);
        output.present();
        Ok(())
    }

    /// Encode + submit one frame into an arbitrary color target. Split out from
    /// `render()` so the headless pipeline audit can drive real rendering into an
    /// offscreen texture and capture render-time wgpu validation errors.
    pub fn encode_frame(&mut self, view: &wgpu::TextureView) {
        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Frame Render Encoder"),
        });

        // Update time and frame counters in global state
        self.uniform_state.update_global_only();

        if self.pipelines.is_empty() {
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Screen Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        } else {
            // Refresh the audio texture(s) for any audio-reactive layer from the
            // shared FFT/waveform. Computed once per frame (the AudioEngine caches
            // the FFT), then uploaded to each audio-using layer's 512×2 texture.
            let audio_bytes = if self.pipelines.iter().any(|l| l.pipeline.audio_texture.is_some()) {
                // Headless thumbnail capture injects a synthetic spectrum; otherwise
                // use the live system-audio FFT.
                if let Some(synth) = &self.headless_audio {
                    Some(synth.clone())
                } else {
                    self.context.audio.as_ref().map(|a| a.texture_rgba())
                }
            } else {
                None
            };
            if let Some(bytes) = &audio_bytes {
                for layer in &self.pipelines {
                    if let Some(tex) = &layer.pipeline.audio_texture {
                        self.context.queue.write_texture(
                            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                            bytes,
                            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(audio::TEX_WIDTH * 4), rows_per_image: Some(audio::TEX_HEIGHT) },
                            wgpu::Extent3d { width: audio::TEX_WIDTH, height: audio::TEX_HEIGHT, depth_or_array_layers: 1 },
                        );
                    }
                }
            }

            // Render-graph ping-pong: every pass writes into the `write_slot`
            // half of its target; every pass reads via its per-parity channel
            // bind group, which was precomputed to bind each buffer to the right
            // (current vs previous frame) slot based on render order. So BOTH the
            // offscreen passes and the image pass select `channel_bind_groups[parity]`.
            // Global quality: below 1.0, render the image passes into a reduced
            // scene target and upscale once at the end. q==1.0 → straight to surface.
            let scaling = self.quality_scale < 0.999;
            let q = if scaling { self.quality_scale } else { 1.0 };
            if scaling {
                let (ew, eh) = self.effective_size();
                self.ensure_scene(ew, eh);
            }

            let parity = self.frame_parity;
            let write_slot = 1 - parity;
            // Image passes target the scene texture while scaling, else the surface.
            let scene_view = if scaling { self.scene.as_ref().map(|s| &s.view) } else { None };
            let image_target_view: &wgpu::TextureView = scene_view.unwrap_or(view);

            for (layer_idx, layer) in self.pipelines.iter_mut().enumerate() {
                let pipeline = &mut layer.pipeline;
                
                // 1. Render internal passes to offscreen targets
                for pass in &pipeline.passes {
                    if pass.name == "image" { continue; }

                    let target = pipeline.targets.get(&pass.name).unwrap();
                    // Render into the slot opposite the one sampled this frame.
                    let target_view = target.view_at(write_slot);
                    let target_size = target.size;

                    // Update this pass's specific uniform data
                    let mut pass_uniforms = self.uniform_state.uniforms;
                    pass_uniforms.iResolution = [target_size.0 as f32, target_size.1 as f32, 1.0];
                    
                    // Update iChannelResolution for the pass
                    let pass_cfg = pipeline.config.render_targets.get(&pass.name).unwrap();
                    for binding in &pass_cfg.bindings {
                        if binding.channel >= 4 {
                            // only 4 iChannels exist; ignore out-of-range bindings.
                        } else if binding.binding_type == "texture" {
                            if let Some(tex) = binding.path.as_ref().and_then(|p| pipeline.textures.get(p)) {
                                let size = tex.texture.size();
                                pass_uniforms.iChannelResolution[binding.channel as usize] = [size.width as f32, size.height as f32, 1.0, 0.0];
                            }
                        } else if binding.binding_type == "buffer" {
                            if let Some(target) = binding.target.as_ref().and_then(|t| pipeline.targets.get(t)) {
                                pass_uniforms.iChannelResolution[binding.channel as usize] = [target.size.0 as f32, target.size.1 as f32, 1.0, 0.0];
                            }
                        } else if binding.binding_type == "audio" {
                            if binding.channel < 4 {
                                pass_uniforms.iChannelResolution[binding.channel as usize] =
                                    [audio::TEX_WIDTH as f32, audio::TEX_HEIGHT as f32, 1.0, 0.0];
                                // Signal that this audio channel is "playing" (some
                                // shaders gate their effect on iChannelTime > 0).
                                pass_uniforms.iChannelTime[binding.channel as usize] = pass_uniforms.iTime;
                            }
                        }
                    }

                    self.context.queue.write_buffer(&pass.uniform_buffer, 0, bytemuck::cast_slice(&[pass_uniforms]));

                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("layer offscreen pass"), // static — avoid per-frame alloc
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target_view,
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

                        rpass.set_pipeline(&pass.pipeline);
                        rpass.set_bind_group(0, &pass.uniform_bind_group, &[]);
                        // Sample the previous frame's slot — distinct from the
                        // write target above, so no read/write aliasing.
                        rpass.set_bind_group(1, &pass.channel_bind_groups[parity], &[]);
                        rpass.draw(0..3, 0..1);
                    }
                    // No swap(): the parity flip at end of frame advances the
                    // ping-pong deterministically for every double-buffered target
                    // at once, which keeps producers and consumers in lockstep.
                }

                // 2. Render final "image" pass to the screen surface
                if let Some(image_pass) = pipeline.passes.iter().find(|p| p.name == "image") {
                    // Update this pass's specific uniform data.
                    // The final image pass renders against the GLOBAL resolution so
                    // span mode works: combined with iMonitorOffset in the shader's
                    // main(), each monitor draws its slice of one canvas.  In the
                    // independent case global_resolution == the monitor size, so
                    // this is identical to the old per-surface behaviour.
                    let mut pass_uniforms = self.uniform_state.uniforms;
                    // "Custom" layers render into a sub-rect (transform = [x,y,w,h]
                    // in surface pixels, top-left origin) via a viewport; the shader
                    // sees the box as its full canvas (iResolution = box size, origin
                    // shifted by iMonitorOffset). Otherwise: full surface / span.
                    // All pixel-space quantities scale by `q` so the shader renders
                    // its full image into the smaller scene target (fragCoord/iResolution
                    // stays 0..1); the upscale blit restores native resolution.
                    let is_custom = layer.positioning == "Custom";
                    if is_custom {
                        let [bx, by, bw, bh] = layer.transform;
                        pass_uniforms.iResolution = [bw.max(1.0) * q, bh.max(1.0) * q, 1.0];
                        pass_uniforms.iMonitorOffset = [-bx * q, -by * q];
                    } else {
                        pass_uniforms.iResolution = [self.global_resolution.0 * q, self.global_resolution.1 * q, 1.0];
                        pass_uniforms.iMonitorOffset = [pass_uniforms.iMonitorOffset[0] * q, pass_uniforms.iMonitorOffset[1] * q];
                    }
                    pass_uniforms.iMouse = [
                        pass_uniforms.iMouse[0] * q, pass_uniforms.iMouse[1] * q,
                        pass_uniforms.iMouse[2] * q, pass_uniforms.iMouse[3] * q,
                    ];
                    // Per-layer compositing controls for the screen pass.
                    pass_uniforms.iOpacity = layer.opacity;
                    pass_uniforms.iBlendMode = blend_mode_to_int(&layer.blend_mode);
                    
                    // Update iChannelResolution for the final image pass
                    let pass_cfg = pipeline.config.render_targets.get("image").unwrap();
                    for binding in &pass_cfg.bindings {
                        if binding.channel >= 4 {
                            // only 4 iChannels exist; ignore out-of-range bindings.
                        } else if binding.binding_type == "texture" {
                            if let Some(tex) = binding.path.as_ref().and_then(|p| pipeline.textures.get(p)) {
                                let size = tex.texture.size();
                                pass_uniforms.iChannelResolution[binding.channel as usize] = [size.width as f32, size.height as f32, 1.0, 0.0];
                            }
                        } else if binding.binding_type == "buffer" {
                            if let Some(target) = binding.target.as_ref().and_then(|t| pipeline.targets.get(t)) {
                                pass_uniforms.iChannelResolution[binding.channel as usize] = [target.size.0 as f32, target.size.1 as f32, 1.0, 0.0];
                            }
                        } else if binding.binding_type == "audio" {
                            if binding.channel < 4 {
                                pass_uniforms.iChannelResolution[binding.channel as usize] =
                                    [audio::TEX_WIDTH as f32, audio::TEX_HEIGHT as f32, 1.0, 0.0];
                                // Signal that this audio channel is "playing" (some
                                // shaders gate their effect on iChannelTime > 0).
                                pass_uniforms.iChannelTime[binding.channel as usize] = pass_uniforms.iTime;
                            }
                        }
                    }

                    self.context.queue.write_buffer(&image_pass.uniform_buffer, 0, bytemuck::cast_slice(&[pass_uniforms]));

                    let load_op = if layer_idx == 0 {
                        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                    } else {
                        wgpu::LoadOp::Load
                    };

                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("layer image pass"), // static — avoid per-frame alloc
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: image_target_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: load_op,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });

                        rpass.set_pipeline(&image_pass.pipeline);
                        rpass.set_bind_group(0, &image_pass.uniform_bind_group, &[]);
                        // Sample the slot just written by the offscreen passes.
                        rpass.set_bind_group(1, &image_pass.channel_bind_groups[parity], &[]);
                        if is_custom {
                            // Clamp the box to the (scaled) target so set_viewport is valid.
                            let [bx, by, bw, bh] = layer.transform;
                            let sw = self.size.width as f32 * q;
                            let sh = self.size.height as f32 * q;
                            let vx = (bx * q).max(0.0).min(sw - 1.0);
                            let vy = (by * q).max(0.0).min(sh - 1.0);
                            let vw = (bw * q).max(1.0).min(sw - vx);
                            let vh = (bh * q).max(1.0).min(sh - vy);
                            rpass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
                        }
                        rpass.draw(0..3, 0..1);
                    }
                }
            }

            // Upscale the reduced-resolution scene onto the surface.
            if scaling {
                if let Some(scene) = &self.scene {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("quality upscale pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
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
                    rpass.set_pipeline(&scene.pipeline);
                    rpass.set_bind_group(0, &scene.bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }
            }
        }

        self.context.queue.submit(std::iter::once(encoder.finish()));
        // Advance the ping-pong for the next frame.
        self.frame_parity ^= 1;
    }

    pub fn estimate_vram_mb(&self) -> f32 {
        let mut total_bytes = 0u64;
        // Swapchain: always native size (2 images × 4 bytes/px) — can't be scaled.
        total_bytes += self.size.width as u64 * self.size.height as u64 * 4 * 2;
        // Offscreen buffer targets below are already sized at the effective
        // (quality-scaled) resolution, so lowering quality shrinks this total.

        for layer in &self.pipelines {
            let pipeline = &layer.pipeline;
            for tex in pipeline.textures.values() {
                let size = tex.texture.size();
                total_bytes += size.width as u64 * size.height as u64 * 4;
            }
            for target in pipeline.targets.values() {
                // Offscreen targets are Rgba16Float → 8 bytes per pixel.
                let count = target.textures.len() as u64;
                total_bytes += target.size.0 as u64 * target.size.1 as u64 * 8 * count;
            }
            // Add internal 1x1 default texture
            total_bytes += 4;
        }

        // Quality-mode scene target (surface format, 4 bytes/px).
        if let Some(s) = &self.scene {
            total_bytes += s.size.0 as u64 * s.size.1 as u64 * 4;
        }

        (total_bytes as f32) / 1024.0 / 1024.0
    }
}
