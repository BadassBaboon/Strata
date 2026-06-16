use std::collections::HashMap;
use std::path::Path;

use crate::manifest::{WallpaperConfig, BindingConfig};
use crate::preprocessor::{preprocess_shader, compile_shader, compile_shader_mapped};

pub struct LoadedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// Run a closure that drives naga's shader backend (SPIR-V/HLSL codegen happens
/// inside `create_shader_module` / `create_render_pipeline`). naga can *panic*
/// there on some shadertoy shaders (e.g. "Expression is not cached"); convert
/// that unwind into a recoverable `Err` so one shader can't crash the engine.
fn catch<T>(what: &str, f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .map_err(|_| format!("shader backend panicked during {}", what))
}

/// wgpu blend state for a layer compositing mode.  The fragment shader outputs
/// premultiplied color (for normal/additive) or a multiply-ready color, so:
///   normal   = premultiplied "over"      (One, OneMinusSrcAlpha)
///   additive = add                       (One, One)
///   multiply = src*dst                   (Dst, Zero)
pub fn blend_for_mode(mode: &str) -> wgpu::BlendState {
    use wgpu::{BlendComponent, BlendFactor, BlendOperation, BlendState};
    match mode {
        "additive" => BlendState {
            color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
            alpha: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
        },
        "multiply" => BlendState {
            color: BlendComponent { src_factor: BlendFactor::Dst, dst_factor: BlendFactor::Zero, operation: BlendOperation::Add },
            alpha: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::Zero, operation: BlendOperation::Add },
        },
        "screen" => BlendState { // 1-(1-src)(1-dst) = src + dst - src*dst
            color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrc, operation: BlendOperation::Add },
            alpha: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add },
        },
        _ => BlendState { // normal — premultiplied alpha over
            color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add },
            alpha: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add },
        },
    }
}

pub struct RenderTargetData {
    pub textures: Vec<wgpu::Texture>,
    pub views: Vec<wgpu::TextureView>, // always 2 (ping-pong slots 0 and 1)
    pub size: (u32, u32),
}

impl RenderTargetData {
    /// `history = true` allocates 2 ping-pong slots so the render-graph can give
    /// a pass the previous-frame version of this buffer (Shadertoy semantics:
    /// self-feedback or read-before-produce). Buffers only ever read at their
    /// current-frame slot get a single texture — at full resolution in
    /// Rgba16Float that halves the offscreen memory for plain multipass shaders.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, history: bool) -> Self {
        let slots = if history { 2 } else { 1 };
        let mut textures = Vec::with_capacity(slots);
        let mut views = Vec::with_capacity(slots);
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };

        for i in 0..slots {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Render Target Texture {}", i)),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Float target (not 8-bit) so Shadertoy-style feedback/accumulation
                // buffers don't quantize each frame into banding artifacts. 16F is
                // filterable + renderable on all target backends (Vulkan/Metal/DX/GL).
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            textures.push(texture);
            views.push(view);
        }

        Self { textures, views, size: (width, height) }
    }

    /// View for ping-pong slot 0 or 1 (single-slot targets map both to slot 0).
    pub fn view_at(&self, slot: usize) -> &wgpu::TextureView {
        &self.views[(slot & 1).min(self.views.len() - 1)]
    }
}

pub struct RenderPassState {
    pub name: String,
    pub pipeline: wgpu::RenderPipeline,
    pub double_buffered: bool,
    pub blend: String,
    // Per-pass uniform buffer to avoid race conditions with multiple passes/resolutions
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    // Pre-created bind groups for channel bindings: [0] for read_index = 0, [1] for read_index = 1
    pub channel_bind_groups: [wgpu::BindGroup; 2],
}

pub struct WallpaperPipeline {
    pub config: WallpaperConfig,
    pub passes: Vec<RenderPassState>,
    pub screen_format: wgpu::TextureFormat,
    pub textures: HashMap<String, LoadedTexture>,
    pub targets: HashMap<String, RenderTargetData>,
    pub default_view: wgpu::TextureView,
    pub default_sampler: wgpu::Sampler,
    pub channel_bind_group_layout: wgpu::BindGroupLayout,
    // 512×2 audio texture (row0=FFT, row1=waveform) for shaders with an `audio`
    // iChannel; updated every frame from the shared AudioEngine. None if unused.
    pub audio_texture: Option<wgpu::Texture>,
    pub audio_view: Option<wgpu::TextureView>,
}

impl WallpaperPipeline {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: WallpaperConfig,
        wallpaper_dir: &Path,
        screen_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        global_uniform_layout: &wgpu::BindGroupLayout,
        layer_blend: &str, // per-layer blend mode for the screen 'image' pass
    ) -> Result<Self, String> {
        let wname = wallpaper_dir.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        log::info!(
            "Loading shader '{}': passes={:?} blend='{}' target={:?} size={}x{}",
            wname, config.wallpaper.passes, layer_blend, screen_format, width, height
        );
        // Create a default 1x1 texture and sampler for unbound channels
        let default_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Texture 1x1"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let default_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // If any pass binds an `audio` channel, create the 512×2 audio texture
        // up front so the channel bind groups can reference it (vs. the 1×1
        // default). render() fills it each frame from the AudioEngine.
        let has_audio = config.render_targets.values()
            .any(|p| p.bindings.iter().any(|b| b.binding_type == "audio"));
        let (audio_texture, audio_view) = if has_audio {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Audio Texture 512x2"),
                size: wgpu::Extent3d { width: crate::audio::TEX_WIDTH, height: crate::audio::TEX_HEIGHT, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (Some(tex), Some(view))
        } else {
            (None, None)
        };

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Load static textures
        let mut textures = HashMap::new();
        for (_, pass_cfg) in &config.render_targets {
            for binding in &pass_cfg.bindings {
                if binding.binding_type == "texture" {
                    if let Some(ref path) = binding.path {
                        if !textures.contains_key(path) {
                            let full_path = wallpaper_dir.join(path);
                            let loaded = Self::load_texture_file(device, queue, &full_path)?;
                            textures.insert(path.clone(), loaded);
                        }
                    }
                }
            }
        }

        // Initialize Render Targets for each pass (except the final 'image' pass if it goes to screen,
        // but wait! If 'image' pass is the final pass, we render directly to screen swapchain).
        let mut targets = HashMap::new();
        for pass_name in &config.wallpaper.passes {
            if pass_name != "image" {
                let pass_cfg = config.render_targets.get(pass_name)
                    .ok_or_else(|| format!("Pass '{}' configuration not found in manifest", pass_name))?;
                let target_width = (width as f32 * pass_cfg.scale).max(1.0) as u32;
                let target_height = (height as f32 * pass_cfg.scale).max(1.0) as u32;
                let history = Self::target_needs_history(&config, pass_name);
                let target = RenderTargetData::new(device, target_width, target_height, history);
                targets.insert(pass_name.clone(), target);
            }
        }

        // Setup the Set 1 (Channels) Bind Group Layout
        // It has 8 entries: 4 pairs of (Texture, Sampler)
        let mut layout_entries = Vec::new();
        for i in 0..4 {
            let binding_idx = i * 2;
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding_idx,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding_idx + 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let channel_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Channel Bind Group Layout"),
            entries: &layout_entries,
        });

        // Compile Vertex Shader Module
        // We optimize vertex execution using the gl_VertexIndex full-screen triangle trick
        let vs_source = r#"#version 450
void main() {
    float x = -1.0 + float((gl_VertexIndex & 1) << 2);
    float y = -1.0 + float((gl_VertexIndex & 2) << 1);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
"#;
        let vs_module = compile_shader(vs_source, wgpu::naga::ShaderStage::Vertex)?;
        let vs_wgpu = catch("vertex shader codegen", || device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FullScreen Quad VS"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(vs_module)),
        }))?;

        // Compile and create render pipelines for each pass
        let mut passes = Vec::new();

        // Load common.glsl if it exists
        let common_path = wallpaper_dir.join("common.glsl");
        let common_glsl = if common_path.exists() {
            Some(std::fs::read_to_string(&common_path)
                .map_err(|e| format!("Failed to read common.glsl: {}", e))?)
        } else {
            None
        };

        for (pass_index, pass_name) in config.wallpaper.passes.iter().enumerate() {
            let pass_cfg = config.render_targets.get(pass_name)
                .ok_or_else(|| format!("Pass '{}' configuration not found in manifest", pass_name))?;

            // Load and preprocess fragment shader
            let fs_path = wallpaper_dir.join(&pass_cfg.source);
            let fs_raw = std::fs::read_to_string(&fs_path)
                .map_err(|e| format!("Failed to read pass shader file {}: {}", pass_cfg.source, e))?;

            let (fs_preprocessed, fs_map) = preprocess_shader(&fs_raw, common_glsl.as_deref(), wgpu::naga::ShaderStage::Fragment, pass_name == "image");
            let fs_module = compile_shader_mapped(&fs_preprocessed, wgpu::naga::ShaderStage::Fragment, Some(&fs_map))
                .map_err(|e| format!("'{}' pass '{}' GLSL compile failed:\n{}", wname, pass_name, e))?;
            log::debug!("  '{}' pass '{}' GLSL→naga OK ({} src lines)", wname, pass_name, fs_preprocessed.lines().count());

            let fs_wgpu = catch(&format!("'{}' pass '{}' shader codegen", wname, pass_name), || device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Pass '{}' FS", pass_name)),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(fs_module)),
            }))?;

            // Pipeline Layout linking uniform bindings (set 0) and channel bindings (set 1)
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("Pass '{}' Pipeline Layout", pass_name)),
                bind_group_layouts: &[
                    Some(global_uniform_layout),
                    Some(&channel_bind_group_layout),
                ],
                immediate_size: 0,
            });

            // Target format is screen format if 'image' pass, else float for offscreen
            let target_format = if pass_name == "image" {
                screen_format
            } else {
                wgpu::TextureFormat::Rgba16Float // must match the offscreen target texture
            };

            // The screen 'image' pass composites the whole layer using its chosen
            // blend mode; offscreen/buffer passes keep their manifest blend.
            let blend_state = if pass_name == "image" {
                blend_for_mode(layer_blend)
            } else if pass_cfg.blend == "alpha" {
                wgpu::BlendState::ALPHA_BLENDING
            } else {
                wgpu::BlendState::REPLACE
            };

            let pipeline = catch(&format!("'{}' pass '{}' pipeline creation", wname, pass_name), || device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("Pass '{}' Render Pipeline", pass_name)),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs_wgpu,
                    entry_point: Some("main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_wgpu,
                    entry_point: Some("main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(blend_state),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            }))?;

            // Create pre-created channel bind groups for read_index = 0 and read_index = 1
            // Two precomputed bind groups, one per frame parity; render() picks the
            // current parity's group, which already binds every buffer to the
            // correct (current vs previous frame) ping-pong slot for this pass.
            let channel_bind_groups = [
                Self::create_channel_bind_group(
                    device, &channel_bind_group_layout, &pass_cfg.bindings,
                    &textures, &targets, &default_view, &default_sampler,
                    audio_view.as_ref(), pass_index, &config.wallpaper.passes, 0,
                ),
                Self::create_channel_bind_group(
                    device, &channel_bind_group_layout, &pass_cfg.bindings,
                    &textures, &targets, &default_view, &default_sampler,
                    audio_view.as_ref(), pass_index, &config.wallpaper.passes, 1,
                ),
            ];

            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pass '{}' Uniform Buffer", pass_name)),
                size: std::mem::size_of::<crate::uniform::ShaderUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Pass '{}' Uniform Bind Group", pass_name)),
                layout: global_uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            passes.push(RenderPassState {
                name: pass_name.clone(),
                pipeline,
                double_buffered: pass_cfg.double_buffered,
                blend: pass_cfg.blend.clone(),
                uniform_buffer,
                uniform_bind_group,
                channel_bind_groups,
            });
        }

        log::info!("Shader '{}' ready: {} pass(es), {} static texture(s), {} render target(s)",
            wname, passes.len(), textures.len(), targets.len());

        Ok(Self {
            config,
            passes,
            screen_format,
            textures,
            targets,
            default_view,
            default_sampler,
            channel_bind_group_layout,
            audio_texture,
            audio_view,
        })
    }

    /// True if some pass reads `target_name`'s PREVIOUS-frame contents — i.e.
    /// the buffer binds itself (feedback) or binds into a pass that runs at or
    /// before its producer. Only such buffers need a second ping-pong texture.
    fn target_needs_history(config: &WallpaperConfig, target_name: &str) -> bool {
        let passes = &config.wallpaper.passes;
        let Some(producer) = passes.iter().position(|p| p == target_name) else { return false };
        passes.iter().enumerate().any(|(pass_index, pass)| {
            config.render_targets.get(pass).is_some_and(|cfg| {
                cfg.bindings.iter().any(|b| {
                    b.binding_type == "buffer"
                        && b.target.as_deref() == Some(target_name)
                        && producer >= pass_index
                })
            })
        })
    }

    fn create_channel_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        bindings: &[BindingConfig],
        loaded_textures: &HashMap<String, LoadedTexture>,
        targets: &HashMap<String, RenderTargetData>,
        default_view: &wgpu::TextureView,
        default_sampler: &wgpu::Sampler,
        audio_view: Option<&wgpu::TextureView>,
        // Render-graph slot resolution: `this_index` is this pass's position in
        // `pass_order` (the per-frame execution order, e.g. bufferA,B,C,D,image),
        // and `parity` is the frame's ping-pong parity.
        this_index: usize,
        pass_order: &[String],
        parity: usize,
    ) -> wgpu::BindGroup {
        // Collect bindings. There are 8 slots (4 pairs). Start with default 1x1 mappings.
        let mut resources: Vec<(u32, &wgpu::TextureView, &wgpu::Sampler)> = (0..4)
            .map(|i| (i * 2, default_view, default_sampler))
            .collect();

        // Override with manifest bindings
        for binding in bindings {
            if binding.channel < 4 {
                let binding_idx = binding.channel * 2;
                match binding.binding_type.as_str() {
                    "texture" => {
                        if let Some(ref path) = binding.path {
                            if let Some(tex) = loaded_textures.get(path) {
                                resources[binding.channel as usize] = (binding_idx, &tex.view, default_sampler);
                            }
                        }
                    }
                    "buffer" => {
                        if let Some(ref target_name) = binding.target {
                            if let Some(target) = targets.get(target_name) {
                                // Shadertoy semantics: a buffer produced EARLIER this
                                // frame is read at its just-written (current) slot;
                                // the buffer itself, or one produced later, is read at
                                // the previous-frame slot.
                                let produced_earlier = pass_order.iter()
                                    .position(|p| p == target_name)
                                    .is_some_and(|pi| pi < this_index);
                                let slot = if produced_earlier { (1 - parity) & 1 } else { parity & 1 };
                                resources[binding.channel as usize] = (binding_idx, target.view_at(slot), default_sampler);
                            }
                        }
                    }
                    "audio" => {
                        // Bind the live 512×2 audio texture (FFT + waveform);
                        // falls back to the silent default if audio is disabled.
                        let v = audio_view.unwrap_or(default_view);
                        resources[binding.channel as usize] = (binding_idx, v, default_sampler);
                    }
                    _ => {}
                }
            }
        }

        let mut entries = Vec::new();
        for (binding_idx, view, sampler) in resources {
            entries.push(wgpu::BindGroupEntry {
                binding: binding_idx,
                resource: wgpu::BindingResource::TextureView(view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: binding_idx + 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Channel Bind Group"),
            layout,
            entries: &entries,
        })
    }

    fn load_texture_file(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &Path,
    ) -> Result<LoadedTexture, String> {
        let img = image::open(path)
            .map_err(|e| format!("Failed to open texture image {:?}: {}", path, e))?;
        let rgba = img.to_rgba8();
        let dimensions = rgba.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Texture {:?}", path.file_name())),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Raw (non-sRGB): our surface is linear UNORM and shaders output
            // display-ready color (Shadertoy convention), so sampled texels must
            // pass through unchanged. An sRGB texture format would decode on
            // sample and render photos too dark and depth maps nonlinearly.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(LoadedTexture { texture, view })
    }

    // Call this if the window resizes to recreate all offscreen target dimensions
    pub fn resize_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        for (name, target) in &mut self.targets {
            let pass_cfg = self.config.render_targets.get(name).unwrap();
            let target_width = (width as f32 * pass_cfg.scale).max(1.0) as u32;
            let target_height = (height as f32 * pass_cfg.scale).max(1.0) as u32;
            let history = target.views.len() > 1; // keep the slot count chosen at build time
            *target = RenderTargetData::new(device, target_width, target_height, history);
        }

        // Recreate the channel bind groups for all passes since target views have changed!
        for (pass_index, pass) in self.passes.iter_mut().enumerate() {
            let pass_cfg = self.config.render_targets.get(&pass.name).unwrap();
            pass.channel_bind_groups = [
                Self::create_channel_bind_group(
                    device, &self.channel_bind_group_layout, &pass_cfg.bindings,
                    &self.textures, &self.targets, &self.default_view, &self.default_sampler,
                    self.audio_view.as_ref(), pass_index, &self.config.wallpaper.passes, 0,
                ),
                Self::create_channel_bind_group(
                    device, &self.channel_bind_group_layout, &pass_cfg.bindings,
                    &self.textures, &self.targets, &self.default_view, &self.default_sampler,
                    self.audio_view.as_ref(), pass_index, &self.config.wallpaper.passes, 1,
                ),
            ];
        }
    }
}
