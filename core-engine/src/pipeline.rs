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
        _ => BlendState { // normal - premultiplied alpha over
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
    /// current-frame slot get a single texture - at full resolution in
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
    // This pass's channel bind-group layout. Per-pass (not shared) because a pass
    // may declare some channels as cubemaps (Cube view dimension) while another
    // pass uses plain 2D textures for the same channel index.
    pub channel_layout: wgpu::BindGroupLayout,
    // Per-channel cubemap flag for this pass (mirrors the layout); used to rebuild
    // bind groups on resize.
    pub cube_channels: [bool; 4],
}

pub struct WallpaperPipeline {
    pub config: WallpaperConfig,
    pub passes: Vec<RenderPassState>,
    pub screen_format: wgpu::TextureFormat,
    pub textures: HashMap<String, LoadedTexture>,
    pub targets: HashMap<String, RenderTargetData>,
    pub default_view: wgpu::TextureView,
    // 1×1 default cube view for unbound/failed cubemap channels, so a Cube layout
    // entry always has a Cube view to bind.
    pub default_cube_view: wgpu::TextureView,
    pub default_sampler: wgpu::Sampler,
    pub clamp_sampler: wgpu::Sampler,
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

        // 1×1 white cube (6 faces) used for any cubemap channel slot that has no
        // loaded texture, so a Cube layout entry always has a matching Cube view.
        let default_cube_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Cube 1x1"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_cube_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8; 4 * 6],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
        );
        let default_cube_view = default_cube_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

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

        // Clamp-to-edge variant, selected per channel for bindings with wrap = "clamp".
        // Fluid/feedback sims need this so neighbour reads at the edges don't wrap around
        // and corrupt the border solve. Same linear filtering as the default sampler.
        let clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Clamp Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
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
                            let full_path = crate::resolve_asset(wallpaper_dir, path);
                            let loaded = Self::load_texture_file(device, queue, &full_path)?;
                            textures.insert(path.clone(), loaded);
                        }
                    }
                } else if binding.binding_type == "cubemap" {
                    if let Some(ref path) = binding.path {
                        if !textures.contains_key(path) {
                            let full_path = crate::resolve_asset(wallpaper_dir, path);
                            let loaded = Self::load_cube_texture(device, queue, &full_path)?;
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

        // The Set 1 (Channels) bind-group layout is built per pass (below) from
        // that pass's cubemap mask, since a channel may be a cubemap in one pass
        // and a 2D texture in another.

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

            // Which channels this pass declares as cubemaps (for shader header +
            // matching bind-group layout view dimension).
            let cube_channels = Self::pass_cube_channels(&pass_cfg.bindings);
            let channel_layout = Self::channel_layout_for(device, cube_channels);

            let (fs_preprocessed, fs_map) = preprocess_shader(&fs_raw, common_glsl.as_deref(), wgpu::naga::ShaderStage::Fragment, pass_name == "image", cube_channels);
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
                    Some(&channel_layout),
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
                    device, &channel_layout, &pass_cfg.bindings,
                    &textures, &targets, &default_view, &default_cube_view, &default_sampler, &clamp_sampler,
                    audio_view.as_ref(), pass_index, &config.wallpaper.passes, 0,
                ),
                Self::create_channel_bind_group(
                    device, &channel_layout, &pass_cfg.bindings,
                    &textures, &targets, &default_view, &default_cube_view, &default_sampler, &clamp_sampler,
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
                channel_layout,
                cube_channels,
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
            default_cube_view,
            default_sampler,
            clamp_sampler,
            audio_texture,
            audio_view,
        })
    }

    /// True if some pass reads `target_name`'s PREVIOUS-frame contents - i.e.
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
        default_cube_view: &wgpu::TextureView,
        default_sampler: &wgpu::Sampler,
        clamp_sampler: &wgpu::Sampler,
        audio_view: Option<&wgpu::TextureView>,
        // Render-graph slot resolution: `this_index` is this pass's position in
        // `pass_order` (the per-frame execution order, e.g. bufferA,B,C,D,image),
        // and `parity` is the frame's ping-pong parity.
        this_index: usize,
        pass_order: &[String],
        parity: usize,
    ) -> wgpu::BindGroup {
        // Collect bindings. There are 8 slots (4 pairs). Start with default mappings:
        // cubemap channels get the 1×1 default cube view (matching their Cube layout
        // entry), all others the 1×1 default 2D view.
        let cube_channels = Self::pass_cube_channels(bindings);
        let mut resources: Vec<(u32, &wgpu::TextureView, &wgpu::Sampler)> = (0..4)
            .map(|i| {
                let dv = if cube_channels[i as usize] { default_cube_view } else { default_view };
                (i * 2, dv, default_sampler)
            })
            .collect();

        // Override with manifest bindings
        for binding in bindings {
            if binding.channel < 4 {
                let binding_idx = binding.channel * 2;
                // Per-channel wrap mode: "clamp" picks the clamp sampler, anything else
                // (incl. omitted) keeps the default repeat sampler.
                let samp = if binding.wrap.as_deref() == Some("clamp") { clamp_sampler } else { default_sampler };
                match binding.binding_type.as_str() {
                    "texture" | "cubemap" => {
                        if let Some(ref path) = binding.path {
                            if let Some(tex) = loaded_textures.get(path) {
                                resources[binding.channel as usize] = (binding_idx, &tex.view, samp);
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
                                resources[binding.channel as usize] = (binding_idx, target.view_at(slot), samp);
                            }
                        }
                    }
                    "audio" => {
                        // Bind the live 512×2 audio texture (FFT + waveform);
                        // falls back to the silent default if audio is disabled.
                        let v = audio_view.unwrap_or(default_view);
                        resources[binding.channel as usize] = (binding_idx, v, samp);
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

    /// Which of the 4 channels a pass declares as a cubemap, from its bindings.
    fn pass_cube_channels(bindings: &[BindingConfig]) -> [bool; 4] {
        let mut mask = [false; 4];
        for b in bindings {
            if b.binding_type == "cubemap" && b.channel < 4 {
                mask[b.channel as usize] = true;
            }
        }
        mask
    }

    /// Set-1 channel bind-group layout for a pass: 4 (texture, sampler) pairs,
    /// with the cubemap channels declared as `Cube` view dimension to match the
    /// `samplerCube` the preprocessor emits for them.
    fn channel_layout_for(device: &wgpu::Device, cube_channels: [bool; 4]) -> wgpu::BindGroupLayout {
        let mut entries = Vec::with_capacity(8);
        for i in 0..4 {
            let binding_idx = i as u32 * 2;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding_idx,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: if cube_channels[i] {
                        wgpu::TextureViewDimension::Cube
                    } else {
                        wgpu::TextureViewDimension::D2
                    },
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding_idx + 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Channel Bind Group Layout"),
            entries: &entries,
        })
    }

    /// Load a Shadertoy cubemap into a 6-layer Cube texture. The base-face file is
    /// `<name>.<ext>`; the other five faces are `<name>_1..<name>_5.<ext>` (the
    /// export convention), in OpenGL/wgpu face order +X,-X,+Y,-Y,+Z,-Z.
    fn load_cube_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base_path: &Path,
    ) -> Result<LoadedTexture, String> {
        let ext = base_path.extension().and_then(|e| e.to_str()).unwrap_or("png");
        let stem = base_path.file_stem().and_then(|s| s.to_str())
            .ok_or_else(|| format!("Bad cubemap path {:?}", base_path))?;
        let dir = base_path.parent().unwrap_or_else(|| Path::new("."));

        // Decode all 6 faces first (so dimensions are known + any failure aborts early).
        let mut faces: Vec<image::RgbaImage> = Vec::with_capacity(6);
        for f in 0..6 {
            let face_path = if f == 0 {
                base_path.to_path_buf()
            } else {
                dir.join(format!("{}_{}.{}", stem, f, ext))
            };
            let img = image::open(&face_path)
                .map_err(|e| format!("Failed to open cubemap face {:?}: {}", face_path, e))?;
            faces.push(img.to_rgba8());
        }
        let (w, h) = faces[0].dimensions();
        if faces.iter().any(|f| f.dimensions() != (w, h)) {
            return Err(format!("Cubemap faces have mismatched sizes: {:?}", base_path));
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Cubemap {:?}", base_path.file_name())),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 6 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (layer, face) in faces.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                face,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        Ok(LoadedTexture { texture, view })
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
                    device, &pass.channel_layout, &pass_cfg.bindings,
                    &self.textures, &self.targets, &self.default_view, &self.default_cube_view, &self.default_sampler, &self.clamp_sampler,
                    self.audio_view.as_ref(), pass_index, &self.config.wallpaper.passes, 0,
                ),
                Self::create_channel_bind_group(
                    device, &pass.channel_layout, &pass_cfg.bindings,
                    &self.textures, &self.targets, &self.default_view, &self.default_cube_view, &self.default_sampler, &self.clamp_sampler,
                    self.audio_view.as_ref(), pass_index, &self.config.wallpaper.passes, 1,
                ),
            ];
        }
    }
}
