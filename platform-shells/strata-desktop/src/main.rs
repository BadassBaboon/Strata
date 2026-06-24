// Build the shipped (release) binary as a Windows GUI app so launching it never
// spawns a console window. Debug builds keep the console for development. CLI
// output (`--version`, `--help`) still reaches the terminal via the parent-console
// attach in `main()`.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, atomic::AtomicBool};
#[cfg(target_os = "windows")]
use std::sync::Mutex;
use std::sync::mpsc::{channel, Sender};
use std::rc::Rc;
use clap::Parser;
use slint::{ComponentHandle, VecModel, SharedString, ModelRc, Image, Model, SharedPixelBuffer, Rgba8Pixel};

mod platform;
mod controller;
mod config;
mod parallax;
mod library_sync;
#[cfg(windows)]
mod video_decode;

use platform::EngineCommand;
use controller::{AppState, import_wallpaper_zip, discover_monitors, LayerInfo};

slint::include_modules!();

struct SlintLogger {
    sender: Sender<String>,
    file: std::sync::Mutex<Option<std::fs::File>>,
}

impl log::Log for SlintLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            // Live feed for the Diagnostics tab.
            let log_line = format!("[{}] {}", record.level(), record.args());
            let _ = self.sender.send(log_line);
            // Persistent file log (survives crashes for post-mortem debugging).
            if let Ok(mut guard) = self.file.lock() {
                if let Some(f) = guard.as_mut() {
                    use std::io::Write;
                    let _ = writeln!(f, "{} [{}] {}", now_hms(), record.level(), record.args());
                    let _ = f.flush();
                }
            }
        }
    }

    fn flush(&self) {}
}

/// Open (append) the persistent log file alongside the config, writing a session
/// marker. Returns None if the directory can't be resolved/created.
fn open_log_file() -> Option<std::fs::File> {
    let dir = directories::ProjectDirs::from("com", "Strata", "engine")?
        .config_dir()
        .to_path_buf();
    std::fs::create_dir_all(&dir).ok()?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("strata.log"))
        .ok()?;
    use std::io::Write;
    let _ = writeln!(f, "\n========== Strata session started {} ==========", now_hms());
    Some(f)
}

/// Live Parallax-preview render target: a headless renderer with the preview
/// package loaded, plus a persistent readback buffer. Driven by the UI timer.
struct ParallaxPreviewState {
    renderer: core_engine::Renderer,
    _target: core_engine::wgpu::Texture,
    view: core_engine::wgpu::TextureView,
    readback: core_engine::wgpu::Buffer,
    w: u32,
    h: u32,
    time: f32,
}

impl ParallaxPreviewState {
    /// Build a preview renderer for an already-exported parallax package dir.
    fn new(ctx: std::sync::Arc<core_engine::GraphicsContext>, dir: &std::path::Path) -> Result<Self, String> {
        use core_engine::wgpu;
        let (w, h) = (640u32, 360u32); // w*4 = 2560 → 256-aligned readback, no padding
        let mut renderer = core_engine::Renderer::new_headless(ctx.clone(), w, h, wgpu::TextureFormat::Rgba8Unorm);
        renderer.add_layer(dir, 1.0, 1.0, "Fill".into(), [0.0, 0.0, 1.0, 1.0], "normal".into())?;
        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("parallax preview target"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("parallax preview readback"),
            size: (w * 4 * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self { renderer, _target: target, view, readback, w, h, time: 0.0 })
    }

    /// Render one frame, read it back into a Slint image. The wallpaper itself only
    /// parallaxes from the real cursor, so the preview feeds a gentle synthetic
    /// mouse sweep to demonstrate the 3D effect the user will get.
    fn frame(&mut self, ctx: &core_engine::GraphicsContext) -> Option<Image> {
        use core_engine::wgpu;
        self.time += 0.1; // ~10 fps from the 100 ms UI timer
        let t = self.time;
        let cx = self.w as f32 * (0.5 + 0.30 * (t * 0.8).sin());
        let cy = self.h as f32 * (0.5 + 0.18 * (t * 0.6).sin());
        self.renderer.uniform_state.set_mouse_position(cx, cy);
        self.renderer.uniform_state.set_mouse_down(true); // iMouse.z>0.5 → shader follows it
        self.renderer.encode_frame(&self.view);

        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self._target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &self.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.w * 4), rows_per_image: Some(self.h) } },
            wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
        );
        ctx.queue.submit(std::iter::once(enc.finish()));

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range();
        let mut pb = SharedPixelBuffer::<Rgba8Pixel>::new(self.w, self.h);
        pb.make_mut_bytes().copy_from_slice(&data);
        drop(data);
        self.readback.unmap();
        Some(Image::from_rgba8(pb))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Declare Per-Monitor-V2 DPI awareness up front, before any window exists. The old
    // GPU UI path (FemtoVG→ANGLE→D3D11) established this as a side effect of D3D init;
    // the software backend does not, which left the process effectively system-DPI-aware
    // and broke maximize/positioning geometry across our mixed-DPI monitors (window
    // oversized and parked bottom-right). This is the correct, explicit fix and is a
    // no-op if winit later sets the same context.
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::UI::HiDpi::{
            SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        };
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }

    // Render Slint's UI in SOFTWARE (CPU), not on the GPU. Slint's default GPU renderer
    // (FemtoVG→ANGLE/D3D11) is a second GPU presenter; on NVIDIA it conflicts with the
    // wallpaper windows' wgpu swapchains - when desktop icons are hidden, the UI's GPU
    // present blacks out the primary monitor's wallpaper. The wallpaper content is drawn
    // entirely by our own wgpu surface, so the UI doesn't need the GPU.
    //
    // We tried (2026-06-16) rendering the UI on the SAME wgpu device as the wallpapers to
    // get GPU UI without the conflict; it failed - two swapchains on one device clash at
    // DXGI swapchain creation ("Access is denied"), because the wallpaper swapchain is
    // created cross-thread on a window the shared DXGI factory also drives. See
    // [[gpu-ui-prototype]] in memory. Software is the proven path; it needs the size-nudge
    // full-repaint workarounds (`needs_repaint` / `move_settle`) as it only presents the
    // damaged region.
    #[cfg(windows)]
    if std::env::var_os("SLINT_BACKEND").is_none() {
        std::env::set_var("SLINT_BACKEND", "winit-software");
    }

    // Register the shared asset roots so shaders resolve textures/cubemaps by name
    // from the library `external/` dir instead of carrying copies per-wallpaper.
    core_engine::set_asset_dirs(controller::library_asset_dirs());

    // GUI-subsystem release builds have no console of their own. If we were
    // launched from a terminal (e.g. `strata-desktop.exe --version`), attach to
    // the parent's console so clap's version/help text is actually visible.
    #[cfg(windows)]
    unsafe {
        windows_sys::Win32::System::Console::AttachConsole(
            windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS,
        );
    }

    let args = Args::parse();

    // Elevated one-shot: write the MPO registry value and exit immediately (before any
    // UI or single-instance handling). The Settings toggle relaunches us as admin with
    // this flag so it can write HKLM without the main app running elevated.
    #[cfg(windows)]
    if let Some(v) = args.set_mpo.as_deref() {
        let ok = set_mpo_disabled(v.eq_ignore_ascii_case("on"));
        std::process::exit(if ok { 0 } else { 1 });
    }

    if args.logs {
        std::env::set_var("RUST_LOG", "info");
    }
    
    #[cfg(target_os = "windows")]
    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::SetProcessDPIAware();
    }

    // Phase-2/3 dev harness: play a single .mp4 in a window to validate the Media
    // Foundation decoder + texture upload + blit on real hardware before the full
    // video-wallpaper integration. `strata-desktop --video <path>`.
    #[cfg(windows)]
    if let Some(v) = args.video {
        env_logger::init();
        return run_video_mode(v);
    }
    if let Some(v) = args.video_web {
        env_logger::init();
        return run_video_web_mode(v);
    }
    if args.video_daemon {
        env_logger::init();
        return run_video_daemon();
    }

    if let Some(wallpaper_path) = args.cli {
        env_logger::init();
        run_cli_mode(wallpaper_path)
    } else {
        run_ui_mode(args.minimized)
    }
}

/// Sanitize a string into a safe folder name (alphanumerics, space, dash, underscore).
fn sanitize_name(s: &str) -> String {
    let cleaned: String = s.chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let trimmed = cleaned.trim().trim_matches('_').trim();
    if trimmed.is_empty() { "video".to_string() } else { trimmed.to_string() }
}

/// Import a movie (.mp4) as a self-contained video wallpaper pack under
/// `%APPDATA%/Strata/import-video/<name>/` (`video.mp4` + `manifest.toml` +
/// `thumbnail.png`). The name defaults to the file's stem. Returns the pack directory.
fn import_video_wallpaper(src: &std::path::Path, display_name: &str) -> Result<std::path::PathBuf, String> {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
    // User-supplied name wins; blank falls back to the file's stem.
    let raw = if display_name.trim().is_empty() { stem } else { display_name };
    let name = sanitize_name(raw);
    let base = controller::import_video_dir().ok_or("could not resolve the user data directory")?;
    std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;
    let dest = base.join(&name);
    if dest.exists() {
        return Err(format!("A wallpaper named \"{name}\" already exists"));
    }
    std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
    // Preserve the source container (.mp4 / .webm); the WebView plays both natively.
    let ext = src.extension().and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase()).unwrap_or_else(|| "mp4".into());
    let file = format!("video.{ext}");
    let video_dst = dest.join(&file);
    std::fs::copy(src, &video_dst).map_err(|e| format!("copy video: {e}"))?;
    let manifest = format!(
        "[wallpaper]\nname = \"{}\"\nauthor = \"\"\nversion = \"1.0.0\"\ntags = [\"video\"]\npasses = []\nvideo = \"{}\"\n",
        name.replace('"', "'"), file,
    );
    std::fs::write(dest.join("manifest.toml"), manifest).map_err(|e| e.to_string())?;

    // Thumbnail from the first frame (best-effort: a failure just leaves a blank card).
    // Media Foundation decodes mp4/H.264 natively, in-process, and instantly - no WebView2.
    // webm/VP9 isn't MF-decodable, so those cards stay blank (mp4 is the common case).
    #[cfg(windows)]
    if ext == "mp4" {
        if let Err(e) = generate_video_thumbnail(&video_dst, &dest.join("thumbnail.png")) {
            log::warn!("Video thumbnail generation failed: {e}");
        }
    }
    Ok(dest)
}

/// Decode the first frame of `video` and save a 480x270 PNG thumbnail (Media Foundation).
#[cfg(windows)]
fn generate_video_thumbnail(video: &std::path::Path, out: &std::path::Path) -> Result<(), String> {
    use core_engine::video::VideoDecoder;
    let mut dec = crate::video_decode::MfVideoDecoder::new(video)?;
    let start = std::time::Instant::now();
    let frame = loop {
        if let Some(f) = dec.next_frame() { break f; }
        if start.elapsed() > std::time::Duration::from_secs(4) {
            return Err("timed out waiting for the first frame".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    drop(dec); // stop decoding
    let img = nv12_to_rgb_image(&frame);
    let thumb = image::imageops::thumbnail(&img, 480, 270);
    thumb.save(out).map_err(|e| format!("save thumbnail: {e}"))?;
    Ok(())
}

/// Convert an NV12 frame to an RGB image (BT.709 limited range). One-time, thumbnail-only.
#[cfg(windows)]
fn nv12_to_rgb_image(f: &core_engine::video::VideoFrame) -> image::RgbImage {
    let w = f.width as usize;
    let h = f.height as usize;
    let mut img = image::RgbImage::new(f.width, f.height);
    for y in 0..h {
        let uv_row = (y / 2) * w;
        for x in 0..w {
            let yy = f.y.get(y * w + x).copied().unwrap_or(0) as f32;
            let ui = uv_row + (x / 2) * 2;
            let u = f.uv.get(ui).copied().unwrap_or(128) as f32;
            let v = f.uv.get(ui + 1).copied().unwrap_or(128) as f32;
            let yf = 1.164383 * (yy - 16.0);
            let uf = u - 128.0;
            let vf = v - 128.0;
            let r = (yf + 1.792741 * vf).clamp(0.0, 255.0) as u8;
            let g = (yf - 0.213249 * uf - 0.532909 * vf).clamp(0.0, 255.0) as u8;
            let b = (yf + 2.112402 * uf).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    img
}

fn run_cli_mode(wallpaper_dir: String) -> Result<(), Box<dyn std::error::Error>> {
    use winit::application::ApplicationHandler;
    use winit::event::{WindowEvent, ElementState, MouseButton};
    use winit::event_loop::{ControlFlow, EventLoop};
    use winit::window::Window;

    struct App {
        wallpaper_dir: String,
        window: Option<Arc<Window>>,
        renderer: Option<core_engine::Renderer>,
        context: Option<Arc<core_engine::GraphicsContext>>,
        rt: tokio::runtime::Runtime,
        last_metrics_update: std::time::Instant,
        frame_count: u32,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            let attributes = Window::default_attributes()
                .with_title("Strata Wallpaper Runner")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
            let window = Arc::new(event_loop.create_window(attributes).unwrap());
            self.window = Some(window.clone());

            let context = Arc::new(self.rt.block_on(core_engine::GraphicsContext::new()).unwrap());
            self.context = Some(context.clone());

            let surface = context.instance.create_surface(window.clone()).unwrap();
            let mut renderer = core_engine::Renderer::new(context, window.clone(), surface, window.inner_size()).unwrap();
            
            if let Err(e) = renderer.add_layer(std::path::Path::new(&self.wallpaper_dir), 1.0, 1.0, "Fill".to_string(), [0.0, 0.0, 1.0, 1.0], "normal".to_string()) {
                log::error!("Shader ingestion error: {}", e);
            }
            self.renderer = Some(renderer);
        }

        fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
            let renderer = self.renderer.as_mut().unwrap();
            let window = self.window.as_ref().unwrap();

            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    renderer.resize(physical_size);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    renderer.uniform_state.set_mouse_position(position.x as f32, position.y as f32);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        renderer.uniform_state.set_mouse_down(state == ElementState::Pressed);
                    }
                }
                WindowEvent::RedrawRequested => {
                    self.frame_count += 1;
                    let now = std::time::Instant::now();
                    let elapsed = now.duration_since(self.last_metrics_update);
                    
                    if elapsed >= std::time::Duration::from_millis(500) {
                        let fps = self.frame_count as f32 / elapsed.as_secs_f32();
                        let frame_time = elapsed.as_secs_f32() * 1000.0 / self.frame_count as f32;
                        let vram = renderer.estimate_vram_mb();
                        window.set_title(&format!("Strata Engine | FPS: {:.1} | {:.2}ms | VRAM: {:.1}MB", fps, frame_time, vram));
                        self.last_metrics_update = now;
                        self.frame_count = 0;
                    }

                    match renderer.render() {
                        Ok(_) => {}
                        Err(core_engine::wgpu::CurrentSurfaceTexture::Outdated) => {
                            renderer.resize(window.inner_size());
                        }
                        Err(e) => log::error!("Render error: {:?}", e),
                    }
                    window.request_redraw();
                }
                _ => ()
            }
        }
    }

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        wallpaper_dir,
        window: None,
        renderer: None,
        context: None,
        rt,
        last_metrics_update: std::time::Instant::now(),
        frame_count: 0,
    };

    event_loop.run_app(&mut app).map_err(|e| e.into())
}

/// Dev harness: decode a single .mp4 with the Media Foundation decoder and blit each
/// frame to a window. Lets us validate decode + upload + orientation + 4K performance
/// on real hardware before integrating video into the full wallpaper system.
#[cfg(windows)]
fn run_video_mode(video_path: String) -> Result<(), Box<dyn std::error::Error>> {
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ControlFlow, EventLoop};
    use winit::window::Window;
    use core_engine::wgpu;
    use core_engine::video::VideoDecoder;

    // NV12 -> RGB on the GPU: sample the luma (Y) and half-res interleaved chroma (UV)
    // planes, then BT.709 limited-range conversion. Keeps colour conversion off the CPU.
    const BLIT_WGSL: &str = r#"
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

    struct App {
        path: String,
        window: Option<Arc<Window>>,
        context: Option<Arc<core_engine::GraphicsContext>>,
        surface: Option<wgpu::Surface<'static>>,
        config: Option<wgpu::SurfaceConfiguration>,
        decoder: Option<Box<dyn core_engine::video::VideoDecoder>>,
        y_tex: Option<wgpu::Texture>,
        uv_tex: Option<wgpu::Texture>,
        bind_group: Option<wgpu::BindGroup>,
        pipeline: Option<wgpu::RenderPipeline>,
        rt: tokio::runtime::Runtime,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            let decoder = match crate::video_decode::MfVideoDecoder::new(std::path::Path::new(&self.path)) {
                Ok(d) => d,
                Err(e) => { log::error!("Video open failed: {e}"); event_loop.exit(); return; }
            };
            let (vw, vh) = decoder.dimensions();
            log::info!("Video opened: {}x{}", vw, vh);

            let attributes = Window::default_attributes()
                .with_title("Strata Video Test")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
            let window = Arc::new(event_loop.create_window(attributes).unwrap());
            let context = Arc::new(self.rt.block_on(core_engine::GraphicsContext::new()).unwrap());
            let surface = context.instance.create_surface(window.clone()).unwrap();
            let caps = surface.get_capabilities(&context.adapter);
            let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&context.device, &config);

            let mk_tex = |label, w, h, fmt| context.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Y = full-res R8 (luma); UV = half-res Rg8 (interleaved chroma).
            let y_tex = mk_tex("video Y", vw, vh, wgpu::TextureFormat::R8Unorm);
            let uv_tex = mk_tex("video UV", vw / 2, vh / 2, wgpu::TextureFormat::Rg8Unorm);
            let y_view = y_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let uv_view = uv_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("video blit"), source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
            });
            let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
                binding, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            };
            let bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    tex_entry(0),
                    tex_entry(1),
                    wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                ],
            });
            let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            let layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
            });
            let pipeline = context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None, layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None, multisample: wgpu::MultisampleState::default(),
                multiview_mask: None, cache: None,
            });

            self.window = Some(window);
            self.context = Some(context);
            self.surface = Some(surface);
            self.config = Some(config);
            self.decoder = Some(Box::new(decoder));
            self.y_tex = Some(y_tex);
            self.uv_tex = Some(uv_tex);
            self.bind_group = Some(bind_group);
            self.pipeline = Some(pipeline);
        }

        fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(sz) => {
                    if let (Some(ctx), Some(surface), Some(config)) = (&self.context, &self.surface, &mut self.config) {
                        config.width = sz.width.max(1);
                        config.height = sz.height.max(1);
                        surface.configure(&ctx.device, config);
                    }
                }
                WindowEvent::RedrawRequested => {
                    let (Some(ctx), Some(surface), Some(decoder), Some(y_tex), Some(uv_tex), Some(bg), Some(pipe), Some(window)) =
                        (&self.context, &self.surface, &mut self.decoder, &self.y_tex, &self.uv_tex, &self.bind_group, &self.pipeline, &self.window)
                    else { return };

                    if let Some(frame) = decoder.next_frame() {
                        // Y plane -> R8 texture (full res); UV plane -> Rg8 texture (half res).
                        ctx.queue.write_texture(
                            wgpu::TexelCopyTextureInfo { texture: y_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                            &frame.y,
                            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(frame.width), rows_per_image: Some(frame.height) },
                            wgpu::Extent3d { width: frame.width, height: frame.height, depth_or_array_layers: 1 },
                        );
                        ctx.queue.write_texture(
                            wgpu::TexelCopyTextureInfo { texture: uv_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                            &frame.uv,
                            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(frame.width), rows_per_image: Some(frame.height / 2) },
                            wgpu::Extent3d { width: frame.width / 2, height: frame.height / 2, depth_or_array_layers: 1 },
                        );
                    }

                    let surface_tex = match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                        _ => { window.request_redraw(); return; }
                    };
                    let view = surface_tex.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    {
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view, depth_slice: None, resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
                        });
                        rp.set_pipeline(pipe);
                        rp.set_bind_group(0, bg, &[]);
                        rp.draw(0..3, 0..1);
                    }
                    ctx.queue.submit([enc.finish()]);
                    surface_tex.present();
                    window.request_redraw();
                }
                _ => {}
            }
        }
    }

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        path: video_path, window: None, context: None, surface: None, config: None,
        decoder: None, y_tex: None, uv_tex: None, bind_group: None, pipeline: None, rt,
    };
    event_loop.run_app(&mut app).map_err(|e| e.into())
}

/// Turn an absolute filesystem path into a `file:///C:/...` URL (forward slashes).
fn file_url(p: &std::path::Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    format!("file:///{}", s.trim_start_matches('/'))
}

/// Dev harness: play a single .mp4 via a system WebView `<video>` element. Validates that
/// the WebView path reaches the low CPU/RAM the browser's media pipeline gives (the
/// approach Seelen uses), before we make it the real video-wallpaper renderer.
fn run_video_web_mode(video_path: String) -> Result<(), Box<dyn std::error::Error>> {
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ControlFlow, EventLoop};
    use winit::window::Window;
    use wry::WebViewBuilder;

    let abs = if std::path::Path::new(&video_path).is_absolute() {
        std::path::PathBuf::from(&video_path)
    } else {
        std::env::current_dir()?.join(&video_path)
    };
    let video_url = file_url(&abs);
    // The page itself is loaded from file:// so the file:// <video> source is same-origin,
    // and muted autoplay is allowed without a user gesture.
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>\
         html,body{{margin:0;height:100%;background:#000;overflow:hidden}}\
         video{{position:fixed;inset:0;width:100%;height:100%;object-fit:cover}}\
         </style></head><body>\
         <video src=\"{video_url}\" autoplay loop muted playsinline preload=\"auto\"></video>\
         </body></html>"
    );
    let html_path = std::env::temp_dir().join("strata-video-web-test.html");
    std::fs::write(&html_path, html)?;
    let page_url = file_url(&html_path);
    log::info!("WebView video test page: {page_url}");

    struct App {
        page_url: String,
        window: Option<Arc<Window>>,
        _webview: Option<wry::WebView>,
    }
    impl ApplicationHandler for App {
        fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
            let window = Arc::new(el.create_window(
                Window::default_attributes()
                    .with_title("Strata Video (WebView) Test")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
            ).unwrap());
            let webview = WebViewBuilder::new(&*window)
                .with_url(&self.page_url)
                .build()
                .expect("build webview");
            self.window = Some(window);
            self._webview = Some(webview);
        }
        fn window_event(&mut self, el: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
            if let WindowEvent::CloseRequested = event { el.exit(); }
        }
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App { page_url, window: None, _webview: None };
    event_loop.run_app(&mut app)?;
    Ok(())
}

// Tracks every live wallpaper window so they can be closed and recreated when
// the monitor configuration changes.  The third tuple element is the monitor
// id, so we can map a monitor → its window for per-monitor teardown.  Rc is
// fine - accessed only on the Slint main thread (spawn_local, timer callbacks).
type WallpaperWindowStore =
    std::rc::Rc<std::cell::RefCell<Vec<(WallpaperWindow, winit::window::WindowId, String)>>>;

// Wallpaper windows queued for destruction.  When a monitor loses its last
// shader we send WindowClosed (so the render thread shuts down and releases its
// surface) but DEFER dropping the Slint component - dropping it destroys the
// HWND, and doing that out from under a still-running render thread would race
// surface.get_current_texture() against a dead window.  The UI timer drops these
// a few hundred ms later, by which point the render thread has exited.
type PendingCloseStore =
    std::rc::Rc<std::cell::RefCell<Vec<(WallpaperWindow, std::time::Instant)>>>;

/// One video wallpaper the daemon is currently showing, tracked in the MAIN process so it
/// can reconcile changes and decide when to kill the daemon.
#[derive(Clone, PartialEq)]
struct VideoEntry { path: std::path::PathBuf, fit: String, geom: (i32, i32, u32, u32) }

/// Handle to the running video-daemon child process (main process side).
struct DaemonHandle { child: std::process::Child, stdin: std::process::ChildStdin }

/// Commands the main process streams to the daemon over stdin (one JSON object per line).
#[derive(serde::Serialize, serde::Deserialize)]
enum DaemonCmd {
    SetVideo { monitor_id: String, x: i32, y: i32, w: u32, h: u32, path: String, fit: String },
    RemoveVideo { monitor_id: String },
    SetFit { monitor_id: String, fit: String },
    SetPaused { monitor_id: String, paused: bool },
    Shutdown,
}

thread_local! {
    /// MAIN PROCESS: the running video-daemon child (None until the first video wallpaper).
    static VIDEO_DAEMON: std::cell::RefCell<Option<DaemonHandle>> =
        const { std::cell::RefCell::new(None) };
    /// MAIN PROCESS: what we've told the daemon to show, per monitor (for reconciliation).
    static VIDEO_STATE: std::cell::RefCell<std::collections::HashMap<String, VideoEntry>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// MAIN PROCESS: last pause state sent per monitor (so we only send on change).
    static VIDEO_PAUSED: std::cell::RefCell<std::collections::HashMap<String, bool>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    /// DAEMON PROCESS: shared WebView2 data context (folder under %LOCALAPPDATA%\Strata, not
    /// next to the exe). Only ever touched inside the daemon child.
    static WEB_CONTEXT: std::cell::RefCell<Option<wry::WebContext>> =
        const { std::cell::RefCell::new(None) };
}

/// Monitor ids that currently have a video wallpaper (as told to the daemon).
fn video_daemon_monitors() -> Vec<String> {
    VIDEO_STATE.with(|s| s.borrow().keys().cloned().collect())
}

/// Map a layer's positioning preset to the CSS `object-fit` for a video wallpaper.
/// Fill = cover the screen (crop, keep aspect); Fit = whole frame letterboxed; Stretch =
/// distort to fill; Center/Custom = native size, centred. Shaders ignore this (they map
/// via their own positioning); it only drives the `<video>` element.
fn positioning_to_object_fit(positioning: &str) -> &'static str {
    match positioning {
        "Fit" => "contain",
        "Stretch" => "fill",
        "Center" | "Custom" => "none",
        _ => "cover", // "Fill" and any unknown default
    }
}

/// Spawn the video-daemon child if it isn't already running. WebView2 lives ONLY in this
/// child, so killing it fully reclaims that memory from the main app.
fn daemon_ensure_spawned() {
    VIDEO_DAEMON.with(|d| {
        let mut h = d.borrow_mut();
        if h.is_some() { return; }
        let exe = match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => { log::error!("video daemon: current_exe: {e}"); return; }
        };
        match std::process::Command::new(exe).arg("--video-daemon")
            .stdin(std::process::Stdio::piped()).spawn()
        {
            Ok(mut child) => {
                let stdin = child.stdin.take().expect("piped stdin");
                *h = Some(DaemonHandle { child, stdin });
                log::info!("video daemon spawned");
            }
            Err(e) => log::error!("video daemon spawn failed: {e}"),
        }
    });
}

/// Send one command to the daemon (no-op if it isn't running).
fn daemon_send(cmd: &DaemonCmd) {
    VIDEO_DAEMON.with(|d| {
        if let Some(h) = d.borrow_mut().as_mut() {
            use std::io::Write;
            if let Ok(line) = serde_json::to_string(cmd) {
                if writeln!(h.stdin, "{line}").and_then(|_| h.stdin.flush()).is_err() {
                    log::warn!("video daemon: write failed (process gone?)");
                }
            }
        }
    });
}

/// Kill the daemon entirely (instant, full WebView2 RAM reclaim) and forget our state.
fn daemon_shutdown() {
    VIDEO_DAEMON.with(|d| {
        if let Some(mut h) = d.borrow_mut().take() {
            let _ = h.child.kill();
            let _ = h.child.wait();
            log::info!("video daemon stopped");
        }
    });
    VIDEO_STATE.with(|s| s.borrow_mut().clear());
}

/// Live-update the fit of a running video wallpaper without recreating it.
fn daemon_set_fit(monitor_id: &str, object_fit: &str) {
    daemon_send(&DaemonCmd::SetFit { monitor_id: monitor_id.to_string(), fit: object_fit.to_string() });
    VIDEO_STATE.with(|s| { if let Some(e) = s.borrow_mut().get_mut(monitor_id) { e.fit = object_fit.to_string(); } });
}

/// Pause/resume a running video wallpaper (used by the cover-detection check).
fn daemon_set_paused(monitor_id: &str, paused: bool) {
    daemon_send(&DaemonCmd::SetPaused { monitor_id: monitor_id.to_string(), paused });
}

/// True if a true-fullscreen app covers this monitor (shared z-order detector in
/// platform::windows; per-monitor + focus-independent).
#[cfg(windows)]
fn monitor_covered(origin: (i32, i32), size: (u32, u32)) -> bool {
    platform::windows::monitor_covered(origin, size)
}
#[cfg(not(windows))]
fn monitor_covered(_origin: (i32, i32), _size: (u32, u32)) -> bool { false }

/// Pause each video wallpaper whose monitor is covered by a fullscreen app, resume it when
/// uncovered. Only sends on a state change. Called on a slow throttle from the UI timer so
/// gaming / fullscreen video drops the movie's decode to ~0.
fn update_video_pause(app_state: &controller::SharedState) {
    let monitors = video_daemon_monitors();
    if monitors.is_empty() {
        VIDEO_PAUSED.with(|p| p.borrow_mut().clear());
        return;
    }
    let geoms: Vec<(String, (i32, i32), (u32, u32))> = {
        let state = app_state.read().unwrap();
        monitors.iter().filter_map(|id|
            state.monitors.iter().find(|m| &m.id == id).map(|m| (id.clone(), m.position, m.resolution))
        ).collect()
    };
    VIDEO_PAUSED.with(|p| {
        let mut map = p.borrow_mut();
        map.retain(|k, _| monitors.contains(k));
        for (id, origin, size) in geoms {
            let covered = monitor_covered(origin, size);
            if map.get(&id) != Some(&covered) {
                daemon_set_paused(&id, covered);
                map.insert(id, covered);
            }
        }
    });
}

/// Reconcile the daemon with the desired per-monitor video state: spawn it on demand, send
/// Set/Remove as assignments change, and kill it when no video remains. Video is always
/// per-monitor (never spanned). Call after any wallpaper/monitor change.
fn reconcile_video_daemon(app_state: &controller::SharedState) {
    let desired: std::collections::HashMap<String, VideoEntry> = {
        let state = app_state.read().unwrap();
        state.monitors.iter().filter_map(|m| {
            let (path, fit) = m.layers.iter().find_map(|l|
                controller::video_wallpaper_path(&l.wallpaper_path)
                    .map(|p| (p, positioning_to_object_fit(&l.positioning).to_string())))?;
            Some((m.id.clone(), VideoEntry {
                path, fit, geom: (m.position.0, m.position.1, m.resolution.0, m.resolution.1),
            }))
        }).collect()
    };

    let empty = VIDEO_STATE.with(|s| {
        let mut cur = s.borrow_mut();

        // Drop monitors that no longer want a video.
        let stale: Vec<String> = cur.keys().filter(|k| !desired.contains_key(*k)).cloned().collect();
        for id in stale {
            daemon_send(&DaemonCmd::RemoveVideo { monitor_id: id.clone() });
            cur.remove(&id);
        }

        // Create/replace monitors whose video path or geometry changed (fit-only changes
        // go through daemon_set_fit, which keeps `cur` in sync so they don't re-trigger).
        for (id, want) in &desired {
            let needs_set = match cur.get(id) {
                Some(have) => have.path != want.path || have.geom != want.geom,
                None => true,
            };
            if needs_set {
                daemon_ensure_spawned();
                daemon_send(&DaemonCmd::SetVideo {
                    monitor_id: id.clone(),
                    x: want.geom.0, y: want.geom.1, w: want.geom.2, h: want.geom.3,
                    path: want.path.to_string_lossy().into_owned(),
                    fit: want.fit.clone(),
                });
                cur.insert(id.clone(), want.clone());
            }
        }
        cur.is_empty()
    });

    // No videos left anywhere -> tear the whole daemon down (reclaims all its memory).
    if empty { daemon_shutdown(); }
}

/// Attach a system WebView playing `video_path` as the wallpaper, filling `window`. The
/// browser's hardware media pipeline does the heavy lifting (zero-copy), so the host
/// process stays tiny. `window` must already be reparented into WorkerW. `object_fit` is
/// the CSS fit (see `positioning_to_object_fit`).
fn attach_video_webview(window: &winit::window::Window, video_path: &std::path::Path, monitor_id: &str, object_fit: &str) -> Result<wry::WebView, String> {
    let abs = if video_path.is_absolute() {
        video_path.to_path_buf()
    } else {
        std::env::current_dir().map_err(|e| e.to_string())?.join(video_path)
    };
    let video_url = file_url(&abs);
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>\
         html,body{{margin:0;height:100%;background:#000;overflow:hidden}}\
         video{{position:fixed;inset:0;width:100%;height:100%;object-fit:{object_fit}}}\
         </style></head><body>\
         <video src=\"{video_url}\" autoplay loop muted playsinline preload=\"auto\"></video>\
         </body></html>"
    );
    // Load the page from file:// so the file:// <video> source is same-origin and muted
    // autoplay is allowed without a user gesture.
    let html_path = std::env::temp_dir().join(format!("strata-wallpaper-{}.html", sanitize_name(monitor_id)));
    std::fs::write(&html_path, html).map_err(|e| e.to_string())?;
    let page_url = file_url(&html_path);

    // One shared WebContext for all video wallpapers, with its data folder under
    // %LOCALAPPDATA%\Strata. Without this, WebView2 drops a writable
    // "<exe>.exe.WebView2" cache next to the executable - which fails (or needs admin)
    // when Strata is installed under Program Files. Keeping one context also avoids the
    // "data folder in use with different options" error from multiple environments.
    WEB_CONTEXT.with(|c| -> Result<wry::WebView, String> {
        let mut guard = c.borrow_mut();
        if guard.is_none() {
            let base = std::env::var("LOCALAPPDATA")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::env::temp_dir());
            let dir = base.join("Strata").join("WebView2");
            std::fs::create_dir_all(&dir).ok();
            *guard = Some(wry::WebContext::new(Some(dir)));
        }
        let ctx = guard.as_mut().unwrap();

        let mut builder = wry::WebViewBuilder::new(window)
            .with_url(&page_url)
            .with_transparent(false) // opaque: skip per-frame alpha compositing
            .with_web_context(ctx);
        // WebView2 (Windows): trim features we never use and guarantee autoplay.
        // Marginal CPU/RAM win, but free. Other platforms keep the defaults.
        #[cfg(target_os = "windows")]
        {
            use wry::WebViewBuilderExtWindows;
            builder = builder.with_additional_browser_args(
                "--autoplay-policy=no-user-gesture-required \
                 --disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection \
                 --disable-background-timer-throttling",
            );
        }
        builder.build().map_err(|e| e.to_string())
    })
}

/// Create one WorkerW-parented video window for the daemon and attach its WebView. Mirrors
/// the shader path's Win32 sequence (prepare -> show -> suppress-activation -> SetParent).
#[cfg(target_os = "windows")]
fn create_daemon_video_window(
    el: &winit::event_loop::ActiveEventLoop,
    x: i32, y: i32, w: u32, h: u32, path: &str, fit: &str, monitor_id: &str,
) -> Result<(std::sync::Arc<winit::window::Window>, wry::WebView), String> {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let window = std::sync::Arc::new(el.create_window(
        winit::window::Window::default_attributes()
            .with_visible(false)
            .with_decorations(false)
            .with_inner_size(winit::dpi::PhysicalSize::new(w, h))
            .with_position(winit::dpi::PhysicalPosition::new(x, y)),
    ).map_err(|e| e.to_string())?);

    if let Ok(handle) = window.window_handle() {
        if let RawWindowHandle::Win32(hh) = handle.as_raw() {
            let hwnd = hh.hwnd.get() as isize;
            platform::windows::prepare_wallpaper_window(hwnd as _, x, y, w, h);
            window.set_visible(true);
            platform::windows::suppress_activation(hwnd as _);
            match platform::windows::get_wallpaper_window() {
                Some(parent) => platform::windows::setup_wallpaper_window(hwnd as _, parent, x, y, w, h),
                None => log::warn!("daemon: WorkerW not found - video at ({x},{y}) above icons"),
            }
        }
    }
    let webview = attach_video_webview(window.as_ref(), std::path::Path::new(path), monitor_id, fit)?;
    Ok((window, webview))
}

/// The video-daemon child process (`--video-daemon`). Owns all video-wallpaper windows +
/// WebViews so WebView2 never loads into the main app. Reads `DaemonCmd`s from stdin; exits
/// when told to (or when stdin closes, i.e. the parent died).
fn run_video_daemon() -> Result<(), Box<dyn std::error::Error>> {
    use winit::application::ApplicationHandler;
    use winit::event_loop::EventLoop;

    let event_loop = EventLoop::<DaemonCmd>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    // Stream commands from stdin on a worker thread.
    std::thread::Builder::new().name("strata-video-daemon-stdin".into()).spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) if l.trim().is_empty() => continue,
                Ok(l) => match serde_json::from_str::<DaemonCmd>(&l) {
                    Ok(cmd) => { if proxy.send_event(cmd).is_err() { return; } }
                    Err(e) => eprintln!("video daemon: bad command: {e}"),
                },
                Err(_) => break,
            }
        }
        // stdin closed (parent exited) -> ask the loop to shut down.
        let _ = proxy.send_event(DaemonCmd::Shutdown);
    }).ok();

    struct Daemon {
        windows: std::collections::HashMap<String, (std::sync::Arc<winit::window::Window>, wry::WebView)>,
    }
    impl ApplicationHandler<DaemonCmd> for Daemon {
        fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
            // Idle between commands - the daemon only reacts to stdin events, so don't
            // busy-poll a CPU core.
            el.set_control_flow(winit::event_loop::ControlFlow::Wait);
        }
        fn user_event(&mut self, el: &winit::event_loop::ActiveEventLoop, cmd: DaemonCmd) {
            match cmd {
                DaemonCmd::SetVideo { monitor_id, x, y, w, h, path, fit } => {
                    self.windows.remove(&monitor_id); // replace any existing
                    #[cfg(target_os = "windows")]
                    match create_daemon_video_window(el, x, y, w, h, &path, &fit, &monitor_id) {
                        Ok(pair) => { self.windows.insert(monitor_id, pair); }
                        Err(e) => eprintln!("video daemon SetVideo failed: {e}"),
                    }
                    #[cfg(not(target_os = "windows"))]
                    { let _ = (el, x, y, w, h, path, fit, monitor_id); }
                }
                DaemonCmd::RemoveVideo { monitor_id } => { self.windows.remove(&monitor_id); }
                DaemonCmd::SetFit { monitor_id, fit } => {
                    if let Some((_, wv)) = self.windows.get(&monitor_id) {
                        let _ = wv.evaluate_script(&format!(
                            "{{var e=document.querySelector('video');if(e)e.style.objectFit='{fit}';}}"));
                    }
                }
                DaemonCmd::SetPaused { monitor_id, paused } => {
                    if let Some((_, wv)) = self.windows.get(&monitor_id) {
                        let js = if paused {
                            "{var e=document.querySelector('video');if(e)e.pause();}"
                        } else {
                            "{var e=document.querySelector('video');if(e)e.play();}"
                        };
                        let _ = wv.evaluate_script(js);
                    }
                }
                DaemonCmd::Shutdown => el.exit(),
            }
        }
        fn window_event(&mut self, _el: &winit::event_loop::ActiveEventLoop,
            _id: winit::window::WindowId, _event: winit::event::WindowEvent) {}
    }

    let mut daemon = Daemon { windows: std::collections::HashMap::new() };
    event_loop.run_app(&mut daemon)?;
    Ok(())
}

/// Create one wallpaper window per monitor and reparent each to the desktop WorkerW.
/// Shader monitors get a wgpu surface + render thread (stored in `store`); video monitors
/// are handed off to the daemon by `reconcile_video_daemon`. Safe to call repeatedly
/// (idempotent per monitor).
fn spawn_wallpaper_windows(
    app_state: controller::SharedState,
    command_tx: std::sync::mpsc::Sender<EngineCommand>,
    context: std::sync::Arc<core_engine::GraphicsContext>,
    store: WallpaperWindowStore,
) {
    slint::spawn_local(async move {
        // Snapshot the monitor list and span setting before the first await.
        let (monitors, span_monitors) = {
            let state = app_state.read().unwrap();
            (state.monitors.clone(), state.span_monitors)
        };

        // In span mode every monitor renders the PRIMARY display's shader as a
        // slice of one unified canvas. Video wallpapers never span (a single clip can't
        // be sliced across WebViews), so they're filtered out here and handled strictly
        // per-monitor below.
        let span_layers: Vec<LayerInfo> = controller::primary_monitor(&monitors)
            .map(|m| m.layers.iter()
                .filter(|l| controller::video_wallpaper_path(&l.wallpaper_path).is_none())
                .cloned().collect())
            .unwrap_or_default();

        // Hand video monitors off to the daemon process (spawns/kills it as needed). This
        // covers startup + every monitor/wallpaper change, since sync + refresh both call
        // here. Video monitors are then skipped in the shader loop below.
        reconcile_video_daemon(&app_state);

        // Idempotent: skip monitors that already have a shader window in the store, so
        // this can be called incrementally when a monitor gains its first shader.
        let existing: Vec<String> =
            store.borrow().iter().map(|(_, _, id)| id.clone()).collect();

        let mut min_x = 0;
        let mut min_y = 0;
        let mut max_x = 0;
        let mut max_y = 0;

        if span_monitors {
            for m in &monitors {
                min_x = min_x.min(m.position.0);
                min_y = min_y.min(m.position.1);
                max_x = max_x.max(m.position.0 + m.resolution.0 as i32);
                max_y = max_y.max(m.position.1 + m.resolution.1 as i32);
            }
        }

        let global_res = if span_monitors {
            ((max_x - min_x) as f32, (max_y - min_y) as f32)
        } else {
            (0.0, 0.0) // Will be set per-monitor below
        };

        // Obtain the WorkerW handle once for all monitors.
        // WorkerW is a global shell window - it appears on ALL virtual desktops
        // automatically, so our wallpaper children do too.
        #[cfg(target_os = "windows")]
        let wallpaper_parent = platform::windows::get_wallpaper_window();

        for m in monitors.iter() {
            let (screen_x, screen_y, mon_w, mon_h, monitor_id) = (
                m.position.0,
                m.position.1,
                m.resolution.0,
                m.resolution.1,
                m.id.clone(),
            );

            // Video monitors are owned by the daemon (reconcile_video_daemon above), not
            // the wgpu store - skip them here entirely so we never build a shader window
            // for a monitor that's playing a movie.
            let has_own_video = m.layers.iter()
                .any(|l| controller::video_wallpaper_path(&l.wallpaper_path).is_some());
            if has_own_video {
                continue;
            }

            // Effective shader layers: span mode shows the primary display's shader on
            // every monitor (each its own slice of the canvas); otherwise each monitor
            // shows its own assigned shader layers (video excluded).
            let layers: Vec<LayerInfo> = if span_monitors {
                span_layers.clone()
            } else {
                m.layers.iter()
                    .filter(|l| controller::video_wallpaper_path(&l.wallpaper_path).is_none())
                    .cloned().collect()
            };

            // Release-desktop rule: a monitor with no shader gets NO window at all, so the
            // user's real Windows wallpaper shows through and no swapchain VRAM / render
            // thread is spent on it.
            if layers.is_empty() {
                continue;
            }

            // Idempotent: this monitor already has a live shader window - leave it be.
            if existing.contains(&monitor_id) {
                continue;
            }

            let wall_ui = WallpaperWindow::new().unwrap();
            wall_ui.window().set_size(slint::PhysicalSize::new(mon_w, mon_h));

            let command_tx_clone  = command_tx.clone();
            let command_tx_closed = command_tx.clone();
            let context_clone = context.clone();

            let wall_ui_win = wall_ui.window();
            #[cfg(target_os = "windows")]
            let mut raw_hwnd: Option<isize> = None;

            // `winit_window` resolves when Slint has created the underlying Win32
            // window.  We harvest the HWND here (before show) to configure Win32
            // styles without a flash on the wrong monitor.
            // NOTE: We do NOT create the wgpu surface here.  The surface must be
            // created AFTER show() + SetParent so that DXGI initialises its swap
            // chain against the window's final reparented geometry.  If the surface
            // is created before SetParent, DXGI caches the pre-reparent window rect
            // and only presents to the top-left portion of the monitor.
            let mut captured_window_id: Option<winit::window::WindowId> = None;
            let mut captured_window: Option<std::sync::Arc<winit::window::Window>> = None;

            if let Ok(w) = slint::winit_030::WinitWindowAccessor::winit_window(wall_ui_win).await {
                let window_id = w.id();
                captured_window_id = Some(window_id);

                // Wallpaper windows have a FIXED physical size (the monitor's
                // native resolution). We intentionally do NOT forward
                // WindowEvent::Resized here because show(), SW_SHOWNA, and
                // SetParent all generate WM_SIZE messages that arrive AFTER the
                // current async task yields - they would overwrite the surface
                // config set by initial_size with whatever Windows transiently
                // reports.  The renderer uses initial_size directly and only
                // the Renderer::resize() recovery path re-reads the size.
                slint::winit_030::WinitWindowAccessor::on_winit_window_event(
                    wall_ui_win,
                    move |_, event| {
                        if let winit::event::WindowEvent::CloseRequested = event {
                            let _ = command_tx_closed
                                .send(EngineCommand::WindowClosed(window_id));
                        }
                        slint::winit_030::EventResult::Propagate
                    },
                );

                #[cfg(target_os = "windows")]
                {
                    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
                    if let Ok(handle) = w.window_handle() {
                        if let RawWindowHandle::Win32(h) = handle.as_raw() {
                            let hwnd_val = h.hwnd.get() as isize;
                            raw_hwnd = Some(hwnd_val);
                            log::info!(
                                "Wallpaper HWND {:x} for monitor at ({},{}) {}x{}",
                                hwnd_val, screen_x, screen_y, mon_w, mon_h
                            );
                            platform::windows::prepare_wallpaper_window(
                                hwnd_val as _, screen_x, screen_y, mon_w, mon_h,
                            );
                        }
                    }
                }

                // Store window handle - surface creation happens below after Win32 setup.
                captured_window = Some(w);
            }

            wall_ui.show().unwrap();

            #[cfg(target_os = "windows")]
            if let Some(hwnd) = raw_hwnd {
                platform::windows::suppress_activation(hwnd as _);
            }

            #[cfg(target_os = "windows")]
            match (raw_hwnd, wallpaper_parent) {
                (Some(hwnd), Some(parent)) => {
                    platform::windows::setup_wallpaper_window(
                        hwnd as _, parent, screen_x, screen_y, mon_w, mon_h,
                    );
                }
                (Some(_), None) => {
                    log::warn!("WorkerW not found - wallpaper at ({},{}) above icons", screen_x, screen_y);
                }
                _ => {
                    log::error!("No HWND captured for monitor at ({},{})", screen_x, screen_y);
                }
            }

            // ── Create wgpu surface AFTER all Win32 setup ─────────────────────────
            // show() + suppress_activation() + setup_wallpaper_window() (SetParent)
            // must all complete before the surface is created so that DXGI binds
            // the swap chain to the window in its final reparented state.
            // w.inner_size() now reflects the physical client area after all Win32
            // calls; we use it as the authoritative surface size and log both values
            // for DPI diagnostics.
            if let (Some(w), Some(wid)) = (captured_window.take(), captured_window_id) {
                let inner = w.inner_size();
                let surface_size = if inner.width > 0 && inner.height > 0 {
                    inner
                } else {
                    winit::dpi::PhysicalSize::new(mon_w, mon_h)
                };
                log::info!(
                    "Surface post-setup ({},{}) inner={}x{} display_info={}x{}",
                    screen_x, screen_y,
                    inner.width, inner.height,
                    mon_w, mon_h
                );
                let surface = context_clone.instance.create_surface(w.clone()).unwrap();
                let (offset, res) = if span_monitors {
                    (( (screen_x - min_x) as f32, (screen_y - min_y) as f32 ), global_res)
                } else {
                    ((0.0, 0.0), (surface_size.width as f32, surface_size.height as f32))
                };

                command_tx_clone.send(EngineCommand::AddWindow {
                    window: w,
                    surface,
                    initial_size: surface_size,
                    offset,
                    global_res: res,
                    layers,
                    monitor_id: monitor_id.clone(),
                }).ok();

                // Keep the Slint component alive (and tagged with its monitor id)
                // so the window can be torn down later when its shader is removed.
                store.borrow_mut().push((wall_ui, wid, monitor_id));
            } else {
                // Window creation failed - keep the component alive to avoid
                // destroying the window mid-initialisation.
                Box::leak(Box::new(wall_ui));
            }
        }
    }).unwrap();
}

/// Reconcile the live wallpaper windows with the current monitor/layer state.
///
/// Call this whenever a change could flip a monitor between "has a shader" and
/// "has no shader" (assigning the first layer, or removing the last one):
///   * Monitors that no longer have any shader have their window torn down so
///     the real Windows wallpaper returns and their swapchain VRAM is freed.
///   * Monitors that just gained their first shader get a fresh window.
///   * Surviving windows are refreshed via `Reload`.
///
/// Window destruction is deferred through `pending_close` (see its type docs)
/// to avoid dropping an HWND out from under a still-running render thread.
fn sync_wallpaper_windows(
    app_state: controller::SharedState,
    command_tx: std::sync::mpsc::Sender<EngineCommand>,
    context: std::sync::Arc<core_engine::GraphicsContext>,
    store: WallpaperWindowStore,
    pending_close: PendingCloseStore,
) {
    // A monitor's shader layers (video excluded - video never gets a wgpu window; the
    // daemon reconcile, invoked by spawn_wallpaper_windows below, owns video teardown).
    let has_shader = |m: &controller::MonitorInfo| {
        m.layers.iter().any(|l| controller::video_wallpaper_path(&l.wallpaper_path).is_none())
    };

    // Which monitors should currently own a wgpu (shader) window?  Mirror the spawn
    // rule: in span mode every monitor wants one iff the PRIMARY display has a shader
    // (they all show its canvas slice); otherwise a monitor wants one iff it has its
    // own shader.
    let want_window: Vec<String> = {
        let state = app_state.read().unwrap();
        let span = state.span_monitors;
        let primary_has_shader = controller::primary_monitor(&state.monitors)
            .map(|m| has_shader(m))
            .unwrap_or(false);
        state.monitors.iter()
            .filter(|m| if span { primary_has_shader } else { has_shader(m) })
            .map(|m| m.id.clone())
            .collect()
    };

    // Tear down windows whose monitor no longer wants one.  Send WindowClosed
    // now (so the render thread shuts down and releases its surface) but defer
    // the Slint-component drop to the timer.
    {
        let mut windows = store.borrow_mut();
        let mut survivors = Vec::with_capacity(windows.len());
        for (win, wid, mid) in windows.drain(..) {
            if want_window.contains(&mid) {
                survivors.push((win, wid, mid));
            } else {
                log::info!("Releasing desktop for monitor {} (no shader assigned)", mid);
                command_tx.send(EngineCommand::WindowClosed(wid)).ok();
                win.hide().ok();
                pending_close.borrow_mut().push((win, std::time::Instant::now()));
            }
        }
        *windows = survivors;
    }

    // Refresh layer content on the windows that remain.
    command_tx.send(EngineCommand::Reload).ok();

    // Create windows for monitors that just gained their first shader.
    // spawn_wallpaper_windows is idempotent and applies the same empty-monitor
    // filter, so this only ever creates the windows that are actually missing.
    spawn_wallpaper_windows(app_state, command_tx, context, store);
}

/// Returns `true` if another Strata instance already holds the single-instance
/// lock. Uses a named mutex (per-login-session via the `Local\` namespace) whose
/// handle is intentionally leaked so it stays open for this process's whole life -
/// any later instance then sees `ERROR_ALREADY_EXISTS`. Fails open (returns false)
/// if the mutex can't be created, so a quirk never prevents the app from launching.
#[cfg(windows)]
fn another_instance_running() -> bool {
    use windows_sys::Win32::System::Threading::CreateMutexW;
    use windows_sys::Win32::Foundation::{GetLastError, CloseHandle, ERROR_ALREADY_EXISTS};
    unsafe {
        let name: Vec<u16> = "Local\\Strata-Desktop-SingleInstance".encode_utf16().chain(std::iter::once(0)).collect();
        let handle = CreateMutexW(std::ptr::null(), 1, name.as_ptr());
        if handle == 0 {
            return false; // couldn't create the lock - allow launch rather than block it
        }
        if GetLastError() == ERROR_ALREADY_EXISTS {
            CloseHandle(handle);
            return true;
        }
        // Deliberately never CloseHandle: the mutex must stay open for this process's
        // whole life so later instances keep seeing ERROR_ALREADY_EXISTS. The handle is
        // a plain OS handle (no Drop) and Windows releases it automatically on exit.
        false
    }
}

#[cfg(not(windows))]
fn another_instance_running() -> bool { false }

/// Reads the *actual* autostart state from the registry: whether a value named
/// "Strata" exists under HKCU\...\Run. This is the source of truth so the in-app
/// toggle reflects reality regardless of who created the entry - the installer's
/// "Start with Windows" option, the app's own toggle (`auto_launch`, same value
/// name), or a hand edit. Format-independent: we only check the value exists, and
/// `auto_launch.disable()` removes it by the same name, so the two stay in sync.
#[cfg(windows)]
fn autostart_is_enabled() -> bool {
    use windows_sys::Win32::System::Registry::{
        RegOpenKeyExW, RegQueryValueExW, RegCloseKey, HKEY, HKEY_CURRENT_USER, KEY_READ,
    };
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    unsafe {
        let subkey: Vec<u16> = "Software\\Microsoft\\Windows\\CurrentVersion\\Run"
            .encode_utf16().chain(std::iter::once(0)).collect();
        let value: Vec<u16> = "Strata".encode_utf16().chain(std::iter::once(0)).collect();
        let mut hkey: HKEY = 0;
        if RegOpenKeyExW(HKEY_CURRENT_USER, subkey.as_ptr(), 0, KEY_READ, &mut hkey) != ERROR_SUCCESS {
            return false;
        }
        let res = RegQueryValueExW(
            hkey, value.as_ptr(),
            std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
        );
        RegCloseKey(hkey);
        res == ERROR_SUCCESS
    }
}

#[cfg(not(windows))]
fn autostart_is_enabled() -> bool { false }

const MPO_SUBKEY: &str = "SOFTWARE\\Microsoft\\Windows\\Dwm";
const MPO_VALUE: &str = "OverlayTestMode";

/// True if MPO is currently disabled, i.e. `HKLM\…\Dwm\OverlayTestMode == 5`. Reading
/// HKLM needs no elevation. Uses the 64-bit view to match what the installer writes.
#[cfg(windows)]
fn read_mpo_disabled() -> bool {
    use windows_sys::Win32::System::Registry::{
        RegOpenKeyExW, RegQueryValueExW, RegCloseKey, HKEY, HKEY_LOCAL_MACHINE, KEY_READ, KEY_WOW64_64KEY,
    };
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    unsafe {
        let subkey: Vec<u16> = MPO_SUBKEY.encode_utf16().chain(std::iter::once(0)).collect();
        let value: Vec<u16> = MPO_VALUE.encode_utf16().chain(std::iter::once(0)).collect();
        let mut hkey: HKEY = 0;
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, subkey.as_ptr(), 0, KEY_READ | KEY_WOW64_64KEY, &mut hkey) != ERROR_SUCCESS {
            return false;
        }
        let mut data: u32 = 0;
        let mut size: u32 = 4;
        let res = RegQueryValueExW(
            hkey, value.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut(),
            &mut data as *mut u32 as *mut u8, &mut size,
        );
        RegCloseKey(hkey);
        res == ERROR_SUCCESS && data == 5
    }
}

#[cfg(not(windows))]
fn read_mpo_disabled() -> bool { false }

/// Write (`disabled=true` → OverlayTestMode=5) or clear (`false` → delete the value) the
/// MPO registry key. Requires the process to be elevated (HKLM). Returns success.
#[cfg(windows)]
fn set_mpo_disabled(disabled: bool) -> bool {
    use windows_sys::Win32::System::Registry::{
        RegCreateKeyExW, RegSetValueExW, RegDeleteValueW, RegCloseKey, HKEY, HKEY_LOCAL_MACHINE,
        KEY_SET_VALUE, KEY_WOW64_64KEY, REG_DWORD, REG_OPTION_NON_VOLATILE,
    };
    use windows_sys::Win32::Foundation::{ERROR_SUCCESS, ERROR_FILE_NOT_FOUND};
    unsafe {
        let subkey: Vec<u16> = MPO_SUBKEY.encode_utf16().chain(std::iter::once(0)).collect();
        let value: Vec<u16> = MPO_VALUE.encode_utf16().chain(std::iter::once(0)).collect();
        let mut hkey: HKEY = 0;
        let mut disp: u32 = 0;
        if RegCreateKeyExW(
            HKEY_LOCAL_MACHINE, subkey.as_ptr(), 0, std::ptr::null(),
            REG_OPTION_NON_VOLATILE, KEY_SET_VALUE | KEY_WOW64_64KEY, std::ptr::null(), &mut hkey, &mut disp,
        ) != ERROR_SUCCESS {
            return false;
        }
        let ok = if disabled {
            let data: u32 = 5;
            RegSetValueExW(hkey, value.as_ptr(), 0, REG_DWORD, &data as *const u32 as *const u8, 4) == ERROR_SUCCESS
        } else {
            let r = RegDeleteValueW(hkey, value.as_ptr());
            r == ERROR_SUCCESS || r == ERROR_FILE_NOT_FOUND
        };
        RegCloseKey(hkey);
        ok
    }
}

#[cfg(not(windows))]
fn set_mpo_disabled(_disabled: bool) -> bool { false }

/// Relaunch ourselves elevated (UAC) to write the MPO value, waiting for the child to
/// finish. Returns true if the elevated write succeeded (false if the user declined the
/// UAC prompt or it failed). Run off the UI thread — it blocks on the child.
#[cfg(windows)]
fn set_mpo_elevated(disabled: bool) -> bool {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::UI::Shell::{ShellExecuteExW, SHELLEXECUTEINFOW, SEE_MASK_NOCLOSEPROCESS};
    use windows_sys::Win32::System::Threading::{WaitForSingleObject, GetExitCodeProcess, INFINITE};
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_HIDE;
    let Ok(exe) = std::env::current_exe() else { return false };
    let exe_w: Vec<u16> = exe.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let verb: Vec<u16> = "runas".encode_utf16().chain(std::iter::once(0)).collect();
    let params: Vec<u16> = format!("--set-mpo {}", if disabled { "on" } else { "off" })
        .encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        let mut sei: SHELLEXECUTEINFOW = std::mem::zeroed();
        sei.cbSize = std::mem::size_of::<SHELLEXECUTEINFOW>() as u32;
        sei.fMask = SEE_MASK_NOCLOSEPROCESS;
        sei.lpVerb = verb.as_ptr();
        sei.lpFile = exe_w.as_ptr();
        sei.lpParameters = params.as_ptr();
        sei.nShow = SW_HIDE as i32;
        if ShellExecuteExW(&mut sei) == 0 || sei.hProcess == 0 {
            return false; // user declined UAC, or launch failed
        }
        WaitForSingleObject(sei.hProcess, INFINITE);
        let mut code: u32 = 1;
        GetExitCodeProcess(sei.hProcess, &mut code);
        CloseHandle(sei.hProcess);
        code == 0
    }
}

#[cfg(not(windows))]
fn set_mpo_elevated(_disabled: bool) -> bool { false }

/// Best-effort: tell the user (who just double-clicked the exe) that Strata is
/// already running in the tray, so a silent no-op isn't confusing.
#[cfg(windows)]
fn notify_already_running() {
    use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_OK, MB_ICONINFORMATION};
    let text: Vec<u16> = "Strata is already running - look for its icon in the system tray.".encode_utf16().chain(std::iter::once(0)).collect();
    let title: Vec<u16> = "Strata".encode_utf16().chain(std::iter::once(0)).collect();
    unsafe { MessageBoxW(0, text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONINFORMATION); }
}

/// Filesystem "added" time for a wallpaper (its dir's mtime). Used by date sorts.
fn entry_mtime(w: &controller::WallpaperEntry) -> Option<std::time::SystemTime> {
    std::fs::metadata(&w.path).and_then(|m| m.modified()).ok()
}

/// Map each bundled shader slug -> the library version it was added in (`added_in` in the
/// library index.toml). Lets date sorts order bundled shaders by WHEN they entered the
/// library (they all share a folder mtime, so the filesystem can't tell them apart).
fn library_added_versions() -> std::collections::HashMap<String, (u32, u32, u32)> {
    let mut map = std::collections::HashMap::new();
    let parse = |s: &str| -> (u32, u32, u32) {
        let mut it = s.trim().trim_start_matches(['v', 'V']).split(['.', '-', '+'])
            .filter_map(|p| p.parse::<u32>().ok());
        (it.next().unwrap_or(0), it.next().unwrap_or(0), it.next().unwrap_or(0))
    };
    if let Some(root) = controller::fetched_library_root() {
        if let Ok(text) = std::fs::read_to_string(root.join("index.toml")) {
            if let Ok(val) = toml::from_str::<toml::Value>(&text) {
                if let Some(arr) = val.get("shader").and_then(|s| s.as_array()) {
                    for s in arr {
                        if let (Some(slug), Some(ver)) = (
                            s.get("slug").and_then(|v| v.as_str()),
                            s.get("added_in").and_then(|v| v.as_str()),
                        ) { map.insert(slug.to_string(), parse(ver)); }
                    }
                }
            }
        }
    }
    map
}

/// A wallpaper's "added" sort key: bundled shaders rank by the library version they were
/// added in (from the index); user content (imports/parallax/movies, not in the index)
/// ranks as newest, ordered among themselves by folder mtime.
fn entry_added_key(w: &controller::WallpaperEntry,
    versions: &std::collections::HashMap<String, (u32, u32, u32)>) -> ((u32, u32, u32), std::time::SystemTime) {
    let slug = w.path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    match versions.get(slug) {
        Some(v) => (*v, std::time::SystemTime::UNIX_EPOCH),
        None => ((u32::MAX, u32::MAX, u32::MAX), entry_mtime(w).unwrap_or(std::time::SystemTime::UNIX_EPOCH)),
    }
}

/// Sort the wallpaper list IN PLACE per the chosen mode. Done on `state.wallpapers` itself
/// (not just the view model) so row indices stay consistent with the grid - the delete
/// handler resolves by index. Modes: "default" (bundled shaders A-Z, then user content
/// A-Z), "name-asc", "name-desc", "date-newest", "date-oldest".
fn sort_entries(entries: &mut [controller::WallpaperEntry], sort_mode: &str) {
    match sort_mode {
        "name-asc"    => entries.sort_by_cached_key(|w| w.name.to_lowercase()),
        "name-desc"   => entries.sort_by_cached_key(|w| std::cmp::Reverse(w.name.to_lowercase())),
        "date-newest" => {
            let v = library_added_versions();
            entries.sort_by_cached_key(|w| std::cmp::Reverse(entry_added_key(w, &v)));
        }
        "date-oldest" => {
            let v = library_added_versions();
            entries.sort_by_cached_key(|w| entry_added_key(w, &v));
        }
        // "default": user content (parallax / imported / movie) sinks to the bottom,
        // each group alphabetical.
        _ => entries.sort_by_cached_key(|w| (controller::is_user_deletable(&w.path), w.name.to_lowercase())),
    }
}

/// Build the Library's Slint card list from already-ordered entries (call `sort_entries`
/// on `state.wallpapers` first). Centralised so every rebuild stays consistent.
fn build_library_items(
    wallpapers: &[controller::WallpaperEntry],
    monitors: &[controller::MonitorInfo],
) -> Vec<WallpaperItem> {
    wallpapers.iter().map(|w| {
        let mut item = WallpaperItem::default();
        item.name = SharedString::from(&w.name);
        item.author = SharedString::from(&w.author);
        item.source_url = SharedString::from(&w.source_url);
        item.tags = ModelRc::from(Rc::new(VecModel::from(
            w.tags.iter().map(|t| SharedString::from(t.as_str())).collect::<Vec<_>>())));
        item.is_parallax = w.tags.iter().any(|t| t.eq_ignore_ascii_case("Parallax"));
        item.is_imported = w.tags.iter().any(|t| t.eq_ignore_ascii_case("Imported"))
            || controller::is_user_deletable(&w.path);
        item.is_video = w.tags.iter().any(|t| t.eq_ignore_ascii_case("video"));
        item.visible = true;
        if let Some(ref thumb) = w.thumbnail {
            if let Ok(slint_img) = Image::load_from_path(thumb) {
                item.thumbnail = slint_img;
                item.has_thumbnail = true;
            }
        }
        let counts: Vec<i32> = monitors.iter()
            .map(|m| m.layers.iter().filter(|l| l.wallpaper_path == w.path).count() as i32)
            .collect();
        item.is_active = counts.iter().any(|&c| c > 0);
        item.usage_counts = Rc::new(VecModel::from(counts)).into();
        item
    }).collect()
}

fn run_ui_mode(start_minimized: bool) -> Result<(), Box<dyn std::error::Error>> {
    // Enforce a single running instance - a second launch exits immediately instead
    // of spawning duplicate tray icons / wallpaper windows fighting over the desktop.
    if another_instance_running() {
        log::info!("Another Strata instance is already running - exiting this one.");
        #[cfg(windows)]
        notify_already_running();
        return Ok(());
    }

    let app_state = Arc::new(std::sync::RwLock::new(AppState::default()));
    let (command_tx, command_rx) = channel::<EngineCommand>();
    let running = Arc::new(AtomicBool::new(true));

    // Create ONE shared graphics context. Both the main thread (surface creation) and the
    // renderer thread (adapter / device / queue) must use the same wgpu::Instance - if they
    // use different instances the adapter IDs are incompatible and wgpu panics.
    let context = {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime for graphics context init");
        Arc::new(
            rt.block_on(core_engine::GraphicsContext::new())
                .expect("Failed to create wgpu graphics context"),
        )
    };

    // Load configuration
    let mut config = config::Config::load();

    // Reconcile the autostart flag with the real registry state. The installer can
    // create the Run entry ("Start with Windows" option) and the user can toggle it
    // in-app or by hand - the registry is authoritative, so sync config to it on
    // every launch (and persist) so the Settings toggle never lies.
    #[cfg(windows)]
    {
        let actual = autostart_is_enabled();
        if actual != config.autostart {
            config.autostart = actual;
            config.save().ok();
        }
    }

    // Shared, live-tunable frame-rate cap read by every monitor render loop.
    // The Settings slider stores into this; render threads pick it up next frame.
    let target_fps = Arc::new(std::sync::atomic::AtomicU32::new(config.target_fps.clamp(1, 240)));

    // Audio sensitivity: authoritative value lives in the AudioEngine (read by the
    // render threads); this Cell mirrors it for the debounced config flush.
    let audio_sensitivity = std::rc::Rc::new(std::cell::Cell::new(config.audio_sensitivity));
    if let Some(a) = &context.audio { a.set_sensitivity(config.audio_sensitivity); }

    // Mouse interactivity: shared atomics read live by every monitor render loop
    // (feeds the desktop cursor into shaders' iMouse). f32 sensitivity is bit-cast.
    // `mouse_mode`: 0=Off, 1=All, 2=Only shaders, 3=Only Parallax (see mouse_mode_to_u8).
    let mouse_mode = Arc::new(std::sync::atomic::AtomicU8::new(mouse_mode_to_u8(&config.mouse_mode)));
    let mouse_sensitivity = Arc::new(std::sync::atomic::AtomicU32::new(config.mouse_sensitivity.to_bits()));

    // Global render-quality scale (f32 bits), read live by every monitor loop.
    let quality_scale = Arc::new(std::sync::atomic::AtomicU32::new(
        shader_quality_to_scale(&config.shader_quality).to_bits(),
    ));

    {
        let mut state = app_state.write().unwrap();
        state.theme_mode = config.theme_mode.clone();
        state.span_monitors = config.span_monitors;
        state.autostart = config.autostart;
        state.monitors = discover_monitors();
        
        // Restore assignments from config, filtering out layers whose wallpaper paths no longer exist
        for m in &mut state.monitors {
            if let Some(m_config) = config.monitors.iter().find(|mc| mc.id == m.id) {
                m.layers = m_config.layers.iter()
                    .filter(|l| l.wallpaper_path.exists())
                    .cloned()
                    .collect();
                if !m_config.name.is_empty()  { m.name = m_config.name.clone(); }
                if !m_config.color.is_empty() { m.color = m_config.color.clone(); }
            }
            // Default hardware color tag (cycled) when none saved yet.
            if m.color.is_empty() {
                m.color = default_monitor_color(&m.id);
            }
        }
    }

    // Bundled UI fonts (Space Grotesk / Inter / Source Code Pro) are embedded at
    // compile time via `import "*.ttf"` statements in ui/main.slint.

    let ui = AppWindow::new()?;

    // Set up themes
    let theme_mode = {
        let state = app_state.read().unwrap();
        state.theme_mode.clone()
    };
    ui.global::<Theme>().set_mode(SharedString::from(&theme_mode));
    let is_dark = match theme_mode.as_str() {
        "dark" => true,
        "light" => false,
        _ => dark_light::detect() == dark_light::Mode::Dark,
    };
    ui.global::<Theme>().set_is_dark(is_dark);
    // Main-window HWND, captured once the winit window exists (startup spawn_local below).
    // Used to save/restore the window placement across tray hide/show. Windows only.
    #[cfg(target_os = "windows")]
    let ui_hwnd: std::rc::Rc<std::cell::Cell<isize>> = std::rc::Rc::new(std::cell::Cell::new(0));
    ui.set_autostart(config.autostart);
    // Reflect the real (registry) MPO state in the Settings toggle.
    ui.set_disable_mpo(read_mpo_disabled());
    ui.set_fps_cap(config.target_fps.clamp(1, 240) as i32);
    ui.set_audio_sensitivity(config.audio_sensitivity);
    ui.set_mouse_mode(SharedString::from(&config.mouse_mode));
    ui.set_mouse_sensitivity(config.mouse_sensitivity);
    ui.set_shader_quality(SharedString::from(&config.shader_quality));
    ui.set_update_version_badge(SharedString::from(format!("v{} (LATEST)", env!("CARGO_PKG_VERSION"))));
    // Trust the version actually on disk (index.toml) over the possibly-stale config, and
    // reconcile config to it so the updater doesn't needlessly re-download a current library.
    let lib_version = controller::installed_library_version().unwrap_or_else(|| config.library_version.clone());
    if lib_version != config.library_version {
        let mut c = config::Config::load();
        c.library_version = lib_version.clone();
        c.save().ok();
    }
    ui.set_lib_update_version_badge(SharedString::from(format!("v{} (LATEST)", lib_version)));

    // Parallax Studio: populate the depth-model dropdown (heuristic + tiers).
    ui.set_parallax_model_options(ModelRc::from(Rc::new(VecModel::from(
        parallax::model_options().iter().map(SharedString::from).collect::<Vec<_>>(),
    ))));
    ui.set_parallax_model_current(SharedString::from(
        parallax::model_options().first().map(String::as_str).unwrap_or("")));
    ui.set_parallax_seg_options(ModelRc::from(Rc::new(VecModel::from(
        parallax::seg_model_options().iter().map(SharedString::from).collect::<Vec<_>>(),
    ))));
    ui.set_parallax_seg_current(SharedString::from(&parallax::seg_model_options()[0]));
    ui.set_parallax_style_options(ModelRc::from(Rc::new(VecModel::from(
        parallax::parallax_style_options().iter().map(SharedString::from).collect::<Vec<_>>(),
    ))));
    ui.set_parallax_style_current(SharedString::from(&parallax::parallax_style_options()[0]));
    ui.set_parallax_upscaler_options(ModelRc::from(Rc::new(VecModel::from(
        parallax::upscaler_options().iter().map(SharedString::from).collect::<Vec<_>>(),
    ))));
    ui.set_parallax_upscaler_current(SharedString::from(&parallax::upscaler_options()[0]));
    // Automatic-mode quality presets (data-driven from presets.toml) + the pro-tip.
    ui.set_parallax_preset_options(ModelRc::from(Rc::new(VecModel::from(
        parallax::preset_options().iter().map(SharedString::from).collect::<Vec<_>>(),
    ))));
    ui.set_parallax_preset_current(SharedString::from(
        parallax::preset_options().first().map(String::as_str).unwrap_or("")));
    ui.set_parallax_protip(SharedString::from(&core_engine::depth::preset_meta().protip));
    // Initial "needs download?" for the default preset's models.
    refresh_parallax_download_state(&ui);

    #[cfg(target_os = "windows")]
    let (tray, tray_toggle_item) = {
        use tray_icon::{TrayIconBuilder, menu::{Menu, MenuItem}};
        let tray_menu = Menu::new();
        // The window starts visible, so the toggle reads "Hide Strata". Its text is
        // flipped as the window is hidden to / shown from the tray (id stays "show").
        let toggle_item = MenuItem::with_id("show", "Hide Strata", true, None);
        let quit_item = MenuItem::with_id("quit", "Quit", true, None);
        let _ = tray_menu.append_items(&[&toggle_item, &quit_item]);

        (Arc::new(Mutex::new(TrayIconBuilder::new()
            .with_menu(Box::new(tray_menu))
            .with_tooltip("Strata")
            .with_icon(load_tray_icon(is_dark))
            .build()
            .unwrap())),
         toggle_item)
    };
    #[cfg(target_os = "windows")]
    let tray_handle = tray.clone();
    // True while the window is hidden to the tray (not just minimized). Drives the
    // tray menu toggle direction + label.
    #[cfg(target_os = "windows")]
    let hidden_to_tray = std::rc::Rc::new(std::cell::Cell::new(false));

    // Library sort mode (persisted in config). Shared by every model rebuild + the
    // sort menu so the chosen order survives refresh / import / delete.
    let sort_mode = Rc::new(std::cell::RefCell::new(config.library_sort.clone()));
    ui.set_sort_mode(SharedString::from(config.library_sort.as_str()));

    // Set up Wallpaper Library: bundled library + user roots (parallax + imports).
    let wallpapers_model = {
        let mut entries = controller::scan_all_wallpapers();
        sort_entries(&mut entries, &sort_mode.borrow());
        {
            let mut state = app_state.write().unwrap();
            state.wallpapers = entries.clone();
        }
        // Snapshot monitors so each card shows its applied-monitor avatars on first paint.
        let monitor_snapshot = { app_state.read().unwrap().monitors.clone() };
        Rc::new(VecModel::<WallpaperItem>::from(build_library_items(&entries, &monitor_snapshot)))
    };
    ui.set_wallpapers(ModelRc::from(wallpapers_model.clone()));

    // Set up Monitors and Canvas
    let monitors_model = Rc::new(VecModel::<MonitorItem>::from(Vec::new()));
    {
        let state = app_state.read().unwrap();
        let mut min_x = 0;
        let mut min_y = 0;
        let mut max_x = 0;
        let mut max_y = 0;

        for m in &state.monitors {
            min_x = min_x.min(m.position.0);
            min_y = min_y.min(m.position.1);
            max_x = max_x.max(m.position.0 + m.resolution.0 as i32);
            max_y = max_y.max(m.position.1 + m.resolution.1 as i32);

            monitors_model.push(MonitorItem {
                id: SharedString::from(&m.id),
                name: SharedString::from(&m.name),
                color: SharedString::from(&m.color),
                width: m.resolution.0 as i32,
                height: m.resolution.1 as i32,
                x: m.position.0,
                y: m.position.1,
            });
        }
        ui.set_canvas_min_x(min_x);
        ui.set_canvas_min_y(min_y);
        ui.set_canvas_max_x(max_x);
        ui.set_canvas_max_y(max_y);
        ui.set_monitors(ModelRc::from(monitors_model.clone()));
    }

    let layers_model = Rc::new(VecModel::<LayerItem>::from(Vec::new()));
    ui.set_layers(ModelRc::from(layers_model.clone()));
    // Seed the Compositor's layer list with the default-selected monitor's layers
    // (selected-monitor-index defaults to 0), so restored assignments show up
    // immediately instead of only after the user clicks a display.
    {
        let state = app_state.read().unwrap();
        if let Some(m) = state.monitors.first() {
            let initial: Vec<LayerItem> = m.layers.iter().map(|l| LayerItem {
                name: SharedString::from(&l.name),
                opacity: l.opacity,
                resolution_scale: l.resolution_scale,
                positioning: SharedString::from(&l.positioning),
                visible: l.visible,
                blend_mode: SharedString::from(&l.blend_mode),
                is_video: controller::video_wallpaper_path(&l.wallpaper_path).is_some(),
                x: l.transform[0], y: l.transform[1], width: l.transform[2], height: l.transform[3],
            }).collect();
            layers_model.set_vec(initial);
        }
    }

    // ── Toast notifications ─────────────────────────────────────────────────
    // Bottom-right transient notices (matches the ui-prototype toaster). The
    // model holds the visible toasts; a parallel expiry list (same order) is
    // pruned by the UI timer so toasts auto-dismiss ~4s after they appear.
    // `push_toast` is shared into every handler that wants to notify the user.
    let toasts_model = Rc::new(VecModel::<ToastData>::from(Vec::new()));
    ui.set_toasts(ModelRc::from(toasts_model.clone()));
    let toast_expiry: Rc<std::cell::RefCell<Vec<(i32, std::time::Instant)>>> =
        Rc::new(std::cell::RefCell::new(Vec::new()));
    let toast_counter: Rc<std::cell::Cell<i32>> = Rc::new(std::cell::Cell::new(0));
    let push_toast: Rc<dyn Fn(&str, &str, bool)> = {
        let model = toasts_model.clone();
        let expiry = toast_expiry.clone();
        let counter = toast_counter.clone();
        Rc::new(move |title: &str, description: &str, destructive: bool| {
            let id = counter.get() + 1;
            counter.set(id);
            model.push(ToastData {
                id,
                title: SharedString::from(title),
                description: SharedString::from(description),
                destructive,
            });
            expiry.borrow_mut().push((id, std::time::Instant::now() + std::time::Duration::from_secs(4)));
            // Cap the stack so a burst of rapid actions can't fill the screen.
            while model.row_count() > 4 {
                model.remove(0);
                expiry.borrow_mut().remove(0);
            }
        })
    };

    // Close (✕) on a toast - drop it from both the model and the expiry list.
    {
        let model = toasts_model.clone();
        let expiry = toast_expiry.clone();
        ui.on_toast_dismissed(move |id| {
            if let Some(idx) = (0..model.row_count()).find(|&i| model.row_data(i).map(|t| t.id) == Some(id)) {
                model.remove(idx);
                let mut e = expiry.borrow_mut();
                if idx < e.len() { e.remove(idx); }
            }
        });
    }

    // ── Thumbnail generation plumbing ───────────────────────────────────────
    // A background thread generates missing cache thumbnails and sends each
    // (wallpaper_dir, thumb_path) here; the UI timer loads them into the library.
    // `thumbnails_busy` is mirrored to the AppWindow `refreshing` flag (spinner).
    let (thumb_tx, thumb_rx) = std::sync::mpsc::channel::<(std::path::PathBuf, std::path::PathBuf)>();
    let thumbnails_busy = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // ── Parallax Studio plumbing ────────────────────────────────────────────
    // The selected source photo, a busy guard, and a completion channel the UI
    // timer drains (Ok(wallpaper name) / Err(message)) to refresh + toast.
    let parallax_image: Rc<std::cell::RefCell<Option<std::path::PathBuf>>> = Rc::new(std::cell::RefCell::new(None));
    let parallax_busy = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (parallax_tx, parallax_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    // Depth-model selection (dropdown label) + on-demand download state.
    // Manual-mode depth model (defaults to the first real model - no "no model" option).
    let parallax_model: Rc<std::cell::RefCell<String>> = Rc::new(std::cell::RefCell::new(
        parallax::model_options().first().cloned().unwrap_or_default()));
    let parallax_downloading = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let download_pct = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let (download_tx, download_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    // Live download-queue line ("Depth Anything V2 Base · 195 MB (1/3)") updated by the
    // download thread, mirrored to the UI by the timer (paired with download_pct).
    let parallax_dl_text: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
    // Automatic-mode quality preset (display name); generation is always cinematic now.
    let parallax_preset: Rc<std::cell::RefCell<String>> = Rc::new(std::cell::RefCell::new(
        parallax::preset_options().first().cloned().unwrap_or_default()));
    // Selected masking (segmentation) model label for Manual mode.
    let parallax_seg: Rc<std::cell::RefCell<String>> =
        Rc::new(std::cell::RefCell::new(parallax::seg_model_options()[0].clone()));
    // Selected parallax style label (Coherent 3D vs Billboard) for cinematic mode.
    let parallax_style: Rc<std::cell::RefCell<String>> =
        Rc::new(std::cell::RefCell::new(parallax::parallax_style_options()[0].clone()));
    // Selected upscaler label (restores detail to the LaMa fill; "Off" by default).
    let parallax_upscaler: Rc<std::cell::RefCell<String>> =
        Rc::new(std::cell::RefCell::new(parallax::upscaler_options()[0].clone()));
    // Render progress (0..100) the build thread reports at stages; mirrored to the
    // previewer's timeline strip by the UI timer.
    let parallax_progress = Arc::new(std::sync::atomic::AtomicU32::new(0));
    // Live preview: a render thread builds the preview package and sends its dir
    // here; the UI timer stands up a headless renderer and animates it.
    let (preview_ready_tx, preview_ready_rx) = std::sync::mpsc::channel::<Result<std::path::PathBuf, String>>();
    let preview_state: Rc<std::cell::RefCell<Option<ParallaxPreviewState>>> = Rc::new(std::cell::RefCell::new(None));
    let parallax_params: Rc<std::cell::Cell<core_engine::parallax::ParallaxParams>> =
        Rc::new(std::cell::Cell::new(core_engine::parallax::ParallaxParams::default()));
    // Set when a tuning slider moves; the timer debounces it and re-bakes the preview
    // shader live (reusing cached depth/inpaint) once dragging settles.
    let parallax_tune_settle: Rc<std::cell::Cell<Option<std::time::Instant>>> =
        Rc::new(std::cell::Cell::new(None));
    // NOTE: thumbnail generation is deliberately NOT kicked off at startup - we
    // want the lightest possible launch footprint (no extra GPU context, no
    // compile burst) so Strata starts fast alongside the user's other autostart
    // apps. Missing thumbnails are generated only on an explicit Refresh or Import.

    // ── Wallpaper window lifecycle stores ───────────────────────────────────
    // Created here (before the callbacks) so layer-assignment callbacks can
    // create/tear down per-monitor windows on demand.  See the type docs.
    let wallpaper_store: WallpaperWindowStore =
        std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let pending_close: PendingCloseStore =
        std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

    // Debounced config persistence: continuous sliders (opacity, FPS cap) set this
    // flag instead of writing the file every tick; the UI timer flushes it (and the
    // quit handler flushes any pending change so nothing is lost).
    let config_dirty: std::rc::Rc<std::cell::Cell<bool>> =
        std::rc::Rc::new(std::cell::Cell::new(false));

    let app_state_search = app_state.clone();
    let wallpapers_model_search = wallpapers_model.clone();
    ui.on_search_edited(move |query| {
        let q = query.to_lowercase();
        let state = app_state_search.read().unwrap();
        for (i, w) in state.wallpapers.iter().enumerate() {
            if let Some(mut item) = wallpapers_model_search.row_data(i) {
                item.visible = w.name.to_lowercase().contains(&q) || w.tags.join(" ").to_lowercase().contains(&q);
                wallpapers_model_search.set_row_data(i, item);
            }
        }
    });

    // Holds the picked .mp4 path between the file dialog and the naming dialog's confirm.
    let pending_video_import: Rc<std::cell::RefCell<Option<std::path::PathBuf>>> =
        Rc::new(std::cell::RefCell::new(None));

    let app_state_import = app_state.clone();
    let wallpapers_model_import = wallpapers_model.clone();
    let push_toast_import = push_toast.clone();
    let thumb_tx_import = thumb_tx.clone();
    let thumbnails_busy_import = thumbnails_busy.clone();
    let ui_handle_import = ui.as_weak();
    let pending_video_set = pending_video_import.clone();
    let sort_mode_import = sort_mode.clone();
    ui.on_import_requested(move || {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Wallpaper (.mp4, .webm, .zip, .json)", &["mp4", "webm", "zip", "json"])
            .add_filter("Video (.mp4, .webm)", &["mp4", "webm"])
            .add_filter("Shadertoy export (.json)", &["json"])
            .add_filter("Wallpaper / Shadertoy ZIP (.zip)", &["zip"])
            .pick_file()
        {
            let is_video = path.extension().and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("mp4") || e.eq_ignore_ascii_case("webm"))
                .unwrap_or(false);

            // Movie wallpaper: don't import yet - open the naming dialog first (default =
            // file name), then finish in on_video_import_confirmed. Mirrors parallax save.
            if is_video {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("video").to_string();
                *pending_video_set.borrow_mut() = Some(path.clone());
                if let Some(ui) = ui_handle_import.upgrade() {
                    ui.set_video_name_default(SharedString::from(stem));
                    ui.set_show_video_name_dialog(true);
                }
                return;
            }

            let import_result = {
                // Imported shaders/packs go to %APPDATA%/Strata/import (user data).
                let dest_base = match controller::import_library_dir() {
                    Some(d) => d,
                    None => {
                        push_toast_import("Import Failed", "Could not resolve the user data directory.", true);
                        return;
                    }
                };
                std::fs::create_dir_all(&dest_base).ok();

                // Route the import: a `.json` (or a `.zip` that isn't a native Strata
                // pack) is a Shadertoy export and is converted to our manifest+glsl
                // format; a `.zip` that contains manifest.toml is a native pack.
                let is_json = path.extension().and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("json")).unwrap_or(false);
                let is_shadertoy = is_json || !controller::zip_is_native_pack(&path);
                if is_shadertoy {
                    controller::import_shadertoy(&path, &dest_base)
                        .map(|(p, warnings)| {
                            for w in &warnings { log::warn!("Shadertoy import note: {}", w); }
                            p
                        })
                } else {
                    import_wallpaper_zip(&path, &dest_base)
                }
            };

            match import_result {
                Ok(_) => {
                    let mut state = app_state_import.write().unwrap();
                    let mut new_wallpapers = controller::scan_all_wallpapers();
                    sort_entries(&mut new_wallpapers, &sort_mode_import.borrow());
                    state.wallpapers = new_wallpapers.clone();
                    let monitors = state.monitors.clone();

                    let slint_walls = build_library_items(&new_wallpapers, &monitors);
                    let dirs: Vec<std::path::PathBuf> = new_wallpapers.iter().map(|w| w.path.clone()).collect();
                    drop(state);
                    wallpapers_model_import.set_vec(slint_walls);
                    log::info!("Imported wallpaper from {:?}", path.file_name().unwrap());
                    push_toast_import("Wallpaper Imported", "Added to your library.", false);
                    // Generate thumbnails for the (new) wallpapers in the background.
                    spawn_thumbnail_generation(dirs, thumb_tx_import.clone(), thumbnails_busy_import.clone());
                }
                Err(e) => {
                    log::error!("Import failed: {}", e);
                    push_toast_import("Import Failed", "Shader couldn't be imported - see logs for details.", true);
                }
            }
        }
    });

    // Finish a movie-wallpaper import once the user has named it in the dialog.
    let push_toast_vid = push_toast.clone();
    let ui_handle_vid = ui.as_weak();
    let pending_video_take = pending_video_import.clone();
    ui.on_video_import_confirmed(move |name| {
        let Some(path) = pending_video_take.borrow_mut().take() else { return; };
        match import_video_wallpaper(&path, name.as_str()) {
            Ok(_) => {
                if let Some(ui) = ui_handle_vid.upgrade() {
                    ui.invoke_refresh_library(); // rescan: adds the new card + its thumbnail
                }
                push_toast_vid("Wallpaper Imported", "Your movie wallpaper was added to the library.", false);
            }
            Err(e) => {
                log::error!("Video import failed: {}", e);
                push_toast_vid("Import Failed", &e, true);
            }
        }
    });

    // Open a shader's original source page (attribution link) in the default
    // browser. Restricted to http(s) URLs so manifest data can't launch a file
    // or command via the shell handler.
    ui.on_open_url(move |url| {
        let u = url.trim().to_string();
        if !(u.starts_with("https://") || u.starts_with("http://")) {
            log::warn!("Refusing to open non-http URL: {:?}", u);
            return;
        }
        if let Err(e) = open::that(&u) {
            log::warn!("Failed to open URL {}: {}", u, e);
        }
    });

    // Check for updates: query the GitHub "latest release" of the Strata repo on a
    // background thread. If a newer version exists the button turns into "DOWNLOAD
    // UPDATE" and opens the releases page; otherwise it reports up-to-date. The repo
    // having no releases yet (404) is treated as "up to date".
    {
        let ui_upd = ui.as_weak();
        ui.on_check_for_updates(move || {
            let Some(ui) = ui_upd.upgrade() else { return };
            // Second click while an update is available → open the releases page.
            if ui.get_update_available() {
                let url = ui.get_update_url().to_string();
                if url.starts_with("https://") || url.starts_with("http://") {
                    if let Err(e) = open::that(&url) { log::warn!("Failed to open releases page: {}", e); }
                }
                return;
            }
            if ui.get_update_checking() { return; }
            ui.set_update_checking(true);
            ui.set_update_button_label("CHECKING…".into());
            let weak = ui.as_weak();
            std::thread::spawn(move || {
                let result = check_github_latest();
                let _ = slint::invoke_from_event_loop(move || {
                    let Some(ui) = weak.upgrade() else { return };
                    ui.set_update_checking(false);
                    match result {
                        Ok(Some((tag, url))) => {
                            ui.set_update_url(url.into());
                            ui.set_update_available(true);
                            ui.set_update_version_badge(slint::format!("{} AVAILABLE", tag));
                            ui.set_update_button_label("DOWNLOAD UPDATE".into());
                        }
                        Ok(None) => {
                            ui.set_update_version_badge(slint::format!("v{} (LATEST)", env!("CARGO_PKG_VERSION")));
                            ui.set_update_button_label("CHECK FOR UPDATES".into());
                        }
                        Err(e) => {
                            log::warn!("Update check failed: {}", e);
                            ui.set_update_version_badge("CHECK FAILED".into());
                            ui.set_update_button_label("CHECK FOR UPDATES".into());
                        }
                    }
                });
            });
        });
    }

    // Asset-Library (Strata-Library) button: discover the latest `library-v*` tag and,
    // if it's newer than what's installed (or nothing is installed yet), download +
    // install it from the repo's zipball into %APPDATA%/strata, then refresh the grid.
    {
        let ui_lib = ui.as_weak();
        ui.on_check_for_library_updates(move || {
            let Some(ui) = ui_lib.upgrade() else { return };
            if ui.get_lib_update_checking() { return; }
            ui.set_lib_update_checking(true);
            ui.set_lib_update_button_label("WORKING…".into());
            // Compare against what's actually on disk (index.toml), not the config value -
            // a stale config would make this re-download a library that's already current.
            let installed = controller::library_installed();
            let current = controller::installed_library_version()
                .unwrap_or_else(|| config::Config::load().library_version);
            let weak = ui.as_weak();
            std::thread::spawn(move || {
                // -> Ok(Some(version)) = synced to that version; Ok(None) = already current.
                let result: Result<Option<String>, String> = (|| {
                    let (owner, repo, version, tag) = library_sync::latest_library()?;
                    if !installed || library_sync::is_newer(&version, &current) {
                        library_sync::sync_library(&owner, &repo, &tag)?;
                        Ok(Some(version))
                    } else {
                        Ok(None)
                    }
                })();
                let _ = slint::invoke_from_event_loop(move || {
                    let Some(ui) = weak.upgrade() else { return };
                    ui.set_lib_update_checking(false);
                    ui.set_lib_update_available(false);
                    ui.set_lib_update_button_label("CHECK FOR UPDATES".into());
                    match result {
                        Ok(Some(version)) => {
                            let mut c = config::Config::load();
                            c.library_version = version.clone();
                            c.save().ok();
                            ui.set_lib_update_version_badge(slint::format!("v{} (LATEST)", version));
                            ui.invoke_refresh_library(); // repopulate the grid from the fetched library
                        }
                        Ok(None) => {
                            ui.set_lib_update_version_badge(slint::format!("v{} (LATEST)", current));
                        }
                        Err(e) => {
                            log::warn!("Library sync failed: {}", e);
                            ui.set_lib_update_version_badge("SYNC FAILED".into());
                        }
                    }
                });
            });
        });
    }

    // Rescan the wallpaper folder and rebuild the library model (preserving each
    // card's applied-monitor state). Triggered by the Library toolbar refresh.
    let app_state_refresh = app_state.clone();
    let wallpapers_model_refresh = wallpapers_model.clone();
    let push_toast_refresh = push_toast.clone();
    let thumb_tx_refresh = thumb_tx.clone();
    let thumbnails_busy_refresh = thumbnails_busy.clone();
    let sort_mode_refresh = sort_mode.clone();
    ui.on_refresh_library(move || {
        // Already refreshing? Don't rescan or kick off a second generator - just
        // tell the user. This stops rapid clicks from pinning the CPU.
        if thumbnails_busy_refresh.load(std::sync::atomic::Ordering::SeqCst) {
            push_toast_refresh("Refresh In Progress", "Thumbnails are still generating - please wait.", false);
            return;
        }
        let mut state = app_state_refresh.write().unwrap();
        let mut scanned = controller::scan_all_wallpapers();
        sort_entries(&mut scanned, &sort_mode_refresh.borrow());
        // Only rebuild the Slint model when the shader SET actually changed (added/
        // removed/renamed or a thumbnail appeared/disappeared). Re-decoding every
        // thumbnail PNG into a fresh GPU texture on each refresh is what made RAM
        // creep up; freshly-generated thumbnails already stream into individual rows
        // via the timer's thumb_rx drain, so an unchanged library needs no rebuild.
        let set_changed = state.wallpapers.len() != scanned.len()
            || state.wallpapers.iter().zip(&scanned)
                .any(|(a, b)| a.path != b.path || a.thumbnail != b.thumbnail);
        let count = scanned.len();
        let dirs: Vec<std::path::PathBuf> = scanned.iter().map(|w| w.path.clone()).collect();

        if set_changed {
            state.wallpapers = scanned.clone();
            let monitors = state.monitors.clone();
            let walls = build_library_items(&scanned, &monitors);
            drop(state);
            wallpapers_model_refresh.set_vec(walls);
            log::info!("Library refreshed: {} wallpaper(s)", count);
            push_toast_refresh("Library Updated", &format!("Shader manifests synchronized - {} found.", count), false);
        } else {
            drop(state);
            push_toast_refresh("Library Up To Date", &format!("{} shaders - no changes found.", count), false);
        }
        // Generate any missing thumbnails in the background (spinner shows progress).
        // No-op if every shader already has one.
        spawn_thumbnail_generation(dirs, thumb_tx_refresh.clone(), thumbnails_busy_refresh.clone());
    });

    // ── Library sort: reorder state.wallpapers + rebuild the grid, persist choice ──
    let ui_sort = ui.as_weak();
    let app_state_sort = app_state.clone();
    let wallpapers_model_sort = wallpapers_model.clone();
    let sort_mode_sort = sort_mode.clone();
    ui.on_sort_changed(move |mode| {
        *sort_mode_sort.borrow_mut() = mode.to_string();
        let mut cfg = config::Config::load();
        cfg.library_sort = mode.to_string();
        cfg.save().ok();
        let items = {
            let mut state = app_state_sort.write().unwrap();
            sort_entries(&mut state.wallpapers, &mode);
            let monitors = state.monitors.clone();
            build_library_items(&state.wallpapers, &monitors)
        };
        wallpapers_model_sort.set_vec(items);
        if let Some(ui) = ui_sort.upgrade() { ui.set_sort_mode(mode); }
    });

    // ── Delete a wallpaper (parallax assets get a trash button) ─────────────
    {
        let ui_del = ui.as_weak();
        let app_state_del = app_state.clone();
        let push_toast_del = push_toast.clone();
        let command_tx_del = command_tx.clone();
        let context_del = context.clone();
        let store_del = wallpaper_store.clone();
        let pending_close_del = pending_close.clone();
        let layers_model_del = layers_model.clone();
        ui.on_wallpaper_delete_requested(move |index| {
            let Some(ui) = ui_del.upgrade() else { return };
            // Resolve the wallpaper directory from app state by row index.
            let path = app_state_del.read().ok()
                .and_then(|s| s.wallpapers.get(index as usize).map(|w| w.path.clone()));
            let Some(path) = path else { return };
            // Safety: only delete user-generated content (parallax creations or
            // imported shaders), never the bundled library or anything elsewhere.
            if !controller::is_user_deletable(&path) {
                log::warn!("Refusing to delete {:?} - not a user-generated wallpaper", path);
                return;
            }

            // First, drop any active layers that use this wallpaper from every
            // monitor + persist, so a deleted wallpaper can't stay on the desktop.
            let mut removed_layers = false;
            {
                let mut state = app_state_del.write().unwrap();
                for m in &mut state.monitors {
                    let before = m.layers.len();
                    m.layers.retain(|l| l.wallpaper_path != path);
                    removed_layers |= m.layers.len() != before;
                }
                if removed_layers {
                    let mut cfg = config::Config::load();
                    cfg.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                    cfg.save().ok();
                }
            }

            if let Err(e) = std::fs::remove_dir_all(&path) {
                log::error!("Delete failed for {:?}: {}", path, e);
                push_toast_del("Delete Failed", "Could not remove the wallpaper folder.", true);
                return;
            }
            log::info!("Deleted wallpaper {:?}", path);

            // If it was applied, reconcile the desktop (close/rebuild windows) and
            // refresh the Compositor's layer panel for the selected monitor.
            if removed_layers {
                sync_wallpaper_windows(
                    app_state_del.clone(), command_tx_del.clone(), context_del.clone(),
                    store_del.clone(), pending_close_del.clone(),
                );
                let sel = ui.get_selected_monitor_index();
                if sel >= 0 {
                    if let Ok(state) = app_state_del.read() {
                        if let Some(m) = state.monitors.get(sel as usize) {
                            let slint_layers: Vec<LayerItem> = m.layers.iter().map(|l| LayerItem {
                                name: SharedString::from(&l.name),
                                opacity: l.opacity,
                                resolution_scale: l.resolution_scale,
                                positioning: SharedString::from(&l.positioning),
                                visible: l.visible,
                                blend_mode: SharedString::from(&l.blend_mode),
                        is_video: controller::video_wallpaper_path(&l.wallpaper_path).is_some(),
                                x: l.transform[0], y: l.transform[1], width: l.transform[2], height: l.transform[3],
                            }).collect();
                            layers_model_del.set_vec(slint_layers);
                        }
                    }
                }
            }

            push_toast_del("Wallpaper Deleted", "The wallpaper was removed.", false);
            ui.invoke_refresh_library(); // rescan to drop it from the grid
        });
    }

    // ── Parallax Studio: pick a source photo ───────────────────────────────
    {
        let ui_pick = ui.as_weak();
        let parallax_image_pick = parallax_image.clone();
        let preview_state_pick = preview_state.clone();
        ui.on_parallax_pick_image(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp"])
                .pick_file()
            {
                let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                *parallax_image_pick.borrow_mut() = Some(path);
                // New photo → drop any existing preview (Save disabled until re-render).
                *preview_state_pick.borrow_mut() = None;
                if let Some(ui) = ui_pick.upgrade() {
                    ui.set_parallax_image_name(SharedString::from(&name));
                    ui.set_parallax_status(SharedString::from("Click Render to preview."));
                    ui.set_parallax_has_preview(false);
                    ui.set_parallax_preview_image(Image::default());
                }
            }
        });
    }

    // ── Parallax Studio: Automatic-mode quality preset ──────────────────────
    {
        let ui_ps = ui.as_weak();
        let preset_sel = parallax_preset.clone();
        ui.on_parallax_preset_changed(move |label| {
            *preset_sel.borrow_mut() = label.to_string();
            if let Some(ui) = ui_ps.upgrade() {
                ui.set_parallax_preset_current(label.clone());
                refresh_parallax_download_state(&ui);
            }
        });
    }

    // ── Parallax Studio: Manual-mode depth model ────────────────────────────
    {
        let ui_mc = ui.as_weak();
        let model_sel = parallax_model.clone();
        ui.on_parallax_model_changed(move |label| {
            *model_sel.borrow_mut() = label.to_string();
            if let Some(ui) = ui_mc.upgrade() {
                ui.set_parallax_model_current(label.clone());
                refresh_parallax_download_state(&ui);
            }
        });
    }

    // ── Parallax Studio: Manual-mode masking (segmentation) model ────────────
    {
        let ui_seg = ui.as_weak();
        let seg_sel = parallax_seg.clone();
        ui.on_parallax_seg_changed(move |label| {
            *seg_sel.borrow_mut() = label.to_string();
            if let Some(ui) = ui_seg.upgrade() {
                ui.set_parallax_seg_current(label.clone());
                refresh_parallax_download_state(&ui);
            }
        });
    }

    // ── Parallax Studio: parallax style (Coherent 3D vs Billboard; both modes) ─
    {
        let ui_style = ui.as_weak();
        let style_sel = parallax_style.clone();
        ui.on_parallax_style_changed(move |label| {
            *style_sel.borrow_mut() = label.to_string();
            if let Some(ui) = ui_style.upgrade() {
                ui.set_parallax_style_current(label.clone());
            }
        });
    }

    // ── Parallax Studio: Manual-mode inpaint upscaler ───────────────────────
    {
        let ui_up = ui.as_weak();
        let up_sel = parallax_upscaler.clone();
        ui.on_parallax_upscaler_changed(move |label| {
            *up_sel.borrow_mut() = label.to_string();
            if let Some(ui) = ui_up.upgrade() {
                ui.set_parallax_upscaler_current(label.clone());
                refresh_parallax_download_state(&ui);
            }
        });
    }

    // ── Parallax Studio: download ALL models the current selection needs ─────
    // One queued download (depth + matte + LaMa + upscaler) with a "(i/N)" + name +
    // size + live %, reported via parallax_dl_text + download_pct → the UI timer.
    {
        let ui_dl = ui.as_weak();
        let downloading = parallax_downloading.clone();
        let pct = download_pct.clone();
        let dtx = download_tx.clone();
        let dl_text = parallax_dl_text.clone();
        ui.on_parallax_download_models(move || {
            let Some(ui) = ui_dl.upgrade() else { return };
            if !parallax::onnx_available() { return; }
            let missing = parallax::missing_models(&parallax_required_models(&ui));
            if missing.is_empty() { ui.set_parallax_models_need_download(false); return; }
            if downloading.swap(true, std::sync::atomic::Ordering::SeqCst) { return; }
            ui.set_parallax_downloading(true);
            pct.store(0, std::sync::atomic::Ordering::SeqCst);
            if let Ok(mut t) = dl_text.lock() { *t = "Starting download…".to_string(); }
            let pct_t = pct.clone();
            let dtx_t = dtx.clone();
            let downloading_t = downloading.clone();
            let dl_text_t = dl_text.clone();
            std::thread::Builder::new().name("strata-models-dl".into()).spawn(move || {
                let res = parallax::download_models_queue(&missing, |i, total, name, size_mb, p| {
                    if let Ok(mut t) = dl_text_t.lock() {
                        *t = format!("{} · {} MB  ({}/{})", name, size_mb, i, total);
                    }
                    pct_t.store(p as u32, std::sync::atomic::Ordering::SeqCst);
                }).map(|_| "models".to_string());
                downloading_t.store(false, std::sync::atomic::Ordering::SeqCst);
                let _ = dtx_t.send(res);
            }).ok();
        });
    }

    // ── Parallax Studio: Manual tuning sliders (height / zoom / steps) ───────
    // Update the baked params + arm a debounced live re-bake (timer reuses the cached
    // depth/inpaint, so tuning is instant - no model inference).
    {
        let ui_pc = ui.as_weak();
        let params_pc = parallax_params.clone();
        let tune_pc = parallax_tune_settle.clone();
        ui.on_parallax_param_changed(move |name, value| {
            let mut p = params_pc.get();
            match name.as_str() {
                "height" => p.height = value,
                "zoom"   => p.zoom = value,
                "steps"  => p.steps = value.round().max(1.0) as u32,
                _ => {}
            }
            params_pc.set(p);
            if let Some(ui) = ui_pc.upgrade() {
                // Mirror back (e.g. steps rounded to an integer) so the value label is exact.
                ui.set_parallax_param_height(p.height);
                ui.set_parallax_param_zoom(p.zoom);
                ui.set_parallax_param_steps(p.steps as f32);
            }
            tune_pc.set(Some(std::time::Instant::now()));
        });
    }

    // ── Parallax Studio: RENDER → estimate depth + build the live preview ────
    {
        let ui_render = ui.as_weak();
        let parallax_image_render = parallax_image.clone();
        let parallax_busy_render = parallax_busy.clone();
        let preview_ready_render = preview_ready_tx.clone();
        let parallax_model_render = parallax_model.clone();
        let parallax_params_render = parallax_params.clone();
        let parallax_preset_render = parallax_preset.clone();
        let parallax_progress_render = parallax_progress.clone();
        let parallax_seg_render = parallax_seg.clone();
        let parallax_style_render = parallax_style.clone();
        let parallax_upscaler_render = parallax_upscaler.clone();
        ui.on_parallax_create(move || {
            let Some(ui) = ui_render.upgrade() else { return };
            let Some(path) = parallax_image_render.borrow().clone() else {
                ui.set_parallax_status(SharedString::from("Select an image first."));
                return;
            };
            if parallax_busy_render.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return; // already working
            }
            ui.set_parallax_busy(true);
            ui.set_parallax_status(SharedString::from("Generating depth, inpainting background…"));
            parallax_progress_render.store(0, std::sync::atomic::Ordering::SeqCst);
            ui.set_parallax_progress(0);

            // Automatic mode → resolve the chosen preset to its models and use the FIXED
            // default tuning params, so it's foolproof and identical no matter what the
            // Manual sliders are set to. Manual mode → the dropdown selections + the tuned
            // params. Generation is always cinematic.
            let (model, seg, upscaler, billboard, params) = if ui.get_parallax_mode() == "automatic" {
                let p = parallax::preset_for_label(&parallax_preset_render.borrow());
                (
                    core_engine::depth::model_by_id(&p.depth).cloned(),
                    core_engine::depth::model_by_id(&p.segment).cloned()
                        .unwrap_or_else(core_engine::depth::u2net_model),
                    core_engine::depth::preset_upscaler(p).cloned(),
                    parallax::style_is_billboard(&parallax_style_render.borrow()),
                    core_engine::parallax::ParallaxParams::default(),
                )
            } else {
                (
                    parallax::tier_for_label(&parallax_model_render.borrow()),
                    parallax::seg_choice_for_label(&parallax_seg_render.borrow()),
                    parallax::upscaler_choice_for_label(&parallax_upscaler_render.borrow()),
                    parallax::style_is_billboard(&parallax_style_render.borrow()),
                    parallax_params_render.get(),
                )
            };
            let tx = preview_ready_render.clone();
            let busy = parallax_busy_render.clone();
            let prog = parallax_progress_render.clone();
            std::thread::Builder::new().name("strata-parallax-preview".into()).spawn(move || {
                let res = parallax::build_preview(&path, &params, model.as_ref(), &seg, true, billboard, upscaler.as_ref(), |p| {
                    prog.store(p, std::sync::atomic::Ordering::SeqCst);
                });
                busy.store(false, std::sync::atomic::Ordering::SeqCst);
                let _ = tx.send(res);
            }).ok();
        });
    }

    // ── Parallax Studio: SAVE TO LIBRARY → promote the preview package ───────
    {
        let ui_save = ui.as_weak();
        let parallax_image_save = parallax_image.clone();
        let parallax_busy_save = parallax_busy.clone();
        let parallax_tx_save = parallax_tx.clone();
        let parallax_params_save = parallax_params.clone();
        ui.on_parallax_save(move |custom_name| {
            let Some(ui) = ui_save.upgrade() else { return };
            let Some(preview) = parallax::preview_dir().filter(|d| d.join("depth.png").exists()) else {
                ui.set_parallax_status(SharedString::from("Render a preview first."));
                return;
            };
            if parallax_busy_save.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            ui.set_parallax_busy(true);
            ui.set_parallax_status(SharedString::from("Saving to Library…"));
            // Use the user's custom name if they typed one; otherwise fall back to
            // the source image's file name.
            let typed = custom_name.trim().to_string();
            let name = if !typed.is_empty() {
                typed
            } else {
                parallax_image_save.borrow().as_ref()
                    .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "Parallax".into())
            };
            let params = parallax_params_save.get();
            let tx = parallax_tx_save.clone();
            let busy = parallax_busy_save.clone();
            std::thread::Builder::new().name("strata-parallax-save".into()).spawn(move || {
                let res = parallax::save_to_library(&preview, &name, &params)
                    .map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
                busy.store(false, std::sync::atomic::Ordering::SeqCst);
                let _ = tx.send(res);
            }).ok();
        });
    }

    // Callbacks

    let app_state_theme = app_state.clone();
    let ui_handle_theme = ui.as_weak();
    ui.on_theme_changed(move |mode| {
        let mut state = app_state_theme.write().unwrap();
        state.theme_mode = mode.to_string();

        let is_dark = match state.theme_mode.as_str() {
            "dark" => true,
            "light" => false,
            _ => dark_light::detect() == dark_light::Mode::Dark,
        };

        if let Some(ui) = ui_handle_theme.upgrade() {
            ui.global::<Theme>().set_is_dark(is_dark);
        }

        #[cfg(target_os = "windows")]
        {
            let icon = load_tray_icon(is_dark);
            let _ = tray_handle.lock().unwrap().set_icon(Some(icon));
        }

        let mut config = config::Config::load();
        config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
        config.save().ok();
    });

    let ui_handle_mon = ui.as_weak();
    let app_state_mon = app_state.clone();
    let layers_model_mon = layers_model.clone();
    ui.on_monitor_selected(move |idx| {
        if let Some(ui) = ui_handle_mon.upgrade() {
            ui.set_selected_monitor_index(idx);
            let state = app_state_mon.read().unwrap();
            if let Some(monitor) = state.monitors.get(idx as usize) {
                let slint_layers: Vec<LayerItem> = monitor.layers.iter().map(|l| {
                    LayerItem {
                        name: SharedString::from(&l.name),
                        opacity: l.opacity,
                        resolution_scale: l.resolution_scale,
                        positioning: SharedString::from(&l.positioning),
                        visible: l.visible,
                        blend_mode: SharedString::from(&l.blend_mode),
                        is_video: controller::video_wallpaper_path(&l.wallpaper_path).is_some(),
                        x: l.transform[0],
                        y: l.transform[1],
                        width: l.transform[2],
                        height: l.transform[3],
                    }
                }).collect();
                layers_model_mon.set_vec(slint_layers);
            }
        }
    });

    // Movie/shader exclusivity: when applying would destroy the other type on a display,
    // we stash the request, show a confirm dialog, and only proceed once the user clicks
    // Apply (which re-invokes on_apply_to_monitor with `force_conflict_apply` set).
    let force_conflict_apply = Rc::new(std::cell::Cell::new(false));
    let pending_conflict = Rc::new(std::cell::RefCell::new(None::<(i32, String)>));

    let app_state_apply = app_state.clone();
    let command_tx_apply = command_tx.clone();
    let layers_model_apply = layers_model.clone();
    let wallpapers_model_apply = wallpapers_model.clone();
    let context_apply = context.clone();
    let store_apply = wallpaper_store.clone();
    let pending_close_apply = pending_close.clone();
    let push_toast_apply = push_toast.clone();
    let ui_handle_apply = ui.as_weak();
    let force_apply_a = force_conflict_apply.clone();
    let pending_conflict_a = pending_conflict.clone();
    ui.on_apply_to_monitor(move |mon_idx, wall_name| {
        let forced = force_apply_a.replace(false);
        // ── Conflict detection (read-only) ──────────────────────────────────────
        // A display holds EITHER shaders OR one movie. Detect whether this apply would
        // remove the other type and, if so, confirm first (unless already confirmed).
        if !forced {
            let conflict: Option<String> = {
                let state = app_state_apply.read().unwrap();
                let wp = state.wallpapers.iter().find(|w| wall_name == w.name).map(|w| w.path.clone());
                match (wp, state.monitors.get(mon_idx as usize)) {
                    (Some(path), Some(monitor)) => {
                        let is_video = |p: &std::path::Path| controller::video_wallpaper_path(p).is_some();
                        let applying_movie = is_video(&path);
                        let already = monitor.layers.iter().any(|l| l.wallpaper_path == path);
                        let has_movie = monitor.layers.iter().any(|l| is_video(&l.wallpaper_path));
                        let has_shader = monitor.layers.iter().any(|l| !is_video(&l.wallpaper_path));
                        let mon_name = monitor.name.clone();
                        let span = state.span_monitors;
                        let primary_has_shader = controller::primary_monitor(&state.monitors)
                            .map(|m| m.layers.iter().any(|l| !is_video(&l.wallpaper_path))).unwrap_or(false);
                        if already {
                            None // toggling off - never a conflict
                        } else if applying_movie && span && primary_has_shader {
                            Some("Compositing shaders with movie-based wallpapers is not supported. Applying this movie wallpaper will remove the shader layers spanning your displays.".to_string())
                        } else if applying_movie && has_shader {
                            Some(format!("Compositing shaders with movie-based wallpapers is not supported. Applying this movie wallpaper will remove the current shader layers on {mon_name}."))
                        } else if !applying_movie && has_movie {
                            Some(format!("Compositing shaders with movie-based wallpapers is not supported. Applying this shader will remove the current movie wallpaper on {mon_name}."))
                        } else {
                            None // empty display, or movie-replaces-movie (silent)
                        }
                    }
                    _ => None,
                }
            };
            if let Some(msg) = conflict {
                if let Some(ui) = ui_handle_apply.upgrade() {
                    *pending_conflict_a.borrow_mut() = Some((mon_idx, wall_name.to_string()));
                    ui.set_conflict_message(SharedString::from(msg));
                    ui.set_show_conflict_dialog(true);
                }
                return;
            }
        }
        // Applying a shader makes that monitor the Compositor's active selection,
        // so switching tabs shows it selected (and its layers) without an extra
        // click - keeps the Library and Compositor in sync.
        if let Some(ui) = ui_handle_apply.upgrade() {
            ui.set_selected_monitor_index(mon_idx);
        }
        // `changed` flips true only if we actually mutated the assignment, so we
        // know whether to reconcile windows (which could add/remove a monitor's
        // first/last shader).
        let mut changed = false;
        {
        let mut state = app_state_apply.write().unwrap();
        let wallpaper_info = state.wallpapers.iter().find(|w| wall_name == w.name).map(|w| (w.path.clone(), w.name.clone()));

        if let Some((path, name)) = wallpaper_info {
            let applying_movie = controller::video_wallpaper_path(&path).is_some();
            let is_video = |p: &std::path::Path| controller::video_wallpaper_path(p).is_some();
            // Movies never span and are exclusive: turn off span and clear the spanning
            // shaders (the primary's layers) before placing a movie on any display.
            let already_here = state.monitors.get(mon_idx as usize)
                .map(|m| m.layers.iter().any(|l| l.wallpaper_path == path)).unwrap_or(false);
            if applying_movie && !already_here && state.span_monitors {
                state.span_monitors = false;
                let primary_idx = state.monitors.iter().position(|m| m.is_primary)
                    .unwrap_or(0);
                if let Some(pm) = state.monitors.get_mut(primary_idx) { pm.layers.clear(); }
            }
            let span_now = state.span_monitors;

            if let Some(monitor) = state.monitors.get_mut(mon_idx as usize) {
                let mon_name = monitor.name.clone();
                let shader_name = name.clone();
                let added;
                // If it's already there, toggle it (remove it)
                if let Some(pos) = monitor.layers.iter().position(|l| l.wallpaper_path == path) {
                    monitor.layers.remove(pos);
                    added = false;
                } else {
                    added = true;
                    // Enforce exclusivity on THIS display: a movie clears everything
                    // (shaders + any old movie = silent replace); a shader clears only a
                    // movie. Same type stacks as layers (shaders) as before.
                    if applying_movie {
                        monitor.layers.clear();
                    } else {
                        monitor.layers.retain(|l| !is_video(&l.wallpaper_path));
                    }
                    // Insert at the TOP of the list (index 0) so the shader the
                    // user just activated composites on top and is immediately
                    // visible - they can move it down later to reorder.
                    monitor.layers.insert(0, LayerInfo {
                        wallpaper_path: path,
                        name: name,
                        opacity: 1.0,
                        resolution_scale: 1.0,
                        positioning: "Fill".to_string(),
                        transform: [0.0, 0.0, 1.0, 1.0],
                        visible: true,
                        blend_mode: "normal".to_string(),
                    });
                }

                let slint_layers: Vec<LayerItem> = monitor.layers.iter().map(|l| {
                    LayerItem {
                        name: SharedString::from(&l.name),
                        opacity: l.opacity,
                        resolution_scale: l.resolution_scale,
                        positioning: SharedString::from(&l.positioning),
                        visible: l.visible,
                        blend_mode: SharedString::from(&l.blend_mode),
                        is_video: controller::video_wallpaper_path(&l.wallpaper_path).is_some(),
                        x: l.transform[0],
                        y: l.transform[1],
                        width: l.transform[2],
                        height: l.transform[3],
                    }
                }).collect();
                layers_model_apply.set_vec(slint_layers);

                // Notify the user that the engine picked up the change.
                if added {
                    let desc = if span_now {
                        "Shader deployed to all monitors (Spanned mode).".to_string()
                    } else {
                        format!("{} applied to {}.", shader_name, mon_name)
                    };
                    push_toast_apply("Engine Synchronizing", &desc, false);
                } else {
                    push_toast_apply("Shader Removed", &format!("{} removed from {}.", shader_name, mon_name), false);
                }

                // Update usage counts in the wallpaper model
                let current_monitors = state.monitors.clone();
                for (i, w) in state.wallpapers.iter().enumerate() {
                    let mut item = wallpapers_model_apply.row_data(i).unwrap();
                    let mut counts = Vec::new();
                    for m in &current_monitors {
                        counts.push(m.layers.iter().filter(|l| l.wallpaper_path == w.path).count() as i32);
                    }
                    item.is_active = counts.iter().any(|&c| c > 0);
                    item.usage_counts = Rc::new(VecModel::from(counts)).into();
                    wallpapers_model_apply.set_row_data(i, item);
                }
                
                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();

                changed = true;
            }
        }
        } // drop the app_state write guard before reconciling windows

        // Assigning the first shader to a monitor (or removing its last one)
        // can create or destroy that monitor's window - reconcile rather than a
        // plain Reload.  sync_wallpaper_windows re-reads app_state, so the guard
        // above MUST be released first.
        if changed {
            // Applying a movie may have turned span off - reflect that in the UI toggle.
            if let Some(ui) = ui_handle_apply.upgrade() {
                let span = app_state_apply.read().map(|s| s.span_monitors).unwrap_or(false);
                ui.set_span_monitors(span);
            }
            sync_wallpaper_windows(
                app_state_apply.clone(),
                command_tx_apply.clone(),
                context_apply.clone(),
                store_apply.clone(),
                pending_close_apply.clone(),
            );
        }
    });

    // Confirm a movie/shader exclusivity swap: re-run the stashed apply with the force
    // flag set so it bypasses the conflict check and performs the destructive replace.
    let ui_handle_conf = ui.as_weak();
    let force_apply_c = force_conflict_apply.clone();
    let pending_conflict_c = pending_conflict.clone();
    ui.on_conflict_confirmed(move || {
        if let Some((mon_idx, wall_name)) = pending_conflict_c.borrow_mut().take() {
            force_apply_c.set(true);
            if let Some(ui) = ui_handle_conf.upgrade() {
                ui.invoke_apply_to_monitor(mon_idx, SharedString::from(wall_name));
            }
        }
    });

    let app_state_rem = app_state.clone();
    let layers_model_rem = layers_model.clone();
    let wallpapers_model_rem = wallpapers_model.clone();
    let ui_handle_rem = ui.as_weak();
    let command_tx_rem = command_tx.clone();
    let context_rem = context.clone();
    let store_rem = wallpaper_store.clone();
    let pending_close_rem = pending_close.clone();
    ui.on_layer_remove_requested(move |layer_idx| {
        let ui = ui_handle_rem.unwrap();
        let mon_idx = ui.get_selected_monitor_index();
        if mon_idx < 0 { return; }
        let mut changed = false;
        {
        let mut state = app_state_rem.write().unwrap();
        if let Some(monitor) = state.monitors.get_mut(mon_idx as usize) {
            if (layer_idx as usize) < monitor.layers.len() {
                monitor.layers.remove(layer_idx as usize);
                let slint_layers: Vec<LayerItem> = monitor.layers.iter().map(|l| {
                    LayerItem {
                        name: SharedString::from(&l.name),
                        opacity: l.opacity,
                        resolution_scale: l.resolution_scale,
                        positioning: SharedString::from(&l.positioning),
                        visible: l.visible,
                        blend_mode: SharedString::from(&l.blend_mode),
                        is_video: controller::video_wallpaper_path(&l.wallpaper_path).is_some(),
                        x: l.transform[0],
                        y: l.transform[1],
                        width: l.transform[2],
                        height: l.transform[3],
                    }
                }).collect();
                layers_model_rem.set_vec(slint_layers);

                // Keep the Library in sync: a removed layer may make its shader
                // inactive (no longer applied to any monitor), so recompute the
                // active flag + per-monitor usage counts across the library.
                let current_monitors = state.monitors.clone();
                for (i, w) in state.wallpapers.iter().enumerate() {
                    if let Some(mut item) = wallpapers_model_rem.row_data(i) {
                        let counts: Vec<i32> = current_monitors.iter()
                            .map(|m| m.layers.iter().filter(|l| l.wallpaper_path == w.path).count() as i32)
                            .collect();
                        item.is_active = counts.iter().any(|&c| c > 0);
                        item.usage_counts = Rc::new(VecModel::from(counts)).into();
                        wallpapers_model_rem.set_row_data(i, item);
                    }
                }

                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();

                changed = true;
            }
        }
        } // drop the app_state write guard before reconciling windows

        // Removing the last layer tears the monitor's window down (real desktop
        // returns); removing a non-last layer just refreshes it.  Either way,
        // reconcile - and only after the write guard is released, since
        // sync_wallpaper_windows re-reads app_state.
        if changed {
            sync_wallpaper_windows(
                app_state_rem.clone(),
                command_tx_rem.clone(),
                context_rem.clone(),
                store_rem.clone(),
                pending_close_rem.clone(),
            );
        }
    });

    let app_state_op = app_state.clone();
    let layers_model_op = layers_model.clone();
    let ui_handle_op = ui.as_weak();
    let command_tx_op = command_tx.clone();
    let config_dirty_op = config_dirty.clone();
    ui.on_layer_opacity_changed(move |layer_index, opacity| {
        let ui = ui_handle_op.unwrap();
        let monitor_index = ui.get_selected_monitor_index();
        if monitor_index < 0 { return; }
        let mut state = app_state_op.write().unwrap();
        // Update opacity + resolve the engine pipeline index (visible layers are
        // added in REVERSE order). Scoped so the &mut monitor borrow ends before
        // we read &state.monitors for the config save.
        let (monitor_id, pidx, visible) = {
            let monitor = match state.monitors.get_mut(monitor_index as usize) {
                Some(m) if (layer_index as usize) < m.layers.len() => m,
                _ => return,
            };
            let li = layer_index as usize;
            let pidx = monitor.layers.iter().enumerate().filter(|(j, l)| *j > li && l.visible).count();
            let visible = monitor.layers[li].visible;
            monitor.layers[li].opacity = opacity;
            (monitor.id.clone(), pidx, visible)
        };
        if let Some(mut slint_layer) = layers_model_op.row_data(layer_index as usize) {
            slint_layer.opacity = opacity;
            layers_model_op.set_row_data(layer_index as usize, slint_layer);
        }
        // Persist via the debounced flag, not a per-tick disk write.
        config_dirty_op.set(true);

        // Lightweight live update - no pipeline rebuild / shader recompile.
        if visible {
            command_tx_op.send(EngineCommand::SetLayerOpacity { monitor_id, pipeline_index: pidx, opacity }).ok();
        }
    });

    // ── Layer visibility toggle ─────────────────────────────────────────────
    let app_state_vis = app_state.clone();
    let layers_model_vis = layers_model.clone();
    let ui_handle_vis = ui.as_weak();
    let command_tx_vis = command_tx.clone();
    ui.on_layer_visibility_toggled(move |layer_index, visible| {
        let ui = ui_handle_vis.unwrap();
        let mut state = app_state_vis.write().unwrap();
        let monitor_index = ui.get_selected_monitor_index();
        if monitor_index < 0 { return; }
        if let Some(monitor) = state.monitors.get_mut(monitor_index as usize) {
            if let Some(layer) = monitor.layers.get_mut(layer_index as usize) {
                layer.visible = visible;
                if let Some(mut sl) = layers_model_vis.row_data(layer_index as usize) {
                    sl.visible = visible;
                    layers_model_vis.set_row_data(layer_index as usize, sl);
                }
                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();
                command_tx_vis.send(EngineCommand::Reload).ok();
            }
        }
    });

    // ── Layer blend mode ────────────────────────────────────────────────────
    let app_state_blend = app_state.clone();
    let layers_model_blend = layers_model.clone();
    let ui_handle_blend = ui.as_weak();
    let command_tx_blend = command_tx.clone();
    ui.on_layer_blend_changed(move |layer_index, mode| {
        let ui = ui_handle_blend.unwrap();
        let mut state = app_state_blend.write().unwrap();
        let monitor_index = ui.get_selected_monitor_index();
        if monitor_index < 0 { return; }
        if let Some(monitor) = state.monitors.get_mut(monitor_index as usize) {
            if let Some(layer) = monitor.layers.get_mut(layer_index as usize) {
                layer.blend_mode = mode.to_string();
                if let Some(mut sl) = layers_model_blend.row_data(layer_index as usize) {
                    sl.blend_mode = mode;
                    layers_model_blend.set_row_data(layer_index as usize, sl);
                }
                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();
                command_tx_blend.send(EngineCommand::Reload).ok();
            }
        }
    });

    // ── Monitor rename / color tag (cosmetic; persisted to config) ──────────
    let app_state_rename = app_state.clone();
    let monitors_model_rename = monitors_model.clone();
    let config_dirty_rename = config_dirty.clone();
    ui.on_monitor_rename(move |mon_idx, name| {
        let mut state = app_state_rename.write().unwrap();
        if let Some(m) = state.monitors.get_mut(mon_idx as usize) {
            m.name = name.to_string();
            if let Some(mut item) = monitors_model_rename.row_data(mon_idx as usize) {
                item.name = name;
                monitors_model_rename.set_row_data(mon_idx as usize, item);
            }
            // Per-keystroke from the rename field - debounce the disk write.
            config_dirty_rename.set(true);
        }
    });

    let app_state_moncolor = app_state.clone();
    let monitors_model_color = monitors_model.clone();
    ui.on_monitor_color_changed(move |mon_idx, color| {
        let mut state = app_state_moncolor.write().unwrap();
        if let Some(m) = state.monitors.get_mut(mon_idx as usize) {
            m.color = color.to_string();
            if let Some(mut item) = monitors_model_color.row_data(mon_idx as usize) {
                item.color = color;
                monitors_model_color.set_row_data(mon_idx as usize, item);
            }
            let mut config = config::Config::load();
            config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
            config.save().ok();
        }
    });

    let app_state_pos = app_state.clone();
    let layers_model_pos = layers_model.clone();
    let ui_handle_pos = ui.as_weak();
    let command_tx_pos = command_tx.clone();
    ui.on_layer_positioning_changed(move |layer_index, mode| {
        let ui = ui_handle_pos.unwrap();
        let mut state = app_state_pos.write().unwrap();
        let monitor_index = ui.get_selected_monitor_index();
        if monitor_index < 0 { return; }
        if let Some(monitor) = state.monitors.get_mut(monitor_index as usize) {
            let monitor_id = monitor.id.clone();
            if let Some(layer) = monitor.layers.get_mut(layer_index as usize) {
                layer.positioning = mode.to_string();
                let is_video = controller::video_wallpaper_path(&layer.wallpaper_path).is_some();
                if let Some(mut slint_layer) = layers_model_pos.row_data(layer_index as usize) {
                    slint_layer.positioning = mode.clone();
                    layers_model_pos.set_row_data(layer_index as usize, slint_layer);
                }

                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();

                if is_video {
                    // Video: live-update the WebView fit (no render-thread Reload needed).
                    daemon_set_fit(&monitor_id, positioning_to_object_fit(&mode));
                } else {
                    command_tx_pos.send(EngineCommand::Reload).ok();
                }
            }
        }
    });

    let app_state_scale = app_state.clone();
    let layers_model_scale = layers_model.clone();
    let ui_handle_scale = ui.as_weak();
    let command_tx_scale = command_tx.clone();
    ui.on_layer_scale_changed(move |layer_index, scale| {
        let ui = ui_handle_scale.unwrap();
        let mut state = app_state_scale.write().unwrap();
        let monitor_index = ui.get_selected_monitor_index();
        if monitor_index < 0 { return; }
        if let Some(monitor) = state.monitors.get_mut(monitor_index as usize) {
            if let Some(layer) = monitor.layers.get_mut(layer_index as usize) {
                layer.resolution_scale = scale;
                if let Some(mut slint_layer) = layers_model_scale.row_data(layer_index as usize) {
                    slint_layer.resolution_scale = scale;
                    layers_model_scale.set_row_data(layer_index as usize, slint_layer);
                }
                
                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();
                
                command_tx_scale.send(EngineCommand::Reload).ok();
            }
        }
    });

    let app_state_up = app_state.clone();
    let layers_model_up = layers_model.clone();
    let ui_handle_up = ui.as_weak();
    let command_tx_up = command_tx.clone();
    ui.on_layer_move_up(move |layer_index| {
        let ui = ui_handle_up.unwrap();
        let mut state = app_state_up.write().unwrap();
        let monitor_index = ui.get_selected_monitor_index();
        if monitor_index < 0 { return; }
        if let Some(monitor) = state.monitors.get_mut(monitor_index as usize) {
            if layer_index > 0 && (layer_index as usize) < monitor.layers.len() {
                monitor.layers.swap(layer_index as usize, layer_index as usize - 1);
                let slint_layers: Vec<LayerItem> = monitor.layers.iter().map(|l| {
                    LayerItem {
                        name: SharedString::from(&l.name),
                        opacity: l.opacity,
                        resolution_scale: l.resolution_scale,
                        positioning: SharedString::from(&l.positioning),
                        visible: l.visible,
                        blend_mode: SharedString::from(&l.blend_mode),
                        is_video: controller::video_wallpaper_path(&l.wallpaper_path).is_some(),
                        x: l.transform[0],
                        y: l.transform[1],
                        width: l.transform[2],
                        height: l.transform[3],
                    }
                }).collect();
                layers_model_up.set_vec(slint_layers);
                
                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();
                
                command_tx_up.send(EngineCommand::Reload).ok();
            }
        }
    });

    let app_state_down = app_state.clone();
    let layers_model_down = layers_model.clone();
    let ui_handle_down = ui.as_weak();
    let command_tx_down = command_tx.clone();
    ui.on_layer_move_down(move |layer_index| {
        let ui = ui_handle_down.unwrap();
        let mut state = app_state_down.write().unwrap();
        let monitor_index = ui.get_selected_monitor_index();
        if monitor_index < 0 { return; }
        if let Some(monitor) = state.monitors.get_mut(monitor_index as usize) {
            if (layer_index as usize) < monitor.layers.len() - 1 {
                monitor.layers.swap(layer_index as usize, layer_index as usize + 1);
                let slint_layers: Vec<LayerItem> = monitor.layers.iter().map(|l| {
                    LayerItem {
                        name: SharedString::from(&l.name),
                        opacity: l.opacity,
                        resolution_scale: l.resolution_scale,
                        positioning: SharedString::from(&l.positioning),
                        visible: l.visible,
                        blend_mode: SharedString::from(&l.blend_mode),
                        is_video: controller::video_wallpaper_path(&l.wallpaper_path).is_some(),
                        x: l.transform[0],
                        y: l.transform[1],
                        width: l.transform[2],
                        height: l.transform[3],
                    }
                }).collect();
                layers_model_down.set_vec(slint_layers);
                
                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();
                
                command_tx_down.send(EngineCommand::Reload).ok();
            }
        }
    });
    
    // Live drag preview: push the rect straight to the engine (no config write,
    // no pipeline rebuild) so the desktop updates in real time while dragging.
    let app_state_live = app_state.clone();
    let ui_handle_live = ui.as_weak();
    let command_tx_live = command_tx.clone();
    ui.on_layer_transform_live(move |layer_index, x, y, w, h| {
        let ui = ui_handle_live.unwrap();
        let mon_idx = ui.get_selected_monitor_index();
        if mon_idx < 0 { return; }
        // READ-ONLY: the preview only touches the engine, never app_state, so
        // Discard can revert by reloading the (still-original) app_state.
        let state = app_state_live.read().unwrap();
        // Resolve the engine pipeline index: visible layers are added in REVERSE,
        // so this layer's pipeline index = number of visible layers after it.
        let (monitor_id, pipeline_index, visible) = match state.monitors.get(mon_idx as usize) {
            Some(m) if (layer_index as usize) < m.layers.len() => {
                let li = layer_index as usize;
                let pidx = m.layers.iter().enumerate()
                    .filter(|(j, l)| *j > li && l.visible).count();
                (m.id.clone(), pidx, m.layers[li].visible)
            }
            _ => return,
        };
        if visible {
            command_tx_live.send(EngineCommand::SetLayerTransform {
                monitor_id, pipeline_index, transform: [x, y, w, h],
            }).ok();
        }
    });

    // Discard: revert the live engine preview back to the saved app_state.
    let command_tx_edit_cancel = command_tx.clone();
    ui.on_layer_edit_cancelled(move || {
        command_tx_edit_cancel.send(EngineCommand::Reload).ok();
    });

    let app_state_transform = app_state.clone();
    let layers_model_transform = layers_model.clone();
    let ui_handle_transform = ui.as_weak();
    let command_tx_transform = command_tx.clone();
    ui.on_layer_transform_changed(move |layer_index, x, y, w, h| {
        let ui = ui_handle_transform.unwrap();
        let mut state = app_state_transform.write().unwrap();
        let monitor_index = ui.get_selected_monitor_index();
        if monitor_index < 0 { return; }
        if let Some(monitor) = state.monitors.get_mut(monitor_index as usize) {
            if let Some(layer) = monitor.layers.get_mut(layer_index as usize) {
                layer.transform = [x, y, w, h];
                layer.positioning = "Custom".to_string(); // engine renders into the rect
                if let Some(mut slint_layer) = layers_model_transform.row_data(layer_index as usize) {
                    slint_layer.x = x;
                    slint_layer.y = y;
                    slint_layer.width = w;
                    slint_layer.height = h;
                    slint_layer.positioning = SharedString::from("Custom");
                    layers_model_transform.set_row_data(layer_index as usize, slint_layer);
                }
                
                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();
                
                command_tx_transform.send(EngineCommand::Reload).ok();
            }
        }
    });

    let app_state_span = app_state.clone();
    let ui_handle_span = ui.as_weak();
    ui.on_span_toggled(move |span| {
        // Persist the new mode, then rebuild the wallpaper windows so the
        // span/independent topology takes effect.
        {
            let mut state = app_state_span.write().unwrap();
            state.span_monitors = span;
            let mut config = config::Config::load();
            config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
            config.save().ok();
        } // drop the write guard BEFORE invoke_refresh_monitors - that callback
          // re-acquires the same lock, and std RwLock is not reentrant (the old
          // code held the guard here and deadlocked the whole UI thread).

        if let Some(ui) = ui_handle_span.upgrade() {
            ui.invoke_refresh_monitors();
        }
    });

    ui.on_debug_toggled(move |_debug| {
        // Log debug toggle
    });

    let app_state_auto = app_state.clone();
    ui.on_autostart_toggled(move |enabled| {
        {
            let mut state = app_state_auto.write().unwrap();
            state.autostart = enabled;
        }
        let state = app_state_auto.read().unwrap();
        // Implement auto-launch
        let exe_path = std::env::current_exe().unwrap();
        let auto = auto_launch::AutoLaunchBuilder::new()
            .set_app_name("Strata")
            .set_app_path(&exe_path.to_string_lossy())
            // Boot into the tray, not the open window.
            .set_args(&["--minimized"])
            .build()
            .unwrap();
        
        if enabled {
            auto.enable().ok();
        } else {
            auto.disable().ok();
        }

        let mut config = config::Config::load();
        config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
        config.save().ok();
    });

    // MPO toggle: writing HKLM\…\Dwm\OverlayTestMode needs admin, so relaunch ourselves
    // elevated (UAC) on a worker thread. The toast can't be raised off-thread (push_toast
    // isn't Send), so the worker stores a status the UI timer polls: 1 = changed (refresh
    // the toggle + show a "restart required" toast), 2 = declined/failed (just snap the
    // toggle back to reality).
    let mpo_status = Arc::new(std::sync::atomic::AtomicU8::new(0));
    {
        let status = mpo_status.clone();
        ui.on_mpo_toggled(move |enabled| {
            let status = status.clone();
            std::thread::spawn(move || {
                let changed = set_mpo_elevated(enabled) && read_mpo_disabled() == enabled;
                status.store(if changed { 1 } else { 2 }, std::sync::atomic::Ordering::Relaxed);
            });
        });
    }

    let command_tx_debug = command_tx.clone();
    let app_state_debug = app_state.clone();
    let context_debug = context.clone();
    let command_tx_shutdown = command_tx.clone();
    let config_dirty_quit = config_dirty.clone();
    let app_state_quit = app_state.clone();
    let target_fps_quit = target_fps.clone();
    let audio_sensitivity_quit = audio_sensitivity.clone();
    let mouse_mode_quit = mouse_mode.clone();
    let mouse_sensitivity_quit = mouse_sensitivity.clone();
    let quality_scale_quit = quality_scale.clone();
    ui.on_quit_requested(move || {
        // Flush any pending debounced config change before exiting abruptly.
        if config_dirty_quit.get() {
            if let Ok(st) = app_state_quit.read() {
                flush_config(&st, target_fps_quit.load(std::sync::atomic::Ordering::Relaxed), audio_sensitivity_quit.get(),
                    u8_to_mouse_mode(mouse_mode_quit.load(std::sync::atomic::Ordering::Relaxed)),
                    f32::from_bits(mouse_sensitivity_quit.load(std::sync::atomic::Ordering::Relaxed)),
                    f32::from_bits(quality_scale_quit.load(std::sync::atomic::Ordering::Relaxed)));
            }
        }
        let _ = command_tx_shutdown.send(EngineCommand::Shutdown);
        daemon_shutdown(); // kill the video daemon child now (it would also self-exit on EOF)
        std::process::exit(0);
    });

    let command_tx_vsync = command_tx.clone();
    ui.on_vsync_changed(move |mode| {
        // Match on a substring so the label wording can vary
        // (e.g. "Immediate (VSync Off)", "Mailbox (Ultra-Fast)").
        let present_mode = if mode.contains("Immediate") {
            core_engine::wgpu::PresentMode::Immediate
        } else if mode.contains("Mailbox") {
            core_engine::wgpu::PresentMode::Mailbox
        } else {
            core_engine::wgpu::PresentMode::Fifo
        };
        let _ = command_tx_vsync.send(EngineCommand::SetVSync(present_mode));
    });

    // FPS cap slider - store into the shared atomic (render threads pick it up
    // next frame); persistence is debounced via the dirty flag.
    let target_fps_ui = target_fps.clone();
    let config_dirty_fps = config_dirty.clone();
    ui.on_fps_cap_changed(move |fps| {
        let clamped = (fps.clamp(1, 240)) as u32;
        target_fps_ui.store(clamped, std::sync::atomic::Ordering::Relaxed);
        config_dirty_fps.set(true);
    });

    // Shader Quality preset - maps the label to a global render scale that every
    // monitor loop picks up live; persistence is debounced via the dirty flag.
    let quality_scale_ui = quality_scale.clone();
    let config_dirty_quality = config_dirty.clone();
    ui.on_shader_quality_changed(move |label| {
        let scale = shader_quality_to_scale(&label);
        quality_scale_ui.store(scale.to_bits(), std::sync::atomic::Ordering::Relaxed);
        config_dirty_quality.set(true);
    });

    // Audio sensitivity slider - retunes the AudioEngine gain live; debounced save.
    let context_audio = context.clone();
    let audio_sensitivity_ui = audio_sensitivity.clone();
    let config_dirty_audio = config_dirty.clone();
    ui.on_audio_sensitivity_changed(move |v| {
        let v = v.clamp(0.0, 4.0);
        if let Some(a) = &context_audio.audio { a.set_sensitivity(v); }
        audio_sensitivity_ui.set(v);
        config_dirty_audio.set(true);
    });

    // Mouse interactivity mode - render loops pick it up live.
    let mouse_mode_ui = mouse_mode.clone();
    let config_dirty_mouse = config_dirty.clone();
    ui.on_mouse_mode_changed(move |label| {
        mouse_mode_ui.store(mouse_mode_to_u8(&label), std::sync::atomic::Ordering::Relaxed);
        config_dirty_mouse.set(true);
    });

    // Mouse sensitivity slider.
    let mouse_sensitivity_ui = mouse_sensitivity.clone();
    let config_dirty_msens = config_dirty.clone();
    ui.on_mouse_sensitivity_changed(move |v| {
        let v = v.clamp(0.1, 4.0);
        mouse_sensitivity_ui.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
        config_dirty_msens.set(true);
    });


    ui.on_open_debugger(move || {
        let _ = command_tx_debug.send(EngineCommand::OpenDebugger);
        let app_state = app_state_debug.clone();
        let command_tx = command_tx_debug.clone();
        let context = context_debug.clone();
        
        let state = app_state.read().unwrap();
        let mut min_x = 0i32;
        let mut min_y = 0i32;
        let mut max_x = 0i32;
        let mut max_y = 0i32;

        for m in &state.monitors {
            min_x = min_x.min(m.position.0);
            min_y = min_y.min(m.position.1);
            max_x = max_x.max(m.position.0 + m.resolution.0 as i32);
            max_y = max_y.max(m.position.1 + m.resolution.1 as i32);
        }

        let total_w = (max_x - min_x) as u32;
        let total_h = (max_y - min_y) as u32;
        let scale = 0.5f32;

        let debug_ui = WallpaperWindow::new().unwrap();
        debug_ui.window().set_size(slint::PhysicalSize::new((total_w as f32 * scale) as u32, (total_h as f32 * scale) as u32));
        
        let mut layers = Vec::new();
        for m in &state.monitors {
            let mut monitor_layers = m.layers.clone();
            for layer in &mut monitor_layers {
                // If it's not a custom transform, it covers the whole monitor.
                // We convert it to a custom transform for the composite debugger view.
                if layer.positioning != "Custom" {
                    layer.transform = [
                        (m.position.0 - min_x) as f32,
                        (m.position.1 - min_y) as f32,
                        m.resolution.0 as f32,
                        m.resolution.1 as f32,
                    ];
                    layer.positioning = "Custom".to_string();
                } else {
                    // Offset custom transform by monitor position
                    layer.transform[0] += (m.position.0 - min_x) as f32;
                    layer.transform[1] += (m.position.1 - min_y) as f32;
                }
            }
            layers.extend(monitor_layers);
        }

        let debug_win_weak = debug_ui.as_weak();
        let command_tx_resized = command_tx.clone();
        let command_tx_closed = command_tx.clone();
        slint::spawn_local(async move {
            let win = debug_win_weak.unwrap();
        let slint_win = win.window();
        if let Ok(w) = slint::winit_030::WinitWindowAccessor::winit_window(slint_win).await {
                let window_id: winit::window::WindowId = w.id();
            slint::winit_030::WinitWindowAccessor::on_winit_window_event(slint_win, move |_, event| {
                    match event {
                        winit::event::WindowEvent::Resized(size) => {
                            let _ = command_tx_resized.send(EngineCommand::WindowResized(window_id, size.clone()));
                        }
                        winit::event::WindowEvent::CloseRequested => {
                            let _ = command_tx_closed.send(EngineCommand::WindowClosed(window_id));
                        }
                        _ => {}
                    }
                    slint::winit_030::EventResult::Propagate
                });
                let surface = context.instance.create_surface(w.clone()).unwrap();
                command_tx.send(EngineCommand::AddWindow {
                    window: w.clone(),
                    surface,
                    initial_size: w.inner_size(),  // debug window: size is always correct here
                    offset: (0.0, 0.0),
                    global_res: (total_w as f32, total_h as f32),
                    layers,
                    monitor_id: "debug-composite".to_string(),
                }).ok();
            }
        }).unwrap();

        debug_ui.show().unwrap();
        Box::leak(Box::new(debug_ui));
    });

    // Spawn Engine Thread
    let engine_state = app_state.clone();
    let engine_running = running.clone();
    let engine_context = context.clone();
    let engine_target_fps = target_fps.clone();
    let engine_mouse_mode = mouse_mode.clone();
    let engine_mouse_sensitivity = mouse_sensitivity.clone();
    let engine_quality = quality_scale.clone();
    let telemetry = Arc::new(std::sync::Mutex::new(platform::EngineTelemetry { fps: 0.0, frame_time: 0.0, vram_usage: 0.0 }));
    let telemetry_thread = telemetry.clone();
    std::thread::spawn(move || {
        platform::renderer::run_renderer(engine_running, engine_state, command_rx, telemetry_thread, engine_context, engine_target_fps, engine_mouse_mode, engine_mouse_sensitivity, engine_quality);
    });

    // ── Initial wallpaper window creation ───────────────────────────────────
    // Only monitors that already have a saved shader assignment get a window;
    // unassigned monitors keep showing the real Windows wallpaper.
    spawn_wallpaper_windows(
        app_state.clone(),
        command_tx.clone(),
        context.clone(),
        wallpaper_store.clone(),
    );

    // Weekly background update check (engine + asset library). Runs at most ~once a
    // week so launches stay fast and offline-friendly. On finding an update it sets
    // the Settings → Updates badges/buttons and raises `update_toast_pending`; the UI
    // timer shows the "Updates available" toast once the window is actually visible
    // (so a tray/minimized start still surfaces it when the user opens the UI).
    let update_toast_pending = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        let cfg = config::Config::load();
        const WEEK: i64 = 7 * 24 * 60 * 60;
        if now - cfg.last_update_check >= WEEK {
            let weak = ui.as_weak();
            // Prefer the on-disk version so a stale config doesn't trigger a re-download.
            let lib_ver = controller::installed_library_version().unwrap_or_else(|| cfg.library_version.clone());
            let pending = update_toast_pending.clone();
            std::thread::spawn(move || {
                let app = check_github_latest();
                let lib = check_library_update(&lib_ver);
                // Only record the check time if we actually reached the network. A failed
                // attempt (e.g. booted offline) leaves last_update_check untouched so the
                // next launch retries instead of going quiet for a whole week.
                if app.is_ok() || lib.is_ok() {
                    let mut c = config::Config::load(); c.last_update_check = now; c.save().ok();
                }
                let _ = slint::invoke_from_event_loop(move || {
                    let Some(ui) = weak.upgrade() else { return };
                    let mut any = false;
                    if let Ok(Some((tag, url))) = app {
                        ui.set_update_url(url.into());
                        ui.set_update_available(true);
                        ui.set_update_version_badge(slint::format!("{} AVAILABLE", tag));
                        ui.set_update_button_label("DOWNLOAD UPDATE".into());
                        any = true;
                    }
                    if let Ok(Some(ver)) = lib {
                        ui.set_lib_update_available(true);
                        ui.set_lib_update_version_badge(slint::format!("v{} AVAILABLE", ver));
                        ui.set_lib_update_button_label("DOWNLOAD UPDATE".into());
                        any = true;
                    }
                    if any { pending.store(true, std::sync::atomic::Ordering::Relaxed); }
                });
            });
        }
    }

    // First-run library fetch: the app ships no wallpapers, so if nothing has been
    // downloaded yet, pull the latest Strata-Library in the background and refresh
    // the grid when it lands. A toast (shown immediately, since this runs on the UI
    // thread) tells the user a one-time download is happening so an empty grid on
    // first launch doesn't look broken. The result toast is raised via the timer
    // (`library_fetch_status`) because push_toast isn't Send. (Subsequent updates go
    // through Settings → Updates.)
    let library_fetch_status = Arc::new(std::sync::atomic::AtomicU8::new(0)); // 0 idle, 1 ok, 2 fail
    if !controller::library_installed() {
        push_toast("Downloading Library", "Fetching the wallpaper library - this happens only once.", false);
        let weak = ui.as_weak();
        let status = library_fetch_status.clone();
        std::thread::spawn(move || {
            let result: Result<String, String> = (|| {
                let (owner, repo, version, tag) = library_sync::latest_library()?;
                library_sync::sync_library(&owner, &repo, &tag)?;
                Ok(version)
            })();
            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = weak.upgrade() else { return };
                match result {
                    Ok(version) => {
                        let mut c = config::Config::load();
                        c.library_version = version.clone();
                        c.save().ok();
                        ui.set_lib_update_version_badge(slint::format!("v{} (LATEST)", version));
                        ui.invoke_refresh_library();
                        log::info!("Fetched wallpaper library v{}", version);
                        status.store(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Err(e) => {
                        log::warn!("First-run library fetch failed (will retry from Settings): {}", e);
                        status.store(2, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        });
    }

    // ── Diagnostics / telemetry / log / tray timer ─────────────────────────
    let ui_handle_timer = ui.as_weak();
    let telemetry_ui = telemetry.clone();
    let update_toast_pending_timer = update_toast_pending.clone();
    let library_fetch_status_timer = library_fetch_status.clone();
    let mpo_status_timer = mpo_status.clone();
    let timer = slint::Timer::default();
    let logs_model = Rc::new(VecModel::<LogEntry>::from(Vec::new()));
    ui.set_logs(ModelRc::from(logs_model.clone()));
    let pending_close_timer = pending_close.clone();
    let config_dirty_timer = config_dirty.clone();
    let app_state_timer = app_state.clone();
    let target_fps_timer = target_fps.clone();
    let mouse_mode_timer = mouse_mode.clone();
    let mouse_sensitivity_timer = mouse_sensitivity.clone();
    let context_timer = context.clone();
    let thumbnails_busy_timer = thumbnails_busy.clone();
    let wallpapers_model_timer = wallpapers_model.clone();
    let app_state_thumb = app_state.clone();
    // The software renderer leaves stale (white/transparent) regions whenever the OS
    // discards/suspends the window's buffer - tray restore, taskbar un-minimize, or
    // waking a long-dormant window. Forcing a full repaint requires a real resize, so
    // `needs_repaint` (set on those events) makes the timer nudge the size +2px and
    // restore it next tick (two resize events → full repaint). `restore_size` carries
    // the pending revert.
    let restore_size = std::cell::Cell::new(None::<slint::PhysicalSize>);
    let needs_repaint = std::rc::Rc::new(std::cell::Cell::new(false));
    let needs_repaint_timer = needs_repaint.clone();
    // Tracks whether the main window is actually on-screen (not hidden to tray /
    // minimized). Used to skip work the user can't see - e.g. the parallax preview's
    // offscreen GPU render. Updated on tray show/hide and winit Occluded events.
    let ui_visible = std::rc::Rc::new(std::cell::Cell::new(true));
    let ui_visible_timer = ui_visible.clone();
    // Moving the window can reveal regions that were off-screen (dragged past a screen
    // edge, or the reposition after restoring a maximized window). The software renderer
    // never marked those dirty, so they stay stale until a full repaint. winit fires
    // `Moved` continuously during a drag, so we debounce: record the last move time and
    // request the full-repaint nudge only once movement has settled (avoids size jitter
    // mid-drag). A resize, by contrast, self-heals (it resets softbuffer's buffer age).
    let move_settle = std::rc::Rc::new(std::cell::Cell::new(None::<std::time::Instant>));
    let move_settle_timer = move_settle.clone();
    // The repaint nudge resizes the window by a pixel - but a `set_size` on a MAXIMIZED
    // window un-maximizes and repositions it (the "maximize shifts to bottom-right" bug).
    // Maximizing already fires a real resize that self-heals the buffer, so we simply
    // skip the nudge while maximized. Kept current by the winit hook.
    let ui_maximized = std::rc::Rc::new(std::cell::Cell::new(false));
    let ui_maximized_timer = ui_maximized.clone();
    #[cfg(target_os = "windows")]
    let ui_hwnd_timer = ui_hwnd.clone();
    #[cfg(target_os = "windows")]
    let toggle_item_timer = tray_toggle_item.clone();
    #[cfg(target_os = "windows")]
    let hidden_to_tray_timer = hidden_to_tray.clone();
    // Uptime guard for device-loss recovery: don't auto-relaunch in the first few
    // seconds, so a device that's broken right at startup can't restart-loop.
    let app_start = std::time::Instant::now();
    let toasts_model_timer = toasts_model.clone();
    let toast_expiry_timer = toast_expiry.clone();
    let push_toast_timer = push_toast.clone();
    let audio_sensitivity_timer = audio_sensitivity.clone();
    let quality_scale_timer = quality_scale.clone();
    let preview_state_timer = preview_state.clone();
    let parallax_params_timer = parallax_params.clone();
    let parallax_tune_settle_timer = parallax_tune_settle.clone();
    // Throttle for the periodic "needs download?" recheck on the Parallax tab.
    let parallax_dl_recheck = std::cell::Cell::new(std::time::Instant::now());
    // Throttle for the video cover-detection check (pause movies under fullscreen apps).
    let video_pause_recheck = std::cell::Cell::new(std::time::Instant::now());
    let app_state_vpause = app_state.clone();

    // Copy the diagnostics log to the clipboard.
    let logs_model_copy = logs_model.clone();
    let push_toast_copy = push_toast.clone();
    ui.on_copy_logs(move || {
        let mut text = String::new();
        for i in 0..logs_model_copy.row_count() {
            if let Some(e) = logs_model_copy.row_data(i) {
                text.push_str(&format!("[{}] {} {}\n", e.time, e.level, e.message));
            }
        }
        match arboard::Clipboard::new().and_then(|mut cb| cb.set_text(text)) {
            Ok(_) => {
                log::info!("Diagnostics log copied to clipboard");
                push_toast_copy("Logs Copied", "Diagnostics data has been copied to clipboard.", false);
            }
            Err(e) => {
                log::warn!("Clipboard copy failed: {}", e);
                push_toast_copy("Copy Failed", "Could not access the clipboard.", true);
            }
        }
    });

    // Set up Logger
    let (log_tx, log_rx) = channel::<String>();
    let logger = SlintLogger { sender: log_tx, file: std::sync::Mutex::new(open_log_file()) };
    let _ = log::set_boxed_logger(Box::new(logger));
    log::set_max_level(log::LevelFilter::Info);

    timer.start(slint::TimerMode::Repeated, std::time::Duration::from_millis(100), move || {
        // ── Pause movie wallpapers under fullscreen apps (every ~1s) ──
        if video_pause_recheck.get().elapsed() >= std::time::Duration::from_millis(1000) {
            video_pause_recheck.set(std::time::Instant::now());
            update_video_pause(&app_state_vpause);
        }

        // ── "Updates available" toast ──
        // The weekly check sets this flag from a background thread; show the toast
        // only once the window is actually visible, so a minimized/tray start still
        // greets the user with it when they open the UI.
        if ui_visible_timer.get()
            && update_toast_pending_timer.swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            push_toast_timer("Updates Available", "Open Settings \u{2192} Updates to get the latest.", false);
        }

        // ── First-run library fetch result toast ──
        // Raised here (not from the fetch thread) because push_toast isn't Send.
        if ui_visible_timer.get() {
            match library_fetch_status_timer.swap(0, std::sync::atomic::Ordering::Relaxed) {
                1 => push_toast_timer("Library Ready", "Your wallpaper library has been downloaded.", false),
                2 => push_toast_timer("Library Download Failed", "Couldn't fetch the library - check your connection, then retry in Settings \u{2192} Updates.", true),
                _ => {}
            }
        }

        // MPO toggle result (set from the elevated worker; toast raised here since
        // push_toast isn't Send). Snap the toggle to the real registry state either way.
        match mpo_status_timer.swap(0, std::sync::atomic::Ordering::Relaxed) {
            1 => {
                if let Some(ui) = ui_handle_timer.upgrade() { ui.set_disable_mpo(read_mpo_disabled()); }
                push_toast_timer("Restart Required", "Restart your PC for the Multi-Plane Overlay change to take effect.", false);
            }
            2 => {
                if let Some(ui) = ui_handle_timer.upgrade() { ui.set_disable_mpo(read_mpo_disabled()); }
            }
            _ => {}
        }

        // ── GPU device-loss recovery ──
        // A lost device (driver TDR/reset, GPU hang, driver update) can't be
        // revived in place. Persist state and relaunch - Strata restores monitors,
        // layers and settings from config, so the wallpaper comes back on its own.
        if context_timer.device_lost.load(std::sync::atomic::Ordering::SeqCst)
            && app_start.elapsed() > std::time::Duration::from_secs(20)
        {
            log::error!("GPU device lost - relaunching Strata to recover the wallpaper");
            if let Ok(st) = app_state_timer.read() {
                flush_config(
                    &st,
                    target_fps_timer.load(std::sync::atomic::Ordering::Relaxed),
                    audio_sensitivity_timer.get(),
                    u8_to_mouse_mode(mouse_mode_timer.load(std::sync::atomic::Ordering::Relaxed)),
                    f32::from_bits(mouse_sensitivity_timer.load(std::sync::atomic::Ordering::Relaxed)),
                    f32::from_bits(quality_scale_timer.load(std::sync::atomic::Ordering::Relaxed)),
                );
            }
            if let Ok(exe) = std::env::current_exe() {
                let _ = std::process::Command::new(exe).spawn();
            }
            std::process::exit(0);
        }

        // ── Deferred wallpaper-window destruction ──
        // Windows queued for teardown are dropped here, ~300 ms after their
        // render thread was told to shut down, so the HWND outlives the last
        // surface present.  Dropping the component destroys the Win32 window.
        {
            let now = std::time::Instant::now();
            pending_close_timer.borrow_mut().retain(|(_, queued)| {
                now.duration_since(*queued) < std::time::Duration::from_millis(300)
            });
        }

        // ── Toast auto-dismiss ──
        // Drop any toast whose 4s lifetime has elapsed. Iterating front-to-back
        // and removing by matching id keeps the model and expiry list in sync.
        {
            let now = std::time::Instant::now();
            let expired: Vec<i32> = toast_expiry_timer.borrow().iter()
                .filter(|(_, deadline)| now >= *deadline)
                .map(|(id, _)| *id)
                .collect();
            if !expired.is_empty() {
                toast_expiry_timer.borrow_mut().retain(|(_, deadline)| now < *deadline);
                for id in expired {
                    if let Some(idx) = (0..toasts_model_timer.row_count())
                        .find(|&i| toasts_model_timer.row_data(i).map(|t| t.id) == Some(id)) {
                        toasts_model_timer.remove(idx);
                    }
                }
            }
        }

        // ── Debounced config flush ──
        if config_dirty_timer.get() {
            if let Ok(st) = app_state_timer.read() {
                flush_config(&st, target_fps_timer.load(std::sync::atomic::Ordering::Relaxed), audio_sensitivity_timer.get(),
                    u8_to_mouse_mode(mouse_mode_timer.load(std::sync::atomic::Ordering::Relaxed)),
                    f32::from_bits(mouse_sensitivity_timer.load(std::sync::atomic::Ordering::Relaxed)),
                    f32::from_bits(quality_scale_timer.load(std::sync::atomic::Ordering::Relaxed)));
            }
            config_dirty_timer.set(false);
        }

        // ── Thumbnail results → library ──
        // Drain any thumbnails the background thread finished and load them into
        // the matching library card (matched by wallpaper path).
        while let Ok((wallpaper, thumb)) = thumb_rx.try_recv() {
            if let Ok(mut st) = app_state_thumb.write() {
                if let Some(idx) = st.wallpapers.iter().position(|w| w.path == wallpaper) {
                    if let Some(mut item) = wallpapers_model_timer.row_data(idx) {
                        if let Ok(img) = Image::load_from_path(&thumb) {
                            item.thumbnail = img;
                            item.has_thumbnail = true;
                            wallpapers_model_timer.set_row_data(idx, item);
                            // Keep AppState authoritative so a later Refresh sees no
                            // phantom change and skips a needless model rebuild.
                            st.wallpapers[idx].thumbnail = Some(thumb);
                        }
                    }
                }
            }
        }

        // ── Parallax Studio results ──
        while let Ok(res) = parallax_rx.try_recv() {
            if let Some(ui) = ui_handle_timer.upgrade() {
                ui.set_parallax_busy(false);
                match res {
                    Ok(name) => {
                        ui.set_parallax_status(SharedString::from(format!("Created \"{}\" - see the Library.", name)));
                        push_toast_timer("Parallax Created", "Your 3D wallpaper was added to the Library.", false);
                        ui.invoke_refresh_library(); // rescan so the new wallpaper appears
                    }
                    Err(e) => {
                        log::error!("Parallax create failed: {}", e);
                        ui.set_parallax_status(SharedString::from("Failed - see Diagnostics."));
                        push_toast_timer("Parallax Failed", "Could not create the wallpaper.", true);
                    }
                }
            }
        }

        // While a render is running, mirror the build thread's staged progress into
        // the previewer's timeline strip.
        if parallax_busy.load(std::sync::atomic::Ordering::SeqCst) {
            if let Some(ui) = ui_handle_timer.upgrade() {
                ui.set_parallax_progress(parallax_progress.load(std::sync::atomic::Ordering::SeqCst) as i32);
            }
        }

        // ── Parallax preview: build on render-completion, then animate ──
        while let Ok(res) = preview_ready_rx.try_recv() {
            if let Some(ui) = ui_handle_timer.upgrade() {
                ui.set_parallax_busy(false); // the render worker has finished, either way
                match res {
                    Ok(dir) => match ParallaxPreviewState::new(context_timer.clone(), &dir) {
                        Ok(state) => {
                            *preview_state_timer.borrow_mut() = Some(state);
                            ui.set_parallax_has_preview(true);
                            ui.set_parallax_progress(100);
                            ui.set_parallax_status(SharedString::from("Preview ready - Save to add it to your Library."));
                        }
                        Err(e) => {
                            log::error!("Parallax preview build failed: {}", e);
                            ui.set_parallax_progress(0);
                            ui.set_parallax_status(SharedString::from("Preview failed - see Diagnostics."));
                        }
                    },
                    Err(e) => {
                        log::error!("Parallax depth/preview failed: {}", e);
                        ui.set_parallax_progress(0);
                        ui.set_parallax_status(SharedString::from("Failed - see Diagnostics."));
                    }
                }
            }
        }
        // Animate the preview (~10 fps) only while the Parallax tab is open AND the
        // window is on-screen - no point rendering frames nobody can see.
        if let Some(ui) = ui_handle_timer.upgrade() {
            if ui_visible_timer.get() && ui.get_active_tab() == "parallax" {
                if let Some(state) = preview_state_timer.borrow_mut().as_mut() {
                    if let Some(img) = state.frame(&context_timer) {
                        ui.set_parallax_preview_image(img);
                    }
                }
            }
        }

        // ── Parallax model download queue: live progress + completion ──
        // The download thread writes "Name · NN MB (i/N)" into parallax_dl_text and the
        // current-file % into download_pct; mirror both to the UI while downloading.
        if parallax_downloading.load(std::sync::atomic::Ordering::SeqCst) {
            if let Some(ui) = ui_handle_timer.upgrade() {
                ui.set_parallax_downloading(true);
                if let Ok(t) = parallax_dl_text.lock() {
                    ui.set_parallax_download_text(SharedString::from(t.as_str()));
                }
                ui.set_parallax_download_percent(download_pct.load(std::sync::atomic::Ordering::SeqCst) as i32);
            }
        }
        while let Ok(res) = download_rx.try_recv() {
            if let Some(ui) = ui_handle_timer.upgrade() {
                ui.set_parallax_downloading(false);
                match res {
                    Ok(_) => {
                        refresh_parallax_download_state(&ui); // recompute → hides the button
                        push_toast_timer("Models Ready", "All required models downloaded - ready to generate.", false);
                    }
                    Err(e) => {
                        log::error!("Model download failed: {}", e);
                        ui.set_parallax_download_text(SharedString::from("Download failed - see Diagnostics."));
                        push_toast_timer("Download Failed", "Could not download the models.", true);
                    }
                }
            }
        }
        // Keep the "needs download?" flag fresh on the Parallax tab (mode/preset/selection
        // can change without a callback, e.g. switching Automatic↔Manual). Cheap fs checks,
        // throttled to ~1/s.
        if parallax_dl_recheck.get().elapsed() >= std::time::Duration::from_millis(1000) {
            parallax_dl_recheck.set(std::time::Instant::now());
            if let Some(ui) = ui_handle_timer.upgrade() {
                if ui.get_active_tab() == "parallax"
                    && !parallax_downloading.load(std::sync::atomic::Ordering::SeqCst) {
                    refresh_parallax_download_state(&ui);
                }
            }
        }

        // Live re-bake: ~180ms after the tuning sliders settle, re-bake the preview's
        // shader with the new params (reusing the cached depth/inpaint via style.toml) and
        // reload the preview renderer - instant tuning with NO model inference.
        if let Some(t) = parallax_tune_settle_timer.get() {
            if t.elapsed() >= std::time::Duration::from_millis(180) {
                parallax_tune_settle_timer.set(None);
                if let Some(dir) = parallax::preview_dir() {
                    if preview_state_timer.borrow().is_some() && dir.join("style.toml").exists() {
                        match core_engine::parallax::rebake_layered(&dir, &parallax_params_timer.get()) {
                            Ok(()) => match ParallaxPreviewState::new(context_timer.clone(), &dir) {
                                Ok(state) => *preview_state_timer.borrow_mut() = Some(state),
                                Err(e) => log::warn!("parallax tuning reload failed: {e}"),
                            },
                            Err(e) => log::warn!("parallax re-bake failed: {e}"),
                        }
                    }
                }
            }
        }

        // Once the window has stopped moving for ~150ms, schedule a full repaint so any
        // regions revealed by the move get painted (see `move_settle`).
        if let Some(t) = move_settle_timer.get() {
            if t.elapsed() >= std::time::Duration::from_millis(150) {
                move_settle_timer.set(None);
                needs_repaint_timer.set(true);
            }
        }

        if let Some(ui) = ui_handle_timer.upgrade() {
            // Force a full repaint after a restore/un-occlude (software renderer leaves
            // stale regions otherwise): revert the previous tick's size nudge, or apply
            // a fresh one if requested. `else if` keeps the two on separate ticks so each
            // is a real resize event.
            if let Some(sz) = restore_size.take() {
                ui.window().set_size(sz);
            } else if needs_repaint_timer.replace(false) && !ui_maximized_timer.get() {
                // Skip while maximized - resizing a maximized window would shift it.
                let sz = ui.window().size();
                ui.window().set_size(slint::PhysicalSize::new(sz.width, sz.height + 2));
                restore_size.set(Some(sz));
            }

            // Accent the library refresh icon while the thumbnail thread runs.
            let busy = thumbnails_busy_timer.load(std::sync::atomic::Ordering::SeqCst);
            ui.set_refreshing(busy);

            // ── Telemetry ──
            if let Ok(tel) = telemetry_ui.try_lock() {
                ui.set_fps(tel.fps);
                ui.set_frame_time(tel.frame_time);
                ui.set_vram_usage(tel.vram_usage);
            }
            // Whether anything is actually being rendered - drives the "Inactive"
            // state on the diagnostics cards (avoids a scary 0 FPS / LOW / -0 MB
            // when no shader is assigned).
            // Movies don't feed the engine telemetry (no wgpu render thread), so a
            // movie-only desktop should read "inactive", not 0 FPS. Count only visible
            // SHADER layers (path NOT under the import-video dir - cheap, no disk read).
            let video_dir = controller::import_video_dir();
            let (engine_active, any_movie) = app_state_timer.read()
                .map(|s| {
                    let is_movie = |l: &controller::LayerInfo| video_dir.as_ref().is_some_and(|vd| l.wallpaper_path.starts_with(vd));
                    let active = s.monitors.iter().any(|m| m.layers.iter().any(|l| l.visible && !is_movie(l)));
                    let movie = s.monitors.iter().any(|m| m.layers.iter().any(|l| is_movie(l)));
                    (active, movie)
                })
                .unwrap_or((false, false));
            ui.set_engine_active(engine_active);
            // Span is unavailable while a movie wallpaper is active (movies don't span).
            ui.set_span_disabled(any_movie);

            // ── Logs ──
            // Parse each "[LEVEL] message" line into a structured entry with a
            // wall-clock timestamp for the Diagnostics log panel.
            while let Ok(line) = log_rx.try_recv() {
                let (level, message) = if line.starts_with('[') {
                    if let Some(end) = line.find(']') {
                        (line[1..end].trim().to_string(), line[end + 1..].trim().to_string())
                    } else {
                        ("INFO".to_string(), line.clone())
                    }
                } else {
                    ("INFO".to_string(), line.clone())
                };
                // Surface a discrete shader-apply failure as a destructive toast.
                // Only the one-shot "Layer reload error" (a compile/build failure
                // at apply time) triggers this - per-frame wgpu errors are left to
                // the log to avoid toast spam.
                if level == "ERROR" && message.starts_with("Layer reload error") {
                    push_toast_timer(
                        "Failed to Apply Shader",
                        "A shader could not be loaded - see the Diagnostics log for details.",
                        true,
                    );
                }

                logs_model.push(LogEntry {
                    time: SharedString::from(now_hms()),
                    level: SharedString::from(level),
                    message: SharedString::from(message),
                });
                if logs_model.row_count() > 200 { logs_model.remove(0); }
            }

            // ── Tray events ──
            #[cfg(target_os = "windows")]
            {
                use tray_icon::menu::MenuEvent;
                if let Ok(event) = MenuEvent::receiver().try_recv() {
                    if event.id == "show" {
                        // Toggle: hide to tray if currently shown, else restore.
                        if hidden_to_tray_timer.get() {
                            ui.show().ok();
                            ui_visible_timer.set(true);
                            hidden_to_tray_timer.set(false);
                            let _ = toggle_item_timer.set_text("Hide Strata");
                            // Restore the exact placement saved at close (maximized state +
                            // monitor). If it landed MAXIMIZED, that resize repaints the
                            // window on its own. Otherwise the restored size may equal the
                            // current size (no resize → no repaint), so we still nudge a
                            // resize to force the full repaint - without it the small window
                            // comes back as white blocks until a mouse-move / resize.
                            let maximized = platform::windows::restore_window_placement(ui_hwnd_timer.get());
                            if !maximized && restore_size.get().is_none() {
                                let sz = ui.window().size();
                                ui.window().set_size(slint::PhysicalSize::new(sz.width, sz.height + 2));
                                restore_size.set(Some(sz));
                            }
                        } else {
                            // Currently shown → hide to tray (mirror the X button).
                            platform::windows::save_window_placement(ui_hwnd_timer.get());
                            ui.hide().ok();
                            ui_visible_timer.set(false);
                            hidden_to_tray_timer.set(true);
                            let _ = toggle_item_timer.set_text("Show Strata");
                        }
                    } else if event.id == "quit" { std::process::exit(0); }
                }
            }
        }
    });

    // ── "Refresh Monitors" button - rebuild wallpaper windows on demand ─────
    // Replaces the old automatic timer-based detection; the user triggers this
    // manually after changing monitor configuration (same idea as Windows'
    // "Identify" button in Display Settings).
    {
        let app_state_ref  = app_state.clone();
        let command_tx_ref = command_tx.clone();
        let context_ref    = context.clone();
        let store_ref      = wallpaper_store.clone();
        let pending_close_ref = pending_close.clone();
        let push_toast_mon = push_toast.clone();

        ui.on_refresh_monitors(move || {
            log::info!("Refresh Monitors requested - rebuilding wallpaper windows");

            // Tear down existing windows.  Send WindowClosed (render threads shut
            // down + release their surfaces), hide each window immediately, and
            // DEFER the Slint-component drop via pending_close so we never destroy
            // an HWND out from under a thread that's still presenting.
            {
                let mut windows = store_ref.borrow_mut();
                for (win, window_id, _) in windows.drain(..) {
                    command_tx_ref.send(EngineCommand::WindowClosed(window_id)).ok();
                    win.hide().ok();
                    pending_close_ref.borrow_mut().push((win, std::time::Instant::now()));
                }
            }

            // Re-discover monitors and update app state, preserving layer
            // assignments for monitors whose IDs haven't changed.
            let new_mons = discover_monitors();
            {
                let mut state = app_state_ref.write().unwrap();
                let cfg = config::Config::load();
                let mut monitors = new_mons;
                for m in &mut monitors {
                    if let Some(mc) = cfg.monitors.iter().find(|mc| mc.id == m.id) {
                        m.layers = mc.layers.iter()
                            .filter(|l| l.wallpaper_path.exists())
                            .cloned()
                            .collect();
                        if !mc.name.is_empty()  { m.name = mc.name.clone(); }
                        if !mc.color.is_empty() { m.color = mc.color.clone(); }
                    }
                    if m.color.is_empty() {
                        m.color = default_monitor_color(&m.id);
                    }
                }
                state.monitors = monitors;
            }

            let mon_count = app_state_ref.read().map(|s| s.monitors.len()).unwrap_or(0);

            // spawn_wallpaper_windows calls reconcile_video_daemon, which moves/recreates
            // or tears down video windows to match the new topology (geometry change ->
            // re-SetVideo; monitor gone -> RemoveVideo).
            spawn_wallpaper_windows(
                app_state_ref.clone(),
                command_tx_ref.clone(),
                context_ref.clone(),
                store_ref.clone(),
            );

            push_toast_mon("Monitors Refreshed", &format!("Detected {} display(s).", mon_count), false);
        });
    }

    let ui_handle_close = ui.as_weak();
    let ui_visible_close = ui_visible.clone();
    #[cfg(target_os = "windows")]
    let ui_hwnd_close = ui_hwnd.clone();
    #[cfg(target_os = "windows")]
    let toggle_item_close = tray_toggle_item.clone();
    #[cfg(target_os = "windows")]
    let hidden_to_tray_close = hidden_to_tray.clone();
    ui.window().on_close_requested(move || {
        if let Some(ui) = ui_handle_close.upgrade() {
            // Snapshot placement (maximized state + monitor) BEFORE hiding, so "Show
            // Strata" can restore the exact geometry - hide()/show() loses it otherwise.
            #[cfg(target_os = "windows")]
            {
                platform::windows::save_window_placement(ui_hwnd_close.get());
                hidden_to_tray_close.set(true);
                let _ = toggle_item_close.set_text("Show Strata");
            }
            ui.hide().ok();
            ui_visible_close.set(false); // hidden to tray - pause unseen work
        }
        slint::CloseRequestResponse::KeepWindowShown
    });

    // When the window becomes visible again after being hidden/minimized/occluded
    // (taskbar restore, waking a dormant window), Windows may have discarded the
    // software framebuffer, leaving stale/transparent regions. Flag a full repaint so
    // the timer nudges a resize. `Occluded(false)` is the un-hide/un-minimize signal.
    #[cfg(target_os = "windows")]
    {
        use slint::winit_030::WinitWindowAccessor;
        use slint::winit_030::winit::event::WindowEvent;
        let ui_visible_evt = ui_visible.clone();
        let move_settle_evt = move_settle.clone();
        let ui_maximized_evt = ui_maximized.clone();
        let needs_repaint_evt = needs_repaint.clone();
        ui.window().on_winit_window_event(move |win, event| {
            // NOTE: this fires for EVERY event (incl. high-frequency CursorMoved), so do
            // only cheap flag-setting here - never per-event syscalls.
            match event {
                WindowEvent::Resized(_) => {
                    // Maximized state only changes via a resize; query the syscall here,
                    // not on every event.
                    ui_maximized_evt.set(win.is_maximized());
                    // A maximize/restore fires Moved (which arms the move-debounce) AND a
                    // Resized. The resize already self-heals the buffer, so cancel any
                    // pending move-nudge - otherwise the nudge would resize (and shift) the
                    // just-maximized window a moment later. This is what was dragging the
                    // maximized window down on the portrait monitor.
                    move_settle_evt.set(None);
                }
                WindowEvent::Occluded(occluded) => {
                    // `Occluded(true)` = minimized/fully hidden; `(false)` = visible again.
                    ui_visible_evt.set(!occluded);
                    if !occluded {
                        // Un-minimize/un-hide: request_redraw alone often leaves the
                        // software UI transparent-but-stale (Slint's partial renderer
                        // only repaints dirty regions, and a click/hover then paints
                        // piecemeal). Flag a full-repaint nudge (timer resizes ±2px →
                        // softbuffer reallocates → guaranteed full repaint).
                        win.request_redraw();
                        needs_repaint_evt.set(true);
                    }
                }
                WindowEvent::Focused(true) => {
                    // Some taskbar minimize/restore cycles signal only via focus (no
                    // Occluded). Only force the full-repaint nudge if we believed the
                    // window was hidden (a genuine restore) - NOT on every focus gain,
                    // which would cause a 2px resize flicker on normal alt-tab/click.
                    win.request_redraw();
                    if !ui_visible_evt.get() {
                        needs_repaint_evt.set(true);
                    }
                    ui_visible_evt.set(true);
                }
                WindowEvent::Moved(_) => {
                    // Debounced in the timer - a move can reveal stale (unpainted) regions.
                    move_settle_evt.set(Some(std::time::Instant::now()));
                }
                _ => {}
            }
            slint::winit_030::EventResult::Propagate
        });
    }

    // Capture the main window's HWND once the winit window exists, for tray placement
    // save/restore (see on_close_requested / the tray "show" handler).
    #[cfg(target_os = "windows")]
    {
        let ui_hwnd_init = ui_hwnd.clone();
        let win_weak = ui.as_weak();
        slint::spawn_local(async move {
            if let Some(ui) = win_weak.upgrade() {
                if let Ok(w) = slint::winit_030::WinitWindowAccessor::winit_window(ui.window()).await {
                    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
                    if let Ok(handle) = w.window_handle() {
                        if let RawWindowHandle::Win32(h) = handle.as_raw() {
                            ui_hwnd_init.set(h.hwnd.get());
                        }
                    }
                }
            }
        }).ok();
    }

    log::info!("UI ready - running Slint event loop.");
    // Use run_event_loop_until_quit (NOT ui.run()) so Strata behaves like a tray
    // app: the loop survives the last window closing. Pressing X on the main window
    // hides it to the tray (see on_close_requested); the process only ends via the
    // UI Quit button or the tray Quit item (both call std::process::exit). Without
    // this, hiding the window when NO wallpaper windows exist would quit the loop.
    // `--minimized` (used by the autostart entry): start hidden in the tray so a
    // PC boot doesn't pop the window open. Wallpapers still render on the desktop;
    // the user opens the UI from the tray when they want it. Otherwise show normally.
    if start_minimized {
        log::info!("Starting minimized to tray (--minimized).");
        ui_visible.set(false);
        #[cfg(target_os = "windows")]
        {
            hidden_to_tray.set(true);
            let _ = tray_toggle_item.set_text("Show Strata");
        }
        // Don't call ui.show() - the loop below stays alive without any window.
    } else {
        ui.show()?;
    }
    slint::run_event_loop_until_quit()?;
    ui.hide().ok();
    Ok(())
}

/// Query GitHub for the Strata repo's latest published release. Returns
/// `Ok(Some((tag, html_url)))` if it's newer than the running build, `Ok(None)`
/// if up to date or no release exists yet (404), or `Err` on a network/parse
/// failure. Runs synchronously - call it off the UI thread.
fn check_github_latest() -> Result<Option<(String, String)>, String> {
    let resp = library_sync::http_agent()
        .get("https://api.github.com/repos/BadassBaboon/Strata/releases/latest")
        .set("User-Agent", "Strata-Updater")
        .set("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(10))
        .call();
    match resp {
        Ok(r) => {
            // Parse the body ourselves (no dependency on ureq's optional json feature).
            let body = r.into_string().map_err(|e| e.to_string())?;
            let json: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
            let tag = json.get("tag_name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let url = json.get("html_url").and_then(|v| v.as_str())
                .unwrap_or("https://github.com/BadassBaboon/Strata/releases").to_string();
            if !tag.is_empty() && is_newer_version(&tag, env!("CARGO_PKG_VERSION")) {
                Ok(Some((tag, url)))
            } else {
                Ok(None)
            }
        }
        // No releases published yet → not an error, just "up to date".
        Err(ureq::Error::Status(404, _)) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

/// Check the official content repository for a newer asset-library version by
/// querying its `library-v*` git tags (see `library_sync::latest_library`). Returns
/// `Ok(Some(version))` if the latest tag is newer than `current`, else `Ok(None)`.
/// Run off the UI thread.
fn check_library_update(current: &str) -> Result<Option<String>, String> {
    let (_owner, _repo, version, _tag) = library_sync::latest_library()?;
    if library_sync::is_newer(&version, current) {
        Ok(Some(version))
    } else {
        Ok(None)
    }
}

/// Parse a `vMAJOR.MINOR.PATCH`(-ish) string into a comparable tuple.
fn parse_semver(s: &str) -> (u32, u32, u32) {
    let s = s.trim().trim_start_matches(['v', 'V']);
    let mut it = s.split(['.', '-', '+', ' ']).filter_map(|p| p.parse::<u32>().ok());
    (it.next().unwrap_or(0), it.next().unwrap_or(0), it.next().unwrap_or(0))
}

/// True if release `tag` is a newer version than the running `current`.
fn is_newer_version(tag: &str, current: &str) -> bool {
    parse_semver(tag) > parse_semver(current)
}

/// Persist the full config from current app state + FPS cap. Used by the
/// debounced timer flush and the quit flush.
/// Ask Windows to trim the process working set back to what's actually live,
/// returning peak/reclaimable pages (e.g. from a thumbnail-generation burst) to
/// the OS. Pages re-fault on demand, so this only sheds genuinely-idle memory.
#[cfg(windows)]
fn trim_working_set() {
    use windows_sys::Win32::System::ProcessStatus::EmptyWorkingSet;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe { EmptyWorkingSet(GetCurrentProcess()); }
}
#[cfg(not(windows))]
fn trim_working_set() {}

/// Write a library thumbnail straight from a source image (cover-crop to w×h). Used
/// for parallax packages so the thumbnail is the user's actual photo, not a render.
fn thumbnail_from_source(src: &std::path::Path, out: &std::path::Path, w: u32, h: u32) -> Result<(), String> {
    let img = image::open(src).map_err(|e| format!("open {:?}: {e}", src))?;
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    img.resize_to_fill(w, h, image::imageops::FilterType::Lanczos3)
        .to_rgb8()
        .save(out)
        .map_err(|e| format!("save thumbnail {:?}: {e}", out))
}

/// Background-generate cache thumbnails for any wallpaper folder that lacks one.
///
/// Runs on its own thread with its OWN, dedicated `GraphicsContext` (device) that
/// is dropped when generation finishes. This is the key to keeping the app
/// lightweight: compiling a whole library of shaders piles up hundreds of MB of
/// driver memory, and that memory is only returned to the OS when the *device* is
/// destroyed - so we never let it accumulate on the app's long-lived device.
/// (Measured: shared-device generation left ~1.3 GB committed; a dedicated device
/// dropped afterward settles back to ~18 MB.) Each finished thumbnail is sent over
/// `tx` for the UI timer to load; `busy` drives the refresh spinner.
fn spawn_thumbnail_generation(
    wallpaper_dirs: Vec<std::path::PathBuf>,
    tx: std::sync::mpsc::Sender<(std::path::PathBuf, std::path::PathBuf)>,
    busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering;
    // Only generate for folders that don't already have an in-folder thumbnail.png.
    let missing: Vec<std::path::PathBuf> = wallpaper_dirs.into_iter()
        .filter(|d| !controller::thumbnail_path(d).exists())
        .collect();
    if missing.is_empty() {
        return;
    }
    // Guard against a second concurrent generator (e.g. a rapid second refresh).
    if busy.swap(true, Ordering::SeqCst) {
        return;
    }
    let busy_guard = busy.clone();
    let spawned = std::thread::Builder::new().name("strata-thumbnails".into()).spawn(move || {
        log::info!("Generating {} missing thumbnail(s)…", missing.len());
        // Drive each headless GPU context on a tiny single-threaded runtime.
        let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
            Ok(rt) => rt,
            Err(e) => { log::error!("Thumbnail runtime init failed: {}", e); busy.store(false, Ordering::SeqCst); return; }
        };
        // A fresh, audio-free GPU context. The driver only frees a device's
        // accumulated shader-compilation memory when the device is destroyed, so we
        // recreate the context every few shaders to keep the transient footprint
        // bounded instead of letting it climb across the whole library.
        let make_ctx = || rt.block_on(core_engine::GraphicsContext::new_render_only()).ok().map(std::sync::Arc::new);
        const CTX_BATCH: usize = 6;

        let mut ctx = match make_ctx() {
            Some(c) => Some(c),
            None => { log::error!("Thumbnail GPU context init failed"); busy.store(false, Ordering::SeqCst); return; }
        };
        for (i, wp) in missing.iter().enumerate() {
            if i > 0 && i % CTX_BATCH == 0 {
                ctx = None; // drop the old device FIRST so its memory is reclaimed…
                match make_ctx() {
                    Some(c) => ctx = Some(c), // …then stand up a fresh one.
                    None => { log::error!("Thumbnail GPU context re-init failed"); break; }
                }
            }
            let Some(ctx) = ctx.as_ref() else { break };
            let out = controller::thumbnail_path(wp);
            // Parallax packages (image.png + depth.png) thumbnail straight from the
            // user's source photo - no shader render - so the library shows the
            // actual picture, and it's cheaper than a GPU pass.
            let src = wp.join("image.png");
            let result = if src.exists() && wp.join("depth.png").exists() {
                thumbnail_from_source(&src, &out, 480, 270)
            } else {
                core_engine::thumbnail::generate_thumbnail(ctx.clone(), wp, &out, 480, 270)
                    .map_err(|e| e.to_string())
            };
            match result {
                Ok(()) => { let _ = tx.send((wp.clone(), out)); }
                Err(e) => log::warn!("Thumbnail failed for {:?}: {}", wp.file_name().unwrap_or_default(), e),
            }
            // Throttle: a gentle background task - keeps CPU low and lets the GPU
            // serve the live wallpapers between shaders.
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        drop(ctx); // destroy the final device → driver returns the compilation memory.
        busy.store(false, Ordering::SeqCst);
        // Return the peak working set (shader compilation + render scratch) to the
        // OS so the process settles back to its live footprint.
        trim_working_set();
        log::info!("Thumbnail generation complete");
    });
    if spawned.is_err() {
        // Thread never started - release the guard or the spinner runs forever.
        log::error!("Thumbnail thread spawn failed");
        busy_guard.store(false, Ordering::SeqCst);
    }
}

/// The models the CURRENT Parallax selection needs (always cinematic). Automatic mode →
/// the chosen preset's bundle; Manual mode → the individual dropdown selections. Read
/// straight from the UI's `*_current` properties so it reflects live state.
fn parallax_required_models(ui: &AppWindow) -> Vec<core_engine::depth::ModelChoice> {
    if ui.get_parallax_mode() == "automatic" {
        let p = parallax::preset_for_label(&ui.get_parallax_preset_current());
        parallax::preset_required_models(p)
    } else {
        let mut v = Vec::new();
        if let Some(m) = parallax::tier_for_label(&ui.get_parallax_model_current()) { v.push(m); }
        v.push(parallax::seg_choice_for_label(&ui.get_parallax_seg_current()));
        v.push(core_engine::depth::lama_model());
        if let Some(u) = parallax::upscaler_choice_for_label(&ui.get_parallax_upscaler_current()) { v.push(u); }
        v
    }
}

/// Recompute whether the current selection has any un-downloaded models and update the
/// "Download required models" button visibility.
fn refresh_parallax_download_state(ui: &AppWindow) {
    if !parallax::onnx_available() {
        ui.set_parallax_models_need_download(false);
        return;
    }
    let missing = parallax::missing_models(&parallax_required_models(ui));
    ui.set_parallax_models_need_download(!missing.is_empty());
}

fn flush_config(state: &AppState, target_fps: u32, audio_sensitivity: f32, mouse_mode: &str, mouse_sensitivity: f32, quality_scale: f32) {
    let mut config = config::Config::load();
    config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
    config.target_fps = target_fps;
    config.audio_sensitivity = audio_sensitivity;
    config.mouse_mode = mouse_mode.to_string();
    config.mouse_sensitivity = mouse_sensitivity;
    config.shader_quality = scale_to_shader_quality(quality_scale).to_string();
    config.save().ok();
}

/// Mouse-interactivity mode label → engine code (0=Off, 1=All, 2=Only shaders,
/// 3=Only Parallax). Unknown labels default to Only Parallax.
fn mouse_mode_to_u8(label: &str) -> u8 {
    match label {
        "Off" => 0,
        "On (Everything)" => 1,
        "On (Only Shaders)" => 2,
        _ => 3, // "On (Only Parallax Studio)"
    }
}

/// Inverse of `mouse_mode_to_u8`, for persisting the mode label to config.
fn u8_to_mouse_mode(mode: u8) -> &'static str {
    match mode {
        0 => "Off",
        1 => "On (Everything)",
        2 => "On (Only Shaders)",
        _ => "On (Only Parallax Studio)",
    }
}

/// Map a Shader Quality preset label (from the Settings dropdown) to a global
/// render scale. Unknown/legacy labels fall back to full quality.
fn shader_quality_to_scale(label: &str) -> f32 {
    if label.starts_with("Low") {
        0.5
    } else if label.starts_with("Medium") {
        0.75
    } else {
        1.0
    }
}

/// Inverse of [`shader_quality_to_scale`] - the canonical label for a stored scale.
fn scale_to_shader_quality(scale: f32) -> &'static str {
    if scale < 0.625 {
        "Low (Performance Mode)"
    } else if scale < 0.875 {
        "Medium (Balanced)"
    } else {
        "High (Maximum Fidelity)"
    }
}

/// Default "hardware color tag" for a monitor, cycled by its index so each
/// display reads as distinct in the Compositor canvas until the user picks one.
fn default_monitor_color(id: &str) -> String {
    let idx = id.rsplit('-').next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
    ["orange", "blue", "purple", "emerald", "rose"][idx % 5].to_string()
}

/// Wall-clock time-of-day as HH:MM:SS (UTC) for diagnostics log timestamps.
/// Uses std only - avoids pulling in a date/time crate for a cosmetic stamp.
fn now_hms() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

#[cfg(target_os = "windows")]
/// Resolve a bundled asset by name, anchored to the EXECUTABLE's directory - never
/// the current working directory. This matters for autostart: the Run-key launch at
/// boot has CWD = System32 (not the install dir), so a CWD-relative `assets/…` lookup
/// failed and the tray icon fell back to a solid square. Tries the installed layout
/// (assets next to the exe) then the dev layout (repo `assets/` two levels up from
/// target/release), with a final CWD-relative fallback.
fn asset_path(name: &str) -> std::path::PathBuf {
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        let installed = dir.join("assets").join(name);
        if installed.exists() { return installed; }
        let dev = dir.join("..").join("..").join("assets").join(name);
        if dev.exists() { return dev; }
    }
    let cwd = std::path::Path::new("assets").join(name);
    if cwd.exists() { cwd } else { std::path::Path::new("../../assets").join(name) }
}

fn load_tray_icon(is_dark: bool) -> tray_icon::Icon {
    let name = if is_dark { "app-icon_dark.png" } else { "app-icon_light.png" };
    let path = asset_path(name);
    // Load the real icon, but NEVER panic: a missing/corrupt asset falls back to a
    // small solid accent square so the tray still appears and the app starts.
    let (icon_rgba, icon_width, icon_height) = match image::open(&path) {
        Ok(img) => {
            let img = img.into_rgba8();
            let (w, h) = img.dimensions();
            (img.into_raw(), w, h)
        }
        Err(e) => {
            log::warn!("Tray icon {:?} unavailable ({}); using fallback square.", path, e);
            let mut px = Vec::with_capacity(16 * 16 * 4);
            for _ in 0..16 * 16 { px.extend_from_slice(&[0x4f, 0x8a, 0xf7, 0xff]); }
            (px, 16u32, 16u32)
        }
    };
    tray_icon::Icon::from_rgba(icon_rgba, icon_width, icon_height).unwrap_or_else(|e| {
        log::error!("Tray icon build failed ({}); using 1x1 fallback.", e);
        tray_icon::Icon::from_rgba(vec![0, 0, 0, 0], 1, 1).expect("1x1 transparent icon")
    })
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to a wallpaper manifest.toml to run in CLI mode
    #[arg(long)]
    cli: Option<String>,

    /// Enable verbose logging
    #[arg(long, alias = "logs")]
    logs: bool,

    /// Start hidden in the system tray (used by the autostart entry so booting
    /// the PC doesn't pop the window open).
    #[arg(long)]
    minimized: bool,

    /// Internal: write the MPO (OverlayTestMode) registry value and exit. Invoked
    /// elevated (via the Settings toggle) since it writes HKLM. "on" disables MPO,
    /// "off" re-enables it.
    #[arg(long)]
    set_mpo: Option<String>,

    /// Dev harness: play a single .mp4 in a window (validates the video decoder).
    #[arg(long)]
    video: Option<String>,

    /// Dev harness: play a single .mp4 via a WebView <video> (validates the WebView path).
    #[arg(long)]
    video_web: Option<String>,

    /// Internal: run the video-wallpaper daemon (hosts all movie WebViews in a child
    /// process so WebView2 never loads into the main app). Driven over stdin; spawned and
    /// killed automatically - not for manual use.
    #[arg(long)]
    video_daemon: bool,
}
