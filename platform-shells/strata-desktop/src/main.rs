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

use platform::EngineCommand;
use controller::{AppState, scan_wallpapers, import_wallpaper_zip, discover_monitors, LayerInfo};

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
    let dir = directories::ProjectDirs::from("com", "strata", "engine")?
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
    // wallpaper windows' wgpu swapchains — when desktop icons are hidden, the UI's GPU
    // present blacks out the primary monitor's wallpaper. The wallpaper content is drawn
    // entirely by our own wgpu surface, so the UI doesn't need the GPU.
    //
    // We tried (2026-06-16) rendering the UI on the SAME wgpu device as the wallpapers to
    // get GPU UI without the conflict; it failed — two swapchains on one device clash at
    // DXGI swapchain creation ("Access is denied"), because the wallpaper swapchain is
    // created cross-thread on a window the shared DXGI factory also drives. See
    // [[gpu-ui-prototype]] in memory. Software is the proven path; it needs the size-nudge
    // full-repaint workarounds (`needs_repaint` / `move_settle`) as it only presents the
    // damaged region.
    #[cfg(windows)]
    if std::env::var_os("SLINT_BACKEND").is_none() {
        std::env::set_var("SLINT_BACKEND", "winit-software");
    }

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

    if args.logs {
        std::env::set_var("RUST_LOG", "info");
    }
    
    #[cfg(target_os = "windows")]
    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::SetProcessDPIAware();
    }

    if let Some(wallpaper_path) = args.cli {
        env_logger::init();
        run_cli_mode(wallpaper_path)
    } else {
        run_ui_mode()
    }
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

// Tracks every live wallpaper window so they can be closed and recreated when
// the monitor configuration changes.  The third tuple element is the monitor
// id, so we can map a monitor → its window for per-monitor teardown.  Rc is
// fine — accessed only on the Slint main thread (spawn_local, timer callbacks).
type WallpaperWindowStore =
    std::rc::Rc<std::cell::RefCell<Vec<(WallpaperWindow, winit::window::WindowId, String)>>>;

// Wallpaper windows queued for destruction.  When a monitor loses its last
// shader we send WindowClosed (so the render thread shuts down and releases its
// surface) but DEFER dropping the Slint component — dropping it destroys the
// HWND, and doing that out from under a still-running render thread would race
// surface.get_current_texture() against a dead window.  The UI timer drops these
// a few hundred ms later, by which point the render thread has exited.
type PendingCloseStore =
    std::rc::Rc<std::cell::RefCell<Vec<(WallpaperWindow, std::time::Instant)>>>;

/// Create one wallpaper window per monitor and reparent each to the desktop
/// WorkerW.  Stores the resulting Slint components in `store` so they can be
/// closed later.  Safe to call multiple times (e.g. after a monitor change);
/// old windows must be drained from `store` and their `WindowClosed` commands
/// sent before calling again.
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
        // slice of one unified canvas.
        let span_layers: Vec<LayerInfo> = controller::primary_monitor(&monitors)
            .map(|m| m.layers.clone())
            .unwrap_or_default();

        // Idempotent: skip monitors that already have a window in the store, so
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
        // WorkerW is a global shell window — it appears on ALL virtual desktops
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

            // Effective layers: span mode shows the primary display's shader on
            // every monitor (each as its own slice of the canvas); otherwise each
            // monitor shows its own assigned layers.
            let layers = if span_monitors {
                span_layers.clone()
            } else {
                m.layers.clone()
            };

            // Release-desktop rule: a monitor with no shader gets NO window at
            // all, so the user's real Windows wallpaper shows through and no
            // swapchain VRAM / render thread is spent on it.
            if layers.is_empty() {
                continue;
            }

            // Idempotent: this monitor already has a live window — leave it be.
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
                // current async task yields — they would overwrite the surface
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

                // Store window handle — surface creation happens below after Win32 setup.
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
                    log::warn!("WorkerW not found — wallpaper at ({},{}) above icons", screen_x, screen_y);
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
                // Window creation failed — keep the component alive to avoid
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
    // Which monitors should currently own a window?  Mirror the same
    // effective-layers rule used when spawning: in span mode every monitor wants
    // a window iff the PRIMARY display has a shader (they all show its canvas
    // slice); otherwise a monitor wants a window iff it has its own shader.
    let want_window: Vec<String> = {
        let state = app_state.read().unwrap();
        let span = state.span_monitors;
        let primary_has_layers = controller::primary_monitor(&state.monitors)
            .map(|m| !m.layers.is_empty())
            .unwrap_or(false);
        state
            .monitors
            .iter()
            .filter(|m| {
                if span { primary_has_layers } else { !m.layers.is_empty() }
            })
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

fn run_ui_mode() -> Result<(), Box<dyn std::error::Error>> {
    let app_state = Arc::new(std::sync::RwLock::new(AppState::default()));
    let (command_tx, command_rx) = channel::<EngineCommand>();
    let running = Arc::new(AtomicBool::new(true));

    // Create ONE shared graphics context. Both the main thread (surface creation) and the
    // renderer thread (adapter / device / queue) must use the same wgpu::Instance — if they
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
    let config = config::Config::load();

    // Shared, live-tunable frame-rate cap read by every monitor render loop.
    // The Settings slider stores into this; render threads pick it up next frame.
    let target_fps = Arc::new(std::sync::atomic::AtomicU32::new(config.target_fps.clamp(1, 240)));

    // Audio sensitivity: authoritative value lives in the AudioEngine (read by the
    // render threads); this Cell mirrors it for the debounced config flush.
    let audio_sensitivity = std::rc::Rc::new(std::cell::Cell::new(config.audio_sensitivity));
    if let Some(a) = &context.audio { a.set_sensitivity(config.audio_sensitivity); }

    // Mouse interactivity: shared atomics read live by every monitor render loop
    // (feeds the desktop cursor into shaders' iMouse). f32 sensitivity is bit-cast.
    let mouse_enabled = Arc::new(std::sync::atomic::AtomicBool::new(config.mouse_interactive));
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
    ui.set_fps_cap(config.target_fps.clamp(1, 240) as i32);
    ui.set_audio_sensitivity(config.audio_sensitivity);
    ui.set_mouse_interactive(config.mouse_interactive);
    ui.set_mouse_sensitivity(config.mouse_sensitivity);
    ui.set_shader_quality(SharedString::from(&config.shader_quality));

    // Parallax Studio: populate the depth-model dropdown (heuristic + tiers).
    ui.set_parallax_model_options(ModelRc::from(Rc::new(VecModel::from(
        parallax::model_options().iter().map(SharedString::from).collect::<Vec<_>>(),
    ))));
    ui.set_parallax_model_current(SharedString::from("Fast (no model)"));
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

    // Set up Wallpaper Library
    let wallpapers_base = if std::path::Path::new("wallpapers").exists() {
        std::path::Path::new("wallpapers")
    } else {
        std::path::Path::new("../../wallpapers")
    };
    
    let wallpaper_entries = scan_wallpapers(wallpapers_base);
    {
        let mut state = app_state.write().unwrap();
        state.wallpapers = wallpaper_entries.clone();
    }

    // Snapshot monitors so each card shows its applied-monitor avatars / active
    // glow from the restored config on first paint.
    let monitor_snapshot = { app_state.read().unwrap().monitors.clone() };
    let wallpapers_model = Rc::new(VecModel::<WallpaperItem>::from(
        wallpaper_entries.iter().map(|w| {
            let mut item = WallpaperItem::default();
            item.name = SharedString::from(&w.name);
            item.author = SharedString::from(&w.author);
            item.tags = ModelRc::from(Rc::new(VecModel::from(w.tags.iter().map(|t| SharedString::from(t.as_str())).collect::<Vec<_>>())));
            item.is_parallax = w.tags.iter().any(|t| t.eq_ignore_ascii_case("Parallax"));
            item.visible = true;
            if let Some(ref thumb) = w.thumbnail {
                if let Ok(slint_img) = Image::load_from_path(thumb) {
                    item.thumbnail = slint_img;
                    item.has_thumbnail = true;
                }
            }
            let counts: Vec<i32> = monitor_snapshot.iter()
                .map(|m| m.layers.iter().filter(|l| l.wallpaper_path == w.path).count() as i32)
                .collect();
            item.is_active = counts.iter().any(|&c| c > 0);
            item.usage_counts = Rc::new(VecModel::from(counts)).into();
            item
        }).collect::<Vec<_>>()
    ));
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

    // Close (✕) on a toast — drop it from both the model and the expiry list.
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
    let parallax_model: Rc<std::cell::RefCell<String>> = Rc::new(std::cell::RefCell::new("Fast (no model)".to_string()));
    let parallax_downloading = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let download_pct = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let (download_tx, download_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    // Cinematic (layered/LDI) mode + which download is in flight (depth vs LaMa),
    // so the timer routes the progress text to the right status line.
    let parallax_cinematic = Rc::new(std::cell::Cell::new(false));
    let download_is_lama = Rc::new(std::cell::Cell::new(false));
    // Selected masking (segmentation) model label for cinematic mode.
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
    // NOTE: thumbnail generation is deliberately NOT kicked off at startup — we
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

    let app_state_import = app_state.clone();
    let wallpapers_model_import = wallpapers_model.clone();
    let push_toast_import = push_toast.clone();
    let thumb_tx_import = thumb_tx.clone();
    let thumbnails_busy_import = thumbnails_busy.clone();
    ui.on_import_requested(move || {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Wallpaper ZIP", &["zip"])
            .pick_file() 
        {
            let dest_base = if std::path::Path::new("wallpapers").exists() {
                std::path::Path::new("wallpapers")
            } else {
                std::path::Path::new("../../wallpapers")
            };

            match import_wallpaper_zip(&path, dest_base) {
                Ok(_) => {
                    let mut state = app_state_import.write().unwrap();
                    let new_wallpapers = scan_wallpapers(dest_base);
                    state.wallpapers = new_wallpapers.clone();
                    let monitors = state.monitors.clone();

                    let slint_walls: Vec<WallpaperItem> = new_wallpapers.iter().map(|w| {
                        let mut item = WallpaperItem::default();
                        item.name = SharedString::from(&w.name);
                        item.author = SharedString::from(&w.author);
                        item.tags = ModelRc::from(Rc::new(VecModel::from(w.tags.iter().map(|t| SharedString::from(t.as_str())).collect::<Vec<_>>())));
                        item.is_parallax = w.tags.iter().any(|t| t.eq_ignore_ascii_case("Parallax"));
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
                    }).collect();
                    let dirs: Vec<std::path::PathBuf> = new_wallpapers.iter().map(|w| w.path.clone()).collect();
                    drop(state);
                    wallpapers_model_import.set_vec(slint_walls);
                    log::info!("Imported wallpaper from {:?}", path.file_name().unwrap());
                    push_toast_import("Shader Imported", "New shader pack added to your library.", false);
                    // Generate thumbnails for the (new) wallpapers in the background.
                    spawn_thumbnail_generation(dirs, thumb_tx_import.clone(), thumbnails_busy_import.clone());
                }
                Err(e) => {
                    log::error!("Import error: {}", e);
                    push_toast_import("Import Failed", "The selected file could not be imported.", true);
                }
            }
        }
    });

    // Rescan the wallpaper folder and rebuild the library model (preserving each
    // card's applied-monitor state). Triggered by the Library toolbar refresh.
    let app_state_refresh = app_state.clone();
    let wallpapers_model_refresh = wallpapers_model.clone();
    let push_toast_refresh = push_toast.clone();
    let thumb_tx_refresh = thumb_tx.clone();
    let thumbnails_busy_refresh = thumbnails_busy.clone();
    ui.on_refresh_library(move || {
        // Already refreshing? Don't rescan or kick off a second generator — just
        // tell the user. This stops rapid clicks from pinning the CPU.
        if thumbnails_busy_refresh.load(std::sync::atomic::Ordering::SeqCst) {
            push_toast_refresh("Refresh In Progress", "Thumbnails are still generating — please wait.", false);
            return;
        }
        let base = if std::path::Path::new("wallpapers").exists() {
            std::path::Path::new("wallpapers")
        } else {
            std::path::Path::new("../../wallpapers")
        };
        let mut state = app_state_refresh.write().unwrap();
        let scanned = scan_wallpapers(base);
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
            let walls: Vec<WallpaperItem> = scanned.iter().map(|w| {
                let mut item = WallpaperItem::default();
                item.name = SharedString::from(&w.name);
                item.author = SharedString::from(&w.author);
                item.tags = ModelRc::from(Rc::new(VecModel::from(w.tags.iter().map(|t| SharedString::from(t.as_str())).collect::<Vec<_>>())));
                item.is_parallax = w.tags.iter().any(|t| t.eq_ignore_ascii_case("Parallax"));
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
            }).collect();
            drop(state);
            wallpapers_model_refresh.set_vec(walls);
            log::info!("Library refreshed: {} wallpaper(s)", count);
            push_toast_refresh("Library Updated", &format!("Shader manifests synchronized — {} found.", count), false);
        } else {
            drop(state);
            push_toast_refresh("Library Up To Date", &format!("{} shaders — no changes found.", count), false);
        }
        // Generate any missing thumbnails in the background (spinner shows progress).
        // No-op if every shader already has one.
        spawn_thumbnail_generation(dirs, thumb_tx_refresh.clone(), thumbnails_busy_refresh.clone());
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
            // Safety: only delete inside the wallpapers library, never elsewhere.
            let base = parallax::wallpapers_base();
            let inside = path.canonicalize().ok()
                .zip(base.canonicalize().ok())
                .map(|(p, b)| p.starts_with(&b))
                .unwrap_or(false);
            if !inside {
                log::warn!("Refusing to delete {:?} — outside the wallpaper library", path);
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

    // ── Parallax Studio: choose a depth model ───────────────────────────────
    {
        let ui_mc = ui.as_weak();
        let model_sel = parallax_model.clone();
        ui.on_parallax_model_changed(move |label| {
            *model_sel.borrow_mut() = label.to_string();
            if let Some(ui) = ui_mc.upgrade() {
                ui.set_parallax_model_current(label.clone());
                refresh_parallax_model_status(&ui, &label);
            }
        });
    }

    // ── Parallax Studio: download the selected depth model (background) ──────
    {
        let ui_dl = ui.as_weak();
        let model_dl = parallax_model.clone();
        let downloading = parallax_downloading.clone();
        let pct = download_pct.clone();
        let dtx = download_tx.clone();
        let dl_is_lama = download_is_lama.clone();
        ui.on_parallax_download_model(move || {
            let Some(ui) = ui_dl.upgrade() else { return };
            let label = model_dl.borrow().clone();
            let Some(choice) = parallax::tier_for_label(&label) else { return };
            if !parallax::onnx_available() {
                ui.set_parallax_model_status(SharedString::from("This build can't run models (no ONNX)."));
                return;
            }
            if downloading.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            dl_is_lama.set(false); // route progress text to the model status line
            ui.set_parallax_downloading(true);
            pct.store(0, std::sync::atomic::Ordering::SeqCst);
            let pct_t = pct.clone();
            let dtx_t = dtx.clone();
            let downloading_t = downloading.clone();
            std::thread::Builder::new().name("strata-model-dl".into()).spawn(move || {
                let res = parallax::download_model(&choice, |done, total| {
                    let p = if total > 0 { (done * 100 / total) as u32 } else { 0 };
                    pct_t.store(p, std::sync::atomic::Ordering::SeqCst);
                }).map(|_| label).map_err(|e| e);
                downloading_t.store(false, std::sync::atomic::Ordering::SeqCst);
                let _ = dtx_t.send(res);
            }).ok();
        });
    }

    // ── Parallax Studio: toggle Cinematic (layered) mode ────────────────────
    {
        let ui_cin = ui.as_weak();
        let cinematic = parallax_cinematic.clone();
        let seg_cin = parallax_seg.clone();
        ui.on_parallax_cinematic_toggled(move |on| {
            cinematic.set(on);
            if let Some(ui) = ui_cin.upgrade() {
                ui.set_parallax_cinematic(on);
                if on { refresh_parallax_lama_status(&ui, &seg_cin.borrow()); }
            }
        });
    }

    // ── Parallax Studio: choose the masking (segmentation) model ─────────────
    {
        let ui_seg = ui.as_weak();
        let seg_sel = parallax_seg.clone();
        ui.on_parallax_seg_changed(move |label| {
            *seg_sel.borrow_mut() = label.to_string();
            if let Some(ui) = ui_seg.upgrade() {
                ui.set_parallax_seg_current(label.clone());
                refresh_parallax_lama_status(&ui, &label);
            }
        });
    }

    // ── Parallax Studio: choose the parallax style (Coherent 3D vs Billboard) ─
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
    {
        let ui_up = ui.as_weak();
        let up_sel = parallax_upscaler.clone();
        ui.on_parallax_upscaler_changed(move |label| {
            *up_sel.borrow_mut() = label.to_string();
            if let Some(ui) = ui_up.upgrade() {
                ui.set_parallax_upscaler_current(label.clone());
            }
        });
    }

    // ── Parallax Studio: download the LaMa inpainting model (background) ─────
    {
        let ui_lama = ui.as_weak();
        let downloading = parallax_downloading.clone();
        let pct = download_pct.clone();
        let dtx = download_tx.clone();
        let dl_is_lama = download_is_lama.clone();
        let seg_lama = parallax_seg.clone();
        ui.on_parallax_download_lama(move || {
            let Some(ui) = ui_lama.upgrade() else { return };
            if !parallax::onnx_available() {
                ui.set_parallax_lama_status(SharedString::from("This build can't run models (no ONNX)."));
                return;
            }
            if downloading.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            dl_is_lama.set(true); // route progress text to the cinematic status line
            ui.set_parallax_downloading(true);
            pct.store(0, std::sync::atomic::Ordering::SeqCst);
            let pct_t = pct.clone();
            let dtx_t = dtx.clone();
            let downloading_t = downloading.clone();
            let seg = parallax::seg_choice_for_label(&seg_lama.borrow());
            std::thread::Builder::new().name("strata-cinematic-dl".into()).spawn(move || {
                // Cinematic needs the chosen subject matter + LaMa (inpaint).
                let res = parallax::download_cinematic(&seg, |pct| {
                    pct_t.store(pct, std::sync::atomic::Ordering::SeqCst);
                }).map(|_| "inpaint".to_string()).map_err(|e| e);
                downloading_t.store(false, std::sync::atomic::Ordering::SeqCst);
                let _ = dtx_t.send(res);
            }).ok();
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
        let parallax_cinematic_render = parallax_cinematic.clone();
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
            let cinematic = parallax_cinematic_render.get();
            ui.set_parallax_status(SharedString::from(if cinematic {
                "Generating depth, inpainting background…"
            } else {
                "Generating depth…"
            }));
            parallax_progress_render.store(0, std::sync::atomic::Ordering::SeqCst);
            ui.set_parallax_progress(0);

            let params = parallax_params_render.get();
            // Selected model → ModelChoice (None = heuristic). build_preview falls
            // back to heuristic if the model isn't built-in or isn't downloaded.
            let model = parallax::tier_for_label(&parallax_model_render.borrow());
            let seg = parallax::seg_choice_for_label(&parallax_seg_render.borrow());
            let billboard = parallax::style_is_billboard(&parallax_style_render.borrow());
            let upscaler = parallax::upscaler_choice_for_label(&parallax_upscaler_render.borrow());
            let tx = preview_ready_render.clone();
            let busy = parallax_busy_render.clone();
            let prog = parallax_progress_render.clone();
            std::thread::Builder::new().name("strata-parallax-preview".into()).spawn(move || {
                let res = parallax::build_preview(&path, &params, model.as_ref(), &seg, cinematic, billboard, upscaler.as_ref(), |p| {
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
        ui.on_parallax_save(move || {
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
            let name = parallax_image_save.borrow().as_ref()
                .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
                .unwrap_or_else(|| "Parallax".into());
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

    let app_state_apply = app_state.clone();
    let command_tx_apply = command_tx.clone();
    let layers_model_apply = layers_model.clone();
    let wallpapers_model_apply = wallpapers_model.clone();
    let context_apply = context.clone();
    let store_apply = wallpaper_store.clone();
    let pending_close_apply = pending_close.clone();
    let push_toast_apply = push_toast.clone();
    let ui_handle_apply = ui.as_weak();
    ui.on_apply_to_monitor(move |mon_idx, wall_name| {
        // Applying a shader makes that monitor the Compositor's active selection,
        // so switching tabs shows it selected (and its layers) without an extra
        // click — keeps the Library and Compositor in sync.
        if let Some(ui) = ui_handle_apply.upgrade() {
            ui.set_selected_monitor_index(mon_idx);
        }
        // `changed` flips true only if we actually mutated the assignment, so we
        // know whether to reconcile windows (which could add/remove a monitor's
        // first/last shader).
        let mut changed = false;
        {
        let mut state = app_state_apply.write().unwrap();
        let span_now = state.span_monitors;
        let wallpaper_info = state.wallpapers.iter().find(|w| wall_name == w.name).map(|w| (w.path.clone(), w.name.clone()));

        if let Some((path, name)) = wallpaper_info {
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
                    // Insert at the TOP of the list (index 0) so the shader the
                    // user just activated composites on top and is immediately
                    // visible — they can move it down later to reorder.
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
        // can create or destroy that monitor's window — reconcile rather than a
        // plain Reload.  sync_wallpaper_windows re-reads app_state, so the guard
        // above MUST be released first.
        if changed {
            sync_wallpaper_windows(
                app_state_apply.clone(),
                command_tx_apply.clone(),
                context_apply.clone(),
                store_apply.clone(),
                pending_close_apply.clone(),
            );
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
        // reconcile — and only after the write guard is released, since
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

        // Lightweight live update — no pipeline rebuild / shader recompile.
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
            // Per-keystroke from the rename field — debounce the disk write.
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
            if let Some(layer) = monitor.layers.get_mut(layer_index as usize) {
                layer.positioning = mode.to_string();
                if let Some(mut slint_layer) = layers_model_pos.row_data(layer_index as usize) {
                    slint_layer.positioning = mode;
                    layers_model_pos.set_row_data(layer_index as usize, slint_layer);
                }
                
                let mut config = config::Config::load();
                config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
                config.save().ok();
                
                command_tx_pos.send(EngineCommand::Reload).ok();
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
        } // drop the write guard BEFORE invoke_refresh_monitors — that callback
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

    let command_tx_debug = command_tx.clone();
    let app_state_debug = app_state.clone();
    let context_debug = context.clone();
    let command_tx_shutdown = command_tx.clone();
    let config_dirty_quit = config_dirty.clone();
    let app_state_quit = app_state.clone();
    let target_fps_quit = target_fps.clone();
    let audio_sensitivity_quit = audio_sensitivity.clone();
    let mouse_enabled_quit = mouse_enabled.clone();
    let mouse_sensitivity_quit = mouse_sensitivity.clone();
    let quality_scale_quit = quality_scale.clone();
    ui.on_quit_requested(move || {
        // Flush any pending debounced config change before exiting abruptly.
        if config_dirty_quit.get() {
            if let Ok(st) = app_state_quit.read() {
                flush_config(&st, target_fps_quit.load(std::sync::atomic::Ordering::Relaxed), audio_sensitivity_quit.get(),
                    mouse_enabled_quit.load(std::sync::atomic::Ordering::Relaxed),
                    f32::from_bits(mouse_sensitivity_quit.load(std::sync::atomic::Ordering::Relaxed)),
                    f32::from_bits(quality_scale_quit.load(std::sync::atomic::Ordering::Relaxed)));
            }
        }
        let _ = command_tx_shutdown.send(EngineCommand::Shutdown);
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

    // FPS cap slider — store into the shared atomic (render threads pick it up
    // next frame); persistence is debounced via the dirty flag.
    let target_fps_ui = target_fps.clone();
    let config_dirty_fps = config_dirty.clone();
    ui.on_fps_cap_changed(move |fps| {
        let clamped = (fps.clamp(1, 240)) as u32;
        target_fps_ui.store(clamped, std::sync::atomic::Ordering::Relaxed);
        config_dirty_fps.set(true);
    });

    // Shader Quality preset — maps the label to a global render scale that every
    // monitor loop picks up live; persistence is debounced via the dirty flag.
    let quality_scale_ui = quality_scale.clone();
    let config_dirty_quality = config_dirty.clone();
    ui.on_shader_quality_changed(move |label| {
        let scale = shader_quality_to_scale(&label);
        quality_scale_ui.store(scale.to_bits(), std::sync::atomic::Ordering::Relaxed);
        config_dirty_quality.set(true);
    });

    // Audio sensitivity slider — retunes the AudioEngine gain live; debounced save.
    let context_audio = context.clone();
    let audio_sensitivity_ui = audio_sensitivity.clone();
    let config_dirty_audio = config_dirty.clone();
    ui.on_audio_sensitivity_changed(move |v| {
        let v = v.clamp(0.0, 4.0);
        if let Some(a) = &context_audio.audio { a.set_sensitivity(v); }
        audio_sensitivity_ui.set(v);
        config_dirty_audio.set(true);
    });

    // Mouse interactivity toggle — render loops pick it up live.
    let mouse_enabled_ui = mouse_enabled.clone();
    let config_dirty_mouse = config_dirty.clone();
    ui.on_mouse_interactive_toggled(move |on| {
        mouse_enabled_ui.store(on, std::sync::atomic::Ordering::Relaxed);
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
    let engine_mouse_enabled = mouse_enabled.clone();
    let engine_mouse_sensitivity = mouse_sensitivity.clone();
    let engine_quality = quality_scale.clone();
    let telemetry = Arc::new(std::sync::Mutex::new(platform::EngineTelemetry { fps: 0.0, frame_time: 0.0, vram_usage: 0.0 }));
    let telemetry_thread = telemetry.clone();
    std::thread::spawn(move || {
        platform::renderer::run_renderer(engine_running, engine_state, command_rx, telemetry_thread, engine_context, engine_target_fps, engine_mouse_enabled, engine_mouse_sensitivity, engine_quality);
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

    // ── Diagnostics / telemetry / log / tray timer ─────────────────────────
    let ui_handle_timer = ui.as_weak();
    let telemetry_ui = telemetry.clone();
    let timer = slint::Timer::default();
    let logs_model = Rc::new(VecModel::<LogEntry>::from(Vec::new()));
    ui.set_logs(ModelRc::from(logs_model.clone()));
    let pending_close_timer = pending_close.clone();
    let config_dirty_timer = config_dirty.clone();
    let app_state_timer = app_state.clone();
    let target_fps_timer = target_fps.clone();
    let mouse_enabled_timer = mouse_enabled.clone();
    let mouse_sensitivity_timer = mouse_sensitivity.clone();
    let context_timer = context.clone();
    let thumbnails_busy_timer = thumbnails_busy.clone();
    let wallpapers_model_timer = wallpapers_model.clone();
    let app_state_thumb = app_state.clone();
    // The software renderer leaves stale (white/transparent) regions whenever the OS
    // discards/suspends the window's buffer — tray restore, taskbar un-minimize, or
    // waking a long-dormant window. Forcing a full repaint requires a real resize, so
    // `needs_repaint` (set on those events) makes the timer nudge the size +2px and
    // restore it next tick (two resize events → full repaint). `restore_size` carries
    // the pending revert.
    let restore_size = std::cell::Cell::new(None::<slint::PhysicalSize>);
    let needs_repaint = std::rc::Rc::new(std::cell::Cell::new(false));
    let needs_repaint_timer = needs_repaint.clone();
    // Tracks whether the main window is actually on-screen (not hidden to tray /
    // minimized). Used to skip work the user can't see — e.g. the parallax preview's
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
    // The repaint nudge resizes the window by a pixel — but a `set_size` on a MAXIMIZED
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
    let parallax_seg_timer = parallax_seg.clone();

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
        // ── GPU device-loss recovery ──
        // A lost device (driver TDR/reset, GPU hang, driver update) can't be
        // revived in place. Persist state and relaunch — Strata restores monitors,
        // layers and settings from config, so the wallpaper comes back on its own.
        if context_timer.device_lost.load(std::sync::atomic::Ordering::SeqCst)
            && app_start.elapsed() > std::time::Duration::from_secs(20)
        {
            log::error!("GPU device lost — relaunching Strata to recover the wallpaper");
            if let Ok(st) = app_state_timer.read() {
                flush_config(
                    &st,
                    target_fps_timer.load(std::sync::atomic::Ordering::Relaxed),
                    audio_sensitivity_timer.get(),
                    mouse_enabled_timer.load(std::sync::atomic::Ordering::Relaxed),
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
                    mouse_enabled_timer.load(std::sync::atomic::Ordering::Relaxed),
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
                        ui.set_parallax_status(SharedString::from(format!("Created \"{}\" — see the Library.", name)));
                        push_toast_timer("Parallax Created", "Your 3D wallpaper was added to the Library.", false);
                        ui.invoke_refresh_library(); // rescan so the new wallpaper appears
                    }
                    Err(e) => {
                        log::error!("Parallax create failed: {}", e);
                        ui.set_parallax_status(SharedString::from("Failed — see Diagnostics."));
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
                            ui.set_parallax_status(SharedString::from("Preview ready — Save to add it to your Library."));
                        }
                        Err(e) => {
                            log::error!("Parallax preview build failed: {}", e);
                            ui.set_parallax_progress(0);
                            ui.set_parallax_status(SharedString::from("Preview failed — see Diagnostics."));
                        }
                    },
                    Err(e) => {
                        log::error!("Parallax depth/preview failed: {}", e);
                        ui.set_parallax_progress(0);
                        ui.set_parallax_status(SharedString::from("Failed — see Diagnostics."));
                    }
                }
            }
        }
        // Animate the preview (~10 fps) only while the Parallax tab is open AND the
        // window is on-screen — no point rendering frames nobody can see.
        if let Some(ui) = ui_handle_timer.upgrade() {
            if ui_visible_timer.get() && ui.get_active_tab() == "parallax" {
                if let Some(state) = preview_state_timer.borrow_mut().as_mut() {
                    if let Some(img) = state.frame(&context_timer) {
                        ui.set_parallax_preview_image(img);
                    }
                }
            }
        }

        // ── Parallax model / LaMa download: progress + completion ──
        // `download_is_lama` routes the live "Downloading…%" text to the right
        // status line (depth model vs LaMa), since they share the download channel.
        if parallax_downloading.load(std::sync::atomic::Ordering::SeqCst) {
            if let Some(ui) = ui_handle_timer.upgrade() {
                ui.set_parallax_downloading(true);
                let txt = SharedString::from(format!("Downloading… {}%", download_pct.load(std::sync::atomic::Ordering::SeqCst)));
                if download_is_lama.get() { ui.set_parallax_lama_status(txt); } else { ui.set_parallax_model_status(txt); }
            }
        }
        while let Ok(res) = download_rx.try_recv() {
            if let Some(ui) = ui_handle_timer.upgrade() {
                ui.set_parallax_downloading(false);
                match res {
                    Ok(label) if label == "inpaint" => {
                        refresh_parallax_lama_status(&ui, &parallax_seg_timer.borrow());
                        push_toast_timer("Models Ready", "Cinematic models downloaded — ready to render.", false);
                    }
                    Ok(label) => {
                        refresh_parallax_model_status(&ui, &label);
                        push_toast_timer("Model Ready", "Depth model downloaded — ready to generate.", false);
                    }
                    Err(e) => {
                        log::error!("Model download failed: {}", e);
                        if download_is_lama.get() {
                            ui.set_parallax_lama_status(SharedString::from("Download failed — see Diagnostics."));
                        } else {
                            ui.set_parallax_model_status(SharedString::from("Download failed — see Diagnostics."));
                        }
                        push_toast_timer("Download Failed", "Could not download the model.", true);
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
                // Skip while maximized — resizing a maximized window would shift it.
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
            // Whether anything is actually being rendered — drives the "Inactive"
            // state on the diagnostics cards (avoids a scary 0 FPS / LOW / -0 MB
            // when no shader is assigned).
            let engine_active = app_state_timer.read()
                .map(|s| s.monitors.iter().any(|m| m.layers.iter().any(|l| l.visible)))
                .unwrap_or(false);
            ui.set_engine_active(engine_active);

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
                // at apply time) triggers this — per-frame wgpu errors are left to
                // the log to avoid toast spam.
                if level == "ERROR" && message.starts_with("Layer reload error") {
                    push_toast_timer(
                        "Failed to Apply Shader",
                        "A shader could not be loaded — see the Diagnostics log for details.",
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
                            // resize to force the full repaint — without it the small window
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

    // ── "Refresh Monitors" button — rebuild wallpaper windows on demand ─────
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
            log::info!("Refresh Monitors requested — rebuilding wallpaper windows");

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
            // Strata" can restore the exact geometry — hide()/show() loses it otherwise.
            #[cfg(target_os = "windows")]
            {
                platform::windows::save_window_placement(ui_hwnd_close.get());
                hidden_to_tray_close.set(true);
                let _ = toggle_item_close.set_text("Show Strata");
            }
            ui.hide().ok();
            ui_visible_close.set(false); // hidden to tray — pause unseen work
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
        ui.window().on_winit_window_event(move |win, event| {
            // NOTE: this fires for EVERY event (incl. high-frequency CursorMoved), so do
            // only cheap flag-setting here — never per-event syscalls.
            match event {
                WindowEvent::Resized(_) => {
                    // Maximized state only changes via a resize; query the syscall here,
                    // not on every event.
                    ui_maximized_evt.set(win.is_maximized());
                    // A maximize/restore fires Moved (which arms the move-debounce) AND a
                    // Resized. The resize already self-heals the buffer, so cancel any
                    // pending move-nudge — otherwise the nudge would resize (and shift) the
                    // just-maximized window a moment later. This is what was dragging the
                    // maximized window down on the portrait monitor.
                    move_settle_evt.set(None);
                }
                WindowEvent::Occluded(occluded) => {
                    // `Occluded(true)` = minimized/fully hidden; `(false)` = visible again.
                    // Slint's own backend already forces a full repaint on un-occlude
                    // (renderer.occluded() → NewBuffer + a redraw), so we must NOT add our
                    // own resize nudge here — that only causes the restore flicker. We just
                    // track visibility (to pause the parallax preview while hidden).
                    ui_visible_evt.set(!occluded);
                }
                WindowEvent::Moved(_) => {
                    // Debounced in the timer — a move can reveal stale (unpainted) regions.
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

    log::info!("UI ready — running Slint event loop.");
    ui.run()?;
    Ok(())
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
/// destroyed — so we never let it accumulate on the app's long-lived device.
/// (Measured: shared-device generation left ~1.3 GB committed; a dedicated device
/// dropped afterward settles back to ~18 MB.) Each finished thumbnail is sent over
/// `tx` for the UI timer to load; `busy` drives the refresh spinner.
fn spawn_thumbnail_generation(
    wallpaper_dirs: Vec<std::path::PathBuf>,
    tx: std::sync::mpsc::Sender<(std::path::PathBuf, std::path::PathBuf)>,
    busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    use std::sync::atomic::Ordering;
    // Only generate for ones actually missing a cached thumbnail.
    let missing: Vec<std::path::PathBuf> = wallpaper_dirs.into_iter()
        .filter(|d| controller::cached_thumbnail_path(d).map(|p| !p.exists()).unwrap_or(false))
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
        if let Some(dir) = controller::thumbnails_dir() {
            let _ = std::fs::create_dir_all(&dir);
        }
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
            if let Some(out) = controller::cached_thumbnail_path(wp) {
                // Parallax packages (image.png + depth.png) thumbnail straight from the
                // user's source photo — no shader render — so the library shows the
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
            }
            // Throttle: a gentle background task — keeps CPU low and lets the GPU
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
        // Thread never started — release the guard or the spinner runs forever.
        log::error!("Thumbnail thread spawn failed");
        busy_guard.store(false, Ordering::SeqCst);
    }
}

/// Update the Parallax Studio model status line + Download-button visibility for
/// the currently chosen dropdown label.
fn refresh_parallax_model_status(ui: &AppWindow, label: &str) {
    match parallax::tier_for_label(label) {
        Some(choice) => {
            let (status, need) = parallax::model_status(&choice);
            ui.set_parallax_model_status(SharedString::from(status));
            ui.set_parallax_model_need_download(need);
        }
        None => {
            ui.set_parallax_model_status(SharedString::from(""));
            ui.set_parallax_model_need_download(false);
        }
    }
}

/// Update the Cinematic models status line (chosen matter + LaMa) + Download-button.
fn refresh_parallax_lama_status(ui: &AppWindow, seg_label: &str) {
    let seg = parallax::seg_choice_for_label(seg_label);
    let (status, need) = parallax::cinematic_status(&seg);
    ui.set_parallax_lama_status(SharedString::from(status));
    ui.set_parallax_lama_need_download(need);
}

fn flush_config(state: &AppState, target_fps: u32, audio_sensitivity: f32, mouse_interactive: bool, mouse_sensitivity: f32, quality_scale: f32) {
    let mut config = config::Config::load();
    config.update_from_state(state.theme_mode.clone(), state.span_monitors, state.autostart, &state.monitors);
    config.target_fps = target_fps;
    config.audio_sensitivity = audio_sensitivity;
    config.mouse_interactive = mouse_interactive;
    config.mouse_sensitivity = mouse_sensitivity;
    config.shader_quality = scale_to_shader_quality(quality_scale).to_string();
    config.save().ok();
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

/// Inverse of [`shader_quality_to_scale`] — the canonical label for a stored scale.
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
/// Uses std only — avoids pulling in a date/time crate for a cosmetic stamp.
fn now_hms() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

#[cfg(target_os = "windows")]
fn load_tray_icon(is_dark: bool) -> tray_icon::Icon {
    let (icon_rgba, icon_width, icon_height) = {
        let name = if is_dark { "app-icon_dark.png" } else { "app-icon_light.png" };
        let path = if std::path::Path::new("assets").join(name).exists() {
            std::path::Path::new("assets").join(name)
        } else {
            std::path::Path::new("../../assets").join(name)
        };
        let image = image::open(path).expect("Failed to open icon path")
            .into_rgba8();
        let (width, height) = image.dimensions();
        let rgba = image.into_raw();
        (rgba, width, height)
    };
    tray_icon::Icon::from_rgba(icon_rgba, icon_width, icon_height).expect("Failed to open icon")
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
}
