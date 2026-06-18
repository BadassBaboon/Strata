use std::sync::{Arc, atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering}, Mutex};
use std::sync::mpsc::{Receiver, channel, Sender};
use winit::window::Window;
use crate::platform::EngineCommand;
use crate::controller::SharedState;
use core_engine::{GraphicsContext, Renderer};
use std::collections::HashMap;

pub fn run_renderer(
    running: Arc<AtomicBool>,
    state: SharedState,
    command_rx: Receiver<EngineCommand>,
    telemetry: Arc<Mutex<crate::platform::EngineTelemetry>>,
    context: Arc<GraphicsContext>,
    // Global frame-rate cap, read live by every monitor loop.  Changing it from
    // the UI takes effect within one frame on all monitors.
    target_fps: Arc<AtomicU32>,
    // Mouse interactivity mode (0=Off,1=All,2=Only shaders,3=Only Parallax): every
    // monitor loop feeds the desktop cursor into iMouse when the mode isn't Off; the
    // engine then gates it per-layer by type (sensitivity = gain around centre).
    mouse_mode: Arc<AtomicU8>,
    mouse_sensitivity: Arc<AtomicU32>, // f32 bits
    // Global render-quality scale (f32 bits), read live by every monitor loop.
    quality_scale: Arc<AtomicU32>,
) {

    let mut monitors: HashMap<winit::window::WindowId, Sender<EngineCommand>> = HashMap::new();
    // Maps window_id → monitor_id so Reload can distribute correct layers
    let mut window_monitor_ids: HashMap<winit::window::WindowId, String> = HashMap::new();
    let vram_stats: Arc<Mutex<HashMap<winit::window::WindowId, f32>>> = Arc::new(Mutex::new(HashMap::new()));
    let fps_stats: Arc<Mutex<HashMap<winit::window::WindowId, f32>>> = Arc::new(Mutex::new(HashMap::new()));

    while running.load(Ordering::SeqCst) {
        // Block up to 500 ms for the first command, then drain any further
        // messages that arrived in the meantime.  This replaces the old
        // try_recv + 100 ms sleep pattern so this thread never busy-waits.
        let first = command_rx.recv_timeout(std::time::Duration::from_millis(500));
        for cmd in first.into_iter().chain(command_rx.try_iter()) {
            match cmd {
                EngineCommand::AddWindow { window, surface, initial_size, offset, global_res, layers, monitor_id } => {
                    let (tx, rx) = channel();
                    let window_id = window.id();
                    let context_clone = context.clone();
                    let running_clone = running.clone();
                    let vram_clone = vram_stats.clone();
                    let fps_clone = fps_stats.clone();
                    let target_fps_clone = target_fps.clone();
                    let mouse_mode_clone = mouse_mode.clone();
                    let mouse_sensitivity_clone = mouse_sensitivity.clone();
                    let quality_scale_clone = quality_scale.clone();

                    std::thread::spawn(move || {
                        run_monitor_loop(window, surface, initial_size, offset, global_res, layers, rx, context_clone, running_clone, window_id, vram_clone, fps_clone, target_fps_clone, mouse_mode_clone, mouse_sensitivity_clone, quality_scale_clone);
                    });

                    monitors.insert(window_id, tx);
                    window_monitor_ids.insert(window_id, monitor_id);
                }
                EngineCommand::WindowResized(id, size) => {
                    if let Some(tx) = monitors.get(&id) {
                        let _ = tx.send(EngineCommand::WindowResized(id, size));
                    }
                }
                EngineCommand::WindowClosed(id) => {
                    if let Some(tx) = monitors.remove(&id) {
                        let _ = tx.send(EngineCommand::Shutdown);
                    }
                    window_monitor_ids.remove(&id);
                    vram_stats.lock().unwrap().remove(&id);
                    fps_stats.lock().unwrap().remove(&id);
                }
                EngineCommand::Reload => {
                    let app_state = state.read().unwrap();
                    // In span mode every window renders the primary display's
                    // shader; otherwise each window renders its own monitor's.
                    let span_layers = if app_state.span_monitors {
                        crate::controller::primary_monitor(&app_state.monitors)
                            .map(|m| m.layers.clone())
                            .unwrap_or_default()
                    } else {
                        Vec::new()
                    };
                    for (window_id, tx) in &monitors {
                        let layers = if app_state.span_monitors {
                            span_layers.clone()
                        } else if let Some(monitor_id) = window_monitor_ids.get(window_id) {
                            app_state.monitors.iter()
                                .find(|m| &m.id == monitor_id)
                                .map(|m| m.layers.clone())
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        };
                        let _ = tx.send(EngineCommand::SetLayers(layers));
                    }
                }
                EngineCommand::SetLayerTransform { monitor_id, pipeline_index, transform } => {
                    for (window_id, tx) in &monitors {
                        if window_monitor_ids.get(window_id).map(|m| m == &monitor_id).unwrap_or(false) {
                            let _ = tx.send(EngineCommand::SetLayerTransform {
                                monitor_id: monitor_id.clone(), pipeline_index, transform,
                            });
                        }
                    }
                }
                EngineCommand::SetLayerOpacity { monitor_id, pipeline_index, opacity } => {
                    for (window_id, tx) in &monitors {
                        if window_monitor_ids.get(window_id).map(|m| m == &monitor_id).unwrap_or(false) {
                            let _ = tx.send(EngineCommand::SetLayerOpacity {
                                monitor_id: monitor_id.clone(), pipeline_index, opacity,
                            });
                        }
                    }
                }
                EngineCommand::Shutdown => {
                    for tx in monitors.values() {
                        let _ = tx.send(EngineCommand::Shutdown);
                    }
                    running.store(false, Ordering::SeqCst);
                }
                EngineCommand::OpenDebugger => {
                    log::info!("Shader Debugger Window opened.");
                }
                EngineCommand::SetVSync(mode) => {
                    for tx in monitors.values() {
                        let _ = tx.send(EngineCommand::SetVSync(mode));
                    }
                }
                _ => {}
            }
        }

        // Update telemetry aggregated from per-monitor threads.
        if let Ok(mut tel) = telemetry.try_lock() {
            let vram = vram_stats.lock().unwrap();
            let fps  = fps_stats.lock().unwrap();
            tel.vram_usage = vram.values().sum();
            tel.fps = if !fps.is_empty() {
                fps.values().sum::<f32>() / fps.len() as f32
            } else { 0.0 };
            tel.frame_time = if tel.fps > 0.0 { 1000.0 / tel.fps } else { 0.0 };
        }
    }
}

fn run_monitor_loop(
    window: Arc<Window>,
    surface: core_engine::wgpu::Surface<'static>,
    initial_size: winit::dpi::PhysicalSize<u32>,
    offset: (f32, f32),
    global_res: (f32, f32),
    layers: Vec<crate::controller::LayerInfo>,
    command_rx: Receiver<EngineCommand>,
    context: Arc<GraphicsContext>,
    running: Arc<AtomicBool>,
    window_id: winit::window::WindowId,
    vram_stats: Arc<Mutex<HashMap<winit::window::WindowId, f32>>>,
    fps_stats: Arc<Mutex<HashMap<winit::window::WindowId, f32>>>,
    target_fps: Arc<AtomicU32>,
    mouse_mode: Arc<AtomicU8>,
    mouse_sensitivity: Arc<AtomicU32>,
    quality_scale: Arc<AtomicU32>,
) {
    let mut renderer = Renderer::new(context, window.clone(), surface, initial_size).expect("Failed to create renderer");
    renderer.set_global_info(offset, global_res);

    // This monitor's absolute screen origin, so the global cursor can be mapped to
    // monitor-local pixels for iMouse. Fixed for a wallpaper window, so read once.
    let monitor_origin: (i32, i32) = window.current_monitor()
        .map(|m| { let p = m.position(); (p.x, p.y) })
        .unwrap_or((0, 0));
    let mut mouse_was_down = false;

    log::info!(
        "Renderer started for window {:?} | surface {}×{} | global {:.0}×{:.0} | offset ({:.0},{:.0})",
        window_id,
        initial_size.width, initial_size.height,
        global_res.0, global_res.1,
        offset.0, offset.1
    );

    // ── Store the authoritative surface size ──────────────────────────────
    // Wallpaper windows never forward WM_SIZE events so this stays constant.
    // The only caller of renderer.resize() after this point is the
    // Outdated/Lost recovery path, which must use this same size.
    let fixed_size = initial_size;

    // Reverse so the TOP of the UI layer list composites on top (Photoshop-style):
    // the engine draws pipelines[0] first (bottom) and later ones over it.
    for layer in layers.iter().rev() {
        if !layer.visible { continue; }
        if let Err(e) = renderer.add_layer(&layer.wallpaper_path, layer.opacity, layer.resolution_scale, layer.positioning.clone(), layer.transform, layer.blend_mode.clone()) {
            log::error!("Layer reload error [{}]: {}", layer.wallpaper_path.display(), e);
        }
    }

    // ── Frame pacing ──────────────────────────────────────────────────────
    // Live wallpapers do NOT need to run at the monitor's native refresh rate.
    // A 170 Hz panel would otherwise drive ~170 encode→submit→present cycles
    // per second for a background almost nobody looks at directly.  Capping to
    // a modest target FPS slashes CPU/GPU wakeups: the thread parks between
    // frames (0 % CPU) instead of spinning at native refresh.
    //
    // The cap is read live from `target_fps` each frame, so the Settings slider
    // retunes every monitor within one frame.  Clamped to a sane 1..=240 range
    // to guard against a 0 (division by zero) or absurd values.
    let frame_budget = |fps: &Arc<AtomicU32>| {
        let f = fps.load(Ordering::Relaxed).clamp(1, 240) as u64;
        std::time::Duration::from_micros(1_000_000 / f)
    };

    let mut last_telemetry = std::time::Instant::now();
    let mut frame_count = 0u32;

    // Tracks whether the current empty state has already been painted once.
    // An unassigned monitor draws a single black frame, then sleeps on the
    // command channel instead of busy-presenting nothing forever.  This is the
    // key fix for "CPU stays high even with all shaders disabled".
    let mut painted_empty = false;

    while running.load(Ordering::SeqCst) {
        // ── Telemetry (updated in both idle and active states) ─────────────
        let now = std::time::Instant::now();
        if now.duration_since(last_telemetry) >= std::time::Duration::from_millis(500) {
            let elapsed = now.duration_since(last_telemetry).as_secs_f32();
            let fps = frame_count as f32 / elapsed;
            let vram = renderer.estimate_vram_mb();
            vram_stats.lock().unwrap().insert(window_id, vram);
            fps_stats.lock().unwrap().insert(window_id, fps);
            frame_count = 0;
            last_telemetry = now;
        }

        // ── Idle path: no layers assigned ─────────────────────────────────
        // Paint one black frame so the desktop background is cleared, then
        // block on the command channel.  An empty monitor now costs ~0 % CPU.
        if renderer.pipelines.is_empty() {
            if !painted_empty {
                match renderer.render() {
                    Ok(_) => painted_empty = true,
                    Err(core_engine::wgpu::CurrentSurfaceTexture::Outdated)
                    | Err(core_engine::wgpu::CurrentSurfaceTexture::Lost) => {
                        renderer.resize(fixed_size); // retry next iteration
                    }
                    Err(_) => painted_empty = true, // give up; avoid a spin
                }
            }
            // Block (instead of busy-waiting) until a command arrives.  The
            // 250 ms timeout only exists to re-check the `running` flag.
            match command_rx.recv_timeout(std::time::Duration::from_millis(250)) {
                Ok(cmd) => {
                    if process_command(&mut renderer, cmd, window_id, &mut painted_empty) {
                        return; // Shutdown
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }

        // ── Active path: drain commands, render, then pace the frame ───────
        let frame_start = std::time::Instant::now();

        while let Ok(cmd) = command_rx.try_recv() {
            if process_command(&mut renderer, cmd, window_id, &mut painted_empty) {
                return; // Shutdown
            }
        }

        // A command may have cleared every layer — re-check before drawing so
        // we drop straight into the idle path next iteration.
        if renderer.pipelines.is_empty() {
            continue;
        }

        // Pick up any live change to the global quality scale (cheap no-op when
        // unchanged; rebuilds the scene target only when the value actually moves).
        renderer.set_quality(f32::from_bits(quality_scale.load(Ordering::Relaxed)));

        // Tell the engine the current mode; it gates iMouse per layer by type
        // (and zeroes it for non-eligible layers, so parallax layers auto-animate).
        let mode = mouse_mode.load(Ordering::Relaxed);
        renderer.set_mouse_mode(mode);

        // Feed the desktop cursor into iMouse whenever the mode isn't Off (the
        // engine decides which layers actually receive it). mouse_down is
        // (re)applied every frame rather than latched: a SetLayers rebuilds the
        // pipelines' uniform state, so a one-shot latch would leave freshly-applied
        // wallpapers with iMouse.z=0 until the setting was toggled.
        if mode != 0 {
            if let Some((cx, cy)) = get_cursor_pos() {
                let w = renderer.size.width as f32;
                let h = renderer.size.height as f32;
                let lx = (cx - monitor_origin.0) as f32;
                let ly = (cy - monitor_origin.1) as f32;
                // Sensitivity scales movement around the monitor centre.
                let sens = f32::from_bits(mouse_sensitivity.load(Ordering::Relaxed));
                let mx = w * 0.5 + (lx - w * 0.5) * sens;
                let my = h * 0.5 + (ly - h * 0.5) * sens;
                renderer.uniform_state.set_mouse_down(true);
                renderer.uniform_state.set_mouse_position(mx, my);
                mouse_was_down = true;
            }
        } else if mouse_was_down {
            renderer.uniform_state.set_mouse_down(false);
            mouse_was_down = false;
        }

        match renderer.render() {
            Ok(_) => frame_count += 1,
            Err(core_engine::wgpu::CurrentSurfaceTexture::Outdated)
            | Err(core_engine::wgpu::CurrentSurfaceTexture::Lost) => {
                // Use the fixed size, not window.inner_size() — on high-DPI
                // monitors inner_size() may return logical pixels that differ
                // from the physical size we configured the surface at.
                log::warn!("Surface lost/outdated for {:?}, recovering at {}×{}",
                    window_id, fixed_size.width, fixed_size.height);
                renderer.resize(fixed_size);
            }
            Err(e) => log::error!("Render error on {:?}: {:?}", window_id, e),
        }

        // Sleep off the remainder of the frame budget to hold ~target_fps.
        let budget = frame_budget(&target_fps);
        let elapsed = frame_start.elapsed();
        if elapsed < budget {
            std::thread::sleep(budget - elapsed);
        }
    }
}

/// Apply a single engine command to this monitor's renderer.
/// Returns `true` if the render loop should shut down.
fn process_command(
    renderer: &mut Renderer,
    cmd: EngineCommand,
    window_id: winit::window::WindowId,
    painted_empty: &mut bool,
) -> bool {
    match cmd {
        EngineCommand::WindowResized(_, size) => {
            log::info!("Surface resize {:?}: {}×{}", window_id, size.width, size.height);
            renderer.resize(size);
            *painted_empty = false;
        }
        EngineCommand::SetLayers(new_layers) => {
            log::info!("Monitor {:?} reloading {} layer(s)", window_id, new_layers.len());
            renderer.clear_layers();
            // Reverse: top of the UI list composites on top (see run_monitor_loop).
            for layer in new_layers.iter().rev() {
                if !layer.visible { continue; }
                if let Err(e) = renderer.add_layer(
                    &layer.wallpaper_path,
                    layer.opacity,
                    layer.resolution_scale,
                    layer.positioning.clone(),
                    layer.transform,
                    layer.blend_mode.clone(),
                ) {
                    log::error!("Layer reload error [{}]: {}", layer.wallpaper_path.display(), e);
                }
            }
            // Force a repaint of the new state: empty → one black frame,
            // non-empty → active rendering resumes.
            *painted_empty = false;
        }
        EngineCommand::SetLayerTransform { pipeline_index, transform, .. } => {
            renderer.set_layer_transform(pipeline_index, transform);
        }
        EngineCommand::SetLayerOpacity { pipeline_index, opacity, .. } => {
            renderer.set_layer_opacity(pipeline_index, opacity);
        }
        EngineCommand::SetVSync(mode) => renderer.set_vsync(mode),
        EngineCommand::Shutdown => return true,
        _ => {}
    }
    false
}

/// Current global cursor position in virtual-screen pixels (absolute, can be
/// negative on multi-monitor setups). Windows-only; other platforms get None.
#[cfg(windows)]
fn get_cursor_pos() -> Option<(i32, i32)> {
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;
    let mut p = POINT { x: 0, y: 0 };
    if unsafe { GetCursorPos(&mut p) } != 0 {
        Some((p.x, p.y))
    } else {
        None
    }
}

#[cfg(not(windows))]
fn get_cursor_pos() -> Option<(i32, i32)> {
    None
}
