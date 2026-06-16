pub mod renderer;

#[cfg(target_os = "windows")]
pub mod windows;

#[derive(Debug, Clone)]
pub struct EngineTelemetry {
    pub fps: f32,
    pub frame_time: f32,
    pub vram_usage: f32,
}

pub enum EngineCommand {
    AddWindow {
        window: std::sync::Arc<winit::window::Window>,
        surface: core_engine::wgpu::Surface<'static>,
        /// Physical pixel size of this monitor window.  Passed explicitly so
        /// the renderer never races `window.inner_size()` before the Win32
        /// SetWindowPos call has committed the correct dimensions.
        initial_size: winit::dpi::PhysicalSize<u32>,
        offset: (f32, f32),
        global_res: (f32, f32),
        layers: Vec<crate::controller::LayerInfo>,
        monitor_id: String,
    },
    WindowResized(winit::window::WindowId, winit::dpi::PhysicalSize<u32>),
    WindowClosed(winit::window::WindowId),
    Reload,
    SetLayers(Vec<crate::controller::LayerInfo>),
    /// Live spatial-edit preview: update one layer's rect without recreating the
    /// pipeline. Routed to the monitor whose id matches.
    SetLayerTransform { monitor_id: String, pipeline_index: usize, transform: [f32; 4] },
    /// Live opacity update (just a uniform — no pipeline rebuild), routed by id.
    SetLayerOpacity { monitor_id: String, pipeline_index: usize, opacity: f32 },
    #[allow(dead_code)]
    ReloadLibrary,
    #[allow(dead_code)]
    LoadWallpaper(std::path::PathBuf),
    #[allow(dead_code)]
    SetMousePosition(f32, f32),
    #[allow(dead_code)]
    SetMouseDown(bool),
    OpenDebugger,
    SetVSync(core_engine::wgpu::PresentMode),
    Shutdown,
}
