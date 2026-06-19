// Headless audit: actually build the wgpu pipeline (all passes) for every
// wallpaper and capture validation errors via an error scope. This catches
// failures the naga-only compile audit misses (multi-pass binding/layout, etc).
//
//   cargo test -p core-engine --test pipeline_audit -- --ignored --nocapture

use core_engine::wgpu;
use core_engine::{GraphicsContext, WallpaperPipeline, UniformState, Renderer};
use core_engine::manifest::WallpaperConfig;
use std::sync::Arc;

/// Folder of wallpaper sub-dirs to audit. Defaults to the in-repo `wallpapers/`
/// dir, but `STRATA_AUDIT_DIR` overrides it so we can validate an arbitrary set
/// (e.g. Strata-Library/import) before packing a release.
fn audit_root() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("STRATA_AUDIT_DIR") {
        return std::path::PathBuf::from(p);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("wallpapers")
}

/// Register shared texture/cubemap roots so shaders resolve their `external/`
/// assets by sha-name (mirrors the app's `set_asset_dirs`). Looks beside the
/// audit root for an `external/` dir (e.g. Strata-Library/{import,external}).
fn register_asset_dirs(root: &std::path::Path) {
    let mut dirs = vec![root.join("external")];
    if let Some(parent) = root.parent() {
        dirs.push(parent.join("external"));
    }
    core_engine::set_asset_dirs(dirs);
}

#[test]
#[ignore]
fn audit_pipeline_all_wallpapers() {
    pollster::block_on(run());
}

async fn run() {
    let ctx = GraphicsContext::new().await.expect("gpu");
    let uniforms = UniformState::new(&ctx.device, 1920.0, 1080.0);

    let root = audit_root();
    register_asset_dirs(&root);
    let mut dirs: Vec<_> = std::fs::read_dir(&root).unwrap()
        .filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.join("manifest.toml").exists())
        .collect();
    dirs.sort();

    let mut pass = 0;
    let mut fail = 0;
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let config = match WallpaperConfig::load_from_dir(&dir) {
            Ok(c) => c,
            Err(e) => { fail += 1; println!("  FAIL  {}  -> manifest: {}", name, e); continue; }
        };

        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = WallpaperPipeline::new(
            &ctx.device, &ctx.queue, config, &dir,
            wgpu::TextureFormat::Bgra8UnormSrgb, 1920, 1080,
            &uniforms.bind_group_layout, "normal",
        );
        let scope_err = scope.pop().await;

        match (result, scope_err) {
            (Err(e), _) => { fail += 1; println!("  FAIL  {}  -> build: {}", name, e.lines().next().unwrap_or("")); }
            (Ok(_), Some(e)) => { fail += 1; println!("  FAIL  {}  -> wgpu validation: {:?}", name, e); }
            (Ok(_), None) => { pass += 1; println!("  PASS  {}", name); }
        }
    }
    println!("\n=== pipeline audit: {} passed, {} failed ===", pass, fail);
}

// Render several frames of every wallpaper into an offscreen texture and assert
// no wgpu validation errors fire — catches render-time bugs the build-only audit
// misses, e.g. the double-buffered (BufferA) texture feedback loop in clock-time.
#[test]
#[ignore]
fn audit_render_all_wallpapers() {
    pollster::block_on(render_run());
}

async fn render_run() {
    let ctx = Arc::new(GraphicsContext::new().await.expect("gpu"));
    let format = wgpu::TextureFormat::Bgra8UnormSrgb;

    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d { width: 640, height: 360, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let root = audit_root();
    register_asset_dirs(&root);
    let mut dirs: Vec<_> = std::fs::read_dir(&root).unwrap()
        .filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.join("manifest.toml").exists())
        .collect();
    dirs.sort();

    let mut pass = 0;
    let mut fail = 0;
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let mut renderer = Renderer::new_headless(ctx.clone(), 640, 360, format);
        if renderer.add_layer(&dir, 1.0, 1.0, "Fill".into(), [0.0, 0.0, 1.0, 1.0], "normal".into()).is_err() {
            // build-time failure already covered by the other audit; skip here.
            continue;
        }
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        for _ in 0..4 { renderer.encode_frame(&view); }       // exercise ping-pong both parities
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        match scope.pop().await {
            None => { pass += 1; println!("  PASS  {}", name); }
            Some(e) => { fail += 1; println!("  FAIL  {}  -> render: {:?}", name, e); }
        }
    }
    println!("\n=== render audit: {} passed, {} failed ===", pass, fail);
    assert_eq!(fail, 0, "{} wallpaper(s) produced render-time validation errors", fail);
}
