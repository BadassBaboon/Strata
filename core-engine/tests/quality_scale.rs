//! Verifies the global quality scale actually changes the internal render
//! resolution (and thus the VRAM estimate), and that switching back to 1.0
//! restores full resolution without panicking. Run with: `--ignored`.

use core_engine::wgpu;
use std::sync::Arc;

#[test]
#[ignore]
fn quality_scales_targets_and_vram_then_restores() {
    let ctx = Arc::new(pollster::block_on(core_engine::GraphicsContext::new_render_only()).expect("gpu"));

    // A multipass wallpaper (has offscreen buffer targets) at 1920×1080.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("wallpapers").join("shader-royale");
    let mut r = core_engine::Renderer::new_headless(ctx, 1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    r.add_layer(&dir, 1.0, 1.0, "Fill".into(), [0.0, 0.0, 1.0, 1.0], "normal".into())
        .expect("add multipass layer");

    let vram_full = r.estimate_vram_mb();

    r.set_quality(0.5);
    let vram_half = r.estimate_vram_mb();
    // Offscreen targets shrink to 0.25 area → estimate must drop meaningfully.
    assert!(vram_half < vram_full,
        "VRAM should drop at 0.5 quality: full={vram_full} half={vram_half}");

    // Drive a frame at half quality to make sure the scaled scene path renders.
    let tex = make_target(&r, 1920, 1080);
    let view = tex.create_view(&Default::default());
    r.encode_frame(&view);

    // Back to full — targets restored, estimate returns to (about) the original.
    r.set_quality(1.0);
    let vram_back = r.estimate_vram_mb();
    assert!((vram_back - vram_full).abs() < 0.01,
        "VRAM should restore at 1.0: full={vram_full} back={vram_back}");
    r.encode_frame(&view); // full-quality frame renders cleanly too

    println!("VRAM full={vram_full:.1}MB  half={vram_half:.1}MB  restored={vram_back:.1}MB");
}

fn make_target(r: &core_engine::Renderer, w: u32, h: u32) -> wgpu::Texture {
    r.context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
