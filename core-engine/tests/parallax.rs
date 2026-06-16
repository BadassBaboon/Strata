//! Validates the depth-parallax shader (resources/parallax.glsl) in our engine:
//! it must compile in naga, render real image content, and shift the displaced
//! sample as the camera (iMouse) moves — more so over the near foreground than
//! the far background. Run with: `--ignored`.

use core_engine::wgpu;
use std::sync::Arc;

#[test]
#[ignore]
fn parallax_compiles_and_responds_to_camera() {
    let ctx = Arc::new(pollster::block_on(core_engine::GraphicsContext::new_render_only()).expect("gpu"));

    // ── Build a temp wallpaper: our shader + a synthetic photo & depth map ──
    let dir = std::env::temp_dir().join("strata_parallax_test");
    let _ = std::fs::remove_dir_all(&dir);

    // Photo: vertical color stripes (so horizontal shifts are visible).
    // Depth: a centered bright square (near foreground) on a dark field (far).
    let (w, h) = (256u32, 256u32);
    let mut photo = image::RgbImage::new(w, h);
    let mut depth = image::GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let stripe = if (x / 16) % 2 == 0 { [230, 90, 40] } else { [40, 120, 220] };
            photo.put_pixel(x, y, image::Rgb(stripe));
            let near = x >= w/4 && x < 3*w/4 && y >= h/4 && y < 3*h/4;
            depth.put_pixel(x, y, image::Luma([if near { 255 } else { 0 }]));
        }
    }
    // Stage the source photo, then exercise the REAL export packaging.
    let src_photo = std::env::temp_dir().join("strata_parallax_src.png");
    photo.save(&src_photo).unwrap();
    core_engine::parallax::export_wallpaper(
        &dir, "Parallax Test", "test", &src_photo, &depth,
        &core_engine::parallax::ParallaxParams::default(),
    ).expect("export wallpaper package");

    // ── Renderer ──
    let (rw, rh) = (640u32, 360u32); // rw*4 = 2560, already 256-aligned
    let mut r = core_engine::Renderer::new_headless(ctx.clone(), rw, rh, wgpu::TextureFormat::Rgba8Unorm);
    // add_layer compiling the shader is itself the naga-compatibility check.
    r.add_layer(&dir, 1.0, 1.0, "Fill".into(), [0.0, 0.0, 1.0, 1.0], "normal".into())
        .expect("parallax shader must compile + build");

    // Position first, THEN press: set_mouse_down latches iMouse.zw from the
    // current position, and iMouse.xy only tracks while the button is held.
    r.uniform_state.set_mouse_position(60.0, 180.0);
    r.uniform_state.set_mouse_down(true);
    r.uniform_state.set_mouse_position(60.0, 180.0);  // camera pushed left
    let left = render_rgba(&mut r, rw, rh);
    r.uniform_state.set_mouse_position(580.0, 180.0); // camera pushed right
    let right = render_rgba(&mut r, rw, rh);

    // 1) Produced real content (not an all-black / single-color frame).
    let nonblack = left.chunks(4).filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 30).count();
    assert!(nonblack > (rw * rh / 4) as usize, "frame is mostly black — shader rendered nothing");

    // 2) Parallax responds to the camera: the two offsets must differ.
    let changed = left.chunks(4).zip(right.chunks(4))
        .filter(|(a, b)| (a[0] as i32 - b[0] as i32).abs() + (a[1] as i32 - b[1] as i32).abs() + (a[2] as i32 - b[2] as i32).abs() > 24)
        .count();
    let pct = 100.0 * changed as f32 / (rw * rh) as f32;
    assert!(pct > 2.0, "camera move barely changed the image ({pct:.1}%); parallax not working");

    println!("Parallax OK — {pct:.1}% of pixels shifted between left/right camera, {nonblack} non-black px");
}

#[test]
#[ignore]
fn layered_composites_foreground_over_inpainted_background() {
    let ctx = Arc::new(pollster::block_on(core_engine::GraphicsContext::new_render_only()).expect("gpu"));

    let dir = std::env::temp_dir().join("strata_parallax_layered_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Scene: a RED near subject (center square) on a flat field; the inpainted
    // background paints the subject out as GREEN. depth: square near (255) else far.
    let (w, h) = (256u32, 256u32);
    let mut fg = image::RgbImage::new(w, h);    // foreground photo (subject = red)
    let mut bg = image::RgbImage::new(w, h);    // inpainted background (subject = green)
    let mut depth = image::GrayImage::new(w, h);
    let mut mask = image::GrayImage::new(w, h); // foreground membership
    for y in 0..h {
        for x in 0..w {
            let near = x >= 96 && x < 160 && y >= 96 && y < 160;
            fg.put_pixel(x, y, image::Rgb(if near { [230, 30, 30] } else { [40, 60, 90] }));
            bg.put_pixel(x, y, image::Rgb([30, 200, 80])); // hole-free background
            depth.put_pixel(x, y, image::Luma([if near { 255 } else { 0 }]));
            mask.put_pixel(x, y, image::Luma([if near { 255 } else { 0 }]));
        }
    }
    fg.save(dir.join("image.png")).unwrap();
    bg.save(dir.join("background.png")).unwrap();
    depth.save(dir.join("depth.png")).unwrap();
    mask.save(dir.join("mask.png")).unwrap();
    std::fs::write(dir.join("image.glsl"),
        core_engine::parallax::parallax_shader_layered(&core_engine::parallax::ParallaxParams::default())).unwrap();
    std::fs::write(dir.join("manifest.toml"), r#"
[wallpaper]
name = "Layered Test"
author = "test"
version = "1.0.0"
passes = ["image"]

[render_targets.image]
source = "image.glsl"
bindings = [
    { channel = 0, type = "texture", path = "image.png" },
    { channel = 1, type = "texture", path = "depth.png" },
    { channel = 2, type = "texture", path = "background.png" },
    { channel = 3, type = "texture", path = "mask.png" },
]
"#).unwrap();

    let (rw, rh) = (640u32, 360u32);
    let mut r = core_engine::Renderer::new_headless(ctx, rw, rh, wgpu::TextureFormat::Rgba8Unorm);
    r.add_layer(&dir, 1.0, 1.0, "Fill".into(), [0.0, 0.0, 1.0, 1.0], "normal".into())
        .expect("layered shader must compile + build");

    // Strong camera offset so the near subject parallaxes noticeably.
    r.uniform_state.set_mouse_position(80.0, 180.0);
    r.uniform_state.set_mouse_down(true);
    r.uniform_state.set_mouse_position(80.0, 180.0);
    let frame = render_rgba(&mut r, rw, rh);

    let px = |x: u32, y: u32| { let i = ((y * rw + x) * 4) as usize; (frame[i], frame[i + 1], frame[i + 2]) };
    let greenish = |c: (u8, u8, u8)| c.1 > 120 && c.0 < 120 && c.2 < 140;
    let reddish = |c: (u8, u8, u8)| c.0 > 150 && c.1 < 100 && c.2 < 100;

    // Center samples the near subject → foreground (red). A corner is background
    // → the inpainted green, NOT a smeared red trail.
    let center = px(rw / 2, rh / 2);
    let corner = px(40, rh / 2);
    assert!(reddish(center), "center should show the foreground subject, got {center:?}");
    assert!(greenish(corner), "background area should show inpainted bg, got {corner:?}");

    // The red subject must stay compact (no taffy): red pixels ≈ the square's
    // displaced footprint, not a stretched smear across the frame.
    let red_px = frame.chunks(4).filter(|c| reddish((c[0], c[1], c[2]))).count();
    let red_frac = red_px as f32 / (rw * rh) as f32;
    assert!(red_frac < 0.25, "too much red — looks like taffy smear ({:.0}%)", red_frac * 100.0);
    println!("Layered LDI OK — center {center:?}, bg corner {corner:?}, red {:.1}%", red_frac * 100.0);
}

fn render_rgba(r: &mut core_engine::Renderer, w: u32, h: u32) -> Vec<u8> {
    let device = r.context.device.clone();
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("parallax test target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    r.encode_frame(&view);

    let bpr = w * 4; // 256-aligned by construction (w=640)
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"), size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    r.context.queue.submit(std::iter::once(enc.finish()));
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().to_vec();
    buf.unmap();
    data
}
