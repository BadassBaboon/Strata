//! Headless thumbnail generation: render a representative frame of a wallpaper
//! to a PNG, so the Library can show previews for shaders that ship without one.
//!
//! Reuses the offscreen render path (`Renderer::new_headless` + `encode_frame`)
//! and a deterministic time override so animated/feedback shaders settle into a
//! good-looking frame instead of capturing t≈0. Cross-platform (wgpu + `image`).

use std::path::Path;
use std::sync::Arc;

use crate::{GraphicsContext, Renderer};

/// Render `wallpaper_dir`'s shader to a `width`×`height` PNG at `out_path`.
/// Returns an error (rather than panicking) if the shader fails to build or the
/// readback/encode fails - the caller can skip that wallpaper and continue.
pub fn generate_thumbnail(
    context: Arc<GraphicsContext>,
    wallpaper_dir: &Path,
    out_path: &Path,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let width = width.max(1);
    let height = height.max(1);
    let format = wgpu::TextureFormat::Rgba8Unorm; // non-sRGB: shader output is already display-ready
    let device = context.device.clone();

    let mut renderer = Renderer::new_headless(context.clone(), width, height, format);
    renderer.add_layer(wallpaper_dir, 1.0, 1.0, "Fill".into(), [0.0, 0.0, 1.0, 1.0], "normal".into())?;

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("thumbnail target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // Feed a synthetic spectrum so audio-reactive shaders render a representative
    // frame even though nothing is playing during headless capture.
    renderer.headless_audio = Some(synthetic_audio_texture());

    // Advance time across several frames so feedback buffers build up and
    // animated shaders reach a settled, representative look (~2.5 s in). Kept
    // modest (and the caller throttles between shaders) to limit the CPU/GPU and
    // peak-memory cost of generating a whole library's thumbnails at once.
    const FRAMES: u32 = 40;
    const TARGET_T: f32 = 2.5;
    for i in 0..FRAMES {
        renderer.uniform_state.headless_time = Some(TARGET_T * (i as f32) / (FRAMES as f32));
        renderer.encode_frame(&view);
    }

    // ── Read the texture back (bytes_per_row must be 256-aligned) ──
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("thumbnail readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("thumbnail copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    // Map and block until the GPU work + callback complete.
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv().map_err(|e| e.to_string())?
        .map_err(|e| format!("thumbnail map_async failed: {:?}", e))?;

    // Strip the per-row padding into a tight RGBA buffer.
    let data = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        rgba.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    readback.unmap();

    let img = image::RgbaImage::from_raw(width, height, rgba)
        .ok_or("failed to assemble thumbnail image")?;
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    img.save(out_path).map_err(|e| format!("failed to write thumbnail PNG: {}", e))?;
    Ok(())
}

/// A representative 512×2 audio texture (row0 = bass-heavy decaying spectrum with
/// a few peaks, row1 = a waveform) so visualizer thumbnails aren't blank/silent.
fn synthetic_audio_texture() -> Vec<u8> {
    let w = crate::audio::TEX_WIDTH as usize;
    let h = crate::audio::TEX_HEIGHT as usize;
    let mut tex = vec![0u8; w * h * 4];
    for i in 0..w {
        let f = i as f32 / w as f32;
        let base = (1.0 - f).powf(1.4);
        let peaks = 0.30 * (i as f32 * 0.13).sin().abs() + 0.18 * (i as f32 * 0.37).sin().abs();
        let b = ((base * 0.85 + peaks).clamp(0.0, 1.0) * 255.0) as u8;
        let p = i * 4;
        tex[p] = b; tex[p + 1] = b; tex[p + 2] = b; tex[p + 3] = 255;
    }
    let stride = w * 4;
    for i in 0..w {
        let v = (0.5 + 0.4 * (i as f32 * 0.06).sin()).clamp(0.0, 1.0);
        let b = (v * 255.0) as u8;
        let p = stride + i * 4;
        tex[p] = b; tex[p + 1] = b; tex[p + 2] = b; tex[p + 3] = 255;
    }
    tex
}
