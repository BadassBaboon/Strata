//! Super-resolution for the Parallax Studio's inpainted background.
//!
//! LaMa fills the subject hole at a fixed 512×512, so that fill is soft once stretched
//! to the wallpaper's resolution. An ESRGAN-style upscaler restores plausible texture
//! and sharpness to it before compositing, so the background revealed during parallax
//! disocclusion doesn't look mushy.
//!
//! Behind the `depth-onnx` feature. Real-ESRGAN-x4plus has a FIXED 128×128 input
//! (→512, 4×), so we process the image in 128-px tiles and stitch the 512-px results.
//! For LaMa's 512² fill that's an exact 4×4 grid (no remainder, no overlap needed).

#[cfg(feature = "depth-onnx")]
use std::path::Path;
#[cfg(feature = "depth-onnx")]
use image::RgbImage;

/// Real-ESRGAN's fixed input tile size (→ 4× output).
#[cfg(feature = "depth-onnx")]
const TILE: u32 = 128;
#[cfg(feature = "depth-onnx")]
const SCALE: u32 = 4;

/// 4×-upscale `image` with a fixed-128-tile ESRGAN model. Returns a `4w × 4h` image.
/// Tiles are clamped to the image edge so non-multiples of 128 still work; for the
/// 512² LaMa fill the grid is exact.
#[cfg(feature = "depth-onnx")]
pub fn upscale_4x(model_path: &Path, image: &RgbImage) -> Result<RgbImage, String> {
    use ort::session::Session;
    use ort::value::Tensor;
    use image::GenericImageView;

    let (w, h) = image.dimensions();
    if w < TILE || h < TILE {
        return Err(format!("image {w}x{h} smaller than the {TILE}px upscaler tile"));
    }

    let mut session = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)
        .map_err(|e| format!("ort optimization level: {e}"))?
        // Free the working set when the session drops (see depth.rs) - no retained arena.
        .with_memory_pattern(false)
        .map_err(|e| format!("ort memory pattern: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("load upscaler {:?}: {e}", model_path))?;

    let t = TILE as usize;
    let n = t * t;
    let mut out = RgbImage::new(w * SCALE, h * SCALE);

    // Tile origins, clamped so the last tile in each axis sits flush against the edge.
    let origins = |span: u32| -> Vec<u32> {
        let mut v = Vec::new();
        let mut p = 0u32;
        loop {
            let o = p.min(span - TILE);
            if v.last() != Some(&o) {
                v.push(o);
            }
            if o + TILE >= span {
                break;
            }
            p += TILE;
        }
        v
    };

    for &iy in &origins(h) {
        for &ix in &origins(w) {
            // Pack the 128×128 tile as NCHW RGB in 0..1 (Real-ESRGAN convention).
            let tile = image.view(ix, iy, TILE, TILE).to_image();
            let mut buf = vec![0.0f32; 3 * n];
            for (i, p) in tile.pixels().enumerate() {
                buf[i] = p.0[0] as f32 / 255.0;
                buf[n + i] = p.0[1] as f32 / 255.0;
                buf[2 * n + i] = p.0[2] as f32 / 255.0;
            }
            let input = Tensor::from_array(([1usize, 3, t, t], buf)).map_err(|e| format!("tile tensor: {e}"))?;
            let outputs = session.run(ort::inputs![input]).map_err(|e| format!("upscale inference: {e}"))?;
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| format!("upscale output: {e}"))?;

            let ot = (TILE * SCALE) as usize; // 512
            let om = ot * ot;
            // Detect 0..1 vs 0..255 output once per tile (cheap, robust across exports).
            let maxv = data.iter().take(om).copied().fold(0.0f32, f32::max);
            let mul = if maxv > 1.5 { 1.0 } else { 255.0 };
            let (ox, oy) = (ix * SCALE, iy * SCALE);
            for ty in 0..ot {
                for tx in 0..ot {
                    let p = ty * ot + tx;
                    out.put_pixel(ox + tx as u32, oy + ty as u32, image::Rgb([
                        (data[p] * mul).clamp(0.0, 255.0) as u8,
                        (data[om + p] * mul).clamp(0.0, 255.0) as u8,
                        (data[2 * om + p] * mul).clamp(0.0, 255.0) as u8,
                    ]));
                }
            }
        }
    }
    Ok(out)
}
