//! Background inpainting for the Parallax Studio's layered (LDI) mode.
//!
//! To kill the "taffy" stretching at foreground silhouettes, we decouple the scene
//! into a foreground layer and an inpainted background layer: the regions hidden
//! behind near subjects are filled in (with LaMa) so a viewpoint shift reveals real
//! background pixels instead of smeared ones.
//!
//! The LaMa ONNX path is behind the `depth-onnx` feature (shares ONNX Runtime with
//! depth estimation). Model: `Carve/LaMa-ONNX` `lama_fp32.onnx` (Apache-2.0), fixed
//! 512×512 input - inputs `image` [1,3,512,512] + `mask` [1,1,512,512] (1 = fill),
//! output `output` [1,3,512,512].

use image::{DynamicImage, GrayImage, ImageBuffer, Luma, imageops::FilterType};

// Types used only by the ONNX inpaint path (its signature + body).
#[cfg(feature = "depth-onnx")]
use std::path::Path;
#[cfg(feature = "depth-onnx")]
use image::{GenericImageView, RgbImage};

/// LaMa's fixed ONNX input resolution.
pub const LAMA_SIZE: u32 = 512;

/// Build a foreground mask from a depth map (near = bright): pixels at/above
/// `threshold` (0..1 of the depth range) are foreground (white = inpaint), dilated
/// by `dilate` px so the mask fully over-covers silhouette halos.
pub fn foreground_mask(depth: &GrayImage, threshold: f32, dilate: u32) -> GrayImage {
    let t = (threshold.clamp(0.0, 1.0) * 255.0) as u8;
    let mut m: GrayImage = ImageBuffer::new(depth.width(), depth.height());
    for (mp, dp) in m.pixels_mut().zip(depth.pixels()) {
        mp.0[0] = if dp.0[0] >= t { 255 } else { 0 };
    }
    if dilate > 0 {
        dilate_mask(&m, dilate)
    } else {
        m
    }
}

/// Adaptive foreground/background split (Otsu's method) over the depth histogram,
/// returned as a 0..1 threshold. Adapts per image so the subject is separated from
/// the background wherever the natural near/far gap is - not a fixed cutoff that
/// only ever catches the very nearest pixels. Clamped to a sane range so a smooth
/// (non-bimodal) depth map can't produce a degenerate threshold.
pub fn otsu_threshold(depth: &GrayImage) -> f32 {
    let mut hist = [0u32; 256];
    for p in depth.pixels() {
        hist[p.0[0] as usize] += 1;
    }
    let total = depth.pixels().len() as f64;
    let sum: f64 = (0..256).map(|i| i as f64 * hist[i] as f64).sum();
    let (mut sum_b, mut w_b, mut best_var, mut thr) = (0.0f64, 0.0f64, 0.0f64, 128usize);
    for i in 0..256 {
        w_b += hist[i] as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f <= 0.0 {
            break;
        }
        sum_b += i as f64 * hist[i] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if between > best_var {
            best_var = between;
            thr = i;
        }
    }
    (thr as f32 / 255.0).clamp(0.35, 0.85)
}

/// Soften a binary mask into a feathered 0..255 alpha (separable box blur, radius
/// `r`), so the layered composite transitions smoothly across silhouettes instead
/// of a hard cut.
pub fn feather_mask(mask: &GrayImage, r: u32) -> GrayImage {
    if r == 0 {
        return mask.clone();
    }
    let (w, h) = (mask.width() as i32, mask.height() as i32);
    let r = r as i32;
    let span = (2 * r + 1) as u32;
    let at = |x: i32, y: i32| mask.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32).0[0] as u32;
    let mut horiz: GrayImage = ImageBuffer::new(mask.width(), mask.height());
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u32;
            for dx in -r..=r { s += at(x + dx, y); }
            horiz.put_pixel(x as u32, y as u32, Luma([(s / span) as u8]));
        }
    }
    let hat = |x: i32, y: i32| horiz.get_pixel(x as u32, y.clamp(0, h - 1) as u32).0[0] as u32;
    let mut out: GrayImage = ImageBuffer::new(mask.width(), mask.height());
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u32;
            for dy in -r..=r { s += hat(x, y + dy); }
            out.put_pixel(x as u32, y as u32, Luma([(s / span) as u8]));
        }
    }
    out
}

/// Fill the masked (subject) region of a depth map with a smooth interpolation of
/// the surrounding background depth (Laplacian/Jacobi diffusion), so the background
/// layer can parallax by its own depth without the subject's depth cliff. Depth is
/// low-frequency, so this runs at a reduced resolution and upscales.
pub fn fill_masked_depth(depth: &GrayImage, mask: &GrayImage) -> GrayImage {
    const WORK: u32 = 192;
    let (ow, oh) = depth.dimensions();
    let d = DynamicImage::ImageLuma8(depth.clone()).resize_exact(WORK, WORK, FilterType::Triangle).to_luma8();
    let m = DynamicImage::ImageLuma8(mask.clone()).resize_exact(WORK, WORK, FilterType::Triangle).to_luma8();
    let (w, h) = (WORK as usize, WORK as usize);

    // "known" = background pixels (outside the subject mask) we hold fixed.
    let known: Vec<bool> = m.pixels().map(|p| p.0[0] < 128).collect();
    let mut u: Vec<f32> = d.pixels().map(|p| p.0[0] as f32).collect();
    let mean_known = {
        let (mut s, mut c) = (0.0f32, 0.0f32);
        for (i, &k) in known.iter().enumerate() {
            if k { s += u[i]; c += 1.0; }
        }
        if c > 0.0 { s / c } else { 128.0 }
    };
    for (i, &k) in known.iter().enumerate() {
        if !k { u[i] = mean_known; } // seed holes
    }
    let mut tmp = u.clone();
    for _ in 0..160 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if known[i] {
                    tmp[i] = u[i];
                    continue;
                }
                let l = u[y * w + x.saturating_sub(1)];
                let r = u[y * w + (x + 1).min(w - 1)];
                let up = u[y.saturating_sub(1) * w + x];
                let dn = u[(y + 1).min(h - 1) * w + x];
                tmp[i] = 0.25 * (l + r + up + dn);
            }
        }
        std::mem::swap(&mut u, &mut tmp);
    }
    let mut filled: GrayImage = ImageBuffer::new(WORK, WORK);
    for (i, p) in filled.pixels_mut().enumerate() {
        p.0[0] = u[i].clamp(0.0, 255.0) as u8;
    }
    DynamicImage::ImageLuma8(filled).resize_exact(ow, oh, FilterType::Triangle).to_luma8()
}

/// Separable max-filter dilation of a binary mask by `r` px.
fn dilate_mask(src: &GrayImage, r: u32) -> GrayImage {
    let (w, h) = (src.width() as i32, src.height() as i32);
    let r = r as i32;
    let at = |x: i32, y: i32| src.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32).0[0];
    let mut horiz: GrayImage = ImageBuffer::new(src.width(), src.height());
    for y in 0..h {
        for x in 0..w {
            let mut m = 0u8;
            for dx in -r..=r { m = m.max(at(x + dx, y)); }
            horiz.put_pixel(x as u32, y as u32, Luma([m]));
        }
    }
    let mut out: GrayImage = ImageBuffer::new(src.width(), src.height());
    for y in 0..h {
        for x in 0..w {
            let mut m = 0u8;
            for dy in -r..=r { m = m.max(horiz.get_pixel(x as u32, (y + dy).clamp(0, h - 1) as u32).0[0]); }
            out.put_pixel(x as u32, y as u32, Luma([m]));
        }
    }
    out
}

/// Inpaint `image` wherever `mask` is white (255 = fill, 0 = keep) using a LaMa
/// ONNX model. Returns an RGB image at the original resolution where the masked
/// regions are filled and everything else is the pristine original.
/// `upscaler`, when given, is a 4× ESRGAN model path used to restore detail to LaMa's
/// (512-px, soft) fill before it's stretched to the source resolution.
#[cfg(feature = "depth-onnx")]
pub fn inpaint(model_path: &Path, image: &DynamicImage, mask: &GrayImage, upscaler: Option<&Path>) -> Result<RgbImage, String> {
    use ort::session::Session;
    use ort::value::Tensor;

    let (ow, oh) = image.dimensions();
    let sz = LAMA_SIZE as usize;
    let n = sz * sz;

    // Scale image + mask to 512×512 (LaMa's fixed input).
    let rgb = image.resize_exact(LAMA_SIZE, LAMA_SIZE, FilterType::CatmullRom).to_rgb8();
    let msmall = DynamicImage::ImageLuma8(mask.clone())
        .resize_exact(LAMA_SIZE, LAMA_SIZE, FilterType::Triangle)
        .to_luma8();

    // image: NCHW RGB 0..1; mask: 1 = hole.
    let mut img_t = vec![0.0f32; 3 * n];
    for (i, p) in rgb.pixels().enumerate() {
        img_t[i] = p.0[0] as f32 / 255.0;
        img_t[n + i] = p.0[1] as f32 / 255.0;
        img_t[2 * n + i] = p.0[2] as f32 / 255.0;
    }
    let mut mask_t = vec![0.0f32; n];
    for (i, p) in msmall.pixels().enumerate() {
        mask_t[i] = if p.0[0] > 127 { 1.0 } else { 0.0 };
    }

    let mut session = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)
        .map_err(|e| format!("ort optimization level: {e}"))?
        // Free the working set when the session drops (see depth.rs) - no retained arena.
        .with_memory_pattern(false)
        .map_err(|e| format!("ort memory pattern: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("load LaMa {:?}: {e}", model_path))?;

    let image_in = Tensor::from_array(([1usize, 3, sz, sz], img_t)).map_err(|e| format!("image tensor: {e}"))?;
    let mask_in = Tensor::from_array(([1usize, 1, sz, sz], mask_t)).map_err(|e| format!("mask tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs!["image" => image_in, "mask" => mask_in])
        .map_err(|e| format!("LaMa inference: {e}"))?;
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| format!("LaMa output: {e}"))?;

    // Output may be 0..1 or 0..255 depending on the export - detect and scale.
    let maxv = data.iter().copied().fold(0.0f32, f32::max);
    let scale = if maxv > 1.5 { 1.0 } else { 255.0 };
    let mut filled = RgbImage::new(LAMA_SIZE, LAMA_SIZE);
    for (i, p) in filled.pixels_mut().enumerate() {
        p.0[0] = (data[i] * scale).clamp(0.0, 255.0) as u8;
        p.0[1] = (data[n + i] * scale).clamp(0.0, 255.0) as u8;
        p.0[2] = (data[2 * n + i] * scale).clamp(0.0, 255.0) as u8;
    }

    // Upscale the filled result and composite with the original. LaMa fully replaces
    // the masked region (hard, so no original subject pixels bleed back as a ghost),
    // with only a NARROW seam-blend at the mask boundary to hide the texture step
    // between LaMa's (512-px, upscaled) fill and the sharp surrounding photo. The
    // blend is a few px at full resolution - wide enough to soften the seam, far too
    // narrow to resurrect the subject.
    // Optionally restore detail to the (soft, 512-px) fill with a 4× upscaler before
    // stretching to the source resolution; on any upscaler error fall back to the raw
    // fill so inpainting still succeeds.
    let filled_src = match upscaler {
        Some(up) => crate::upscale::upscale_4x(up, &filled).unwrap_or_else(|e| {
            log::warn!("upscaler failed, using raw LaMa fill: {e}");
            filled
        }),
        None => filled,
    };
    let filled_full = DynamicImage::ImageRgb8(filled_src).resize_exact(ow, oh, FilterType::CatmullRom).to_rgb8();
    let orig = image.to_rgb8();
    let seam = (ow.max(oh) / 500).max(2); // ~8 px @ 4K
    let soft_full = feather_mask(mask, seam);
    let mut out = RgbImage::new(ow, oh);
    for y in 0..oh {
        for x in 0..ow {
            let a = soft_full.get_pixel(x, y).0[0] as f32 / 255.0;
            let o = orig.get_pixel(x, y).0;
            let f = filled_full.get_pixel(x, y).0;
            out.put_pixel(x, y, image::Rgb([
                (o[0] as f32 * (1.0 - a) + f[0] as f32 * a) as u8,
                (o[1] as f32 * (1.0 - a) + f[1] as f32 * a) as u8,
                (o[2] as f32 * (1.0 - a) + f[2] as f32 * a) as u8,
            ]));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreground_mask_thresholds_and_dilates() {
        // Depth: near (255) square in the middle of a far (0) field.
        let (w, h) = (40u32, 40u32);
        let mut depth: GrayImage = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let near = x >= 16 && x < 24 && y >= 16 && y < 24;
                depth.put_pixel(x, y, Luma([if near { 255 } else { 0 }]));
            }
        }
        let mask = foreground_mask(&depth, 0.6, 3);
        // Center is foreground; far corner is background; dilation grew the square.
        assert_eq!(mask.get_pixel(20, 20).0[0], 255);
        assert_eq!(mask.get_pixel(1, 1).0[0], 0);
        let on = mask.pixels().filter(|p| p.0[0] == 255).count();
        assert!(on > 8 * 8, "dilation should expand the 8x8 core, got {on} px");
    }

    #[test]
    fn otsu_separates_bimodal_depth() {
        // 70% far (value 30) + 30% near (value 220): the split should land between.
        let (w, h) = (40u32, 40u32);
        let mut depth: GrayImage = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                depth.put_pixel(x, y, Luma([if y < 28 { 30 } else { 220 } ]));
            }
        }
        let t = otsu_threshold(&depth);
        assert!(t > 30.0 / 255.0 && t < 220.0 / 255.0, "otsu split between clusters, got {t}");
        // feather turns the binary mask into a soft gradient (intermediate values exist)
        let mask = foreground_mask(&depth, t, 2);
        let soft = feather_mask(&mask, 3);
        assert!(soft.pixels().any(|p| p.0[0] > 0 && p.0[0] < 255), "feather should produce a gradient");
    }

    // Real LaMa inpaint, only when a model is available:
    //   STRATA_LAMA_MODEL=path/to/lama_fp32.onnx cargo test -p core-engine \
    //       --features depth-onnx --test ... lama_real -- --ignored --nocapture
    #[cfg(feature = "depth-onnx")]
    #[test]
    #[ignore]
    fn lama_real_inpaint() {
        let Ok(model) = std::env::var("STRATA_LAMA_MODEL") else {
            eprintln!("set STRATA_LAMA_MODEL to run");
            return;
        };
        // A textured image with a solid red block we ask LaMa to remove.
        let (w, h) = (200u32, 150u32);
        let mut img = RgbImage::new(w, h);
        let mut mask: GrayImage = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = (((x / 8 + y / 8) % 2) as u8) * 90 + 60;
                img.put_pixel(x, y, image::Rgb([c, c / 2, 255 - c]));
            }
        }
        for y in 60..100 {
            for x in 80..120 {
                img.put_pixel(x, y, image::Rgb([255, 0, 0]));
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let out = inpaint(Path::new(&model), &DynamicImage::ImageRgb8(img), &mask, None).expect("inpaint");
        assert_eq!(out.dimensions(), (w, h));
        // The red block should be gone (filled with surrounding pattern).
        let p = out.get_pixel(100, 80).0;
        let still_red = p[0] > 200 && p[1] < 60 && p[2] < 60;
        assert!(!still_red, "masked region not inpainted, still red: {:?}", p);
        println!("LaMa inpaint OK - center now {:?}", p);
    }
}
