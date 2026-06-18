//! Salient-subject segmentation (U²-Net) for the Parallax Studio's cinematic mode.
//!
//! Decouples the subject mask from the depth map — depth thresholding ("the Otsu
//! trap") slices a subject's continuous depth gradient and chops off whatever sits
//! past the cutoff (e.g. a horse's tail). A dedicated segmentation model produces a
//! whole-subject alpha regardless of depth.
//!
//! Behind the `depth-onnx` feature (shares ONNX Runtime). Model: Heliosoph/u2net-onnx
//! `u2net.onnx` (Apache-2.0), input `input.1` [1,3,320,320] (ImageNet-normalized,
//! RGB), outputs d0..d6; d0 (the first output) is the fused saliency [1,1,320,320].

#[cfg(feature = "depth-onnx")]
use std::path::Path;
#[cfg(feature = "depth-onnx")]
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer};
#[cfg(feature = "depth-onnx")]
use image::imageops::FilterType;

/// U²-Net's fixed ONNX input resolution.
pub const U2NET_SIZE: u32 = 320;

/// Produce a subject alpha matte (white = subject) for `image` using a salient-
/// object ONNX model (U²-Net @ 320, BiRefNet @ 1024, …). `input_size` is the model's
/// square input. Single-input/single-output; min-max normalized so it handles both
/// probability and logit outputs. Returned at the original resolution.
#[cfg(feature = "depth-onnx")]
pub fn segment_subject(model_path: &Path, image: &DynamicImage, input_size: u32) -> Result<GrayImage, String> {
    use ort::session::Session;
    use ort::value::Tensor;

    let dim = input_size.max(64);
    let sz = dim as usize;
    let n = sz * sz;

    let rgb = image.resize_exact(dim, dim, FilterType::CatmullRom).to_rgb8();
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let mut t = vec![0.0f32; 3 * n];
    for (i, p) in rgb.pixels().enumerate() {
        for c in 0..3 {
            t[c * n + i] = (p.0[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }

    let mut session = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)
        .map_err(|e| format!("ort optimization level: {e}"))?
        // Free the working set when the session drops (see depth.rs) — no retained arena.
        .with_memory_pattern(false)
        .map_err(|e| format!("ort memory pattern: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("load U2Net {:?}: {e}", model_path))?;

    // Positional input binding works for any single-input model regardless of the
    // input tensor's name (U²-Net "input.1", BiRefNet "input_image", …).
    let input = Tensor::from_array(([1usize, 3, sz, sz], t)).map_err(|e| format!("input tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs![input])
        .map_err(|e| format!("segmentation inference: {e}"))?;
    // First output is the saliency / matte map.
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| format!("segmentation output: {e}"))?;

    // Normalize the saliency map to a 0..1 probability. Models differ in what they
    // emit: U²-Net's `d0` is already a sigmoid probability (values in 0..1), while
    // BiRefNet exports raw LOGITS (e.g. −80..25). Min-max normalization is fragile
    // here — one extreme logit (BiRefNet-Full hits −82) stretches the range and
    // drags the background up past the binarize threshold, whitening the whole
    // frame. So: if the output isn't already bounded to 0..1, treat it as logits and
    // squash with a sigmoid (saturates → immune to outliers); otherwise use it as-is.
    let out_of_01 = data.iter().filter(|&&v| v.is_finite() && (v < -0.01 || v > 1.01)).count();
    let is_logits = out_of_01 as f64 / data.len().max(1) as f64 > 0.10;
    let to_prob = |v: f32| -> f32 {
        if !v.is_finite() {
            0.0
        } else if is_logits {
            1.0 / (1.0 + (-v).exp())
        } else {
            v.clamp(0.0, 1.0)
        }
    };
    let norm: Vec<f32> = (0..n).map(|i| to_prob(data.get(i).copied().unwrap_or(0.0))).collect();

    // Auto-polarity safety net: if a model emits an inverted matte, the WHOLE frame
    // reads "on" with the subject as a dark hole. Flip only when both the outer
    // border AND the image as a whole are majority-on — a real salient subject is a
    // centred minority, so this never fires on a legitimately large/central subject.
    let margin = (dim / 20).max(2) as usize;
    let (mut border_sum, mut border_cnt) = (0.0f64, 0usize);
    for y in 0..sz {
        for x in 0..sz {
            if x < margin || x >= sz - margin || y < margin || y >= sz - margin {
                border_sum += norm[y * sz + x] as f64;
                border_cnt += 1;
            }
        }
    }
    let border_on = border_cnt > 0 && border_sum / border_cnt as f64 > 0.5;
    let overall_on = norm.iter().filter(|&&v| v > 0.5).count() as f64 / norm.len() as f64 > 0.5;
    let inverted = border_on && overall_on;

    let mut small: GrayImage = ImageBuffer::new(dim, dim);
    for (i, p) in small.pixels_mut().enumerate() {
        let mut a = norm[i];
        if inverted { a = 1.0 - a; }
        // Contrast-binarize: soft mattes (BiRefNet) come back with a grey field and
        // a faint drop-shadow halo; smoothstep collapses background→0 and subject→1
        // while keeping an anti-aliased edge. No-op-ish for already-crisp U²-Net.
        let a = a * a * (3.0 - 2.0 * a); // smoothstep(0,1)
        let a = ((a - 0.4) / 0.2).clamp(0.0, 1.0); // hard-ish threshold band 0.4..0.6
        let a = a * a * (3.0 - 2.0 * a);
        p.0[0] = (a * 255.0) as u8;
    }

    let (ow, oh) = image.dimensions();
    // Triangle (bilinear), NOT Lanczos3: Lanczos overshoots near edges (ringing),
    // which resurrects faint specks downstream. The matte is feathered anyway, so we
    // don't need Lanczos sharpness here. This is the FULL matte (every salient
    // subject); callers thin it to the main subject with [`drop_small_subjects`].
    Ok(DynamicImage::ImageLuma8(small)
        .resize_exact(ow, oh, FilterType::Triangle)
        .to_luma8())
}

/// Zero out any connected component whose area is less than `rel_thresh` × the
/// largest component's area. The largest component (the primary subject) is always
/// preserved, so this never empties the matte. 4-connectivity. Detection uses a LOW
/// threshold so a dropped subject's faint anti-aliased penumbra is removed too (a
/// leftover fringe would otherwise survive thresholding/dilation downstream).
///
/// A subject only partly separable from the ground (e.g. a distant dog with its legs
/// in the grass) gets a clean cutout that then FLOATS or folds during parallax. The
/// Studio drops such small subjects from the FOREGROUND matte (so they stay in the
/// background plate) while still flattening their depth via the full matte.
#[cfg(feature = "depth-onnx")]
pub fn drop_small_subjects(mask: &mut GrayImage, rel_thresh: f32) {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let on: Vec<bool> = mask.pixels().map(|p| p.0[0] > 24).collect();
    let mut label = vec![0u32; w * h];
    let mut areas: Vec<u32> = vec![0]; // index 0 = background / unlabeled
    let mut stack: Vec<usize> = Vec::new();
    let mut cur = 0u32;
    for start in 0..w * h {
        if !on[start] || label[start] != 0 {
            continue;
        }
        cur += 1;
        let mut area = 0u32;
        label[start] = cur;
        stack.push(start);
        while let Some(idx) = stack.pop() {
            area += 1;
            let (x, y) = (idx % w, idx / w);
            let push = |n: usize, stack: &mut Vec<usize>, label: &mut Vec<u32>| {
                if on[n] && label[n] == 0 {
                    label[n] = cur;
                    stack.push(n);
                }
            };
            if x > 0 { push(idx - 1, &mut stack, &mut label); }
            if x + 1 < w { push(idx + 1, &mut stack, &mut label); }
            if y > 0 { push(idx - w, &mut stack, &mut label); }
            if y + 1 < h { push(idx + w, &mut stack, &mut label); }
        }
        areas.push(area);
    }
    if cur == 0 {
        return;
    }
    let largest = areas.iter().copied().max().unwrap_or(0);
    let min_area = (largest as f32 * rel_thresh) as u32;
    for (i, p) in mask.pixels_mut().enumerate() {
        let l = label[i] as usize;
        if l != 0 && areas[l] < min_area {
            p.0[0] = 0;
        }
    }
}

#[cfg(all(test, feature = "depth-onnx"))]
mod tests {
    use super::*;

    // Real U²-Net run, only when a model is available:
    //   STRATA_U2NET_MODEL=path/to/u2net.onnx cargo test -p core-engine \
    //       --features depth-onnx --lib segment::tests::u2net_real -- --ignored --nocapture
    #[test]
    #[ignore]
    fn u2net_real() {
        let Ok(model) = std::env::var("STRATA_U2NET_MODEL") else {
            eprintln!("set STRATA_U2NET_MODEL to run");
            return;
        };
        // A bright disc on a dark field — U²-Net should flag the disc as salient.
        let (w, h) = (240u32, 180u32);
        let mut img = image::RgbImage::new(w, h);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                let on = d < 55.0;
                img.put_pixel(x, y, if on { image::Rgb([240, 200, 60]) } else { image::Rgb([20, 24, 30]) });
            }
        }
        let size: u32 = std::env::var("STRATA_SEG_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(U2NET_SIZE);
        let mask = segment_subject(Path::new(&model), &DynamicImage::ImageRgb8(img), size).expect("segment");
        assert_eq!(mask.dimensions(), (w, h));
        let center = mask.get_pixel(w / 2, h / 2).0[0];
        let corner = mask.get_pixel(2, 2).0[0];
        println!("U2Net OK — center={center} corner={corner}");
        assert!(center > corner, "subject (center) should be more salient than corner");
    }
}
