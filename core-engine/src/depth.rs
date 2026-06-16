//! Depth estimation for the Parallax Studio: turn a flat photo into a grayscale
//! depth map (near = bright) that [`crate::resources`]' parallax shader displaces.
//!
//! Two backends behind one [`DepthEstimator`] trait:
//!   • [`HeuristicEstimator`] — always available, no download, no ML. A rough
//!     depth from a blurred luminance + radial center bias. Good enough to preview
//!     and to use when the user doesn't want to fetch a model.
//!   • [`OnnxEstimator`] — behind the `depth-onnx` cargo feature; runs a
//!     DepthAnything-style ONNX model via ONNX Runtime (`ort`) for real depth.
//!
//! The ONNX Runtime is an optional, heavy native dependency, so it's feature-gated
//! to keep the base engine lightweight. Models are never bundled — the desktop
//! shell downloads them on demand (see the model registry).

use std::path::Path;

use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use image::imageops::FilterType;

/// Produces a grayscale depth map (near = bright) for an image.
pub trait DepthEstimator {
    fn estimate(&self, image: &DynamicImage) -> Result<GrayImage, String>;
}

// ── Shared image <-> tensor helpers (also used by the ONNX backend) ──────────

/// Resize to `size`×`size` RGB and produce a normalized NCHW f32 tensor using the
/// ImageNet mean/std DepthAnything was trained with. `size` must be a multiple of
/// 14 for DepthAnything (e.g. 518).
pub fn preprocess_imagenet(image: &DynamicImage, size: u32) -> Vec<f32> {
    let rgb = image.resize_exact(size, size, FilterType::CatmullRom).to_rgb8();
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let n = (size * size) as usize;
    let mut out = vec![0.0f32; 3 * n];
    for (i, px) in rgb.pixels().enumerate() {
        for c in 0..3 {
            out[c * n + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

/// Normalize a raw depth field (`in_w`×`in_h`, row-major) to 0..1, build a
/// grayscale image, and resize it to `out_w`×`out_h`. `invert` flips the
/// near/far convention (DepthAnything outputs larger = nearer, which is what our
/// shader wants, so the default is no inversion).
pub fn depth_field_to_map(
    depth: &[f32],
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    invert: bool,
) -> GrayImage {
    // Edge dilate + blur (at model resolution, before upscaling): a max-filter
    // grows near (foreground) regions outward so the depth silhouette slightly
    // overshoots the object, then a light blur softens the cliff. This is the
    // standard fix for the "taffy"/tearing artifact at foreground edges during
    // parallax (DepthAnything's own post-processing does max+gaussian filtering).
    let dilated = max_filter(depth, in_w, in_h, 2);
    let field = box_blur(&dilated, in_w, in_h, 2);

    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &field {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let range = (hi - lo).max(1e-6);
    let mut img: GrayImage = ImageBuffer::new(in_w, in_h);
    for (i, px) in img.pixels_mut().enumerate() {
        let mut t = (field.get(i).copied().unwrap_or(lo) - lo) / range; // 0..1
        if invert {
            t = 1.0 - t;
        }
        *px = Luma([(t.clamp(0.0, 1.0) * 255.0) as u8]);
    }
    if (in_w, in_h) == (out_w, out_h) {
        img
    } else {
        // Lanczos3 upscale preserves fine structure (hair, foliage) far better
        // than bilinear, keeping depth edges crisp so the parallax doesn't tear.
        DynamicImage::ImageLuma8(img)
            .resize_exact(out_w, out_h, FilterType::Lanczos3)
            .to_luma8()
    }
}

/// Separable max filter (dilation) over a single-channel f32 field, radius `r`.
fn max_filter(src: &[f32], w: u32, h: u32, r: i32) -> Vec<f32> {
    let (w, h) = (w as i32, h as i32);
    let mut horiz = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for dx in -r..=r {
                let xx = (x + dx).clamp(0, w - 1);
                m = m.max(src[(y * w + xx) as usize]);
            }
            horiz[(y * w + x) as usize] = m;
        }
    }
    let mut out = vec![0.0f32; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for dy in -r..=r {
                let yy = (y + dy).clamp(0, h - 1);
                m = m.max(horiz[(yy * w + x) as usize]);
            }
            out[(y * w + x) as usize] = m;
        }
    }
    out
}

// ── Heuristic backend (no ML, no download) ───────────────────────────────────

/// A dependency-free depth approximation. Not physically accurate — it assumes
/// the subject is brighter/central and the background darker/edges — but it lets
/// the parallax pipeline and preview work with zero downloads.
pub struct HeuristicEstimator {
    /// Working resolution for the blur pass (keeps it cheap on large photos).
    pub work: u32,
    /// How strongly the image center is treated as nearer (0 = off).
    pub center_bias: f32,
}

impl Default for HeuristicEstimator {
    fn default() -> Self {
        Self { work: 256, center_bias: 0.45 }
    }
}

impl DepthEstimator for HeuristicEstimator {
    fn estimate(&self, image: &DynamicImage) -> Result<GrayImage, String> {
        let (ow, oh) = image.dimensions();
        let small = image.resize_exact(self.work, self.work, FilterType::Triangle).to_rgb8();
        // Luminance as a coarse depth proxy, softened so it reads as surfaces.
        let mut field = vec![0.0f32; (self.work * self.work) as usize];
        for (i, px) in small.pixels().enumerate() {
            field[i] = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
        }
        field = box_blur(&field, self.work, self.work, 6);
        // Radial center bias: nearer toward the middle of the frame.
        let (cx, cy) = (self.work as f32 / 2.0, self.work as f32 / 2.0);
        let maxd = (cx * cx + cy * cy).sqrt();
        for y in 0..self.work {
            for x in 0..self.work {
                let i = (y * self.work + x) as usize;
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / maxd;
                field[i] = field[i] * (1.0 - self.center_bias) + (1.0 - d) * 255.0 * self.center_bias;
            }
        }
        Ok(depth_field_to_map(&field, self.work, self.work, ow, oh, false))
    }
}

/// Simple separable box blur over a single-channel f32 field, `passes` times
/// (approximates a Gaussian). Radius is fixed at 1 per pass for cache-friendliness.
fn box_blur(src: &[f32], w: u32, h: u32, passes: u32) -> Vec<f32> {
    let (w, h) = (w as usize, h as usize);
    let mut a = src.to_vec();
    let mut b = vec![0.0f32; a.len()];
    for _ in 0..passes {
        // Horizontal
        for y in 0..h {
            for x in 0..w {
                let l = a[y * w + x.saturating_sub(1)];
                let c = a[y * w + x];
                let r = a[y * w + (x + 1).min(w - 1)];
                b[y * w + x] = (l + c + r) / 3.0;
            }
        }
        // Vertical
        for y in 0..h {
            for x in 0..w {
                let u = b[y.saturating_sub(1) * w + x];
                let c = b[y * w + x];
                let d = b[(y + 1).min(h - 1) * w + x];
                a[y * w + x] = (u + c + d) / 3.0;
            }
        }
    }
    a
}

// ── ONNX backend (feature `depth-onnx`) ──────────────────────────────────────

/// Runs a DepthAnything-style ONNX model. The `model_path` is a `.onnx` file the
/// desktop shell downloaded (see the model registry). `input_size` must match the
/// model (518 for DepthAnything V2).
#[cfg(feature = "depth-onnx")]
pub struct OnnxEstimator {
    pub model_path: std::path::PathBuf,
    pub input_size: u32,
    /// Flip near/far if a particular model outputs smaller = nearer.
    pub invert: bool,
    /// DepthAnything V3 takes a 5-D `pixel_values` input `[batch, num_images, 3, H, W]`
    /// (and emits metric distance) instead of V2's 4-D `[1, 3, H, W]`.
    pub rank5: bool,
    /// True when the model outputs metric DISTANCE (larger = farther), e.g. V3.
    /// Such output must be converted to disparity (1/distance) before it can drive
    /// parallax — see [`Self::estimate`]. V2 already emits disparity, so this is false.
    pub metric: bool,
}

#[cfg(feature = "depth-onnx")]
impl OnnxEstimator {
    pub fn new(model_path: impl Into<std::path::PathBuf>) -> Self {
        Self { model_path: model_path.into(), input_size: 518, invert: false, rank5: false, metric: false }
    }

    /// Configure from a registry entry (input size, near/far convention, arch).
    pub fn from_choice(model_path: impl Into<std::path::PathBuf>, choice: &ModelChoice) -> Self {
        Self {
            model_path: model_path.into(),
            input_size: choice.input_size,
            invert: choice.invert,
            rank5: choice.is_v3(),
            metric: choice.is_v3(),
        }
    }
}

#[cfg(feature = "depth-onnx")]
impl DepthEstimator for OnnxEstimator {
    fn estimate(&self, image: &DynamicImage) -> Result<GrayImage, String> {
        use ort::session::Session;
        use ort::value::Tensor;

        let size = self.input_size;
        let input = preprocess_imagenet(image, size);

        // Cap graph optimization at Level1 (basic). ORT's *extended* fusions
        // (e.g. SimplifiedLayerNormFusion) crash on the fp16 DepthAnything exports
        // ("GetIndexFromName … InsertedPrecisionFreeCast …"), so we skip them.
        let mut session = Session::builder()
            .map_err(|e| format!("ort session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)
            .map_err(|e| format!("ort optimization level: {e}"))?
            .commit_from_file(&self.model_path)
            .map_err(|e| format!("load model {:?}: {e}", self.model_path))?;

        // V3's `pixel_values` is 5-D (extra leading num_images axis); V2 is 4-D.
        let s = size as usize;
        let outputs = if self.rank5 {
            let tensor = Tensor::from_array(([1usize, 1, 3, s, s], input))
                .map_err(|e| format!("build input tensor: {e}"))?;
            session.run(ort::inputs![tensor]).map_err(|e| format!("inference: {e}"))?
        } else {
            let tensor = Tensor::from_array(([1usize, 3, s, s], input))
                .map_err(|e| format!("build input tensor: {e}"))?;
            session.run(ort::inputs![tensor]).map_err(|e| format!("inference: {e}"))?
        };

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract output: {e}"))?;

        let (ow, oh) = image.dimensions();

        // Parallax/DIBR displaces each pixel by its DISPARITY (screen shift ∝ 1/distance),
        // not by distance. V2 already outputs affine-invariant inverse depth (disparity),
        // which is why it parallaxes cleanly. V3 outputs metric DISTANCE — feeding that to
        // the shader linearly squashes the whole near/mid field into a thin band and leaves
        // a cliff to the far sky, which folds and shears under motion. So for metric models
        // we convert distance → disparity here, restoring a V2-like (parallax-correct)
        // distribution: far recedes toward 0, near/mid spread out.
        if self.metric {
            // Use the 2nd/98th percentiles, not raw min/max, for the near/far bounds.
            // Outdoor scenes put the sky/clouds at an extreme far distance (and the odd
            // stray near pixel); those outliers would stretch the range and re-squash
            // everything between them. Clamping to robust percentiles pins the far field
            // to one plane (flattening distant clutter that otherwise jitters under
            // parallax) and keeps the near/mid spread healthy.
            let mut sorted: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
            if sorted.is_empty() {
                return Ok(depth_field_to_map(data, size, size, ow, oh, self.invert));
            }
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pick = |q: f32| sorted[(((sorted.len() - 1) as f32) * q) as usize];
            let lo = pick(0.02);
            let hi = pick(0.98);
            let range = (hi - lo).max(1e-6);
            // Offset the near plane off zero so the nearest pixel doesn't blow up to a
            // spike that min-max then crushes everything else against. ~15% of the range
            // gives a healthy near:far disparity spread without a single dominating value.
            let eps = 0.15 * range;
            let disparity: Vec<f32> = data
                .iter()
                .map(|&v| 1.0 / ((v.clamp(lo, hi) - lo) + eps))
                .collect();
            // Disparity is large for near (we want near = bright), so no extra invert.
            return Ok(depth_field_to_map(&disparity, size, size, ow, oh, false));
        }

        Ok(depth_field_to_map(data, size, size, ow, oh, self.invert))
    }
}

// ── Model registry (TOML-backed; drives the on-demand downloader) ────────────

/// Rough GPU class, used to pick the right model precision. Derived from the wgpu
/// adapter the engine already created (`GraphicsContext::gpu_class`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuClass {
    /// CPU / software rendering — only the small int8 model is practical.
    Cpu,
    /// Integrated GPU — small int8 is the sweet spot.
    Integrated,
    /// Dedicated GPU — can handle the fp16 base/large models.
    Discrete,
}

/// One downloadable file belonging to a model. `files[0]` is always the `.onnx`
/// graph ORT loads; any further entries (e.g. external `model.onnx_data` weights)
/// just need to sit alongside it in the same directory.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct ModelFile {
    pub name: String,
    pub url: String,
}

/// A model the Parallax Studio can download and run. One entry may span several
/// files. The `id` doubles as the on-disk subdirectory, so identically named
/// files (`model.onnx`) from different models never collide. Weights are NEVER
/// bundled — the shell fetches them on demand, so non-free (CC BY-NC) weights
/// remain the user's choice, not our redistribution.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct ModelChoice {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub commercial_ok: bool,
    #[serde(default)]
    pub size_mb: u32,
    #[serde(default = "default_input_size")]
    pub input_size: u32,
    #[serde(default)]
    pub invert: bool,
    /// Depth only: "v2" (4-D input) or "v3" (5-D `pixel_values`). Empty otherwise.
    #[serde(default)]
    pub arch: String,
    /// Depth only: GPU recommendation bucket ("small" | "base" | "large").
    #[serde(default)]
    pub tier: String,
    pub files: Vec<ModelFile>,
}

fn default_input_size() -> u32 { 518 }

impl ModelChoice {
    /// The `.onnx` graph file name ORT loads (the first listed file).
    pub fn primary_file(&self) -> &str {
        self.files.first().map(|f| f.name.as_str()).unwrap_or("model.onnx")
    }
    /// True for the DepthAnything V3 architecture (5-D input, metric depth).
    pub fn is_v3(&self) -> bool { self.arch == "v3" }
}

#[derive(serde::Deserialize)]
struct Registry {
    #[serde(default)]
    depth: Vec<ModelChoice>,
    #[serde(default)]
    segment: Vec<ModelChoice>,
    #[serde(default)]
    inpaint: Vec<ModelChoice>,
    #[serde(default)]
    upscale: Vec<ModelChoice>,
}

/// The embedded registry, parsed once. A future "update model library" feature can
/// swap the source TOML (same schema) without code changes.
fn registry() -> &'static Registry {
    use std::sync::OnceLock;
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| {
        toml::from_str(include_str!("resources/models.toml"))
            .expect("embedded models.toml is valid")
    })
}

/// All depth-estimator models, in dropdown order.
pub fn depth_models() -> &'static [ModelChoice] { &registry().depth }
/// All subject-matte / segmentation models, in dropdown order.
pub fn segment_models() -> &'static [ModelChoice] { &registry().segment }
/// All upscaler models, in dropdown order.
pub fn upscale_models() -> &'static [ModelChoice] { &registry().upscale }

/// Look up any model (depth, segment, inpaint, or upscale) by its `id`.
pub fn model_by_id(id: &str) -> Option<&'static ModelChoice> {
    let r = registry();
    r.depth.iter().chain(&r.segment).chain(&r.inpaint).chain(&r.upscale).find(|m| m.id == id)
}

/// The LaMa inpainting model (Cinematic/layered background fill).
pub fn lama_model() -> ModelChoice {
    model_by_id("lama").expect("lama in registry").clone()
}
/// The default U²-Net subject-matte model.
pub fn u2net_model() -> ModelChoice {
    model_by_id("u2net").expect("u2net in registry").clone()
}
/// BiRefNet-lite subject-matte model (sharper alternative to U²-Net).
pub fn birefnet_model() -> ModelChoice {
    model_by_id("birefnet-lite").expect("birefnet-lite in registry").clone()
}

/// Hardware-aware recommendation: which depth model to default to for this GPU.
pub fn recommended_tier(gpu: GpuClass) -> &'static ModelChoice {
    let want = match gpu {
        GpuClass::Discrete => "base",
        _ => "small",
    };
    depth_models().iter()
        .find(|m| m.tier == want)
        .unwrap_or(&depth_models()[0])
}

// ── High-level cached pipeline ───────────────────────────────────────────────

/// Estimate depth for `image_path` with `estimator` and write a PNG depth map to
/// `out_path` (skipped if `out_path` already exists and is newer than the source).
pub fn generate_depth_map(
    estimator: &dyn DepthEstimator,
    image_path: &Path,
    out_path: &Path,
) -> Result<(), String> {
    if is_cached(image_path, out_path) {
        return Ok(());
    }
    let image = image::open(image_path)
        .map_err(|e| format!("open image {:?}: {e}", image_path))?;
    let depth = estimator.estimate(&image)?;
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    depth.save(out_path).map_err(|e| format!("save depth {:?}: {e}", out_path))?;
    Ok(())
}

/// True if `out_path` exists and is at least as new as `image_path`.
fn is_cached(image_path: &Path, out_path: &Path) -> bool {
    let (Ok(src), Ok(dst)) = (std::fs::metadata(image_path), std::fs::metadata(out_path)) else {
        return false;
    };
    match (src.modified(), dst.modified()) {
        (Ok(s), Ok(d)) => d >= s,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_shape_and_normalization() {
        let img = DynamicImage::new_rgb8(32, 32);
        let t = preprocess_imagenet(&img, 14);
        assert_eq!(t.len(), 3 * 14 * 14);
        // Black input → (0 - mean)/std on every channel, finite.
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn depth_field_normalizes_and_resizes() {
        // A horizontal gradient field → after dilate/blur/normalize it still spans
        // a wide range and resizes to the requested output size.
        let (w, h) = (32u32, 32u32);
        let mut field = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                field[(y * w + x) as usize] = x as f32; // 0..31 left→right
            }
        }
        let map = depth_field_to_map(&field, w, h, 96, 96, false);
        assert_eq!(map.dimensions(), (96, 96));
        let max = map.pixels().map(|p| p[0]).max().unwrap();
        let min = map.pixels().map(|p| p[0]).min().unwrap();
        assert!(max > 200, "expected near-white somewhere, got max {max}");
        assert!(min < 60, "expected near-black somewhere, got min {min}");
    }

    #[test]
    fn model_registry_and_recommendation() {
        // The embedded TOML parses and the expected models are present.
        let v3 = model_by_id("da3-small").expect("da3-small in registry");
        assert!(v3.is_v3());
        assert!(!v3.invert, "V3 metric distance is handled by disparity conversion, not invert");
        assert_eq!(v3.files.len(), 2, "V3 ships model.onnx + model.onnx_data");
        assert_eq!(v3.primary_file(), "model.onnx");

        let v2 = model_by_id("da2-small").expect("da2-small in registry");
        assert!(!v2.is_v3());
        assert_eq!(v2.primary_file(), "model_int8.onnx");

        assert!(model_by_id("birefnet-lite").is_some(), "BiRefNet lite present");
        assert_eq!(u2net_model().input_size, 320);
        assert_eq!(birefnet_model().input_size, 1024);
        assert_eq!(lama_model().input_size, 512);

        // Hardware drives the recommended tier bucket.
        assert_eq!(recommended_tier(GpuClass::Discrete).tier, "base");
        assert_eq!(recommended_tier(GpuClass::Integrated).tier, "small");
        assert_eq!(recommended_tier(GpuClass::Cpu).tier, "small");
    }

    // Real ONNX inference, only when a model is available. Run with:
    //   STRATA_DEPTH_MODEL=path/to/model.onnx cargo test -p core-engine \
    //       --features depth-onnx --test ... onnx_real -- --ignored --nocapture
    #[cfg(feature = "depth-onnx")]
    #[test]
    #[ignore]
    fn onnx_real_inference() {
        let Ok(model) = std::env::var("STRATA_DEPTH_MODEL") else {
            eprintln!("set STRATA_DEPTH_MODEL to a .onnx file to run this");
            return;
        };
        let mut img = image::RgbImage::new(96, 64);
        for (x, _y, px) in img.enumerate_pixels_mut() {
            let v = (x * 255 / 96) as u8;
            *px = image::Rgb([v, 128, 255 - v]);
        }
        let est = OnnxEstimator::new(model);
        let map = est.estimate(&DynamicImage::ImageRgb8(img)).expect("inference");
        assert_eq!(map.dimensions(), (96, 64));
        println!("ONNX depth OK: {:?}", map.dimensions());
    }

    #[test]
    fn heuristic_produces_full_size_map() {
        let mut img = image::RgbImage::new(64, 48);
        // Bright center, dark edges → center should read nearer (brighter).
        for y in 0..48 {
            for x in 0..64 {
                let near = x > 20 && x < 44 && y > 14 && y < 34;
                let v = if near { 240 } else { 30 };
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let est = HeuristicEstimator::default();
        let map = est.estimate(&DynamicImage::ImageRgb8(img)).unwrap();
        assert_eq!(map.dimensions(), (64, 48));
        let center = map.get_pixel(32, 24)[0];
        let corner = map.get_pixel(1, 1)[0];
        assert!(center > corner, "center {center} should be nearer than corner {corner}");
    }
}
