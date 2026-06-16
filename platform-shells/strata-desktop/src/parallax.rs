//! Parallax Studio orchestration (desktop side): download depth models on demand,
//! estimate a depth map, and export a playable parallax wallpaper into the library.
//!
//! The heavy ML path (ONNX Runtime) is behind the `depth-onnx` feature; without it
//! the Studio falls back to the dependency-free heuristic estimator, so the feature
//! works out of the box with zero downloads.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use core_engine::depth::{self, DepthEstimator, ModelChoice};
use core_engine::parallax::{export_wallpaper, ParallaxParams};

/// `%AppData%/strata/models` — where downloaded depth models live (never bundled).
pub fn models_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|b| b.data_dir().join("strata").join("models"))
}

/// On-disk directory for a model's files (`models/<id>/`).
pub fn model_dir(choice: &ModelChoice) -> Option<PathBuf> {
    models_dir().map(|d| d.join(&choice.id))
}

/// On-disk path of the `.onnx` graph ORT loads (`models/<id>/<primary>.onnx`).
pub fn model_file_path(choice: &ModelChoice) -> Option<PathBuf> {
    model_dir(choice).map(|d| d.join(choice.primary_file()))
}

/// True only when EVERY file of the model is present (multi-file models like
/// DepthAnything V3 ship a separate `model.onnx_data` weights blob).
pub fn is_model_downloaded(choice: &ModelChoice) -> bool {
    match model_dir(choice) {
        Some(dir) => choice.files.iter().all(|f| dir.join(&f.name).exists()),
        None => false,
    }
}

/// Download every file of `choice` to `models/<id>/`, reporting overall
/// `(downloaded, total)` bytes via `progress` across all files. Each file streams
/// to a `.part` and is renamed on success, so a partial download is never mistaken
/// for a complete model. Files already present are skipped.
pub fn download_model(choice: &ModelChoice, mut progress: impl FnMut(u64, u64)) -> Result<PathBuf, String> {
    let dir = model_dir(choice).ok_or("could not resolve models directory")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("create models dir: {e}"))?;

    // Total bytes across all files (Content-Length per file) for a smooth bar.
    let mut sizes = Vec::with_capacity(choice.files.len());
    let mut grand_total = 0u64;
    for f in &choice.files {
        let dest = dir.join(&f.name);
        if dest.exists() {
            let n = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            sizes.push(n);
            grand_total += n;
        } else {
            let n = ureq::head(&f.url).call().ok()
                .and_then(|r| r.header("Content-Length").and_then(|s| s.parse().ok()))
                .unwrap_or(0);
            sizes.push(n);
            grand_total += n;
        }
    }
    let grand_total = grand_total.max(1);

    let mut done_prev = 0u64; // bytes from already-finished files
    for (i, f) in choice.files.iter().enumerate() {
        let dest = dir.join(&f.name);
        if dest.exists() {
            done_prev += sizes[i];
            progress(done_prev, grand_total);
            continue;
        }
        log::info!("Downloading {} file {} ({} of {})", choice.id, f.name, i + 1, choice.files.len());
        let resp = ureq::get(&f.url).call().map_err(|e| format!("download {} failed: {e}", f.name))?;
        let tmp = dest.with_extension("part");
        let mut reader = resp.into_reader();
        let mut file = std::fs::File::create(&tmp).map_err(|e| format!("create {:?}: {e}", tmp))?;
        let mut buf = [0u8; 64 * 1024];
        let mut done = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| format!("read stream: {e}"))?;
            if n == 0 { break; }
            file.write_all(&buf[..n]).map_err(|e| format!("write model: {e}"))?;
            done += n as u64;
            progress(done_prev + done, grand_total);
        }
        file.flush().ok();
        drop(file);
        std::fs::rename(&tmp, &dest).map_err(|e| format!("finalize model: {e}"))?;
        done_prev += done;
    }
    log::info!("Model ready: {:?}", dir);
    model_file_path(choice).ok_or_else(|| "resolve model path".to_string())
}

/// The wallpaper library root (same resolution the rest of the shell uses).
pub fn wallpapers_base() -> PathBuf {
    if Path::new("wallpapers").exists() {
        PathBuf::from("wallpapers")
    } else {
        PathBuf::from("../../wallpapers")
    }
}

/// Estimate depth for `image`. Uses the ONNX model when built with `depth-onnx`
/// and a downloaded model is supplied; otherwise the heuristic estimator.
fn estimate_depth(image: &image::DynamicImage, model: Option<&ModelChoice>) -> Result<image::GrayImage, String> {
    #[cfg(feature = "depth-onnx")]
    {
        if let Some(choice) = model {
            if is_model_downloaded(choice) {
                if let Some(path) = model_file_path(choice) {
                    let est = core_engine::depth::OnnxEstimator::from_choice(path, choice);
                    return est.estimate(image);
                }
            }
        }
    }
    let _ = model; // unused without the onnx feature
    depth::HeuristicEstimator::default().estimate(image)
}

/// `%AppData%/strata/parallax-preview` — scratch package for the live preview
/// (photo + depth + baked shader). Reused across renders so depth is estimated once.
pub fn preview_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|b| b.data_dir().join("strata").join("parallax-preview"))
}

/// Estimate depth for `image_path` and build a parallax package in the preview
/// scratch dir (NOT the library). Returns the preview dir. Background-thread work.
pub fn build_preview(
    image_path: &Path,
    params: &ParallaxParams,
    model: Option<&ModelChoice>,
    seg: &ModelChoice,
    cinematic: bool,
    billboard: bool,
    upscaler: Option<&ModelChoice>,
    progress: impl Fn(u32),
) -> Result<PathBuf, String> {
    let _ = (seg, billboard, upscaler); // used only under the depth-onnx cinematic path
    let dir = preview_dir().ok_or("no preview dir")?;
    let _ = std::fs::remove_dir_all(&dir); // start clean each render
    progress(5);
    let image = image::open(image_path).map_err(|e| format!("open image {:?}: {e}", image_path))?;
    let depth = estimate_depth(&image, model)?;
    progress(if cinematic { 35 } else { 70 });

    // Cinematic (layered): inpaint the background behind the foreground subjects so
    // disocclusion reveals real pixels instead of smearing. Needs the ONNX build +
    // a downloaded LaMa model; otherwise we fall back to single-layer.
    if cinematic {
        #[cfg(feature = "depth-onnx")]
        {
            let lama = core_engine::depth::lama_model();
            if let (Some(seg_path), Some(lama_path)) = (model_file_path(seg), model_file_path(&lama)) {
                if seg_path.exists() && lama_path.exists() {
                    let dim = image.width().max(image.height());
                    // Full matte = every salient subject (incl. small ones like a
                    // distant dog). Main matte = just the primary subject(s); small
                    // ones are dropped so they aren't lifted into the foreground layer.
                    let alpha_full = core_engine::segment::segment_subject(&seg_path, &image, seg.input_size)?;
                    let mut alpha_main = alpha_full.clone();
                    core_engine::segment::drop_small_subjects(&mut alpha_main, 0.3);
                    progress(55);
                    // Inpaint removes ONLY the main subject (dog stays in the plate).
                    // Dilate enough to swallow the subject's soft edge; the inpaint
                    // seam is then blended a few px so the fill is hole-free + ghostless.
                    let inpaint_mask = core_engine::inpaint::foreground_mask(&alpha_main, 0.5, (dim / 150).max(8));
                    // Optional 4× upscaler to restore detail to the LaMa fill. Download
                    // on demand (first use), then pass its path to the inpainter.
                    let up_path = match upscaler {
                        Some(u) => {
                            if !is_model_downloaded(u) {
                                download_model(u, |_, _| {})?;
                            }
                            model_file_path(u)
                        }
                        None => None,
                    };
                    let bg = core_engine::inpaint::inpaint(&lama_path, &image, &inpaint_mask, up_path.as_deref())?;
                    progress(85);
                    // Background depth: flatten over ALL subjects (incl. the dog) by
                    // diffusion, so NOTHING in the bg layer has a depth cliff to fold
                    // around — the dog rides flat with the backdrop, keeping its pixels.
                    let flatten_mask = core_engine::inpaint::foreground_mask(&alpha_full, 0.5, (dim / 150).max(8));
                    let bg_depth = core_engine::inpaint::fill_masked_depth(&depth, &flatten_mask);
                    // Foreground layer matte = primary subject only (drives the composite).
                    let soft = core_engine::inpaint::feather_mask(&alpha_main, (dim / 220).max(2));
                    core_engine::parallax::export_layered_wallpaper(
                        &dir, "Parallax Preview", "Parallax Studio", image_path, &depth, &bg_depth, &bg, &soft, params, billboard,
                    )?;
                    progress(100);
                    log::info!("Cinematic preview: {} matte + LaMa inpaint + diffusion bg-depth", seg.name);
                    return Ok(dir);
                }
            }
        }
        log::warn!("Cinematic requested but segmentation/inpaint models unavailable — single-layer");
    }

    export_wallpaper(&dir, "Parallax Preview", "Parallax Studio", image_path, &depth, params)?;
    progress(100);
    Ok(dir)
}

/// Re-bake just the shader in `dir` with new `params` (depth/photo unchanged) — for
/// live slider tuning without re-estimating depth. (Wired up with the tuning sliders.)
#[allow(dead_code)]
pub fn rebake_params(dir: &Path, params: &ParallaxParams) -> Result<(), String> {
    std::fs::write(dir.join("image.glsl"), core_engine::parallax::parallax_shader(params))
        .map_err(|e| format!("rebake shader: {e}"))
}

/// Promote a preview package into the library under `display_name` with `params`
/// (re-uses the already-estimated depth — no inference). Returns the library dir.
pub fn save_to_library(preview: &Path, display_name: &str, params: &ParallaxParams) -> Result<PathBuf, String> {
    let out = unique_dir(&wallpapers_base(), &slugify(display_name));
    std::fs::create_dir_all(&out).map_err(|e| format!("create {:?}: {e}", out))?;
    // Copy the preview's assets verbatim — including image.glsl, which has the baked
    // shader (the layered shader bakes SUBJECT_DEPTH, so we must NOT re-derive it from
    // params here). Only the manifest is rewritten, to carry the library display name.
    let layered = preview.join("background.png").exists() && preview.join("mask.png").exists();
    for f in ["image.png", "depth.png", "background.png", "mask.png", "image.glsl"] {
        let src = preview.join(f);
        if src.exists() {
            std::fs::copy(&src, out.join(f)).map_err(|e| format!("copy {f}: {e}"))?;
        }
    }
    let _ = params; // shader already baked in the copied image.glsl
    core_engine::parallax::write_manifest(&out, display_name, "Parallax Studio", layered)?;
    log::info!("Parallax wallpaper saved to {:?}", out);
    Ok(out)
}

fn slugify(name: &str) -> String {
    let mut s = String::new();
    let mut last_dash = false;
    for c in name.trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c);
            last_dash = false;
        } else if !last_dash {
            s.push('-');
            last_dash = true;
        }
    }
    let s = s.trim_matches('-').to_string();
    if s.is_empty() { "parallax".to_string() } else { s }
}

/// True if this build can actually run ONNX models (else only the heuristic).
pub fn onnx_available() -> bool {
    cfg!(feature = "depth-onnx")
}

/// Dropdown labels: the no-model heuristic, plus the DepthAnything tiers ONLY when
/// this build can actually run them (avoids offering models that silently fall back
/// to the heuristic in a non-ONNX build).
pub fn model_options() -> Vec<String> {
    let mut v = vec!["Fast (no model)".to_string()];
    if onnx_available() {
        v.extend(depth::depth_models().iter().map(|m| m.name.clone()));
    }
    v
}

/// Map a dropdown label back to its depth model (None = the heuristic option).
pub fn tier_for_label(label: &str) -> Option<ModelChoice> {
    depth::depth_models().iter().find(|m| m.name == label).cloned()
}

/// Status line + whether a Download button should show, for a chosen model file.
pub fn model_status(choice: &ModelChoice) -> (String, bool) {
    if !onnx_available() {
        ("Needs the ONNX build to run".to_string(), false)
    } else if is_model_downloaded(choice) {
        (format!("Ready · {}", choice.license), false)
    } else {
        (format!("Not downloaded · {} MB · {}", choice.size_mb, choice.license), true)
    }
}

/// Upscaler dropdown options: "Off" plus each upscaler model (restores detail to the
/// LaMa-inpainted background). Only meaningful in the ONNX build + Cinematic mode.
pub fn upscaler_options() -> Vec<String> {
    let mut v = vec!["Off".to_string()];
    v.extend(depth::upscale_models().iter().map(|m| m.name.clone()));
    v
}

/// Resolve an upscaler dropdown label to its model (None = "Off").
pub fn upscaler_choice_for_label(label: &str) -> Option<ModelChoice> {
    depth::upscale_models().iter().find(|m| m.name == label).cloned()
}

/// Parallax-style dropdown options for the Cinematic (layered) path. "Coherent 3D"
/// keeps the subject grounded via per-pixel depth; "Billboard" is a rock-steady flat
/// cut-out layer (good for cleanly isolated/graphic subjects).
pub fn parallax_style_options() -> Vec<String> {
    vec!["Coherent 3D (grounded)".to_string(), "Billboard (flat layers)".to_string()]
}

/// True if the dropdown label selects the billboard style.
pub fn style_is_billboard(label: &str) -> bool {
    label.starts_with("Billboard")
}

/// Masking-model dropdown options (subject matte). U²-Net is the lightweight
/// default; BiRefNet-lite is sharper but larger.
pub fn seg_model_options() -> Vec<String> {
    depth::segment_models().iter().map(|m| m.name.clone()).collect()
}

/// Resolve a masking dropdown label to its model (defaults to the first entry).
pub fn seg_choice_for_label(label: &str) -> ModelChoice {
    depth::segment_models().iter()
        .find(|m| m.name == label)
        .or_else(|| depth::segment_models().first())
        .cloned()
        .unwrap_or_else(depth::u2net_model)
}

/// Cinematic mode needs the chosen segmentation model + LaMa (inpaint).
pub fn cinematic_models(seg: &ModelChoice) -> [ModelChoice; 2] {
    [seg.clone(), core_engine::depth::lama_model()]
}

/// Status line + whether a Download button should show for Cinematic mode.
pub fn cinematic_status(seg: &ModelChoice) -> (String, bool) {
    if !onnx_available() {
        return ("Needs the ONNX build to run".to_string(), false);
    }
    let models = cinematic_models(seg);
    if models.iter().all(is_model_downloaded) {
        ("Ready · subject + inpaint models".to_string(), false)
    } else {
        let total: u32 = models.iter().map(|m| m.size_mb).sum();
        (format!("Not downloaded · {} MB total", total), true)
    }
}

/// Download both cinematic models (chosen matter + LaMa), reporting 0..100 %.
pub fn download_cinematic(seg: &ModelChoice, mut progress: impl FnMut(u32)) -> Result<(), String> {
    let models = cinematic_models(seg);
    let total_mb: f64 = models.iter().map(|m| m.size_mb as f64).sum::<f64>().max(1.0);
    let mut done_mb = 0.0f64;
    for m in &models {
        if is_model_downloaded(m) {
            done_mb += m.size_mb as f64;
            progress((done_mb / total_mb * 100.0) as u32);
            continue;
        }
        let base = done_mb;
        let mb = m.size_mb as f64;
        download_model(m, |d, t| {
            let frac = if t > 0 { d as f64 / t as f64 } else { 0.0 };
            progress(((base + frac * mb) / total_mb * 100.0) as u32);
        })?;
        done_mb += mb;
    }
    Ok(())
}

/// Append `-2`, `-3`, … if the slug dir already exists, so creations never clobber.
fn unique_dir(base: &Path, slug: &str) -> PathBuf {
    let first = base.join(slug);
    if !first.exists() {
        return first;
    }
    for n in 2..1000 {
        let p = base.join(format!("{slug}-{n}"));
        if !p.exists() {
            return p;
        }
    }
    first
}
