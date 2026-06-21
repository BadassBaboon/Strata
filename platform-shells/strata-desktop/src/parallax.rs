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

/// `%AppData%/strata/models` - where downloaded depth models live (never bundled).
pub fn models_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|b| b.data_dir().join("Strata").join("models"))
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
            let n = crate::library_sync::http_agent().head(&f.url).call().ok()
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
        let resp = crate::library_sync::http_agent().get(&f.url).call().map_err(|e| format!("download {} failed: {e}", f.name))?;
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

/// `%AppData%/strata/parallax-preview` - scratch package for the live preview
/// (photo + depth + baked shader). Reused across renders so depth is estimated once.
pub fn preview_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|b| b.data_dir().join("Strata").join("parallax-preview"))
}

/// Estimate depth for `image_path` and build a parallax package in the preview
/// scratch dir (NOT the library). Returns the preview dir. Background-thread work.
/// Wipe the preview scratch dir so every render starts from a CLEAN package. A single
/// `remove_dir_all` can fail intermittently on Windows - an antivirus scan (or any
/// transient handle) on the just-written PNGs / model file briefly locks a file. If that
/// failure is ignored, stale files survive: e.g. a previous layered render's
/// `background.png` lingers and the next render's preview shows the OLD inpainted fill
/// (the "inpainted/upscaled background won't apply" bug). So retry briefly, then belt-and-
/// suspenders remove the files that decide layered-vs-single so a leftover can't leak in.
fn prepare_clean_preview_dir(dir: &Path) -> Result<(), String> {
    for attempt in 0..6 {
        match std::fs::remove_dir_all(dir) {
            Ok(_) => break,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
            Err(_) if attempt < 5 => std::thread::sleep(std::time::Duration::from_millis(150)),
            Err(e) => log::warn!("preview dir not fully cleaned ({e}); clearing key files"),
        }
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("create preview dir: {e}"))?;
    // These decide how the preview package is interpreted; never let stale ones survive.
    for f in ["background.png", "mask.png", "image.glsl", "manifest.toml", "depth.png", "image.png"] {
        let _ = std::fs::remove_file(dir.join(f));
    }
    Ok(())
}

pub fn build_preview(
    image_path: &Path,
    params: &ParallaxParams,
    model: Option<&ModelChoice>,
    seg: &ModelChoice,
    _cinematic: bool, // retained for call-site compat; generation is always cinematic now
    billboard: bool,
    upscaler: Option<&ModelChoice>,
    progress: impl Fn(u32),
) -> Result<PathBuf, String> {
    let _ = (seg, billboard, upscaler); // used only under the depth-onnx cinematic path
    let dir = preview_dir().ok_or("no preview dir")?;
    prepare_clean_preview_dir(&dir)?; // reliably clean BEFORE writing the new package
    progress(5);
    let image = image::open(image_path).map_err(|e| format!("open image {:?}: {e}", image_path))?;
    let depth = estimate_depth(&image, model)?;
    progress(35);

    // Generation is ALWAYS layered/cinematic now - it's the only mode (the plain
    // non-layered output was inferior in ~99% of cases). The block below builds the
    // layered package and returns. It only falls through to the single-layer fallback
    // when the ONNX build or the required models are unavailable - and Automatic mode
    // downloads the preset's models first, so that path is just a safety net.
    {
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
                    // around - the dog rides flat with the backdrop, keeping its pixels.
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
        log::warn!("Cinematic models unavailable (ONNX build / models missing) - single-layer fallback");
    }

    export_wallpaper(&dir, "Parallax Preview", "Parallax Studio", image_path, &depth, params)?;
    progress(100);
    Ok(dir)
}

/// Re-bake just the shader in `dir` with new `params` (depth/photo unchanged) - for
/// live slider tuning without re-estimating depth. (Wired up with the tuning sliders.)
#[allow(dead_code)]
pub fn rebake_params(dir: &Path, params: &ParallaxParams) -> Result<(), String> {
    std::fs::write(dir.join("image.glsl"), core_engine::parallax::parallax_shader(params))
        .map_err(|e| format!("rebake shader: {e}"))
}

/// Promote a preview package into the library under `display_name` with `params`
/// (re-uses the already-estimated depth - no inference). Returns the library dir.
pub fn save_to_library(preview: &Path, display_name: &str, params: &ParallaxParams) -> Result<PathBuf, String> {
    // Parallax creations live in %APPDATA%/strata/parallax-wallpapers (user data),
    // not the bundled/read-only install library.
    let base = crate::controller::parallax_library_dir()
        .ok_or_else(|| "Could not resolve the user data directory".to_string())?;
    std::fs::create_dir_all(&base).map_err(|e| format!("create {:?}: {e}", base))?;
    let out = unique_dir(&base, &slugify(display_name));
    std::fs::create_dir_all(&out).map_err(|e| format!("create {:?}: {e}", out))?;
    // Copy the preview's assets verbatim - including image.glsl, which has the baked
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

/// Manual-mode depth-estimator dropdown labels. In the ONNX build the user MUST pick a
/// real model (no heuristic "no model" option). Only a non-ONNX build (which can't run
/// any model) falls back to offering the heuristic so the dropdown isn't empty.
pub fn model_options() -> Vec<String> {
    if onnx_available() {
        depth::depth_models().iter().map(|m| clean_label(&m.name)).collect()
    } else {
        vec!["Fast (no model)".to_string()]
    }
}

/// Map a dropdown label back to its depth model (None = the heuristic option).
pub fn tier_for_label(label: &str) -> Option<ModelChoice> {
    depth::depth_models().iter().find(|m| clean_label(&m.name) == label).cloned()
}

/// Upscaler dropdown options: "Off" plus each upscaler model (restores detail to the
/// LaMa-inpainted background). Only meaningful in the ONNX build + Cinematic mode.
pub fn upscaler_options() -> Vec<String> {
    let mut v = vec!["Off".to_string()];
    v.extend(depth::upscale_models().iter().map(|m| clean_label(&m.name)));
    v
}

/// Resolve an upscaler dropdown label to its model (None = "Off").
pub fn upscaler_choice_for_label(label: &str) -> Option<ModelChoice> {
    depth::upscale_models().iter().find(|m| clean_label(&m.name) == label).cloned()
}

// ── Clean dropdown labels ────────────────────────────────────────────────────────
// Model `name`s in models.toml carry backend detail (precision · size · license), e.g.
// "Depth Anything V2 Base · fp16 (195 MB · non-commercial)". The UI should show only the
// human name; size/license are backend-only (shown in the download queue). Strip at the
// first " · " or " (".
pub fn clean_label(name: &str) -> String {
    let n = name.trim();
    let mut end = n.len();
    if let Some(i) = n.find(" · ") { end = end.min(i); }
    if let Some(i) = n.find(" (") { end = end.min(i); }
    n[..end].trim().to_string()
}

// ── Quality presets (Automatic mode) ──────────────────────────────────────────────

/// Preset dropdown labels (preset display names, in registry order).
pub fn preset_options() -> Vec<String> {
    depth::presets().iter().map(|p| p.name.clone()).collect()
}

/// Resolve a preset dropdown label to its preset (defaults to the first).
pub fn preset_for_label(label: &str) -> &'static depth::Preset {
    depth::presets().iter().find(|p| p.name == label)
        .or_else(|| depth::presets().first())
        .expect("at least one preset")
}

/// Every model a preset needs to generate (depth + matte + LaMa inpaint + optional
/// upscaler), in download order. Used to check availability + drive the download queue.
pub fn preset_required_models(p: &depth::Preset) -> Vec<ModelChoice> {
    let mut v = Vec::new();
    if let Some(m) = depth::model_by_id(&p.depth) { v.push(m.clone()); }
    if let Some(m) = depth::model_by_id(&p.segment) { v.push(m.clone()); }
    v.push(depth::lama_model());
    if let Some(m) = depth::preset_upscaler(p) { v.push(m.clone()); }
    v
}

/// The subset of `models` not yet on disk (the download queue for a preset/selection).
pub fn missing_models(models: &[ModelChoice]) -> Vec<ModelChoice> {
    models.iter().filter(|m| !is_model_downloaded(m)).cloned().collect()
}

/// Download a list of models sequentially. `progress(index, total, name, size_mb, pct)`
/// fires as each file streams (index is 1-based for "(i/N)" UI). Skips ones already on
/// disk. Returns on the first failure.
pub fn download_models_queue(
    models: &[ModelChoice],
    mut progress: impl FnMut(usize, usize, &str, u32, u8),
) -> Result<(), String> {
    let queue: Vec<&ModelChoice> = models.iter().filter(|m| !is_model_downloaded(m)).collect();
    let total = queue.len();
    for (i, m) in queue.iter().enumerate() {
        let label = clean_label(&m.name);
        progress(i + 1, total, &label, m.size_mb, 0);
        download_model(m, |done, all| {
            let pct = if all > 0 { ((done * 100) / all).min(100) as u8 } else { 0 };
            progress(i + 1, total, &label, m.size_mb, pct);
        })?;
        progress(i + 1, total, &label, m.size_mb, 100);
    }
    Ok(())
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
    depth::segment_models().iter().map(|m| clean_label(&m.name)).collect()
}

/// Resolve a masking dropdown label to its model (defaults to the first entry).
pub fn seg_choice_for_label(label: &str) -> ModelChoice {
    depth::segment_models().iter()
        .find(|m| clean_label(&m.name) == label)
        .or_else(|| depth::segment_models().first())
        .cloned()
        .unwrap_or_else(depth::u2net_model)
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
