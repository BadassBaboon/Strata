//! One-shot tool that assembles the `Strata-Library/` prototype from the in-repo
//! sources: it copies every wallpaper folder, the shared `external/` assets, and
//! `models.toml`/`presets.toml`, generates an in-folder `thumbnail.png` for each
//! wallpaper, and emits `index.toml` (schema/library version + per-item hashes).
//!
//! Run it whenever the library content changes:
//!   cargo test -p core-engine --test assemble_library -- --ignored --nocapture
//!
//! It is `#[ignore]`d so normal `cargo test` never does GPU work or rewrites files.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

const LIBRARY_VERSION: &str = "1.0.0";
const SCHEMA_VERSION: u32 = 1;
const TODAY: &str = "2026-06-18";

#[test]
#[ignore]
fn assemble_library() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let src_wallpapers = root.join("wallpapers");
    let ext_src = root.join("assets").join("external");
    let lib = root.join("Strata-Library");
    // Wallpaper folders live under shader-library/; index.toml, models.toml,
    // presets.toml and external/ sit at the library root (mirrors the runtime
    // %APPDATA%/strata/strata-library/ layout — as if the repo were cloned there).
    let shader_lib = lib.join("shader-library");
    std::fs::create_dir_all(&shader_lib).unwrap();

    // models.toml / presets.toml live at the library root (mirrors the runtime
    // %APPDATA%/strata/strata-library/ layout).
    let models_src = root.join("core-engine/src/resources/models.toml");
    let presets_src = root.join("core-engine/src/resources/presets.toml");
    std::fs::copy(&models_src, lib.join("models.toml")).unwrap();
    std::fs::copy(&presets_src, lib.join("presets.toml")).unwrap();
    let models_hash = sha256_file(&models_src);
    let presets_hash = sha256_file(&presets_src);

    // Shared texture/cubemap assets → external/.
    copy_dir(&ext_src, &lib.join("external"));
    // In case any shader references a texture, let the engine resolve from external/.
    core_engine::set_asset_dirs(vec![ext_src.clone()]);

    let ctx = Arc::new(pollster::block_on(core_engine::GraphicsContext::new_render_only()).unwrap());

    // Each wallpaper folder (must have manifest.toml), sorted by slug.
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&src_wallpapers).unwrap()
        .filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("manifest.toml").exists())
        .collect();
    dirs.sort();

    let mut shader_entries = String::new();
    for d in &dirs {
        let slug = d.file_name().unwrap().to_string_lossy().to_string();
        let dst = shader_lib.join(&slug);
        copy_dir(d, &dst);

        // Generate the in-folder thumbnail if the source didn't already ship one.
        let thumb = dst.join("thumbnail.png");
        if !thumb.exists() {
            match core_engine::thumbnail::generate_thumbnail(ctx.clone(), &dst, &thumb, 480, 270) {
                Ok(()) => println!("thumb OK   {slug}"),
                Err(e) => println!("thumb FAIL {slug}: {e}"),
            }
        }

        let cfg = core_engine::WallpaperConfig::load_from_dir(&dst).unwrap();
        let w = &cfg.wallpaper;
        // Hash the AUTHORED content (manifest + every .glsl) so the client can
        // detect a real shader change; the derived thumbnail is excluded.
        let hash = sha256_shader(&dst);

        shader_entries.push_str("\n[[shader]]\n");
        shader_entries.push_str(&format!("slug = {}\n", toml_str(&slug)));
        shader_entries.push_str(&format!("name = {}\n", toml_str(&w.name)));
        shader_entries.push_str(&format!("author = {}\n", toml_str(&w.author)));
        if !w.source_url.is_empty() {
            shader_entries.push_str(&format!("source_url = {}\n", toml_str(&w.source_url)));
        }
        let tags = w.tags.iter().map(|t| toml_str(t)).collect::<Vec<_>>().join(", ");
        shader_entries.push_str(&format!("tags = [{tags}]\n"));
        shader_entries.push_str(&format!("added_in = {}\n", toml_str(LIBRARY_VERSION)));
        shader_entries.push_str(&format!("updated = {}\n", toml_str(TODAY)));
        shader_entries.push_str(&format!("sha256 = {}\n", toml_str(&hash)));
    }

    // index.toml
    let mut index = String::new();
    index.push_str("# Strata-Library manifest. Regenerate with:\n");
    index.push_str("#   cargo test -p core-engine --test assemble_library -- --ignored\n\n");
    index.push_str(&format!("schema_version = {SCHEMA_VERSION}\n"));
    index.push_str(&format!("library_version = {}\n\n", toml_str(LIBRARY_VERSION)));
    index.push_str("# Registry files (hash lets a client know when to re-fetch them).\n");
    index.push_str("[files.models]\n");
    index.push_str(&format!("version = {}\nsha256 = {}\n\n", toml_str(LIBRARY_VERSION), toml_str(&models_hash)));
    index.push_str("[files.presets]\n");
    index.push_str(&format!("version = {}\nsha256 = {}\n", toml_str(LIBRARY_VERSION), toml_str(&presets_hash)));
    index.push_str(&shader_entries);
    std::fs::write(lib.join("index.toml"), index).unwrap();

    println!("\nAssembled {} shaders into {:?}", dirs.len(), lib);
}

/// sha256 of a single file's bytes, lowercase hex.
fn sha256_file(p: &Path) -> String {
    let bytes = std::fs::read(p).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(&bytes);
    hex(&h.finalize())
}

/// sha256 over a shader's authored files (manifest.toml + every *.glsl), sorted by
/// name so the digest is stable; the generated thumbnail/assets are excluded.
fn sha256_shader(dir: &Path) -> String {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).unwrap()
        .filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            n == "manifest.toml" || n.ends_with(".glsl")
        })
        .collect();
    files.sort();
    let mut h = Sha256::new();
    for f in files {
        h.update(f.file_name().unwrap().to_string_lossy().as_bytes());
        h.update(b"\0");
        h.update(&std::fs::read(&f).unwrap_or_default());
    }
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap().filter_map(|e| e.ok()) {
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to);
        } else {
            std::fs::copy(&from, &to).unwrap();
        }
    }
}

fn toml_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
