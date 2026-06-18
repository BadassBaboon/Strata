//! Shadertoy `.json` → Strata wallpaper package converter.
//!
//! A Shadertoy export (the `.json` produced by the Shadertoy export browser
//! plugin, or the inner file of its `.zip`) is a set of render passes with GLSL
//! `code` plus channel `inputs`. This module ports one into Strata's on-disk
//! format: a `manifest.toml`, one `.glsl` per pass (`image.glsl`, `bufferA.glsl`
//! …, `common.glsl`), and any texture/cubemap assets copied in beside them.
//!
//! Assets are resolved from `assets_dir` (the bundled `assets/external`, seeded
//! from the Shadertoy desktop app's media cache) by the basename of the input's
//! `filepath` (`/media/a/<hash>.<ext>` → `<hash>.<ext>`); cubemaps additionally
//! pull the five sibling faces `<hash>_1..<hash>_5.<ext>`.
//!
//! Conversion does a GLSL compile check (naga front-end, no GPU device) of every
//! pass so an incompatible shader fails here with a reason, before it is added to
//! the library. Codegen/pipeline issues that only surface with a real device are
//! caught later by thumbnail generation.

use serde::Deserialize;
use std::path::Path;

use crate::preprocessor::{preprocess_shader, compile_shader_mapped};

#[derive(Deserialize)]
struct StExport {
    #[serde(default)]
    renderpass: Vec<StPass>,
    #[serde(default)]
    info: StInfo,
}

#[derive(Deserialize, Default)]
struct StInfo {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    username: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct StPass {
    #[serde(default)]
    inputs: Vec<StInput>,
    #[serde(default)]
    outputs: Vec<StOutput>,
    #[serde(default)]
    code: String,
    #[serde(default)]
    name: String,
    #[serde(rename = "type", default)]
    ptype: String,
}

#[derive(Deserialize)]
struct StInput {
    channel: u32,
    #[serde(rename = "type", default)]
    itype: String,
    #[serde(default)]
    id: serde_json::Value,
    #[serde(default)]
    filepath: String,
}

#[derive(Deserialize)]
struct StOutput {
    #[serde(default)]
    id: serde_json::Value,
}

/// Result of a successful conversion.
pub struct ImportReport {
    /// Human-readable shader name (from the export's `info.name`).
    pub name: String,
    /// Non-fatal notes (skipped/unsupported inputs or passes) worth logging.
    pub warnings: Vec<String>,
}

/// Peek at a Shadertoy export's display name without doing a full conversion,
/// so the caller can derive a folder slug (e.g. "Clearly a bug" → clearly-a-bug)
/// before creating the destination directory. Returns `None` if unparseable.
pub fn peek_name(json: &str) -> Option<String> {
    let export: StExport = serde_json::from_str(json).ok()?;
    Some(display_name(&export.info))
}

/// Stable key for a Shadertoy render-target id (number or string), so buffer
/// inputs can be matched to the pass that produces them regardless of the
/// export format version.
fn id_key(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Map a Shadertoy pass `type`/`name` to (manifest pass name, glsl filename).
/// Returns `None` for passes Strata doesn't render as a screen/buffer pass
/// (`common` is handled separately; `sound`/`cubemap` render passes are skipped).
fn pass_target(ptype: &str, name: &str) -> Option<(String, String)> {
    match ptype {
        "image" => Some(("image".to_string(), "image.glsl".to_string())),
        "buffer" => {
            // Letter from the pass name, e.g. "Buffer A" → A.
            let letter = name.chars().rev().find(|c| c.is_ascii_alphabetic()).unwrap_or('A')
                .to_ascii_uppercase();
            Some((format!("buffer{letter}"), format!("buffer{letter}.glsl")))
        }
        _ => None,
    }
}

/// Convert a Shadertoy export into a Strata wallpaper package at `dest_dir`
/// (created fresh; must not already exist). Texture/cubemap inputs are referenced
/// by file name and must resolve against a shared asset root registered via
/// [`crate::set_asset_dirs`] (the Strata-Library `external/` dir) — they are NOT
/// copied into the wallpaper folder, so library assets live in one place.
pub fn convert_shadertoy(
    json: &str,
    dest_dir: &Path,
) -> Result<ImportReport, String> {
    let export: StExport = serde_json::from_str(json)
        .map_err(|e| format!("Not a valid Shadertoy export: {e}"))?;
    if export.renderpass.is_empty() {
        return Err("Shadertoy export has no render passes".to_string());
    }

    let mut warnings: Vec<String> = Vec::new();

    // Build output-id → pass-name map (for resolving `buffer` inputs) and gather
    // the common-pass source.
    let mut id_to_pass: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut common_src: Option<String> = None;
    for p in &export.renderpass {
        if p.ptype == "common" {
            common_src = Some(p.code.clone());
            continue;
        }
        if let Some((pass_name, _)) = pass_target(&p.ptype, &p.name) {
            for o in &p.outputs {
                let k = id_key(&o.id);
                if !k.is_empty() {
                    id_to_pass.insert(k, pass_name.clone());
                }
            }
        }
    }

    if !export.renderpass.iter().any(|p| p.ptype == "image") {
        return Err("Shadertoy export has no Image pass".to_string());
    }

    fs_create_dir(dest_dir)?;

    // Emitted .glsl files carry the raw Shadertoy code (only the naga `mat2` fix
    // is applied) so they match shadertoy.com — name/author/source_url live in
    // manifest.toml, not in a comment header.

    // Write common.glsl (injected into every pass by the engine).
    if let Some(ref common) = common_src {
        write_file(&dest_dir.join("common.glsl"), &patch_naga_mat2(common))?;
    }

    // Per-pass: write glsl + collect manifest binding lines.
    struct PassOut {
        name: String,
        source: String,
        bindings: Vec<String>, // manifest binding TOML fragments
    }
    let mut pass_outs: Vec<PassOut> = Vec::new();

    for p in &export.renderpass {
        if p.ptype == "common" {
            continue;
        }
        let Some((pass_name, file_name)) = pass_target(&p.ptype, &p.name) else {
            warnings.push(format!("Skipped unsupported pass type '{}' ({})", p.ptype, p.name));
            continue;
        };

        write_file(&dest_dir.join(&file_name), &patch_naga_mat2(&p.code))?;

        let mut bindings: Vec<String> = Vec::new();
        for inp in &p.inputs {
            if inp.channel >= 4 {
                warnings.push(format!("{pass_name}: ignoring input on channel {} (>3)", inp.channel));
                continue;
            }
            match inp.itype.as_str() {
                "texture" => {
                    let base = basename(&inp.filepath)
                        .ok_or_else(|| format!("{pass_name}: texture input has no filepath"))?;
                    require_asset(&base)?;
                    bindings.push(format!(
                        "    {{ channel = {}, type = \"texture\", path = \"{}\" }}",
                        inp.channel, base
                    ));
                }
                "cubemap" => {
                    let base = basename(&inp.filepath)
                        .ok_or_else(|| format!("{pass_name}: cubemap input has no filepath"))?;
                    require_cubemap(&base)?;
                    bindings.push(format!(
                        "    {{ channel = {}, type = \"cubemap\", path = \"{}\" }}",
                        inp.channel, base
                    ));
                }
                "buffer" => {
                    let key = id_key(&inp.id);
                    if let Some(target) = id_to_pass.get(&key) {
                        bindings.push(format!(
                            "    {{ channel = {}, type = \"buffer\", target = \"{}\" }}",
                            inp.channel, target
                        ));
                    } else {
                        warnings.push(format!(
                            "{pass_name}: channel {} references unknown buffer (id {key}); left unbound",
                            inp.channel
                        ));
                    }
                }
                "music" | "musicstream" | "mic" => {
                    // Strata feeds the live system-audio FFT/waveform texture to any
                    // audio channel; the original track/stream isn't bundled.
                    bindings.push(format!(
                        "    {{ channel = {}, type = \"audio\" }}",
                        inp.channel
                    ));
                }
                other => {
                    warnings.push(format!(
                        "{pass_name}: channel {} input type '{other}' is unsupported; left unbound",
                        inp.channel
                    ));
                }
            }
        }

        pass_outs.push(PassOut { name: pass_name, source: file_name, bindings });
    }

    // Canonical pass order: buffers A..D, then image.
    pass_outs.sort_by_key(|p| match p.name.as_str() {
        "image" => 99,
        n if n.starts_with("buffer") => n.bytes().last().unwrap_or(b'A') as i32 - b'A' as i32,
        _ => 50,
    });

    // Build manifest.toml.
    let mut toml = String::new();
    toml.push_str("[wallpaper]\n");
    toml.push_str(&format!("name = {}\n", toml_str(&display_name(&export.info))));
    toml.push_str(&format!("author = {}\n", toml_str(if export.info.username.is_empty() { "Shadertoy" } else { &export.info.username })));
    toml.push_str("version = \"1.0.0\"\n");
    // Original Shadertoy page for attribution (satisfies the license's credit
    // clause; surfaced as a clickable "Made by …" link in the Library).
    if !export.info.id.is_empty() {
        toml.push_str(&format!("source_url = {}\n", toml_str(&format!("https://www.shadertoy.com/view/{}", export.info.id))));
    }

    // Tags: the export's tags (title-cased-ish, deduped) plus the "Imported"
    // marker that drives the red Library badge + delete affordance.
    let mut tags: Vec<String> = Vec::new();
    for t in &export.info.tags {
        let t = t.trim();
        if !t.is_empty() && !tags.iter().any(|x| x.eq_ignore_ascii_case(t)) {
            tags.push(t.to_string());
        }
    }
    if !tags.iter().any(|t| t.eq_ignore_ascii_case("Imported")) {
        tags.push("Imported".to_string());
    }
    let tag_list = tags.iter().map(|t| toml_str(t)).collect::<Vec<_>>().join(", ");
    toml.push_str(&format!("tags = [{tag_list}]\n"));

    let pass_names = pass_outs.iter().map(|p| toml_str(&p.name)).collect::<Vec<_>>().join(", ");
    toml.push_str(&format!("passes = [{pass_names}]\n\n"));

    for p in &pass_outs {
        toml.push_str(&format!("[render_targets.{}]\n", p.name));
        toml.push_str(&format!("source = {}\n", toml_str(&p.source)));
        if p.bindings.is_empty() {
            toml.push_str("bindings = []\n\n");
        } else {
            toml.push_str("bindings = [\n");
            toml.push_str(&p.bindings.join(",\n"));
            toml.push_str("\n]\n\n");
        }
    }

    write_file(&dest_dir.join("manifest.toml"), &toml)?;

    // GLSL compile check (naga front-end). Catches incompatible shaders here so
    // the caller can surface a clear "import failed" toast and discard the folder.
    let common_for_check = if common_src.is_some() {
        std::fs::read_to_string(dest_dir.join("common.glsl")).ok()
    } else {
        None
    };
    let common_for_check = common_for_check.as_deref();
    for p in &pass_outs {
        let glsl = std::fs::read_to_string(dest_dir.join(&p.source))
            .map_err(|e| format!("read back {}: {e}", p.source))?;
        let cube_channels = cube_mask_from_bindings(&p.bindings);
        let (pre, map) = preprocess_shader(
            &glsl,
            common_for_check,
            wgpu::naga::ShaderStage::Fragment,
            p.name == "image",
            cube_channels,
        );
        compile_shader_mapped(&pre, wgpu::naga::ShaderStage::Fragment, Some(&map))
            .map_err(|e| format!("Pass '{}' failed to compile on the wgpu backend:\n{e}", p.name))?;
    }

    Ok(ImportReport { name: display_name(&export.info), warnings })
}

/// Best display name for the shader.
fn display_name(info: &StInfo) -> String {
    if info.name.trim().is_empty() {
        "Imported Shader".to_string()
    } else {
        info.name.trim().to_string()
    }
}

/// Reconstruct a `[bool;4]` cubemap mask from the manifest binding fragments we
/// just generated (used only for the local compile check).
fn cube_mask_from_bindings(bindings: &[String]) -> [bool; 4] {
    let mut mask = [false; 4];
    for b in bindings {
        if b.contains("type = \"cubemap\"") {
            // channel = N
            if let Some(rest) = b.split("channel = ").nth(1) {
                if let Some(n) = rest.split(|c: char| !c.is_ascii_digit()).find(|s| !s.is_empty()) {
                    if let Ok(ch) = n.parse::<usize>() {
                        if ch < 4 { mask[ch] = true; }
                    }
                }
            }
        }
    }
    mask
}

/// Basename of a Shadertoy `filepath` like `/media/a/<hash>.<ext>`.
fn basename(filepath: &str) -> Option<String> {
    let trimmed = filepath.trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    let name = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
    if name.is_empty() { None } else { Some(name.to_string()) }
}

/// Verify a texture asset is resolvable in a shared asset root (it is referenced
/// by name, not copied into the wallpaper folder).
fn require_asset(base: &str) -> Result<(), String> {
    // Resolve against an empty wallpaper dir so only the shared roots are checked.
    if crate::asset_exists(Path::new(""), base) {
        Ok(())
    } else {
        Err(format!(
            "Required texture '{base}' isn't in the Strata-Library asset directory"
        ))
    }
}

/// Verify a cubemap's 6 faces (`<stem>.<ext>` + `<stem>_1..<stem>_5.<ext>`) all
/// resolve in a shared asset root.
fn require_cubemap(base: &str) -> Result<(), String> {
    let (stem, ext) = match base.rsplit_once('.') {
        Some((s, e)) => (s, e),
        None => return Err(format!("cubemap '{base}' has no extension")),
    };
    require_asset(base)?;
    for f in 1..=5 {
        let face = format!("{stem}_{f}.{ext}");
        if !crate::asset_exists(Path::new(""), &face) {
            return Err(format!("cubemap '{base}' is missing face {face} in the asset directory"));
        }
    }
    Ok(())
}

fn fs_create_dir(dir: &Path) -> Result<(), String> {
    if dir.exists() {
        return Err(format!("destination already exists: {:?}", dir));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("create {:?}: {e}", dir))
}

fn write_file(path: &Path, contents: &str) -> Result<(), String> {
    std::fs::write(path, contents).map_err(|e| format!("write {:?}: {e}", path))
}

/// TOML-quote a string value (handles quotes/backslashes).
fn toml_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Rewrite single-argument `mat2(...)` constructor calls to `_stm2(...)`, a
/// naga-safe overload defined in the engine header. naga's GLSL frontend
/// miscompiles `mat2(vec4)` / `mat2(float)` (the `mat2(cos(a+vec4(...)))`
/// rotation idiom) into a malformed Compose; the multi-argument forms
/// `mat2(a,b,c,d)` / `mat2(a,b)` are fine and left untouched. We detect "single
/// argument" by balanced-paren scanning for the absence of a top-level comma.
fn patch_naga_mat2(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 32);
    let mut i = 0;
    while i < bytes.len() {
        // Match the identifier `mat2` not preceded/followed by an identifier char
        // (so `mat2x2`, `xmat2` don't match), immediately followed by `(`.
        if src[i..].starts_with("mat2") {
            let prev_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after = i + 4;
            // Skip whitespace between `mat2` and `(`.
            let mut j = after;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if prev_ok && j < bytes.len() && bytes[j] == b'(' {
                // Find the matching close paren and whether a top-level comma exists.
                let (close, has_top_comma) = scan_call_args(bytes, j);
                if let Some(close) = close {
                    if !has_top_comma {
                        // Single-argument constructor -> rewrite to the safe overload.
                        out.push_str("_stm2");
                        out.push_str(&src[after..=close]);
                        i = close + 1;
                        continue;
                    }
                }
            }
            // Not a single-arg mat2 call: emit `mat2` verbatim and continue.
            out.push_str("mat2");
            i = after;
            continue;
        }
        // Advance one full UTF-8 char (comments may contain non-ASCII text).
        let ch = src[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Given `bytes` and the index of an opening `(`, return the index of its
/// matching `)` (if balanced) and whether the argument list has a top-level comma.
fn scan_call_args(bytes: &[u8], open: usize) -> (Option<usize>, bool) {
    let mut depth = 0i32;
    let mut has_top_comma = false;
    let mut k = open;
    while k < bytes.len() {
        match bytes[k] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return (Some(k), has_top_comma);
                }
            }
            b',' if depth == 1 => has_top_comma = true,
            _ => {}
        }
        k += 1;
    }
    (None, has_top_comma)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Render real thumbnails for the texture + cubemap samples to exercise the
    // full GPU pipeline (bind-group layouts, cube view), not just the naga check.
    // cargo test -p core-engine --lib render_samples -- --ignored --nocapture
    #[test]
    #[ignore]
    fn render_samples() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let temp = root.join(".temp-prototype");
        crate::set_asset_dirs(vec![root.join("assets").join("external")]);
        let ctx = std::sync::Arc::new(
            pollster::block_on(crate::GraphicsContext::new_render_only()).unwrap(),
        );
        for json_name in ["lsSGRc.json", "332XWd.json", "33cGDj.json"] {
            let json = std::fs::read_to_string(temp.join(json_name)).unwrap();
            let dest = std::env::temp_dir().join(format!("strata-render-{}", json_name.replace('.', "_")));
            let _ = std::fs::remove_dir_all(&dest);
            convert_shadertoy(&json, &dest).unwrap();
            let out = dest.join("thumb.png");
            match crate::thumbnail::generate_thumbnail(ctx.clone(), &dest, &out, 320, 200) {
                Ok(()) => println!("RENDER OK  {json_name} -> {:?}", out),
                Err(e) => panic!("RENDER FAIL {json_name}: {e}"),
            }
        }
    }

    // cargo test -p core-engine convert_samples -- --ignored --nocapture
    #[test]
    #[ignore]
    fn convert_samples() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let temp = root.join(".temp-prototype");
        crate::set_asset_dirs(vec![root.join("assets").join("external")]);
        for json_name in ["33cGDj.json", "lsSGRc.json", "332XWd.json"] {
            let json = std::fs::read_to_string(temp.join(json_name)).unwrap();
            let dest = std::env::temp_dir().join(format!("strata-test-{}", json_name.replace('.', "_")));
            let _ = std::fs::remove_dir_all(&dest);
            println!("\n=== {json_name} ===");
            match convert_shadertoy(&json, &dest) {
                Ok(r) => {
                    println!("OK name={:?} warnings={:?}", r.name, r.warnings);
                    println!("--- manifest.toml ---\n{}", std::fs::read_to_string(dest.join("manifest.toml")).unwrap());
                    let mut files: Vec<_> = std::fs::read_dir(&dest).unwrap()
                        .filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).collect();
                    files.sort();
                    println!("files: {files:?}");
                }
                Err(e) => println!("ERR: {e}"),
            }
        }
    }
}
