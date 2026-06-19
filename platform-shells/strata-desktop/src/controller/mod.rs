use std::sync::{Arc, RwLock};
use std::path::{Path, PathBuf};
use core_engine::manifest::WallpaperConfig;
use zip::ZipArchive;
use std::fs;
use serde::{Serialize, Deserialize};

/// A wallpaper's thumbnail lives INSIDE its own folder as `thumbnail.png`, so the
/// folder is fully self-contained (zip it up and it's a complete, re-uploadable
/// pack — the same layout Strata-Library ships). This replaced the old central
/// `%APPDATA%/strata/thumbnails` cache.
pub fn thumbnail_path(wallpaper_dir: &Path) -> PathBuf {
    wallpaper_dir.join("thumbnail.png")
}

#[derive(Clone, Debug, Default)]
pub struct WallpaperEntry {
    pub name: String,
    pub author: String,
    pub source_url: String,
    pub path: PathBuf,
    pub tags: Vec<String>,
    pub thumbnail: Option<PathBuf>,
}

fn default_true() -> bool { true }
fn default_blend() -> String { "normal".to_string() }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LayerInfo {
    pub wallpaper_path: PathBuf,
    pub name: String,
    pub opacity: f32,
    pub resolution_scale: f32,
    pub positioning: String, // "Fill", "Fit", "Stretch", "Center", "Custom"
    pub transform: [f32; 4], // x, y, width, height (for Custom)
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default = "default_blend")]
    pub blend_mode: String,  // "normal" | "additive" | "multiply"
}

impl Default for LayerInfo {
    fn default() -> Self {
        Self {
            wallpaper_path: PathBuf::new(),
            name: String::new(),
            opacity: 1.0,
            resolution_scale: 1.0,
            positioning: "Fill".to_string(),
            transform: [0.0, 0.0, 1.0, 1.0],
            visible: true,
            blend_mode: "normal".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MonitorInfo {
    pub id: String,
    pub name: String,
    pub color: String,   // hardware color tag: orange|blue|purple|emerald|rose
    pub resolution: (u32, u32),
    pub position: (i32, i32),
    pub is_primary: bool,
    pub layers: Vec<LayerInfo>,
}

pub struct AppState {
    pub wallpapers: Vec<WallpaperEntry>,
    pub monitors: Vec<MonitorInfo>,
    pub theme_mode: String, // "system", "dark", "light"
    pub span_monitors: bool,
    pub autostart: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            wallpapers: Vec::new(),
            monitors: Vec::new(),
            theme_mode: "system".to_string(),
            span_monitors: false,
            autostart: false,
        }
    }
}

/// The primary monitor (the one the OS marks as primary), falling back to the
/// monitor at the desktop origin (0,0), then the first discovered monitor.
/// In span mode this monitor's layers are what gets stretched across the canvas.
pub fn primary_monitor(monitors: &[MonitorInfo]) -> Option<&MonitorInfo> {
    monitors
        .iter()
        .find(|m| m.is_primary)
        .or_else(|| monitors.iter().find(|m| m.position == (0, 0)))
        .or_else(|| monitors.first())
}

#[allow(dead_code)]
pub type SharedState = Arc<RwLock<AppState>>;

pub fn scan_wallpapers(base_path: &Path) -> Vec<WallpaperEntry> {
    let mut wallpapers = Vec::new();
    if let Ok(entries) = fs::read_dir(base_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            // Require an explicit manifest.toml for a folder to appear in the
            // Library. The engine's load_from_dir() has a no-manifest fallback
            // (handy for CLI), but in the curated Library it produced broken
            // "ghost cards" for manifest-less folders (e.g. multipass shaders
            // collapsed to a single white pass).
            if path.is_dir() && path.join("manifest.toml").exists() {
                if let Ok(config) = WallpaperConfig::load_from_dir(&path) {
                    // In-folder thumbnail (thumbnail.png, or a bundled .jpg/.jpeg);
                    // None means it's generated into the folder on next refresh.
                    let mut thumbnail = None;
                    for ext in ["png", "jpg", "jpeg"] {
                        let thumb_path = path.join(format!("thumbnail.{}", ext));
                        if thumb_path.exists() {
                            thumbnail = Some(thumb_path);
                            break;
                        }
                    }

                    wallpapers.push(WallpaperEntry {
                        name: config.wallpaper.name,
                        author: config.wallpaper.author,
                        source_url: config.wallpaper.source_url,
                        tags: config.wallpaper.tags,
                        path: path.to_path_buf(),
                        thumbnail,
                    });
                }
            }
        }
    }
    wallpapers
}

pub fn import_wallpaper_zip(zip_path: &Path, dest_base: &Path) -> Result<PathBuf, String> {
    let file = fs::File::open(zip_path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
    
    let folder_name = zip_path.file_stem().unwrap().to_string_lossy().to_string();
    let dest_path = dest_base.join(&folder_name);
    
    if dest_path.exists() {
        return Err("Wallpaper already exists".to_string());
    }
    
    fs::create_dir_all(&dest_path).map_err(|e| e.to_string())?;
    
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let outpath = match file.enclosed_name() {
            Some(path) => dest_path.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath).map_err(|e| e.to_string())?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p).map_err(|e| e.to_string())?;
                }
            }
            let mut outfile = fs::File::create(&outpath).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut outfile).map_err(|e| e.to_string())?;
        }
    }
    
    if !dest_path.join("manifest.toml").exists() {
        fs::remove_dir_all(&dest_path).ok();
        return Err("No valid live wallpaper found! (Missing manifest.toml)".to_string());
    }
    
    Ok(dest_path)
}

/// `%APPDATA%/strata` (Roaming) — root for user-generated content that must NOT
/// live in the (read-only in release) install directory.
pub fn user_data_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|b| b.data_dir().join("Strata"))
}

/// Where Parallax Studio creations are saved: `%APPDATA%/strata/parallax-wallpapers`.
pub fn parallax_library_dir() -> Option<PathBuf> {
    user_data_dir().map(|d| d.join("parallax-wallpapers"))
}

/// Where imported Shadertoy shaders are saved: `%APPDATA%/strata/import`.
pub fn import_library_dir() -> Option<PathBuf> {
    user_data_dir().map(|d| d.join("import"))
}

/// The bundled, curated wallpaper library shipped with the app. CWD-relative for
/// now; the planned Strata-Library decoupling will move this to
/// `%APPDATA%/strata/strata-library`.
pub fn bundled_library_dir() -> PathBuf {
    if Path::new("wallpapers").exists() {
        PathBuf::from("wallpapers")
    } else {
        PathBuf::from("../../wallpapers")
    }
}

/// The runtime-fetched shader library: `%APPDATA%/strata/strata-library/shader-library`
/// (downloaded from the Strata-Library repo). This is the primary curated library
/// now that the app no longer bundles shaders.
pub fn fetched_library_dir() -> Option<PathBuf> {
    user_data_dir().map(|d| d.join("strata-library").join("shader-library"))
}

/// Root of the fetched library tree (`…/strata-library`) — holds shader-library/,
/// external/, models.toml, presets.toml, index.toml. The sync target.
pub fn fetched_library_root() -> Option<PathBuf> {
    user_data_dir().map(|d| d.join("strata-library"))
}

/// True once at least one shader has been fetched into the library.
pub fn library_installed() -> bool {
    fetched_library_dir()
        .map(|d| std::fs::read_dir(&d).map(|mut e| e.any(|x| {
            x.ok().map(|x| x.path().join("manifest.toml").exists()).unwrap_or(false)
        })).unwrap_or(false))
        .unwrap_or(false)
}

/// All directories the Library scans: the fetched library + the bundled folder (a
/// dev convenience, usually absent) + the user roots (parallax + imports).
pub fn library_roots() -> Vec<PathBuf> {
    let mut roots = vec![bundled_library_dir()];
    roots.extend(fetched_library_dir());
    roots.extend(parallax_library_dir());
    roots.extend(import_library_dir());
    roots
}

/// Owner + repo of the official content repository, parsed from its jsDelivr URL
/// (`…/gh/OWNER/REPO@tag`). Used for the GitHub tags / zipball endpoints.
pub fn official_owner_repo() -> Option<(String, String)> {
    let url = official_repo_url()?;
    // Accept the jsDelivr form (…/gh/OWNER/REPO[@tag]) or the plain GitHub form
    // (github.com/OWNER/REPO[.git]). Any @tag / trailing path is discarded — the
    // version is resolved from the repo's tags, not from this URL.
    let rest = if let Some(r) = url.split("/gh/").nth(1) {
        r
    } else if let Some(r) = url.split("github.com/").nth(1) {
        r
    } else {
        return None;
    };
    let path = rest.split(['@', '#', '?']).next()?;
    let mut parts = path.trim_matches('/').splitn(3, '/');
    let owner = parts.next()?.to_string();
    let repo = parts.next()?.trim_end_matches(".git").trim_end_matches('/').to_string();
    if owner.is_empty() || repo.is_empty() { None } else { Some((owner, repo)) }
}

/// True if `path` lives in a user-writable root (parallax or import) and may
/// therefore be deleted from the Library. The bundled library is never deletable.
pub fn is_user_deletable(path: &Path) -> bool {
    let canon = path.canonicalize().ok();
    [parallax_library_dir(), import_library_dir()]
        .into_iter()
        .flatten()
        .filter_map(|r| r.canonicalize().ok())
        .any(|root| canon.as_ref().is_some_and(|p| p.starts_with(&root)))
}

/// Scan every library root and merge the results into one list.
pub fn scan_all_wallpapers() -> Vec<WallpaperEntry> {
    let mut all = Vec::new();
    for root in library_roots() {
        all.extend(scan_wallpapers(&root));
    }
    all
}

/// Directory of bundled Shadertoy assets (`assets/external`) — the shared texture/
/// cubemap library that shaders reference by name (NOT copied per-wallpaper).
/// CWD-relative like `bundled_library_dir()` for now.
pub fn assets_external_dir() -> PathBuf {
    if Path::new("assets/external").exists() {
        PathBuf::from("assets/external")
    } else {
        PathBuf::from("../../assets/external")
    }
}

/// All shared asset search roots, in priority order. Registered with the engine
/// at startup so shaders resolve textures/cubemaps by name. Includes the current
/// bundled `assets/external` and the future fetched
/// `%APPDATA%/strata/strata-library/external` (empty until the library decoupling
/// lands).
pub fn library_asset_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![assets_external_dir()];
    if let Some(d) = user_data_dir() {
        dirs.push(d.join("strata-library").join("external"));
    }
    dirs
}

/// Base URL of the first `official = true` content repository declared in
/// `repositories.toml` (falls back to the first repository, or None). The update
/// check appends `/index.toml` to read the remote `library_version`.
pub fn official_repo_url() -> Option<String> {
    let path = if Path::new("repositories.toml").exists() {
        PathBuf::from("repositories.toml")
    } else {
        PathBuf::from("../../repositories.toml")
    };
    let text = fs::read_to_string(path).ok()?;
    // Walk [[repository]] blocks; prefer one marked official.
    let mut first_url: Option<String> = None;
    for block in text.split("[[repository]]").skip(1) {
        let url = block.lines().find_map(|l| {
            let l = l.trim();
            l.strip_prefix("url").and_then(|r| r.trim_start().strip_prefix('='))
                .map(|v| v.trim().trim_matches('"').to_string())
        });
        let official = block.lines().any(|l| l.trim().starts_with("official") && l.contains("true"));
        if let Some(u) = url {
            if u.is_empty() { continue; }
            if official { return Some(u); }
            first_url.get_or_insert(u);
        }
    }
    first_url
}

/// True if `zip_path` is a native Strata wallpaper pack (contains a manifest.toml)
/// rather than a Shadertoy export. Used to route `.zip` imports to the right path.
pub fn zip_is_native_pack(zip_path: &Path) -> bool {
    let Ok(file) = fs::File::open(zip_path) else { return false };
    let Ok(mut archive) = ZipArchive::new(file) else { return false };
    for i in 0..archive.len() {
        if let Ok(f) = archive.by_index(i) {
            if let Some(name) = f.enclosed_name() {
                if name.file_name().and_then(|n| n.to_str()) == Some("manifest.toml") {
                    return true;
                }
            }
        }
    }
    false
}

/// Read the Shadertoy export JSON from `path`: the file itself for `.json`, or
/// the first `.json` entry inside a `.zip`.
fn read_shadertoy_json(path: &Path) -> Result<String, String> {
    let is_zip = path.extension().and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("zip")).unwrap_or(false);
    if !is_zip {
        return fs::read_to_string(path).map_err(|e| e.to_string());
    }
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut json_idx = None;
    for i in 0..archive.len() {
        if let Ok(f) = archive.by_index(i) {
            if f.name().to_lowercase().ends_with(".json") {
                json_idx = Some(i);
                break;
            }
        }
    }
    let idx = json_idx.ok_or_else(|| "No .json found inside the ZIP".to_string())?;
    let mut entry = archive.by_index(idx).map_err(|e| e.to_string())?;
    let mut buf = String::new();
    std::io::Read::read_to_string(&mut entry, &mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Import a Shadertoy export (`.json` or `.zip`) by converting it into a Strata
/// wallpaper package under `dest_base`. Texture/cubemap assets are referenced from
/// the shared library asset dirs (registered with the engine), not copied. Returns
/// the new folder path and any non-fatal conversion warnings.
pub fn import_shadertoy(path: &Path, dest_base: &Path)
    -> Result<(PathBuf, Vec<String>), String>
{
    let json = read_shadertoy_json(path)?;
    // Slug from the shader's display name (falls back to file stem), so the
    // folder reads like the library's own (e.g. "clearly-a-bug").
    let name = core_engine::shadertoy_import::peek_name(&json)
        .filter(|n| n != "Imported Shader")
        .or_else(|| path.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string()))
        .unwrap_or_else(|| "imported-shader".to_string());
    let dest = unique_dir(dest_base, &slugify(&name));
    match core_engine::shadertoy_import::convert_shadertoy(&json, &dest) {
        Ok(report) => Ok((dest, report.warnings)),
        Err(e) => {
            // Discard the half-written folder so a failed import leaves no ghost.
            fs::remove_dir_all(&dest).ok();
            Err(e)
        }
    }
}

/// Folder-name slug from a display name (ascii-alnum, dash-separated).
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
    if s.is_empty() { "imported-shader".to_string() } else { s }
}

/// First non-colliding `base/slug`, then `base/slug-2`, `-3`, …
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

pub fn discover_monitors() -> Vec<MonitorInfo> {
    use display_info::DisplayInfo;
    if let Ok(displays) = DisplayInfo::all() {
        displays.into_iter().enumerate().map(|(i, m)| {
            MonitorInfo {
                id: format!("monitor-{}", i),
                name: format!("Display {}", i + 1),
                color: String::new(), // assigned a default tag by the caller
                resolution: (m.width, m.height),
                position: (m.x, m.y),
                is_primary: m.is_primary,
                layers: Vec::new(),
            }
        }).collect()
    } else {
        Vec::new()
    }
}
