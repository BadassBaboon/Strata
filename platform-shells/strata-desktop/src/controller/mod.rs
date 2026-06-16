use std::sync::{Arc, RwLock};
use std::path::{Path, PathBuf};
use core_engine::manifest::WallpaperConfig;
use zip::ZipArchive;
use std::fs;
use serde::{Serialize, Deserialize};

/// Central thumbnail cache directory: `%APPDATA%/strata/thumbnails` on Windows
/// (data dir / strata / thumbnails elsewhere). Thumbnails are stored here keyed
/// by the wallpaper folder name (e.g. `audio-visualizer-raymarching.png`) so the
/// wallpaper folders stay clean and read-only installs still get previews.
pub fn thumbnails_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|b| b.data_dir().join("strata").join("thumbnails"))
}

/// Path where this wallpaper folder's generated thumbnail lives in the cache.
pub fn cached_thumbnail_path(wallpaper_dir: &Path) -> Option<PathBuf> {
    let name = wallpaper_dir.file_name()?.to_str()?;
    Some(thumbnails_dir()?.join(format!("{}.png", name)))
}

#[derive(Clone, Debug, Default)]
pub struct WallpaperEntry {
    pub name: String,
    pub author: String,
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
                    // Prefer the generated cache thumbnail; fall back to one bundled
                    // in the wallpaper folder; else None (generated on next refresh).
                    let mut thumbnail = cached_thumbnail_path(&path).filter(|p| p.exists());
                    if thumbnail.is_none() {
                        for ext in ["png", "jpg", "jpeg"] {
                            let thumb_path = path.join(format!("thumbnail.{}", ext));
                            if thumb_path.exists() {
                                thumbnail = Some(thumb_path);
                                break;
                            }
                        }
                    }

                    wallpapers.push(WallpaperEntry {
                        name: config.wallpaper.name,
                        author: config.wallpaper.author,
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
