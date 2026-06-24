use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::fs;
use directories::ProjectDirs;
use crate::controller::{LayerInfo, MonitorInfo};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Config {
    pub theme_mode: String,
    pub span_monitors: bool,
    // NOTE: a legacy `clone_monitors` key may exist in older config files; serde
    // ignores unknown keys, so dropping the field here is backwards-compatible.
    pub autostart: bool,
    // Frame-rate cap applied to every monitor's render loop.  `#[serde(default)]`
    // keeps older config files (without this key) loadable - they fall back to 30.
    #[serde(default = "default_target_fps")]
    pub target_fps: u32,
    // Audio-reactivity sensitivity (spectrum gain). 1.0 = default.
    #[serde(default = "default_audio_sensitivity")]
    pub audio_sensitivity: f32,
    // Which wallpapers receive the desktop cursor in iMouse. One of:
    // "Off" | "On (Everything)" | "On (Only Shaders)" | "On (Only Parallax Studio)".
    // (Replaces the legacy `mouse_interactive` bool, which serde now ignores.)
    #[serde(default = "default_mouse_mode")]
    pub mouse_mode: String,
    #[serde(default = "default_mouse_sensitivity")]
    pub mouse_sensitivity: f32,
    // Global render-quality preset label (maps to a render scale in the shell).
    #[serde(default = "default_shader_quality")]
    pub shader_quality: String,
    // Installed Strata-Library version (from its index.toml). Compared to the
    // remote library version during update checks. Defaults to the shipped 1.0.0.
    #[serde(default = "default_library_version")]
    pub library_version: String,
    // Unix seconds of the last automatic update check (app + library). 0 = never.
    // The check runs at most ~weekly so launches stay fast and offline-friendly.
    #[serde(default)]
    pub last_update_check: i64,
    // Library sort order: "default" | "name-asc" | "name-desc" | "date-newest" | "date-oldest".
    #[serde(default = "default_library_sort")]
    pub library_sort: String,
    pub monitors: Vec<MonitorConfig>,
}

/// Default library sort: bundled shaders A-Z with user content at the bottom.
pub fn default_library_sort() -> String {
    "default".to_string()
}

/// Default installed asset-library version (matches the shipped Strata-Library).
pub fn default_library_version() -> String {
    "1.0.0".to_string()
}

/// Default wallpaper frame cap - 60 FPS is the industry-standard smooth baseline.
pub fn default_target_fps() -> u32 {
    60
}

/// Default audio sensitivity multiplier.
pub fn default_audio_sensitivity() -> f32 {
    1.0
}

/// Default mouse sensitivity multiplier (1.0 = cursor tracks 1:1).
pub fn default_mouse_sensitivity() -> f32 {
    1.0
}

/// Default mouse-interactivity mode - only Parallax Studio wallpapers follow the
/// cursor (shaders stay non-interactive unless the user opts in).
pub fn default_mouse_mode() -> String {
    "On (Only Parallax Studio)".to_string()
}

/// Default shader-quality preset - full native resolution.
pub fn default_shader_quality() -> String {
    "High (Maximum Fidelity)".to_string()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MonitorConfig {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub color: String,
    pub layers: Vec<LayerInfo>,
}

impl Config {
    pub fn get_path() -> Option<PathBuf> {
        ProjectDirs::from("com", "Strata", "engine").map(|proj_dirs| {
            let config_dir = proj_dirs.config_dir();
            if !config_dir.exists() {
                fs::create_dir_all(config_dir).ok();
            }
            config_dir.join("config.toml")
        })
    }

    pub fn load() -> Self {
        if let Some(path) = Self::get_path() {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(config) = toml::from_str::<Config>(&content) {
                    return config;
                }
            }
        }
        Self {
            theme_mode: "system".to_string(),
            span_monitors: false,
            autostart: false,
            target_fps: default_target_fps(),
            audio_sensitivity: default_audio_sensitivity(),
            mouse_mode: default_mouse_mode(),
            mouse_sensitivity: default_mouse_sensitivity(),
            shader_quality: default_shader_quality(),
            library_version: default_library_version(),
            last_update_check: 0,
            library_sort: default_library_sort(),
            monitors: Vec::new(),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        if let Some(path) = Self::get_path() {
            let content = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
            fs::write(path, content).map_err(|e| e.to_string())?;
            Ok(())
        } else {
            Err("Could not determine config path".to_string())
        }
    }

    pub fn update_from_state(&mut self, theme_mode: String, span: bool, autostart: bool, monitors: &[MonitorInfo]) {
        self.theme_mode = theme_mode;
        self.span_monitors = span;
        self.autostart = autostart;
        self.monitors = monitors.iter().map(|m| MonitorConfig {
            id: m.id.clone(),
            name: m.name.clone(),
            color: m.color.clone(),
            layers: m.layers.clone(),
        }).collect();
    }
}
