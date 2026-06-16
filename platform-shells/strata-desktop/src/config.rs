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
    // keeps older config files (without this key) loadable — they fall back to 30.
    #[serde(default = "default_target_fps")]
    pub target_fps: u32,
    // Audio-reactivity sensitivity (spectrum gain). 1.0 = default.
    #[serde(default = "default_audio_sensitivity")]
    pub audio_sensitivity: f32,
    // Feed the desktop cursor into shaders' iMouse (e.g. 1D Radial Lightmap).
    #[serde(default)]
    pub mouse_interactive: bool,
    #[serde(default = "default_mouse_sensitivity")]
    pub mouse_sensitivity: f32,
    // Global render-quality preset label (maps to a render scale in the shell).
    #[serde(default = "default_shader_quality")]
    pub shader_quality: String,
    pub monitors: Vec<MonitorConfig>,
}

/// Default wallpaper frame cap — 60 FPS is the industry-standard smooth baseline.
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

/// Default shader-quality preset — full native resolution.
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
        ProjectDirs::from("com", "strata", "engine").map(|proj_dirs| {
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
            mouse_interactive: false,
            mouse_sensitivity: default_mouse_sensitivity(),
            shader_quality: default_shader_quality(),
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
