use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WallpaperConfig {
    #[serde(rename = "wallpaper")]
    pub wallpaper: WallpaperInfo,
    #[serde(default)]
    pub render_targets: HashMap<String, PassConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WallpaperInfo {
    pub name: String,
    pub author: String,
    pub version: String,
    /// Original source page (e.g. the Shadertoy shader URL), for attribution.
    /// Empty when unknown.
    #[serde(default)]
    pub source_url: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub passes: Vec<String>,
    /// Set for a VIDEO (movie .mp4) wallpaper: the video file name relative to the
    /// wallpaper folder. When present this is a video wallpaper (no shader passes) and
    /// the engine plays it via the platform video decoder instead of building pipelines.
    #[serde(default)]
    pub video: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassConfig {
    pub source: String,
    #[serde(default)]
    pub double_buffered: bool,
    #[serde(default = "default_blend")]
    pub blend: String, // "replace", "alpha"
    #[serde(default = "default_scale")]
    pub scale: f32,
    #[serde(default)]
    pub bindings: Vec<BindingConfig>,
}

fn default_scale() -> f32 {
    1.0
}

fn default_blend() -> String {
    "replace".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingConfig {
    pub channel: u32,
    #[serde(rename = "type")]
    pub binding_type: String, // "texture", "buffer", "audio"
    pub path: Option<String>,
    pub target: Option<String>,
    pub stream: Option<String>,
    /// Texture wrap mode for this channel: "clamp" or "repeat" (default). Shadertoy
    /// fluid/feedback sims need clamp so neighbour reads at the edges don't wrap around
    /// and corrupt the border solve. Omitted = repeat (the historical behaviour).
    pub wrap: Option<String>,
}

impl WallpaperConfig {
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let manifest_path = dir.as_ref().join("manifest.toml");
        if !manifest_path.exists() {
            // Dynamic fallback: build configuration from folder contents
            let image_path = dir.as_ref().join("image.glsl");
            if image_path.exists() {
                let name = dir.as_ref().file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("Dynamic Wallpaper")
                    .to_string();
                
                let mut bindings = Vec::new();
                if let Ok(entries) = std::fs::read_dir(dir.as_ref()) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            let file_name = entry.file_name().to_string_lossy().to_string();
                            if ext.eq_ignore_ascii_case("mp3") || ext.eq_ignore_ascii_case("wav") {
                                let channel = if file_name.contains("iChannel0") { 0 }
                                    else if file_name.contains("iChannel1") { 1 }
                                    else if file_name.contains("iChannel2") { 2 }
                                    else if file_name.contains("iChannel3") { 3 }
                                    else { 0 };
                                bindings.push(BindingConfig {
                                    channel,
                                    binding_type: "audio".to_string(),
                                    path: None,
                                    target: None,
                                    stream: Some(file_name),
                                    wrap: None,
                                });
                            } else if ext.eq_ignore_ascii_case("png") || ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg") {
                                let channel = if file_name.contains("iChannel0") { 0 }
                                    else if file_name.contains("iChannel1") { 1 }
                                    else if file_name.contains("iChannel2") { 2 }
                                    else if file_name.contains("iChannel3") { 3 }
                                    else { 0 };
                                bindings.push(BindingConfig {
                                    channel,
                                    binding_type: "texture".to_string(),
                                    path: Some(file_name),
                                    target: None,
                                    stream: None,
                                    wrap: None,
                                });
                            }
                        }
                    }
                }

                let mut render_targets = HashMap::new();
                render_targets.insert("image".to_string(), PassConfig {
                    source: "image.glsl".to_string(),
                    double_buffered: false,
                    blend: "replace".to_string(),
                    scale: 1.0,
                    bindings,
                });

                return Ok(WallpaperConfig {
                    wallpaper: WallpaperInfo {
                        name,
                        author: "Unknown".to_string(),
                        version: "1.0.0".to_string(),
                        source_url: String::new(),
                        tags: Vec::new(),
                        passes: vec!["image".to_string()],
                        video: None,
                    },
                    render_targets,
                });
            }
            return Err(format!("No manifest.toml or image.glsl found in {:?}", dir.as_ref()));
        }

        let content = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read manifest.toml: {}", e))?;
        let config: WallpaperConfig = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse manifest.toml: {}", e))?;
        Ok(config)
    }
}
