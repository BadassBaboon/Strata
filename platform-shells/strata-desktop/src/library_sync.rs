//! Runtime fetch of the Strata-Library content (shaders, thumbnails, external
//! assets, models.toml / presets.toml) from its GitHub repo into
//! `%APPDATA%/strata/strata-library`.
//!
//! The library is small, so a sync downloads the tag's **zipball** in one shot and
//! extracts it (atomic swap, no per-file fetching or manifest parsing). Version
//! discovery uses the GitHub **tags** API and picks the highest `library-v*` tag —
//! that's how Strata learns a newer library (e.g. `library-v1.1.0`) was published.

use std::io::Read;
use std::path::PathBuf;

const UA: &str = "Strata-Library-Sync";

/// Parse `MAJOR.MINOR.PATCH`(-ish) into a comparable tuple.
fn semver(s: &str) -> (u32, u32, u32) {
    let s = s.trim().trim_start_matches(['v', 'V']);
    let mut it = s.split(['.', '-', '+']).filter_map(|p| p.parse::<u32>().ok());
    (it.next().unwrap_or(0), it.next().unwrap_or(0), it.next().unwrap_or(0))
}

/// True if `tag_version` is newer than the installed `current`.
pub fn is_newer(tag_version: &str, current: &str) -> bool {
    semver(tag_version) > semver(current)
}

/// Discover the latest published library: `(owner, repo, version, tag)` from the
/// repo's `library-v*` git tags (highest version wins). This is the mechanism by
/// which the app notices a new release on Strata-Library.
pub fn latest_library() -> Result<(String, String, String, String), String> {
    let (owner, repo) = crate::controller::official_owner_repo()
        .ok_or("no official content repository configured in repositories.toml")?;
    let url = format!("https://api.github.com/repos/{}/{}/tags?per_page=100", owner, repo);
    let body = ureq::get(&url)
        .set("User-Agent", UA)
        .set("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(15))
        .call().map_err(|e| e.to_string())?
        .into_string().map_err(|e| e.to_string())?;
    let tags: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    let mut best: Option<((u32, u32, u32), String)> = None;
    if let Some(arr) = tags.as_array() {
        for t in arr {
            if let Some(name) = t.get("name").and_then(|v| v.as_str()) {
                if let Some(ver) = name.strip_prefix("library-v") {
                    let sv = semver(ver);
                    if best.as_ref().is_none_or(|(b, _)| sv > *b) {
                        best = Some((sv, name.to_string()));
                    }
                }
            }
        }
    }
    let (sv, tag) = best.ok_or("no library-v* tags found on the repository")?;
    Ok((owner, repo, format!("{}.{}.{}", sv.0, sv.1, sv.2), tag))
}

/// Download the library at `tag` and install it into
/// `%APPDATA%/strata/strata-library`, replacing any previous copy. Extracts to a
/// staging dir first and atomically swaps, so a failed/partial download never
/// corrupts an existing install.
pub fn sync_library(owner: &str, repo: &str, tag: &str) -> Result<(), String> {
    let root = crate::controller::fetched_library_root()
        .ok_or("could not resolve the user data directory")?;
    let url = format!("https://codeload.github.com/{}/{}/zip/refs/tags/{}", owner, repo, tag);
    let resp = ureq::get(&url)
        .set("User-Agent", UA)
        .timeout(std::time::Duration::from_secs(120))
        .call().map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    resp.into_reader().read_to_end(&mut bytes).map_err(|e| format!("download: {e}"))?;

    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| format!("open zip: {e}"))?;

    // Extract into a sibling staging dir, then atomically swap into place.
    let staging = root.with_file_name("strata-library.tmp");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| format!("mkdir staging: {e}"))?;

    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(|e| e.to_string())?;
        // Zip-slip guard + strip the archive's top-level "<repo>-<tag>/" folder.
        let Some(enclosed) = f.enclosed_name() else { continue };
        let mut comps = enclosed.components();
        comps.next();
        let rel: PathBuf = comps.as_path().to_path_buf();
        if rel.as_os_str().is_empty() { continue; }
        let out = staging.join(&rel);
        if f.name().ends_with('/') {
            std::fs::create_dir_all(&out).ok();
        } else {
            if let Some(p) = out.parent() { std::fs::create_dir_all(p).ok(); }
            let mut o = std::fs::File::create(&out).map_err(|e| format!("write {:?}: {e}", out))?;
            std::io::copy(&mut f, &mut o).map_err(|e| format!("extract {:?}: {e}", out))?;
        }
    }

    // Sanity-check the extracted tree before swapping it in.
    if !staging.join("shader-library").exists() {
        let _ = std::fs::remove_dir_all(&staging);
        return Err("downloaded library has no shader-library/ folder".to_string());
    }

    let _ = std::fs::remove_dir_all(&root);
    if let Some(p) = root.parent() { std::fs::create_dir_all(p).ok(); }
    std::fs::rename(&staging, &root).map_err(|e| format!("install library: {e}"))?;
    Ok(())
}
