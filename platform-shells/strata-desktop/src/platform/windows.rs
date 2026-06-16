#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::{HWND, LPARAM, BOOL, RECT, GetLastError};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::WindowsAndMessaging::{
    EnumWindows, FindWindowExA, SendMessageTimeoutA, ShowWindow, SMTO_NORMAL,
    GetWindowLongA, SetWindowLongA, SetWindowPos, GetWindowRect,
    GWL_EXSTYLE,
    WS_EX_APPWINDOW, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE,
    SWP_NOZORDER, SWP_NOCOPYBITS, SWP_SHOWWINDOW, SWP_NOACTIVATE,
    SWP_NOMOVE, SWP_NOSIZE,
    SW_SHOWNA,
};
#[cfg(target_os = "windows")]
use std::ptr;

// ── Theme-aware window background brush ─────────────────────────────────────────
// winit registers its window class with a null background brush, so before Slint's
// software renderer blits its first frame the OS erases the window to white. On a
// tray restore that white can flash for a frame. Setting the class background brush
// to the active theme colour makes that erase blend into the loading UI instead.
// Called once at startup and again on theme change (rare) — never per-frame.
#[cfg(target_os = "windows")]
pub fn set_window_bg_brush(hwnd: isize, is_dark: bool) {
    use windows_sys::Win32::Graphics::Gdi::{CreateSolidBrush, DeleteObject};
    use windows_sys::Win32::UI::WindowsAndMessaging::{SetClassLongPtrW, GCLP_HBRBACKGROUND};
    if hwnd == 0 { return; }
    // COLORREF is 0x00BBGGRR. Dark #111317 / Light #f3f5f7 — matches AppWindow.background.
    let color: u32 = if is_dark { 0x0017_1311 } else { 0x00f7_f5f3 };
    unsafe {
        let brush = CreateSolidBrush(color);
        if brush == 0 { return; }
        // Returns the previous class brush. winit's default is null (0) on the first
        // call; on later (theme-change) calls it's the brush we set, which we delete to
        // avoid leaking. DeleteObject on a non-handle simply fails harmlessly.
        let old = SetClassLongPtrW(hwnd as HWND, GCLP_HBRBACKGROUND, brush as isize);
        if old != 0 { DeleteObject(old as _); }
    }
}

// ── WorkerW detection ─────────────────────────────────────────────────────────
//
// The Windows desktop rendering layer is a special window called WorkerW.
// After sending the 0x052C message to Progman, a WorkerW is created (or already
// exists) behind the desktop icon container.  Child windows of this WorkerW
// render behind desktop icons on ALL virtual desktops automatically — WorkerW
// is a global shell window, so no special per-virtual-desktop handling is needed.
//
// Because many third-party applications also register windows with the "WorkerW"
// class, we filter by rect size (must be ≥ 640 × 480) and try four strategies
// to reliably identify Explorer's desktop WorkerW.

/// Locate Explorer's desktop WorkerW HWND.
#[cfg(target_os = "windows")]
pub fn get_wallpaper_window() -> Option<HWND> {
    unsafe {
        let progman = FindWindowExA(0, 0, b"Progman\0".as_ptr(), ptr::null());
        if progman == 0 {
            log::error!("WorkerW search: Progman not found — cannot enable wallpaper mode");
            return None;
        }
        log::info!("WorkerW search: Progman HWND = {:x}", progman);

        // Tell Progman to spawn a WorkerW (two wParam variants for cross-build compat).
        let mut _r: usize = 0;
        SendMessageTimeoutA(progman, 0x052C, 0,    0,    SMTO_NORMAL, 1000, &mut _r as *mut _);
        SendMessageTimeoutA(progman, 0x052C, 0x0D, 0x01, SMTO_NORMAL, 1000, &mut _r as *mut _);

        // Helper: check that a WorkerW candidate is monitor-sized (≥ 640×480).
        let large_enough = |hwnd: HWND| -> Option<(i32, i32)> {
            let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            GetWindowRect(hwnd, &mut r);
            let (w, h) = (r.right - r.left, r.bottom - r.top);
            if w >= 640 && h >= 480 { Some((w, h)) } else { None }
        };

        // ── Strategy 1 ── Classic: top-level window owning SHELLDLL_DefView → WorkerW sibling
        {
            let mut workerw: HWND = 0;
            unsafe extern "system" fn s1(hwnd: HWND, lp: LPARAM) -> BOOL {
                let p = FindWindowExA(hwnd, 0, b"SHELLDLL_DefView\0".as_ptr(), ptr::null());
                if p != 0 {
                    *(lp as *mut HWND) = FindWindowExA(0, hwnd, b"WorkerW\0".as_ptr(), ptr::null());
                    return 0;
                }
                1
            }
            EnumWindows(Some(s1), &mut workerw as *mut _ as LPARAM);
            if workerw != 0 {
                if let Some((w, h)) = large_enough(workerw) {
                    log::info!("WorkerW: strategy 1 → {:x} ({}×{})", workerw, w, h);
                    return Some(workerw);
                }
                log::warn!("WorkerW: strategy 1 candidate {:x} too small — skipping", workerw);
            }
        }

        // ── Strategy 2 ── WorkerW directly below Progman in Z-order
        {
            let ww = FindWindowExA(0, progman, b"WorkerW\0".as_ptr(), ptr::null());
            if ww != 0 {
                if let Some((w, h)) = large_enough(ww) {
                    log::info!("WorkerW: strategy 2 (below Progman) → {:x} ({}×{})", ww, w, h);
                    return Some(ww);
                }
                log::warn!("WorkerW: strategy 2 candidate {:x} too small — skipping", ww);
            }
        }

        // ── Strategy 3 ── Enumerate ALL top-level WorkerW windows, filter by size.
        {
            let mut candidates: Vec<(HWND, i32, i32)> = Vec::new();
            let mut hw = FindWindowExA(0, 0, b"WorkerW\0".as_ptr(), ptr::null());
            while hw != 0 {
                let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
                GetWindowRect(hw, &mut r);
                candidates.push((hw, r.right - r.left, r.bottom - r.top));
                hw = FindWindowExA(0, hw, b"WorkerW\0".as_ptr(), ptr::null());
            }
            log::info!(
                "WorkerW: {} top-level WorkerW handles: [{}]",
                candidates.len(),
                candidates.iter().map(|(h,w,ht)| format!("{:x}({}×{})", h, w, ht))
                    .collect::<Vec<_>>().join(", ")
            );
            let large: Vec<_> = candidates.iter().filter(|&&(_, w, h)| w >= 640 && h >= 480).collect();
            log::info!(
                "WorkerW: {} large (≥640×480) candidates: [{}]",
                large.len(),
                large.iter().map(|(h,w,ht)| format!("{:x}({}×{})", h, w, ht))
                    .collect::<Vec<_>>().join(", ")
            );
            // Prefer empty (no children = rendering target).
            for &&(ww, w, h) in &large {
                if FindWindowExA(ww, 0, ptr::null(), ptr::null()) == 0 {
                    log::info!("WorkerW: strategy 3a (large empty) → {:x} ({}×{})", ww, w, h);
                    return Some(ww);
                }
            }
            if let Some(&&(last, w, h)) = large.last() {
                log::info!("WorkerW: strategy 3b (last large) → {:x} ({}×{})", last, w, h);
                return Some(last);
            }
        }

        // ── Strategy 4 ── WorkerW as a direct child of Progman (some Win11 builds)
        {
            let ww = FindWindowExA(progman, 0, b"WorkerW\0".as_ptr(), ptr::null());
            if ww != 0 {
                let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
                GetWindowRect(ww, &mut r);
                log::info!("WorkerW: strategy 4 (child of Progman) → {:x} ({}×{})",
                    ww, r.right - r.left, r.bottom - r.top);
                return Some(ww);
            }
        }

        log::warn!(
            "WorkerW: not found after all strategies — wallpaper will sit above icons. \
             Try: taskkill /f /im explorer.exe && start explorer.exe"
        );
        None
    }
}

// ── Pre-show window preparation ───────────────────────────────────────────────

/// Set extended styles and pre-position the window **before** `show()` so it
/// never flashes in the taskbar and arrives at the right monitor immediately.
#[cfg(target_os = "windows")]
pub fn prepare_wallpaper_window(child: HWND, screen_x: i32, screen_y: i32, width: u32, height: u32) {
    unsafe {
        let ex = GetWindowLongA(child, GWL_EXSTYLE) as u32;
        SetWindowLongA(child, GWL_EXSTYLE,
            ((ex & !WS_EX_APPWINDOW) | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE) as i32);
        SetWindowPos(child, 0, screen_x, screen_y, width as i32, height as i32,
            SWP_NOZORDER | SWP_NOACTIVATE);
        log::info!("prepare_wallpaper_window: HWND {:x} → ({}, {}) {}×{}", child, screen_x, screen_y, width, height);
    }
}

/// Call immediately **after** `wall_ui.show()` to cancel any transient taskbar
/// registration that Slint's `ShowWindow(SW_SHOW)` may have caused.
#[cfg(target_os = "windows")]
pub fn suppress_activation(child: HWND) {
    unsafe {
        ShowWindow(child, SW_SHOWNA);
        SetWindowPos(child, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    }
}

// ── Post-show WorkerW reparenting ─────────────────────────────────────────────

/// Reparent a wallpaper window into WorkerW so it renders **behind** desktop icons.
/// Call **after** `show()` and `suppress_activation()`.
///
/// # Why we keep WS_POPUP
///
/// Changing `GWL_STYLE` from `WS_POPUP` to `WS_CHILD` sends `WM_STYLECHANGED`
/// to the window.  winit processes `WM_STYLECHANGED` on the event-loop thread
/// and calls `GetClientRect` during that handler; the render thread
/// simultaneously calls `window.inner_size() → GetClientRect`.  When the window
/// is in the style-transition state `GetClientRect` can return 0, which causes
/// winit to panic with "Unexpected GetClientRect failure".
///
/// A `WS_POPUP` window reparented via `SetParent` still renders in WorkerW's
/// Z-order layer (behind desktop icons).  The only behavioural difference is
/// that `SetWindowPos` uses **screen coordinates** rather than parent-relative
/// ones, which we pass directly as `screen_x` / `screen_y`.
#[cfg(target_os = "windows")]
pub fn setup_wallpaper_window(
    child: HWND,
    workerw: HWND,
    screen_x: i32,
    screen_y: i32,
    width: u32,
    height: u32,
) {
    unsafe {
        // Log WorkerW rect for diagnostics (does not affect behaviour).
        let mut ww_rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        GetWindowRect(workerw, &mut ww_rect);
        log::info!(
            "setup_wallpaper_window: HWND {:x} → WorkerW {:x} | \
             WorkerW screen rect ({},{})…({},{}) | window screen pos ({},{})",
            child, workerw,
            ww_rect.left, ww_rect.top, ww_rect.right, ww_rect.bottom,
            screen_x, screen_y
        );

        // Belt-and-suspenders: re-apply extended styles (WS_EX_TOOLWINDOW was
        // set in prepare_wallpaper_window but verify it survived show()).
        let ex = GetWindowLongA(child, GWL_EXSTYLE) as u32;
        SetWindowLongA(child, GWL_EXSTYLE,
            ((ex & !WS_EX_APPWINDOW) | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE) as i32);

        // Reparent into WorkerW — places rendering behind desktop icons.
        // We do NOT change GWL_STYLE (keep WS_POPUP) to avoid the winit
        // GetClientRect panic described above.
        let prev = windows_sys::Win32::UI::WindowsAndMessaging::SetParent(child, workerw);
        if prev == 0 {
            let err = GetLastError();
            log::error!(
                "setup_wallpaper_window: SetParent failed — error {} \
                 (5=access denied/UIPI, 1400=invalid HWND). \
                 HWND {:x} → WorkerW {:x}",
                err, child, workerw
            );
            return;
        }

        // Even though we keep WS_POPUP, SetParent makes this window a child.
        // Child windows in Win32 ALWAYS use parent-relative coordinates for
        // SetWindowPos, even if they have the WS_POPUP style.
        let rel_x = screen_x - ww_rect.left;
        let rel_y = screen_y - ww_rect.top;

        SetWindowPos(
            child, 0,
            rel_x, rel_y,
            width as i32, height as i32,
            SWP_NOZORDER | SWP_NOCOPYBITS | SWP_SHOWWINDOW | SWP_NOACTIVATE,
        );

        log::info!("setup_wallpaper_window: success");
    }
}
