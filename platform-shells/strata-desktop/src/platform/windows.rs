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

// ── Main-window placement save/restore ──────────────────────────────────────────
// Slint's hide()/show() (used for the tray) does NOT preserve the window's maximized
// state or monitor - show() re-applies the component's preferred size at a default
// position. So before hiding we snapshot the full WINDOWPLACEMENT (normal rect +
// maximized/minimized flag + which monitor) and re-apply it on show. SetWindowPlacement
// also triggers a resize → Slint repaints fully, so it doubles as the bring-up repaint.
#[cfg(target_os = "windows")]
thread_local! {
    static SAVED_PLACEMENT: std::cell::Cell<
        Option<windows_sys::Win32::UI::WindowsAndMessaging::WINDOWPLACEMENT>
    > = const { std::cell::Cell::new(None) };
}

#[cfg(target_os = "windows")]
pub fn save_window_placement(hwnd: isize) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetWindowPlacement, WINDOWPLACEMENT};
    if hwnd == 0 { return; }
    unsafe {
        let mut wp: WINDOWPLACEMENT = std::mem::zeroed();
        wp.length = std::mem::size_of::<WINDOWPLACEMENT>() as u32;
        if GetWindowPlacement(hwnd as HWND, &mut wp) != 0 {
            SAVED_PLACEMENT.with(|c| c.set(Some(wp)));
        }
    }
}

/// Re-apply the placement saved by `save_window_placement`. Returns true if the window
/// was restored to a MAXIMIZED state - in that case the maximize already resizes the
/// window (so it repaints itself) and the caller must NOT add a size nudge (that would
/// un-maximize it). For a normal restore the size may be unchanged (→ no resize → no
/// repaint), so the caller still needs its nudge.
#[cfg(target_os = "windows")]
pub fn restore_window_placement(hwnd: isize) -> bool {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SetWindowPlacement, SW_SHOWMAXIMIZED, SW_SHOWMINIMIZED, SW_SHOWNORMAL,
        WPF_RESTORETOMAXIMIZED,
    };
    if hwnd == 0 { return false; }
    SAVED_PLACEMENT.with(|c| {
        if let Some(mut wp) = c.get() {
            // Never restore to MINIMIZED - the user asked to show it. Fall back to the
            // "restore-to" target (maximized if it was maximized before minimizing).
            if wp.showCmd == SW_SHOWMINIMIZED as u32 {
                wp.showCmd = if (wp.flags & WPF_RESTORETOMAXIMIZED) != 0 {
                    SW_SHOWMAXIMIZED as u32
                } else {
                    SW_SHOWNORMAL as u32
                };
            }
            unsafe { SetWindowPlacement(hwnd as HWND, &wp); }
            wp.showCmd == SW_SHOWMAXIMIZED as u32
        } else {
            false
        }
    })
}

// ── WorkerW detection ─────────────────────────────────────────────────────────
//
// The Windows desktop rendering layer is a special window called WorkerW.
// After sending the 0x052C message to Progman, a WorkerW is created (or already
// exists) behind the desktop icon container.  Child windows of this WorkerW
// render behind desktop icons on ALL virtual desktops automatically - WorkerW
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
            log::error!("WorkerW search: Progman not found - cannot enable wallpaper mode");
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
                log::warn!("WorkerW: strategy 1 candidate {:x} too small - skipping", workerw);
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
                log::warn!("WorkerW: strategy 2 candidate {:x} too small - skipping", ww);
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
            "WorkerW: not found after all strategies - wallpaper will sit above icons. \
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

        // Reparent into WorkerW - places rendering behind desktop icons.
        // We do NOT change GWL_STYLE (keep WS_POPUP) to avoid the winit
        // GetClientRect panic described above.
        let prev = windows_sys::Win32::UI::WindowsAndMessaging::SetParent(child, workerw);
        if prev == 0 {
            let err = GetLastError();
            log::error!(
                "setup_wallpaper_window: SetParent failed - error {} \
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

/// True if a true-fullscreen app covers this monitor. Walks top-level windows in Z-order
/// (top first) and inspects the TOPMOST real (visible, non-cloaked, non-shell) window that
/// overlaps the monitor: covered iff it spans the whole monitor (incl. the taskbar area -
/// a merely maximised window leaves the taskbar and doesn't qualify). Per-monitor and
/// focus-independent, so a fullscreen app on ANOTHER monitor never counts here.
#[cfg(target_os = "windows")]
pub fn monitor_covered(origin: (i32, i32), size: (u32, u32)) -> bool {
    use windows_sys::Win32::UI::WindowsAndMessaging::{IsWindowVisible, IsIconic, GetClassNameW};
    use windows_sys::Win32::Graphics::Dwm::{DwmGetWindowAttribute, DWMWA_CLOAKED};

    struct Ctx { ox: i32, oy: i32, mw: i32, mh: i32, covered: bool, done: bool }

    unsafe extern "system" fn cb(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let ctx = &mut *(lparam as *mut Ctx);
        if ctx.done { return 0; }
        if IsWindowVisible(hwnd) == 0 || IsIconic(hwnd) != 0 { return 1; }
        let mut cloaked: u32 = 0;
        DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED as u32,
            (&mut cloaked as *mut u32).cast(), std::mem::size_of::<u32>() as u32);
        if cloaked != 0 { return 1; }
        let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        if GetWindowRect(hwnd, &mut r) == 0 { return 1; }
        let intersects = r.left < ctx.ox + ctx.mw && r.right > ctx.ox
            && r.top < ctx.oy + ctx.mh && r.bottom > ctx.oy;
        if !intersects { return 1; }
        let mut buf = [0u16; 64];
        let n = GetClassNameW(hwnd, buf.as_mut_ptr(), buf.len() as i32).max(0) as usize;
        let cls = String::from_utf16_lossy(&buf[..n]);
        if matches!(cls.as_str(),
            "Progman" | "WorkerW" | "Shell_TrayWnd" | "Shell_SecondaryTrayWnd"
            | "Windows.UI.Core.CoreWindow" | "XamlExplorerHostIslandWindow"
        ) { return 1; }
        // Topmost real window over this monitor: covered iff it fully spans it.
        ctx.covered = r.left <= ctx.ox && r.top <= ctx.oy
            && r.right >= ctx.ox + ctx.mw && r.bottom >= ctx.oy + ctx.mh;
        ctx.done = true;
        0
    }

    let mut ctx = Ctx { ox: origin.0, oy: origin.1, mw: size.0 as i32, mh: size.1 as i32, covered: false, done: false };
    unsafe { EnumWindows(Some(cb), &mut ctx as *mut _ as LPARAM); }
    ctx.covered
}
