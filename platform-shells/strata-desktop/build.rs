fn main() {
    slint_build::compile("ui/main.slint").expect("Slint compilation failed");

    // Embed the application icon into the .exe so it shows in Explorer / the
    // shortcut, not just the taskbar (which Windows derives from the window icon).
    #[cfg(windows)]
    {
        let icon = "../../assets/app-icon_dark.ico";
        println!("cargo:rerun-if-changed={icon}");
        let mut res = winresource::WindowsResource::new();
        res.set_icon(icon);
        if let Err(e) = res.compile() {
            println!("cargo:warning=failed to embed exe icon: {e}");
        }
    }
}
