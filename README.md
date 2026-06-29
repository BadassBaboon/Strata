# Strata

A lightweight, hardware-accelerated live wallpaper and screensaver engine.

Strata runs animated, interactive wallpapers directly on your computer's graphics card. Unlike web-based wallpaper tools, it uses native graphics APIs (DirectX 12, Vulkan, and Metal) to reduce memory and CPU usage. The application includes a selection of shaders ported from Shadertoy.com.

![Strata screenshot](assets/screenshot-1.jpg)

![Strata screenshot](assets/screenshot-2.jpg)

## Features

* **Shadertoy compatibility:** Runs standard Shadertoy animations natively.
* **Audio-reactive wallpapers:** Wallpapers react to your system audio. The engine captures audio only when a reactive wallpaper is active.
* **Mouse interaction:** Wallpapers react to your cursor. You can apply this to all wallpapers, specific ones, or turn it off entirely.
* **Parallax Studio:** Converts 2D photos into 3D animated wallpapers that shift with your cursor. An on-device machine learning pipeline separates the subject from the background and fills in the gaps.
* **Wallpaper import:** Import shaders exported from Shadertoy. The application converts them and generates thumbnails automatically.
* **Multi-monitor support:** Assign different wallpapers to each monitor, or stretch one wallpaper across all screens. You can adjust the resolution scale for each layer from 0.25x to 1.0x.
* **Wallpaper library:** Browse, search, and filter your installed wallpapers.
* **Movie-based wallpaper support:** Bring your favorite movie based wallpaper to Strata, you can import .mp4 and .webm and use them as your desktop wallpaper.
* **Automatic updates:** Checks for engine and wallpaper library updates weekly.
* **Native interface:** A lightweight user interface with dark and light themes, a system tray icon, and performance diagnostics.

## Project goals

* **Low resource usage:** Idle CPU usage stays under 1% while rendering two monitors at 60 FPS. The memory footprint is approximately 200MB.
* **On-demand execution:** Resources are only active when needed. Audio capture, thumbnail generation, and wallpaper windows shut down when not in use.

## Supported platforms

| Platform | Status |
| :--- | :--- |
| Windows | Supported |
| Linux | Planned |
| Android | Planned |
| macOS | Planned |
| iOS | Planned |

## To-do list

- [ ] **Game auto-pause:** Pause wallpaper rendering when a full-screen game or heavy application is running.
- [ ] **Daily wallpaper rotation:** Automatically change the active wallpaper each day from your library.
- [ ] **Parallax auto-detect:** Automatically select the optimal quality setting for Parallax Studio based on your hardware.
- [ ] **Translation support:** Add multiple language options to the interface.
- [ ] **Screensaver mode:** Use the engine as a native screensaver.
- [ ] **Distribution packaging:** Create a standard Windows installer.
- [ ] **Spotify integration:** Display track information and album art inside active wallpapers.
- [ ] **Linux support:** Add Wayland and X11 desktop integration.
- [ ] **Android support:** Create a native Android wallpaper service package.
- [ ] **macOS support:** Integrate with macOS desktop spaces and the menu bar.
- [ ] **iOS support:** Build a compatible iOS application.

## License

Strata is free and open-source software, licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). Anyone may use, study, modify, and redistribute Strata, but distributed forks or derivatives must remain open-source under the GPL. There are no paid tiers and no donations.

This repository contains only the engine code. Content is fetched at runtime:
* Shaders, thumbnails, and textures are downloaded from the [Strata-Library](https://github.com/BadassBaboon/Strata-Library) repository.
* Machine learning models are downloaded from Hugging Face.
* Parallax Studio presets are embedded in the engine.
