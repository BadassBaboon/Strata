//! Video frame interface.
//!
//! Movie wallpapers play in a separate WebView daemon process (the system browser's
//! hardware media pipeline), so the engine itself never decodes or blits video. The only
//! remaining use of a `VideoDecoder` is generating a still thumbnail from a clip's first
//! frame at import time; this module keeps that OS-agnostic abstraction (the concrete
//! decoder - Windows Media Foundation - lives in the desktop shell).

/// One decoded video frame in **NV12** - the format hardware decoders output natively, so
/// there is no CPU colour conversion and each frame is only `width*height*3/2` bytes.
///
/// * `y`  - full-resolution luma plane, `width * height` bytes, tightly packed.
/// * `uv` - half-resolution interleaved chroma plane, `width * (height/2)` bytes
///          (`U,V,U,V…`), tightly packed.
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
}

/// A video source that hands over decoded frames. Used at import time to grab the first
/// frame for a thumbnail. Implementations decode on their own worker thread.
pub trait VideoDecoder: Send {
    /// Native video dimensions in pixels (width, height).
    fn dimensions(&self) -> (u32, u32);

    /// The most recent decoded frame if a new one is available since the last call,
    /// else `None`.
    fn next_frame(&mut self) -> Option<VideoFrame>;
}
