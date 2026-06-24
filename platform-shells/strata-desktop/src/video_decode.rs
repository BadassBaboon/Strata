//! Windows video decoder for movie wallpapers, built on Media Foundation.
//!
//! All Media Foundation / COM objects live ENTIRELY on a single worker thread (created
//! in `MfVideoDecoder::new`), so we never move a COM interface across threads. The
//! worker decodes (hardware-accelerated by MF where the GPU supports it), paces itself
//! to the clip's presentation timestamps, and drops the latest BGRA frame into a shared
//! slot. The render thread calls `next_frame()` to take it. The clip loops by reopening
//! the source reader on end-of-stream.
//!
//! This implements the OS-agnostic `core_engine::video::VideoDecoder` trait, so other
//! platforms (Linux VA-API/ffmpeg, macOS VideoToolbox) can provide their own decoder
//! without the renderer knowing the difference.

#![cfg(windows)]

use core_engine::video::{VideoDecoder, VideoFrame};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::core::HSTRING;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

struct Shared {
    frame: Mutex<Option<VideoFrame>>,
    stop: AtomicBool,
}

pub struct MfVideoDecoder {
    dims: (u32, u32),
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl MfVideoDecoder {
    /// Open `path` and start decoding on a worker thread. Blocks only until the worker
    /// has read the video dimensions (or failed); decoding then continues in the
    /// background.
    pub fn new(path: &std::path::Path) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            frame: Mutex::new(None),
            stop: AtomicBool::new(false),
        });
        let path = path.to_path_buf();
        let shared_worker = shared.clone();
        let (tx, rx) = std::sync::mpsc::channel::<Result<(u32, u32), String>>();

        let worker = std::thread::Builder::new()
            .name("strata-video-decode".into())
            .spawn(move || decode_loop(&path, &shared_worker, tx))
            .map_err(|e| format!("spawn video thread: {e}"))?;

        match rx.recv() {
            Ok(Ok(dims)) => Ok(Self { dims, shared, worker: Some(worker) }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("video decode thread exited before reporting dimensions".into()),
        }
    }
}

impl VideoDecoder for MfVideoDecoder {
    fn dimensions(&self) -> (u32, u32) {
        self.dims
    }
    fn next_frame(&mut self) -> Option<VideoFrame> {
        self.shared.frame.lock().ok().and_then(|mut s| s.take())
    }
}

impl Drop for MfVideoDecoder {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

/// Opened reader plus the output frame geometry.
struct OpenReader {
    reader: IMFSourceReader,
    width: u32,
    height: u32,
    /// Bytes between rows in the locked buffer (always positive; we copy top-down and
    /// leave any vertical flip to the renderer).
    stride: usize,
}

const FIRST_VIDEO_STREAM: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;

/// Open a Media Foundation source reader for `path`, configured to output NV12.
///
/// NV12 is the hardware decoder's NATIVE output, so requesting it needs no video
/// processor and costs no CPU colour conversion (the renderer does YUV->RGB on the GPU).
unsafe fn open_reader(path: &std::path::Path) -> windows::core::Result<OpenReader> {
    let url = HSTRING::from(path.as_os_str());
    let reader = MFCreateSourceReaderFromURL(&url, None)?;

    let out_type: IMFMediaType = MFCreateMediaType()?;
    out_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
    out_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
    reader.SetCurrentMediaType(FIRST_VIDEO_STREAM, None, &out_type)?;

    // Read back the negotiated type for the real frame size + row stride (the Y plane's
    // stride; the interleaved UV plane uses the same stride for half as many rows).
    let cur = reader.GetCurrentMediaType(FIRST_VIDEO_STREAM)?;
    let frame_size = cur.GetUINT64(&MF_MT_FRAME_SIZE)?;
    let width = (frame_size >> 32) as u32;
    let height = (frame_size & 0xffff_ffff) as u32;
    let stride = match cur.GetUINT32(&MF_MT_DEFAULT_STRIDE) {
        Ok(s) => (s as i32).unsigned_abs() as usize,
        Err(_) => width as usize,
    };

    Ok(OpenReader { reader, width, height, stride })
}

fn decode_loop(
    path: &std::path::Path,
    shared: &Arc<Shared>,
    init_tx: std::sync::mpsc::Sender<Result<(u32, u32), String>>,
) {
    unsafe {
        // The worker is MTA so MF's COM objects are thread-affine to it only.
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        if MFStartup(MF_VERSION, MFSTARTUP_FULL).is_err() {
            let _ = init_tx.send(Err("MFStartup failed".into()));
            CoUninitialize();
            return;
        }

        let mut open = match open_reader(path) {
            Ok(o) => o,
            Err(e) => {
                let _ = init_tx.send(Err(format!("open video: {e}")));
                MFShutdown().ok();
                CoUninitialize();
                return;
            }
        };
        let _ = init_tx.send(Ok((open.width, open.height)));
        log::info!("video decode loop started ({}x{}, stride {})", open.width, open.height, open.stride);

        let mut clock = Instant::now();
        let mut base_pts: Option<i64> = None;
        let mut frame_count: u64 = 0;

        while !shared.stop.load(Ordering::Relaxed) {
            let mut flags: u32 = 0;
            let mut timestamp: i64 = 0;
            let mut sample: Option<IMFSample> = None;
            if let Err(e) = open.reader.ReadSample(
                FIRST_VIDEO_STREAM, 0, None,
                Some(&mut flags), Some(&mut timestamp), Some(&mut sample),
            ) {
                log::error!("video ReadSample failed: {e}");
                break;
            }

            // End of stream -> reopen to loop the clip.
            if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                match open_reader(path) {
                    Ok(o) => { open = o; base_pts = None; }
                    Err(_) => break,
                }
                continue;
            }

            let Some(sample) = sample else { continue }; // stream tick / gap

            // Pace to the clip's presentation timestamps (100 ns units) so playback runs
            // at real speed instead of decoding as fast as possible.
            if base_pts.is_none() {
                base_pts = Some(timestamp);
                clock = Instant::now();
            }
            let target = Duration::from_nanos(((timestamp - base_pts.unwrap()).max(0) as u64) * 100);
            let elapsed = clock.elapsed();
            if target > elapsed {
                std::thread::sleep(target - elapsed);
            }
            if shared.stop.load(Ordering::Relaxed) { break; }

            // Copy the locked NV12 frame into tightly packed Y + UV planes (the source
            // rows are `stride` bytes apart; we copy `width` bytes per row). The UV plane
            // follows the Y plane in the buffer (Y is `stride * height` bytes).
            let frame = (|| -> windows::core::Result<VideoFrame> {
                let buffer: IMFMediaBuffer = sample.ConvertToContiguousBuffer()?;
                let mut ptr: *mut u8 = std::ptr::null_mut();
                let mut len: u32 = 0;
                buffer.Lock(&mut ptr, None, Some(&mut len))?;
                let src = std::slice::from_raw_parts(ptr, len as usize);

                let w = open.width as usize;
                let h = open.height as usize;
                // The Y plane spans the CODED height (16-aligned, can exceed the display
                // height, e.g. 1080 -> 1088), so the UV plane starts at stride*coded, not
                // stride*display. Derive coded height from the buffer length
                // (total = stride * coded * 3/2) so chroma lines up at any resolution.
                let coded_h = if open.stride > 0 {
                    ((src.len() / open.stride) * 2 / 3).max(h)
                } else { h };
                let mut y_plane = vec![0u8; w * h];
                for row in 0..h {
                    let s = row * open.stride;
                    if s + w <= src.len() {
                        y_plane[row * w..(row + 1) * w].copy_from_slice(&src[s..s + w]);
                    }
                }
                let uv_base = open.stride * coded_h;
                let mut uv_plane = vec![0u8; w * (h / 2)];
                for row in 0..(h / 2) {
                    let s = uv_base + row * open.stride;
                    if s + w <= src.len() {
                        uv_plane[row * w..(row + 1) * w].copy_from_slice(&src[s..s + w]);
                    }
                }

                buffer.Unlock()?;
                Ok(VideoFrame { width: open.width, height: open.height, y: y_plane, uv: uv_plane })
            })();

            match frame {
                Ok(frame) => {
                    if frame_count == 0 {
                        log::info!("video: first frame decoded ({} y bytes, {} uv bytes)", frame.y.len(), frame.uv.len());
                    }
                    frame_count += 1;
                    if let Ok(mut slot) = shared.frame.lock() {
                        *slot = Some(frame); // replace any unconsumed frame (drop it)
                    }
                }
                Err(e) => log::error!("video frame copy failed: {e}"),
            }
        }
        log::info!("video decode loop ended after {frame_count} frames");

        MFShutdown().ok();
        CoUninitialize();
    }
}
