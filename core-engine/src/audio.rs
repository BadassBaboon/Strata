//! System-audio capture → FFT → a Shadertoy-style audio texture.
//!
//! Shadertoy binds an audio/music iChannel to a 512×2 texture: row 0 is the FFT
//! spectrum (frequency magnitudes), row 1 is the raw waveform. Shaders read the
//! `.x` (red) channel. We reproduce that here from the system's output (what's
//! playing on the speakers, via loopback) so audio-reactive wallpapers work.
//!
//! Threading: cpal's `Stream` is `!Send` on most platforms, so it lives on a
//! dedicated capture thread. The audio callback pushes mono samples into a shared
//! ring buffer (tiny critical section). Render threads call `texture_rgba()`,
//! which lazily computes the FFT (cached ~12 ms so multiple monitors don't each
//! recompute) and returns the 512×2 RGBA bytes.
//!
//! On-demand: `texture_rgba()` records a timestamp; the capture thread only opens
//! the OS audio stream while requests are recent and closes it after a short idle
//! period. So when no audio-reactive wallpaper is on screen we capture nothing and
//! let the audio device idle (lightweight / battery-friendly).
//!
//! Robustness: if no device/stream is available the engine simply yields silence
//! (zeros) — audio shaders then render as they did before (no crash).
//!
//! Cross-platform note: system-audio LOOPBACK here relies on cpal's WASAPI
//! behaviour (input stream on the default *output* device → loopback). On
//! Linux/macOS that path won't loop back the system mix; this module is the single
//! place to add a platform-specific capture source later (no engine changes).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rustfft::{num_complex::Complex, Fft, FftPlanner};

pub const TEX_WIDTH: u32 = 512;
pub const TEX_HEIGHT: u32 = 2;
const FFT_SIZE: usize = 1024; // → 512 usable frequency bins
const RING_CAP: usize = 8192; // a few frames of samples at 44.1/48 kHz
const IDLE_STOP_MS: u64 = 2000; // stop capturing this long after the last request

struct Shared {
    ring: VecDeque<f32>,
    fft: Arc<dyn Fft<f32>>,
    scratch: Vec<Complex<f32>>, // reused FFT input buffer (no per-compute alloc)
    smooth: Vec<f32>,    // temporally-smoothed spectrum (TEX_WIDTH)
    tex: Vec<u8>,        // TEX_WIDTH * TEX_HEIGHT * 4 (RGBA)
    last_compute: Option<Instant>,
}

pub struct AudioEngine {
    shared: Arc<Mutex<Shared>>,
    running: Arc<AtomicBool>,
    // User "sensitivity" multiplier applied to the spectrum (f32 bits). 1.0 =
    // default. Atomic so the UI can retune it live without locking the audio.
    gain: std::sync::atomic::AtomicU32,
    // Millis (since `base`) of the most recent `texture_rgba()` call. The capture
    // thread opens the OS audio stream only while this is recent and closes it
    // after IDLE_STOP_MS — so we never capture system audio when no audio-reactive
    // wallpaper is on screen (key to the lightweight, battery-friendly goal).
    last_request: Arc<AtomicU64>,
    base: Instant,
    _capture: Option<std::thread::JoinHandle<()>>,
}

impl AudioEngine {
    pub fn new() -> Arc<Self> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let shared = Arc::new(Mutex::new(Shared {
            ring: VecDeque::with_capacity(RING_CAP),
            fft,
            scratch: vec![Complex { re: 0.0, im: 0.0 }; FFT_SIZE],
            smooth: vec![0.0; TEX_WIDTH as usize],
            tex: vec![0u8; (TEX_WIDTH * TEX_HEIGHT * 4) as usize],
            last_compute: None,
        }));
        let running = Arc::new(AtomicBool::new(true));
        let last_request = Arc::new(AtomicU64::new(0));
        let base = Instant::now();

        let shared_t = shared.clone();
        let running_t = running.clone();
        let last_req_t = last_request.clone();
        // Spawn failure is non-fatal — the engine just yields silence.
        let capture = std::thread::Builder::new()
            .name("strata-audio".into())
            .spawn(move || run_capture(shared_t, running_t, last_req_t, base))
            .map_err(|e| log::warn!("Audio thread spawn failed (silence): {}", e))
            .ok();

        Arc::new(Self {
            shared,
            running,
            gain: std::sync::atomic::AtomicU32::new(1.0f32.to_bits()),
            last_request,
            base,
            _capture: capture,
        })
    }

    /// Set the audio sensitivity (spectrum gain). 1.0 = default; higher = more
    /// reactive. Clamped to a sane range.
    pub fn set_sensitivity(&self, g: f32) {
        let g = g.clamp(0.0, 4.0);
        self.gain.store(g.to_bits(), Ordering::Relaxed);
    }

    pub fn sensitivity(&self) -> f32 {
        f32::from_bits(self.gain.load(Ordering::Relaxed))
    }

    /// Current 512×2 RGBA texture bytes (row 0 = FFT, row 1 = waveform).
    /// Lazily recomputed (cached ~12 ms) so per-monitor calls are cheap.
    pub fn texture_rgba(&self) -> Vec<u8> {
        // Signal demand so the capture thread keeps (or starts) the audio stream.
        self.last_request.store(self.base.elapsed().as_millis() as u64, Ordering::Relaxed);
        let mut s = match self.shared.lock() {
            Ok(s) => s,
            Err(_) => return vec![0u8; (TEX_WIDTH * TEX_HEIGHT * 4) as usize], // poisoned → silence
        };
        let now = Instant::now();
        let stale = s.last_compute.map_or(true, |t| now.duration_since(t) >= Duration::from_millis(12));
        if stale {
            let gain = f32::from_bits(self.gain.load(Ordering::Relaxed));
            compute(&mut s, gain);
            s.last_compute = Some(now);
        }
        s.tex.clone()
    }
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

/// Compute the spectrum + waveform from the ring buffer into `s.tex`.
/// `gain` is the user sensitivity multiplier applied to the spectrum.
fn compute(s: &mut Shared, gain: f32) {
    let n = FFT_SIZE;
    // Pull the most recent `n` samples (zero-padded if we don't have enough yet)
    // into the reused scratch buffer — no per-call allocation.
    let len = s.ring.len();
    let start = len.saturating_sub(n);
    let mut filled = 0;
    for (i, sample) in s.ring.iter().skip(start).enumerate() {
        // Hann window to reduce spectral leakage.
        let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / (n as f32 - 1.0)).cos();
        s.scratch[i] = Complex { re: sample * w, im: 0.0 };
        filled = i + 1;
    }
    for slot in &mut s.scratch[filled..] {
        *slot = Complex { re: 0.0, im: 0.0 };
    }

    let fft = s.fft.clone();
    fft.process(&mut s.scratch);
    let buf = &s.scratch;

    let bins = TEX_WIDTH as usize; // 512, = n/2
    let row_stride = (TEX_WIDTH * 4) as usize;
    // ── Row 0: FFT magnitudes (log-scaled, smoothed) ──
    for i in 0..bins {
        let mag = buf[i].norm() / (n as f32);
        let db = 20.0 * (mag + 1e-6).log10();
        // Match the WebAudio AnalyserNode mapping Shadertoy uses: dB in
        // [minDecibels=-100, maxDecibels=-30] → [0,1]. This is much "hotter" than
        // a -60..0 range — present frequencies land at 0.6-1.0, which is what
        // shaders expecting Shadertoy's spectrum (e.g. logistic-gated ones) need.
        let v = ((db + 100.0) / 70.0 * gain).clamp(0.0, 1.0);
        // Temporal smoothing: fast attack, slow release (nice visualizer feel).
        let prev = s.smooth[i];
        s.smooth[i] = if v > prev { v } else { prev * 0.85 + v * 0.15 };
        let byte = (s.smooth[i] * 255.0) as u8;
        let p = i * 4;
        s.tex[p] = byte; s.tex[p + 1] = byte; s.tex[p + 2] = byte; s.tex[p + 3] = 255;
    }
    // ── Row 1: waveform (most recent samples, centred at 0.5) ──
    for i in 0..bins {
        let idx = len.saturating_sub(bins) + i;
        let sample = s.ring.get(idx).copied().unwrap_or(0.0);
        let v = (0.5 + 0.5 * sample).clamp(0.0, 1.0);
        let byte = (v * 255.0) as u8;
        let p = row_stride + i * 4;
        s.tex[p] = byte; s.tex[p + 1] = byte; s.tex[p + 2] = byte; s.tex[p + 3] = 255;
    }
}

fn run_capture(
    shared: Arc<Mutex<Shared>>,
    running: Arc<AtomicBool>,
    last_request: Arc<AtomicU64>,
    base: Instant,
) {
    use cpal::traits::StreamTrait;
    // The stream is held here only while an audio-reactive shader is actively
    // requesting data; otherwise it's dropped so the OS audio device can idle.
    let mut stream: Option<cpal::Stream> = None;
    let mut warned = false; // suppress repeated build-failure logs within one session

    while running.load(Ordering::SeqCst) {
        let now = base.elapsed().as_millis() as u64;
        let last = last_request.load(Ordering::Relaxed);
        // `last > 0` means at least one request has happened; idle if stale.
        let wanted = last > 0 && now.saturating_sub(last) < IDLE_STOP_MS;

        match (wanted, stream.is_some()) {
            (true, false) => match build_stream(shared.clone()) {
                Ok(s) => match s.play() {
                    Ok(()) => { log::info!("Audio capture started (on demand)"); stream = Some(s); warned = false; }
                    Err(e) => { if !warned { log::warn!("Audio stream failed to start: {}", e); warned = true; } }
                },
                Err(e) => { if !warned { log::warn!("Audio capture unavailable (silence): {}", e); warned = true; } }
            },
            (false, true) => {
                stream = None; // drop → stop capturing
                if let Ok(mut s) = shared.lock() {
                    s.ring.clear();
                    s.smooth.iter_mut().for_each(|v| *v = 0.0);
                }
                log::info!("Audio capture stopped (idle)");
            }
            _ => {}
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn build_stream(shared: Arc<Mutex<Shared>>) -> Result<cpal::Stream, String> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    // Prefer system-audio LOOPBACK: on WASAPI, building an *input* stream on the
    // default *output* device captures what's playing on the speakers. Fall back
    // to the default input device (microphone) if there's no output device.
    let (device, config, source) = if let Some(dev) = host.default_output_device() {
        let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
        (dev, cfg, "output loopback")
    } else if let Some(dev) = host.default_input_device() {
        let cfg = dev.default_input_config().map_err(|e| e.to_string())?;
        (dev, cfg, "input (microphone)")
    } else {
        return Err("no audio device found".into());
    };

    log::info!("Audio source: {} @ {} Hz, {} ch, {:?}",
        source, config.sample_rate().0, config.channels(), config.sample_format());

    let channels = config.channels() as usize;
    let sample_format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();
    let err_fn = |e| log::warn!("audio stream error: {}", e);

    // Downmix a frame of interleaved channels to one mono sample, append to ring.
    fn push(shared: &Arc<Mutex<Shared>>, frame_mono: impl Iterator<Item = f32>) {
        if let Ok(mut s) = shared.lock() {
            for m in frame_mono {
                if s.ring.len() >= RING_CAP { s.ring.pop_front(); }
                s.ring.push_back(m);
            }
        }
    }

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let sh = shared.clone();
            device.build_input_stream(&stream_config,
                move |data: &[f32], _: &_| {
                    push(&sh, data.chunks(channels).map(|f| f.iter().copied().sum::<f32>() / channels as f32));
                }, err_fn, None)
        }
        cpal::SampleFormat::I16 => {
            let sh = shared.clone();
            device.build_input_stream(&stream_config,
                move |data: &[i16], _: &_| {
                    push(&sh, data.chunks(channels).map(|f| f.iter().map(|&s| s as f32 / 32768.0).sum::<f32>() / channels as f32));
                }, err_fn, None)
        }
        cpal::SampleFormat::U16 => {
            let sh = shared.clone();
            device.build_input_stream(&stream_config,
                move |data: &[u16], _: &_| {
                    push(&sh, data.chunks(channels).map(|f| f.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).sum::<f32>() / channels as f32));
                }, err_fn, None)
        }
        other => return Err(format!("unsupported sample format {:?}", other)),
    };

    stream.map_err(|e| e.to_string())
}
