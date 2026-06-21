use std::time::Instant;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
pub struct ShaderUniforms {
    pub iResolution: [f32; 3],      // 0..12
    pub iTime: f32,                 // 12..16
    pub iTimeDelta: f32,            // 16..20
    pub iFrameRate: f32,            // 20..24
    pub iFrame: i32,                // 24..28
    pub iSampleRate: f32,           // 28..32
    pub iMouse: [f32; 4],           // 32..48
    pub iDate: [f32; 4],            // 48..64
    pub iChannelTime: [f32; 4],     // 64..80
    pub iChannelResolution: [[f32; 4]; 4], // 80..144
    pub iGlobalResolution: [f32; 3],       // 144..156
    pub _pad0: f32,                        // 156..160
    pub iMonitorOffset: [f32; 2],          // 160..168
    pub iOpacity: f32,                     // 168..172  per-layer opacity (image pass)
    pub iBlendMode: i32,                   // 172..176  0=normal 1=additive 2=multiply
}

pub struct UniformState {
    pub uniforms: ShaderUniforms,
    pub buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    
    // Tracking state for calculations
    start_time: Instant,
    last_frame_time: Instant,

    // When Some, iTime is forced to this value (instead of wall-clock) - used by
    // headless thumbnail capture to render a deterministic, settled frame.
    pub headless_time: Option<f32>,

    // Mouse state
    mouse_down: bool,
    click_pos: (f32, f32),
    current_pos: (f32, f32),
}

impl UniformState {
    pub fn new(device: &wgpu::Device, width: f32, height: f32) -> Self {
        let now = Instant::now();
        let uniforms = ShaderUniforms {
            iResolution: [width, height, 1.0],
            iTime: 0.0,
            iTimeDelta: 0.016,
            iFrameRate: 60.0,
            iFrame: 0,
            iSampleRate: 44100.0,
            iMouse: [0.0, 0.0, 0.0, 0.0],
            iDate: Self::calculate_date(),
            iChannelTime: [0.0; 4],
            iChannelResolution: [[0.0; 4]; 4],
            iGlobalResolution: [width, height, 1.0],
            _pad0: 0.0,
            iMonitorOffset: [0.0, 0.0],
            iOpacity: 1.0,
            iBlendMode: 0,
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Global Uniform Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Global Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            uniforms,
            buffer,
            bind_group_layout,
            bind_group,
            start_time: now,
            last_frame_time: now,
            headless_time: None,
            mouse_down: false,
            click_pos: (0.0, 0.0),
            current_pos: (0.0, 0.0),
        }
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.uniforms.iResolution = [width, height, 1.0];
    }

    pub fn set_monitor_offset(&mut self, x: f32, y: f32) {
        self.uniforms.iMonitorOffset = [x, y];
    }

    pub fn set_global_resolution(&mut self, width: f32, height: f32) {
        self.uniforms.iGlobalResolution = [width, height, 1.0];
    }

    pub fn set_channel_resolution(&mut self, index: usize, width: f32, height: f32) {
        if index < 4 {
            self.uniforms.iChannelResolution[index] = [width, height, 1.0, 0.0];
        }
    }

    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        // Flip y to match Shadertoy's bottom-left origin
        let flipped_y = self.uniforms.iResolution[1] - y;
        self.current_pos = (x, flipped_y);
        
        // iMouse.xy always tracks current position if button is down.
        // If button is up, it stays at the last position it was when down? 
        // Actually, Shadertoy's iMouse.xy tracks the current position ONLY while the mouse is pressed.
        // When not pressed, iMouse.xy does NOT update.
        if self.mouse_down {
            self.uniforms.iMouse[0] = x;
            self.uniforms.iMouse[1] = flipped_y;
        }
    }

    pub fn set_mouse_down(&mut self, down: bool) {
        self.mouse_down = down;
        if down {
            self.click_pos = self.current_pos;
            self.uniforms.iMouse[0] = self.current_pos.0;
            self.uniforms.iMouse[1] = self.current_pos.1;
            self.uniforms.iMouse[2] = self.current_pos.0;
            self.uniforms.iMouse[3] = self.current_pos.1;
        } else {
            // In Shadertoy: zw becomes negative to signal mouse up.
            // We use .abs() and then negate to ensure it's negative even if it was 0 (as best as we can with f32).
            self.uniforms.iMouse[2] = -self.click_pos.0.abs();
            self.uniforms.iMouse[3] = -self.click_pos.1.abs();
        }
    }

    pub fn update_global_only(&mut self) {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame_time).as_secs_f32();

        if let Some(t) = self.headless_time {
            // Deterministic capture: caller drives iTime; fixed 60fps delta.
            self.uniforms.iTime = t;
            self.uniforms.iTimeDelta = 1.0 / 60.0;
            self.uniforms.iFrameRate = 60.0;
        } else {
            self.uniforms.iTime = now.duration_since(self.start_time).as_secs_f32();
            self.uniforms.iTimeDelta = delta;
            self.uniforms.iFrameRate = if delta > 0.0 { 1.0 / delta } else { 60.0 };
            self.uniforms.iDate = Self::calculate_date();
        }
        self.uniforms.iFrame += 1;
        self.last_frame_time = now;
    }

    pub fn update(&mut self, queue: &wgpu::Queue) {
        self.update_global_only();
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    fn calculate_date() -> [f32; 4] {
        // Shadertoy's iDate = (year, month, day, seconds-since-local-midnight),
        // matching JS Date: month is 0-based, day is 1-based, and crucially the
        // time is LOCAL wall-clock - that's what the clock shaders read from
        // iDate.w to display the user's actual system time (incl. timezone/DST).
        use chrono::{Datelike, Timelike, Local};
        let now = Local::now();
        let seconds_in_day = now.num_seconds_from_midnight() as f32
            + now.nanosecond() as f32 / 1_000_000_000.0;
        [
            now.year() as f32,
            now.month0() as f32, // 0 = January (Shadertoy/JS convention)
            now.day() as f32,    // 1-based
            seconds_in_day,
        ]
    }
}
