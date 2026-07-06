//! Day/night cycle: drives sun direction, sky/fog color, ambient and sun intensity.

use glam::Vec3;

const DAY_LENGTH_SECS: f32 = 1200.0; // S8: a full cycle in 20 minutes (vanilla)

pub struct Environment {
    /// Time of day in [0, 1): 0.25 = sunrise, 0.5 = noon, 0.75 = sunset, 0.0 = midnight.
    pub time: f32,
    pub paused: bool,
}

impl Environment {
    pub fn new(time: f32) -> Self {
        Self {
            time: time.rem_euclid(1.0),
            paused: false,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if !self.paused {
            self.time = (self.time + dt / DAY_LENGTH_SECS).rem_euclid(1.0);
        }
    }

    /// Direction toward the sun (unit vector).
    pub fn sun_dir(&self) -> Vec3 {
        // Sun sweeps the sky; angle 0 at midnight (below horizon), PI/2 at noon.
        let angle = (self.time - 0.25) * std::f32::consts::TAU;
        Vec3::new(angle.cos() * 0.6, angle.sin(), 0.45).normalize()
    }

    /// 0 at night, 1 in full day, with smooth dawn/dusk.
    pub fn day_factor(&self) -> f32 {
        smoothstep(-0.12, 0.22, self.sun_dir().y)
    }

    /// How close the sun is to the horizon (for sunset tinting).
    fn horizon_factor(&self) -> f32 {
        let y = self.sun_dir().y;
        (1.0 - (y.abs() / 0.32)).clamp(0.0, 1.0) * smoothstep(-0.25, 0.0, y)
    }

    pub fn sky_color(&self) -> [f32; 3] {
        let day = Vec3::new(0.46, 0.64, 0.92);
        let night = Vec3::new(0.02, 0.03, 0.08);
        let sunset = Vec3::new(0.92, 0.52, 0.26);
        let base = night.lerp(day, self.day_factor());
        let c = base.lerp(sunset, self.horizon_factor() * 0.7);
        [c.x, c.y, c.z]
    }

    pub fn fog_color(&self) -> [f32; 3] {
        self.sky_color()
    }

    pub fn ambient(&self) -> f32 {
        lerp(0.07, 0.34, self.day_factor())
    }

    pub fn sun_intensity(&self) -> f32 {
        self.day_factor()
    }

    pub fn wgpu_clear(&self) -> wgpu::Color {
        let c = self.sky_color();
        wgpu::Color {
            r: c[0] as f64,
            g: c[1] as f64,
            b: c[2] as f64,
            a: 1.0,
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
