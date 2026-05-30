//! First-person fly camera + a uniform buffer payload, plus an input-driven controller.

use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,   // radians, rotation around +Y
    pub pitch: f32, // radians, look up/down
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            position,
            yaw,
            pitch,
            fovy: 70_f32.to_radians(),
            znear: 0.1,
            zfar: 4000.0,
        }
    }

    /// Unit forward direction from yaw/pitch.
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_to_rh(self.position, self.forward(), Vec3::Y);
        let proj = Mat4::perspective_rh(self.fovy, aspect.max(0.0001), self.znear, self.zfar);
        proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0; 4],
        }
    }

    pub fn update(&mut self, camera: &Camera, aspect: f32) {
        self.view_proj = camera.view_proj(aspect).to_cols_array_2d();
        self.cam_pos = [
            camera.position.x,
            camera.position.y,
            camera.position.z,
            1.0,
        ];
    }
}

/// Keyboard/mouse driven fly controller. Movement keys set booleans; mouse motion accumulates
/// look deltas; `update` integrates them into the camera each frame.
pub struct FlyController {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub fast: bool,
    yaw_delta: f32,
    pitch_delta: f32,
    pub speed: f32,
    pub sensitivity: f32,
}

impl Default for FlyController {
    fn default() -> Self {
        Self {
            forward: false,
            back: false,
            left: false,
            right: false,
            up: false,
            down: false,
            fast: false,
            yaw_delta: 0.0,
            pitch_delta: 0.0,
            speed: 24.0,
            sensitivity: 0.0022,
        }
    }
}

impl FlyController {
    pub fn process_mouse(&mut self, dx: f32, dy: f32) {
        self.yaw_delta += dx;
        self.pitch_delta += dy;
    }

    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        // Look
        camera.yaw += self.yaw_delta * self.sensitivity;
        camera.pitch = (camera.pitch - self.pitch_delta * self.sensitivity)
            .clamp(-1.5533, 1.5533); // ~89 degrees
        self.yaw_delta = 0.0;
        self.pitch_delta = 0.0;

        // Move on the horizontal plane relative to view, plus vertical fly.
        let fwd = camera.forward();
        let flat_fwd = Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero();
        let right = flat_fwd.cross(Vec3::Y).normalize_or_zero();

        let mut dir = Vec3::ZERO;
        if self.forward {
            dir += flat_fwd;
        }
        if self.back {
            dir -= flat_fwd;
        }
        if self.right {
            dir += right;
        }
        if self.left {
            dir -= right;
        }
        if self.up {
            dir += Vec3::Y;
        }
        if self.down {
            dir -= Vec3::Y;
        }

        let speed = if self.fast { self.speed * 4.0 } else { self.speed };
        camera.position += dir.normalize_or_zero() * speed * dt;
    }
}
