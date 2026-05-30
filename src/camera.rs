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
    pub sun_dir: [f32; 4],
    pub sky_color: [f32; 4],
    pub fog_color: [f32; 4],
    /// (fog_start, fog_end, ambient, sun_intensity)
    pub params: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0; 4],
            sun_dir: [0.4, 0.8, 0.45, 0.0],
            sky_color: [0.46, 0.64, 0.92, 1.0],
            fog_color: [0.46, 0.64, 0.92, 1.0],
            params: [280.0, 370.0, 0.34, 1.0],
        }
    }

    pub fn update(&mut self, camera: &Camera, aspect: f32) {
        self.view_proj = camera.view_proj(aspect).to_cols_array_2d();
        self.cam_pos = [camera.position.x, camera.position.y, camera.position.z, 1.0];
    }

    pub fn set_environment(&mut self, env: &crate::environment::Environment, fog_start: f32, fog_end: f32) {
        let s = env.sun_dir();
        self.sun_dir = [s.x, s.y, s.z, 0.0];
        let sky = env.sky_color();
        self.sky_color = [sky[0], sky[1], sky[2], 1.0];
        let fog = env.fog_color();
        self.fog_color = [fog[0], fog[1], fog[2], 1.0];
        self.params = [fog_start, fog_end, env.ambient(), env.sun_intensity()];
    }
}
