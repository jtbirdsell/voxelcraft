//! First-person fly camera + a uniform buffer payload, plus an input-driven controller.

use glam::{Mat4, Vec3, Vec4};

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
    /// The projection the world is rendered with — JITTERED under DLSS (subpixel Halton offset).
    pub view_proj: [[f32; 4]; 4],
    /// Previous frame's UN-jittered view_proj, for screen-space motion vectors. (M33-G3)
    pub prev_view_proj: [[f32; 4]; 4],
    /// Current frame's UN-jittered view_proj. Motion vectors are `nojitter_cur − nojitter_prev` so
    /// they carry only geometric motion — DLSS resolves the jitter separately, and jitter-polluted
    /// motion vectors otherwise make static geometry tremble at the jitter frequency. (M33-G8)
    pub view_proj_nojitter: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sky_color: [f32; 4],
    pub fog_color: [f32; 4],
    /// (fog_start, fog_end, ambient, sun_intensity)
    pub params: [f32; 4],
    /// (elapsed_seconds, day_fraction, _, _) — drives animated water/lava and later effects.
    pub time: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            prev_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            view_proj_nojitter: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0; 4],
            sun_dir: [0.4, 0.8, 0.45, 0.0],
            sky_color: [0.46, 0.64, 0.92, 1.0],
            fog_color: [0.46, 0.64, 0.92, 1.0],
            params: [280.0, 370.0, 0.34, 1.0],
            time: [0.0; 4],
        }
    }

    /// Update the animation clock: elapsed seconds + day fraction.
    pub fn set_time(&mut self, secs: f32, day: f32) {
        self.time = [secs, day, 0.0, 0.0];
    }

    pub fn update(&mut self, camera: &Camera, aspect: f32) {
        let fresh = camera.view_proj(aspect).to_cols_array_2d();
        // Roll the UN-jittered projections forward (prev <- last frame's nojitter); render proj
        // starts unjittered and is offset by `apply_jitter` after this. Motion uses only nojitter.
        self.prev_view_proj = self.view_proj_nojitter;
        self.view_proj_nojitter = fresh;
        self.view_proj = fresh;
        self.cam_pos = [camera.position.x, camera.position.y, camera.position.z, 1.0];
    }

    /// Offset the render projection by a subpixel jitter (in render-resolution pixels) for DLSS
    /// temporal reconstruction. Apply AFTER `update` (jitters `view_proj` only, not the nojitter
    /// copies). The same pixel jitter is handed to NGX.
    ///
    /// This is a CLIP-SPACE translation — `jittered = T · view_proj`, where T adds `j · clip.w` to
    /// clip.xy so the perspective divide yields a constant NDC offset `j`. (Adding `j` to a single
    /// `view_proj` element instead scales by world-space Z — a depth-dependent shear, not a subpixel
    /// shift — which makes the whole scene tremble as the jitter changes each frame.) (M33-G8)
    pub fn apply_jitter(&mut self, jitter_px: (f32, f32), render_w: u32, render_h: u32) {
        let jx = 2.0 * jitter_px.0 / render_w.max(1) as f32;
        let jy = 2.0 * jitter_px.1 / render_h.max(1) as f32;
        let t = Mat4::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::new(jx, jy, 0.0, 1.0));
        self.view_proj = (t * Mat4::from_cols_array_2d(&self.view_proj)).to_cols_array_2d();
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
