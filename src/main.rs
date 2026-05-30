//! Voxelcraft — M1: a window with a fly camera over a single 32^3 chunk.
//!
//! Controls: WASD move, mouse look, Space up, Left-Shift down, Left-Ctrl boost, Esc quit.
//! Set the env var VOXELCRAFT_SHOT=path.png to render one frame offscreen to a PNG and exit
//! (used for headless verification).

mod block;
mod camera;
mod capture;
mod gpu;
mod mesher;
mod renderer;
mod world;

use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use camera::{Camera, CameraUniform, FlyController};
use gpu::Gpu;
use renderer::{ChunkRenderer, GpuMesh};
use world::Chunk;

struct State {
    gpu: Gpu,
    renderer: ChunkRenderer,
    mesh: GpuMesh,
    camera: Camera,
    controller: FlyController,
    camera_uniform: CameraUniform,
    last_frame: Instant,
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
    grabbed: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: None,
            grabbed: false,
        }
    }

    fn set_grab(&mut self, grab: bool) {
        let Some(window) = &self.window else { return };
        if grab {
            let ok = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
            window.set_cursor_visible(false);
            self.grabbed = ok;
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
            self.grabbed = false;
        }
    }

    fn handle_key(&mut self, code: KeyCode, pressed: bool, event_loop: &ActiveEventLoop) {
        if code == KeyCode::Escape && pressed {
            event_loop.exit();
            return;
        }
        let Some(state) = &mut self.state else { return };
        let c = &mut state.controller;
        match code {
            KeyCode::KeyW => c.forward = pressed,
            KeyCode::KeyS => c.back = pressed,
            KeyCode::KeyA => c.left = pressed,
            KeyCode::KeyD => c.right = pressed,
            KeyCode::Space => c.up = pressed,
            KeyCode::ShiftLeft => c.down = pressed,
            KeyCode::ControlLeft => c.fast = pressed,
            _ => {}
        }
    }

    fn frame(&mut self) {
        let Some(state) = &mut self.state else { return };
        let now = Instant::now();
        let dt = (now - state.last_frame).as_secs_f32().min(0.1);
        state.last_frame = now;

        state.controller.update(&mut state.camera, dt);
        let aspect = state.gpu.aspect();
        state.camera_uniform.update(&state.camera, aspect);
        state.renderer.update_camera(&state.gpu, &state.camera_uniform);
        state.renderer.render(&state.gpu, &state.mesh);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Voxelcraft — M1")
            .with_inner_size(LogicalSize::new(1600.0, 900.0));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        let gpu = pollster::block_on(Gpu::new(window.clone()));
        let renderer = ChunkRenderer::new(&gpu);

        let chunk = Chunk::generate_demo();
        let mesh_data = mesher::build_mesh(&chunk);
        log::info!(
            "Chunk mesh: {} vertices, {} indices",
            mesh_data.vertices.len(),
            mesh_data.indices.len()
        );
        let mesh = renderer.upload_mesh(&gpu, &mesh_data);

        // Look down at the hill from the +Z side.
        let camera = Camera::new(
            Vec3::new(16.0, 40.0, 80.0),
            -std::f32::consts::FRAC_PI_2,
            -0.38,
        );
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(&camera, gpu.aspect());
        renderer.update_camera(&gpu, &camera_uniform);

        // Headless screenshot mode: render one frame to a PNG and exit.
        if let Ok(path) = std::env::var("VOXELCRAFT_SHOT") {
            capture::screenshot(&gpu, &renderer, &mesh, gpu.config.width, gpu.config.height, &path);
            event_loop.exit();
            return;
        }

        self.state = Some(State {
            gpu,
            renderer,
            mesh,
            camera,
            controller: FlyController::default(),
            camera_uniform,
            last_frame: Instant::now(),
        });
        self.set_grab(true);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.gpu.resize(size);
                }
            }
            WindowEvent::Focused(false) => self.set_grab(false),
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                if !self.grabbed {
                    self.set_grab(true);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.handle_key(code, pressed, event_loop);
                }
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.grabbed {
                if let Some(state) = &mut self.state {
                    state
                        .controller
                        .process_mouse(delta.0 as f32, delta.1 as f32);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,voxelcraft=info"),
    )
    .init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
