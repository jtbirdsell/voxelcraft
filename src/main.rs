//! Voxelcraft — M2: an infinite, procedurally generated world streamed as chunks around a
//! fly camera, greedy-meshed with cross-chunk face culling and frustum-culled for drawing.
//!
//! Controls: WASD move, mouse look, Space up, Left-Shift down, Left-Ctrl boost, Esc quit.
//! Set VOXELCRAFT_SHOT=path.png to render one frame offscreen (headless verification).

mod block;
mod camera;
mod capture;
mod frustum;
mod game;
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
use frustum::Frustum;
use game::Game;
use gpu::Gpu;
use renderer::ChunkRenderer;

const SEED: u64 = 0x5EED_C0FFEE;
const RENDER_DISTANCE: i32 = 12;

struct State {
    gpu: Gpu,
    renderer: ChunkRenderer,
    game: Game,
    camera: Camera,
    controller: FlyController,
    camera_uniform: CameraUniform,
    last_frame: Instant,
    fps_accum: f32,
    fps_frames: u32,
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

        // Stream chunks around the camera.
        state
            .game
            .update(&state.gpu, &state.renderer, state.camera.position);

        // Cull + draw.
        let frustum = Frustum::from_view_proj(state.camera.view_proj(aspect));
        let visible = state.game.visible_meshes(&frustum);
        state.renderer.render_world(&state.gpu, &visible);

        // Periodic stats.
        state.fps_accum += dt;
        state.fps_frames += 1;
        if state.fps_accum >= 2.0 {
            let fps = state.fps_frames as f32 / state.fps_accum;
            log::info!(
                "{:.0} fps | {} chunks loaded | {} meshes | {} drawn",
                fps,
                state.game.loaded_chunk_count(),
                state.game.mesh_count(),
                visible.len()
            );
            state.fps_accum = 0.0;
            state.fps_frames = 0;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Voxelcraft — M2")
            .with_inner_size(LogicalSize::new(1600.0, 900.0));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        let gpu = pollster::block_on(Gpu::new(window.clone()));
        let renderer = ChunkRenderer::new(&gpu);
        let mut game = Game::new(SEED, RENDER_DISTANCE);

        let camera = Camera::new(Vec3::new(8.0, 120.0, 24.0), -std::f32::consts::FRAC_PI_2, -0.35);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(&camera, gpu.aspect());
        renderer.update_camera(&gpu, &camera_uniform);

        // Headless screenshot: synchronously load the area, then render one frame to a PNG.
        if let Ok(path) = std::env::var("VOXELCRAFT_SHOT") {
            game.load_all_blocking(&gpu, &renderer, camera.position);
            let frustum = Frustum::from_view_proj(camera.view_proj(gpu.aspect()));
            let visible = game.visible_meshes(&frustum);
            log::info!(
                "Headless: {} chunks, {} meshes, {} visible",
                game.loaded_chunk_count(),
                game.mesh_count(),
                visible.len()
            );
            capture::screenshot(&gpu, &renderer, &visible, gpu.config.width, gpu.config.height, &path);
            event_loop.exit();
            return;
        }

        self.state = Some(State {
            gpu,
            renderer,
            game,
            camera,
            controller: FlyController::default(),
            camera_uniform,
            last_frame: Instant::now(),
            fps_accum: 0.0,
            fps_frames: 0,
        });
        self.set_grab(true);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
