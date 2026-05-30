//! Voxelcraft — M4: an interactive voxel sandbox.
//!
//! Controls:
//!   WASD move · mouse look · Space up/jump · Left-Shift down · Left-Ctrl sprint/boost
//!   F toggle fly/walk · Left-click break · Right-click place · 1-9 / scroll select block · Esc quit
//!
//! Set VOXELCRAFT_SHOT=path.png to render one frame offscreen (headless verification).

mod block;
mod camera;
mod capture;
mod frustum;
mod game;
mod gpu;
mod mesher;
mod overlay;
mod player;
mod raycast;
mod renderer;
mod worker;
mod world;
mod worldgen;

use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Vec3};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use camera::{Camera, CameraUniform};
use frustum::Frustum;
use game::Game;
use gpu::Gpu;
use overlay::Hotbar;
use player::{Input, Player};
use renderer::ChunkRenderer;

const SEED: u64 = 0x5EED_C0FFEE;
const RENDER_DISTANCE: i32 = 12;
const REACH: f32 = 6.0;
const SENSITIVITY: f32 = 0.0022;

struct State {
    gpu: Gpu,
    renderer: ChunkRenderer,
    game: Game,
    camera: Camera,
    player: Player,
    input: Input,
    hotbar: Hotbar,
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

    fn handle_key(&mut self, code: KeyCode, pressed: bool, repeat: bool, event_loop: &ActiveEventLoop) {
        if code == KeyCode::Escape && pressed {
            event_loop.exit();
            return;
        }
        let Some(state) = &mut self.state else { return };
        match code {
            KeyCode::KeyW => state.input.forward = pressed,
            KeyCode::KeyS => state.input.back = pressed,
            KeyCode::KeyA => state.input.left = pressed,
            KeyCode::KeyD => state.input.right = pressed,
            KeyCode::Space => state.input.up = pressed,
            KeyCode::ShiftLeft => state.input.down = pressed,
            KeyCode::ControlLeft => state.input.sprint = pressed,
            KeyCode::KeyF if pressed && !repeat => {
                state.player.flying = !state.player.flying;
                state.player.velocity = Vec3::ZERO;
            }
            KeyCode::Digit1 => maybe_select(state, pressed, 0),
            KeyCode::Digit2 => maybe_select(state, pressed, 1),
            KeyCode::Digit3 => maybe_select(state, pressed, 2),
            KeyCode::Digit4 => maybe_select(state, pressed, 3),
            KeyCode::Digit5 => maybe_select(state, pressed, 4),
            KeyCode::Digit6 => maybe_select(state, pressed, 5),
            KeyCode::Digit7 => maybe_select(state, pressed, 6),
            KeyCode::Digit8 => maybe_select(state, pressed, 7),
            KeyCode::Digit9 => maybe_select(state, pressed, 8),
            _ => {}
        }
    }

    fn frame(&mut self) {
        let Some(state) = &mut self.state else { return };
        let now = Instant::now();
        let dt = (now - state.last_frame).as_secs_f32().min(0.1);
        state.last_frame = now;

        // Apply mouse look.
        let (yaw_d, pitch_d) = state.input.take_look();
        state.camera.yaw += yaw_d * SENSITIVITY;
        state.camera.pitch = (state.camera.pitch - pitch_d * SENSITIVITY).clamp(-1.5533, 1.5533);

        // Player physics (disjoint field borrows: player mut, game/input shared).
        let yaw = state.camera.yaw;
        let game_ref = &state.game;
        state
            .player
            .update(dt, yaw, &state.input, |p| game_ref.is_solid_at(p));
        state.camera.position = state.player.eye();

        // Stream chunks around the player.
        state
            .game
            .update(&state.gpu, &state.renderer, state.player.position);

        // Block targeting.
        let eye = state.camera.position;
        let fwd = state.camera.forward();
        let target = raycast::cast(eye, fwd, REACH, |p| state.game.is_solid_at(p));

        // Break / place (edge-triggered).
        if state.input.break_pressed {
            state.input.break_pressed = false;
            if let Some(hit) = &target {
                state
                    .game
                    .set_block(&state.gpu, &state.renderer, hit.block, block::AIR);
            }
        }
        if state.input.place_pressed {
            state.input.place_pressed = false;
            if let Some(hit) = &target {
                let place = hit.block + hit.normal;
                let id = state.hotbar.selected_block();
                let blocks_player = block::is_solid(id) && state.player.intersects_block(place);
                if !blocks_player {
                    state.game.set_block(&state.gpu, &state.renderer, place, id);
                }
            }
        }

        // Render.
        let aspect = state.gpu.aspect();
        state.camera_uniform.update(&state.camera, aspect);
        state.renderer.update_camera(&state.gpu, &state.camera_uniform);

        let frustum = Frustum::from_view_proj(state.camera.view_proj(aspect));
        let visible = state.game.visible_meshes(&frustum);
        let highlight = target.as_ref().map(|h| h.block);
        let ui = overlay::build_ui(state.gpu.config.width, state.gpu.config.height, &state.hotbar);
        state
            .renderer
            .render_frame(&state.gpu, &visible, highlight, &ui);

        // Stats.
        state.fps_accum += dt;
        state.fps_frames += 1;
        if state.fps_accum >= 2.0 {
            log::info!(
                "{:.0} fps | {} chunks | {} meshes | {} drawn | {}",
                state.fps_frames as f32 / state.fps_accum,
                state.game.loaded_chunk_count(),
                state.game.mesh_count(),
                visible.len(),
                if state.player.flying { "fly" } else { "walk" }
            );
            state.fps_accum = 0.0;
            state.fps_frames = 0;
        }
    }
}

fn maybe_select(state: &mut State, pressed: bool, index: usize) {
    if pressed {
        state.hotbar.select(index);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Voxelcraft — M4")
            .with_inner_size(LogicalSize::new(1600.0, 900.0));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        let gpu = pollster::block_on(Gpu::new(window.clone()));
        let renderer = ChunkRenderer::new(&gpu);
        let mut game = Game::new(SEED, RENDER_DISTANCE);

        // Spawn flying above the terrain.
        let player = Player::new(Vec3::new(8.0, 96.0, 24.0), true);
        let mut camera = Camera::new(player.eye(), -std::f32::consts::FRAC_PI_2, -0.30);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(&camera, gpu.aspect());
        renderer.update_camera(&gpu, &camera_uniform);

        if let Ok(path) = std::env::var("VOXELCRAFT_SHOT") {
            game.load_all_blocking(&gpu, &renderer, player.position);

            // Verify set_block + synchronous remesh: build a visible tower + platform and
            // carve a small crater, then highlight a block.
            for y in 80..101 {
                let id = if y % 2 == 0 { block::WOOD } else { block::STONE };
                game.set_block(&gpu, &renderer, IVec3::new(4, y, 8), id);
            }
            for dx in -1..=1 {
                for dz in -1..=1 {
                    game.set_block(&gpu, &renderer, IVec3::new(4 + dx, 101, 8 + dz), block::SNOW);
                }
            }
            let highlight = Some(IVec3::new(4, 101, 8));

            let frustum = Frustum::from_view_proj(camera.view_proj(gpu.aspect()));
            let visible = game.visible_meshes(&frustum);
            let ui = overlay::build_ui(gpu.config.width, gpu.config.height, &Hotbar::new());
            log::info!(
                "Headless: {} chunks, {} meshes, {} visible",
                game.loaded_chunk_count(),
                game.mesh_count(),
                visible.len()
            );
            capture::screenshot(
                &gpu,
                &renderer,
                &visible,
                highlight,
                &ui,
                gpu.config.width,
                gpu.config.height,
                &path,
            );
            event_loop.exit();
            return;
        }

        camera.position = player.eye();
        self.state = Some(State {
            gpu,
            renderer,
            game,
            camera,
            player,
            input: Input::default(),
            hotbar: Hotbar::new(),
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
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                if btn_state == ElementState::Pressed {
                    if !self.grabbed {
                        self.set_grab(true);
                    } else if let Some(state) = &mut self.state {
                        match button {
                            MouseButton::Left => state.input.break_pressed = true,
                            MouseButton::Right => state.input.place_pressed = true,
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(state) = &mut self.state {
                    let dir = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -y.signum() as i32,
                        MouseScrollDelta::PixelDelta(p) => -(p.y.signum() as i32),
                    };
                    if dir != 0 {
                        state.hotbar.scroll(dir);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.handle_key(code, pressed, event.repeat, event_loop);
                }
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.grabbed {
                if let Some(state) = &mut self.state {
                    state.input.yaw_delta += delta.0 as f32;
                    state.input.pitch_delta += delta.1 as f32;
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
