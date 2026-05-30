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
mod entity;
mod environment;
mod frustum;
mod game;
mod gpu;
mod mesher;
mod overlay;
mod persistence;
mod player;
mod raycast;
mod renderer;
mod voxel_volume;
mod worker;
mod world;
mod worldgen;

use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Vec3};
use rustc_hash::FxHashMap;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use camera::{Camera, CameraUniform};
use environment::Environment;
use frustum::Frustum;
use game::Game;
use gpu::Gpu;
use overlay::Hotbar;
use persistence::Level;
use player::{Input, Player};
use renderer::ChunkRenderer;

const SEED: u64 = 0x5EED_C0FFEE;
const RENDER_DISTANCE: i32 = 12;
const REACH: f32 = 6.0;
const SENSITIVITY: f32 = 0.0022;
const FOG_END: f32 = (RENDER_DISTANCE as f32 - 1.0) * 32.0;
const FOG_START: f32 = FOG_END * 0.68;

struct State {
    gpu: Gpu,
    renderer: ChunkRenderer,
    game: Game,
    camera: Camera,
    player: Player,
    input: Input,
    hotbar: Hotbar,
    environment: Environment,
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

    fn save_world(&self) {
        if let Some(state) = &self.state {
            let level = Level {
                seed: state.game.seed(),
                spawn: state.player.position.to_array(),
                yaw: state.camera.yaw,
                pitch: state.camera.pitch,
                time: state.environment.time,
                flying: state.player.flying,
            };
            state.game.save(&level);
        }
    }

    fn handle_key(&mut self, code: KeyCode, pressed: bool, repeat: bool, event_loop: &ActiveEventLoop) {
        if code == KeyCode::Escape && pressed {
            event_loop.exit();
            return;
        }
        if code == KeyCode::KeyP && pressed && !repeat {
            self.save_world();
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
            KeyCode::KeyR if pressed && !repeat => {
                let mode = state.game.cycle_rtx();
                log::info!("RTX lighting: {mode}");
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
        state.environment.update(dt);

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

        // Survival: a hard fall or starvation can kill; respawn at spawn.
        if !state.player.flying && state.player.is_dead() {
            log::info!("You died — respawning at spawn.");
            state.player.respawn();
            state.camera.position = state.player.eye();
        }

        // Stream chunks around the player and advance flowing fluids.
        state
            .game
            .update(&state.gpu, &state.renderer, state.player.position, dt);

        // Block targeting.
        let eye = state.camera.position;
        let fwd = state.camera.forward();
        let target = raycast::cast(eye, fwd, REACH, |p| state.game.is_solid_at(p));

        // Break / place (edge-triggered).
        if state.input.break_pressed {
            state.input.break_pressed = false;
            if let Some(hit) = &target {
                let broken = state.game.block_at(hit.block);
                if state
                    .game
                    .set_block(&state.gpu, &state.renderer, hit.block, block::AIR)
                    && block::is_solid(broken)
                {
                    // Drop the broken block as a collectible item.
                    let center = hit.block.as_vec3() + Vec3::splat(0.5);
                    state.game.spawn_item(center, broken);
                }
            }
        }
        if state.input.place_pressed {
            state.input.place_pressed = false;
            if let Some(hit) = &target {
                let place = hit.block + hit.normal;
                let id = state.hotbar.selected_block();
                let blocks_player = block::is_solid(id) && state.player.intersects_block(place);
                if !blocks_player
                    && state.game.set_block(&state.gpu, &state.renderer, place, id)
                    && block::is_fluid(id)
                {
                    // Placed water/lava becomes a flowing source.
                    state.game.add_fluid_source(place, id);
                }
            }
        }

        // Render.
        let aspect = state.gpu.aspect();
        state.camera_uniform.update(&state.camera, aspect);
        state
            .camera_uniform
            .set_environment(&state.environment, FOG_START, FOG_END);
        state.renderer.update_camera(&state.gpu, &state.camera_uniform);
        state.renderer.set_sky(state.environment.wgpu_clear());

        let frustum = Frustum::from_view_proj(state.camera.view_proj(aspect));
        let entity_mesh = state.game.build_entity_mesh(&state.gpu, &state.renderer);
        let mut visible = state.game.visible_meshes(&frustum);
        if let Some(em) = &entity_mesh {
            visible.push(em);
        }
        let highlight = target.as_ref().map(|h| h.block);
        let ui = overlay::build_ui(
            state.gpu.config.width,
            state.gpu.config.height,
            &state.hotbar,
            state.player.health,
            state.player.hunger,
            !state.player.flying,
        );
        let volume_bg = state.game.volume_bind_group();
        state
            .renderer
            .render_frame(&state.gpu, &visible, volume_bg, highlight, &ui);

        // Stats.
        state.fps_accum += dt;
        state.fps_frames += 1;
        if state.fps_accum >= 2.0 {
            log::info!(
                "{:.0} fps | {} chunks | {} meshes | {} drawn | {} entities | {}",
                state.fps_frames as f32 / state.fps_accum,
                state.game.loaded_chunk_count(),
                state.game.mesh_count(),
                visible.len(),
                state.game.entity_count(),
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

        // Headless screenshot path: a fresh generated world with a few verification edits.
        if let Ok(path) = std::env::var("VOXELCRAFT_SHOT") {
            let mut game = Game::new(
                &gpu,
                renderer.volume_bgl(),
                SEED,
                RENDER_DISTANCE,
                FxHashMap::default(),
            );
            // Offscreen render is one-shot, so trace many more GI rays for a clean image.
            game.set_rtx_quality(64);
            // A sloped sandy pool to verify depth-driven water clarity: shallow at the near (-X)
            // shore, deepening away, filled to a flat surface (top at y=150). The sandy bottom shows
            // through the shallows and fades to opaque blue-green with depth — driven by depth, not
            // view angle. A colored strip atop the far wall is reflected in the deep end. The camera
            // looks down the slope so the whole shallow->deep gradient is in frame.
            let environment = Environment::new(0.30);
            let player = Player::new(Vec3::new(4.0, 152.38, 20.0), false);
            let camera = Camera::new(player.eye(), 0.0, -0.32);
            let mut camera_uniform = CameraUniform::new();

            game.load_all_blocking(&gpu, &renderer, player.position);

            // Sand bottom that drops ~1 block every 3 along +X (water depth = 149 - floor_y, so it
            // ramps from 1 at the near shore to ~9 at the far wall), with the water column above it.
            for x in 8..=32 {
                let floor_y = 148 - (x - 8) / 3;
                for z in 10..=30 {
                    game.set_block(&gpu, &renderer, IVec3::new(x, floor_y, z), block::SAND);
                    for y in (floor_y + 1)..=149 {
                        game.set_block(&gpu, &renderer, IVec3::new(x, y, z), block::WATER);
                    }
                }
            }
            // Stone side + far walls contain the pool (tops at the waterline, except the far wall).
            for z in 9..=31 {
                for y in 138..=150 {
                    game.set_block(&gpu, &renderer, IVec3::new(33, y, z), block::STONE);
                }
            }
            for x in 8..=33 {
                for y in 138..=150 {
                    game.set_block(&gpu, &renderer, IVec3::new(x, y, 9), block::STONE);
                    game.set_block(&gpu, &renderer, IVec3::new(x, y, 31), block::STONE);
                }
            }
            // A colored strip on the far wall above the water, reflected by the deep end.
            for z in 10..=30 {
                let id = match z % 3 {
                    0 => block::LEAVES,
                    1 => block::SNOW,
                    _ => block::SAND,
                };
                for y in 151..=156 {
                    game.set_block(&gpu, &renderer, IVec3::new(33, y, z), id);
                }
            }
            let highlight: Option<IVec3> = None;

            // Populate the voxel volume so the depth march + reflections trace against the pool.
            game.prime_volume(&gpu, player.position);

            camera_uniform.update(&camera, gpu.aspect());
            camera_uniform.set_environment(&environment, FOG_START, FOG_END);
            renderer.set_sky(environment.wgpu_clear());
            renderer.update_camera(&gpu, &camera_uniform);

            let frustum = Frustum::from_view_proj(camera.view_proj(gpu.aspect()));
            let entity_mesh = game.build_entity_mesh(&gpu, &renderer);
            let mut visible = game.visible_meshes(&frustum);
            if let Some(em) = &entity_mesh {
                visible.push(em);
            }
            let volume_bg = game.volume_bind_group();
            let ui = overlay::build_ui(
                gpu.config.width,
                gpu.config.height,
                &Hotbar::new(),
                20.0,
                20.0,
                false,
            );
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
                volume_bg,
                highlight,
                &ui,
                gpu.config.width,
                gpu.config.height,
                &path,
            );
            event_loop.exit();
            return;
        }

        // Interactive: load a saved world if one exists, else start a new one.
        let dir = persistence::save_dir();
        let level = persistence::load_level(&dir);
        let (seed, spawn, yaw, pitch, time, flying) = match &level {
            Some(l) => (l.seed, Vec3::from(l.spawn), l.yaw, l.pitch, l.time, l.flying),
            None => (
                SEED,
                Vec3::new(8.0, 96.0, 24.0),
                -std::f32::consts::FRAC_PI_2,
                -0.30,
                0.34,
                true,
            ),
        };
        let saved = persistence::load_chunks(&dir);
        let mut game = Game::new(&gpu, renderer.volume_bgl(), seed, RENDER_DISTANCE, saved);
        // A few mobs near spawn; they fall onto terrain as it streams in.
        for (dx, dz) in [(-3, -5), (3, -6), (6, 2), (-5, 3), (1, 7), (7, -2)] {
            game.spawn_mob(Vec3::new(spawn.x + dx as f32, spawn.y, spawn.z + dz as f32));
        }
        let environment = Environment::new(time);
        let player = Player::new(spawn, flying);
        let camera = Camera::new(player.eye(), yaw, pitch);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(&camera, gpu.aspect());
        camera_uniform.set_environment(&environment, FOG_START, FOG_END);
        renderer.update_camera(&gpu, &camera_uniform);
        renderer.set_sky(environment.wgpu_clear());

        self.state = Some(State {
            gpu,
            renderer,
            game,
            camera,
            player,
            input: Input::default(),
            hotbar: Hotbar::new(),
            environment,
            camera_uniform,
            last_frame: Instant::now(),
            fps_accum: 0.0,
            fps_frames: 0,
        });
        self.set_grab(true);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.save_world();
                event_loop.exit();
            }
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
