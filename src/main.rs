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
mod font;
mod frustum;
mod game;
mod gpu;
mod item;
mod light;
mod mesher;
mod overlay;
mod persistence;
mod player;
mod raycast;
mod renderer;
mod texture;
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
use item::Inventory;
use persistence::Level;
use player::{Input, Player};
use renderer::ChunkRenderer;

const SEED: u64 = 0x5EED_C0FFEE;
const RENDER_DISTANCE: i32 = 12;
const REACH: f32 = 6.0;
const SENSITIVITY: f32 = 0.0022;
const FOG_END: f32 = (RENDER_DISTANCE as f32 - 1.0) * 32.0;
const FOG_START: f32 = FOG_END * 0.68;

/// Active GUI screen. Gameplay input is suppressed while any screen is open (M15b; grows later).
#[derive(PartialEq, Clone, Copy)]
enum Screen {
    None,
    Inventory,
}

struct State {
    gpu: Gpu,
    renderer: ChunkRenderer,
    game: Game,
    camera: Camera,
    player: Player,
    input: Input,
    inventory: Inventory,
    environment: Environment,
    camera_uniform: CameraUniform,
    last_frame: Instant,
    fps_accum: f32,
    fps_frames: u32,
    fps_smooth: f32,
    debug_f3: bool,
    elapsed: f32,
    screen: Screen,
    cursor: (f32, f32),
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
            let dir = persistence::save_dir();
            if let Err(e) = persistence::save_inventory(&dir, &state.inventory) {
                log::error!("failed to save inventory: {e}");
            }
        }
    }

    fn toggle_inventory(&mut self) {
        let open = {
            let Some(state) = &mut self.state else { return };
            if state.screen == Screen::Inventory {
                drop_leftover(&mut state.inventory, &mut state.game, state.player.position);
                state.screen = Screen::None;
                false
            } else {
                state.screen = Screen::Inventory;
                state.input = Input::default();
                state.cursor = (
                    state.gpu.config.width as f32 * 0.5,
                    state.gpu.config.height as f32 * 0.5,
                );
                true
            }
        };
        self.set_grab(!open);
    }

    fn close_screen(&mut self) {
        if let Some(state) = &mut self.state {
            if state.screen != Screen::None {
                drop_leftover(&mut state.inventory, &mut state.game, state.player.position);
                state.screen = Screen::None;
            }
        }
        self.set_grab(true);
    }

    fn inventory_click(&mut self, button: MouseButton) {
        let Some(state) = &mut self.state else { return };
        let right = match button {
            MouseButton::Left => false,
            MouseButton::Right => true,
            _ => return,
        };
        let (w, h) = (state.gpu.config.width, state.gpu.config.height);
        let (cx, cy) = state.cursor;
        for (slot_i, x, y) in overlay::inventory_slot_rects(w, h) {
            if cx >= x && cx < x + overlay::INV_SLOT && cy >= y && cy < y + overlay::INV_SLOT {
                state.inventory.click_slot(slot_i, right);
                break;
            }
        }
    }

    fn handle_key(&mut self, code: KeyCode, pressed: bool, repeat: bool, event_loop: &ActiveEventLoop) {
        if code == KeyCode::Escape && pressed {
            if self.state.as_ref().is_some_and(|s| s.screen != Screen::None) {
                self.close_screen();
                return;
            }
            event_loop.exit();
            return;
        }
        if code == KeyCode::KeyP && pressed && !repeat {
            self.save_world();
            return;
        }
        if code == KeyCode::KeyE && pressed && !repeat {
            self.toggle_inventory();
            return;
        }
        // While a GUI screen is open, swallow gameplay key PRESSES but let RELEASES through, so a
        // movement key held across the open/close edge can't get stuck on.
        if pressed && self.state.as_ref().is_some_and(|s| s.screen != Screen::None) {
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
            KeyCode::F3 if pressed && !repeat => {
                state.debug_f3 = !state.debug_f3;
            }
            KeyCode::KeyQ if pressed && !repeat => state.input.drop_pressed = true,
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
        state.elapsed += dt;

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
            // Drop the whole inventory at the death site, then respawn empty.
            let death_pos = state.player.position + Vec3::new(0.0, 1.0, 0.0);
            for stack in state.inventory.drain_all() {
                if let Some(b) = stack.block() {
                    for _ in 0..stack.count {
                        state.game.spawn_item(death_pos, b);
                    }
                }
            }
            state.player.respawn();
            state.camera.position = state.player.eye();
        }

        // Stream chunks, advance fluids/entities; collect any picked-up item drops into inventory.
        let collected =
            state
                .game
                .update(&state.gpu, &state.renderer, state.player.position, dt);
        for b in collected {
            if !state.inventory.add_block(b) {
                // Inventory full — drop it back so blocks aren't vacuum-deleted.
                state
                    .game
                    .spawn_item(state.player.position + Vec3::new(0.0, 1.0, 0.0), b);
            }
        }

        // Block targeting.
        let eye = state.camera.position;
        let fwd = state.camera.forward();
        let target = raycast::cast(eye, fwd, REACH, |p| state.game.is_solid_at(p));

        // Break / place (edge-triggered).
        if state.input.break_pressed {
            state.input.break_pressed = false;
            if let Some(hit) = &target {
                let broken = state.game.block_at(hit.block);
                if broken != block::AIR
                    && state
                        .game
                        .set_block(&state.gpu, &state.renderer, hit.block, block::AIR)
                    && !state.inventory.creative
                {
                    // Drop the block's item (stone→cobblestone, grass→dirt, leaves→nothing, …).
                    if let Some(drop) = block::drops(broken) {
                        let center = hit.block.as_vec3() + Vec3::splat(0.5);
                        state.game.spawn_item(center, drop);
                    }
                }
            }
        }
        if state.input.place_pressed {
            state.input.place_pressed = false;
            if let Some(hit) = &target {
                let place = hit.block + hit.normal;
                let id = state.inventory.selected_block();
                let blocks_player = block::is_solid(id) && state.player.intersects_block(place);
                if id != block::AIR
                    && !blocks_player
                    && state.game.set_block(&state.gpu, &state.renderer, place, id)
                {
                    state.inventory.consume_selected();
                    if block::is_fluid(id) {
                        // Placed water/lava becomes a flowing source.
                        state.game.add_fluid_source(place, id);
                    }
                }
            }
        }

        // Q: drop one of the selected item ahead of the camera.
        if state.input.drop_pressed {
            state.input.drop_pressed = false;
            if let Some(b) = state.inventory.drop_one_selected() {
                let pos = state.camera.position + state.camera.forward() * 1.2;
                state.game.spawn_item(pos, b);
            }
        }

        // Render.
        let aspect = state.gpu.aspect();
        state.camera_uniform.update(&state.camera, aspect);
        state
            .camera_uniform
            .set_environment(&state.environment, FOG_START, FOG_END);
        state.camera_uniform.set_time(state.elapsed, state.environment.time);
        state.renderer.update_camera(&state.gpu, &state.camera_uniform);
        state.renderer.set_sky(state.environment.wgpu_clear());

        let frustum = Frustum::from_view_proj(state.camera.view_proj(aspect));
        let entity_mesh = state.game.build_entity_mesh(&state.gpu, &state.renderer);
        let mut visible = state.game.visible_meshes(&frustum);
        if let Some(em) = &entity_mesh {
            visible.push(em);
        }
        let highlight = target.as_ref().map(|h| h.block);
        let inst_fps = 1.0 / dt.max(1e-4);
        state.fps_smooth = if state.fps_smooth <= 0.0 {
            inst_fps
        } else {
            state.fps_smooth * 0.92 + inst_fps * 0.08
        };
        let debug_lines = if state.debug_f3 {
            let p = state.player.position;
            Some(build_debug_lines(
                state.fps_smooth,
                p,
                state.camera.forward(),
                state.camera.yaw,
                state.camera.pitch,
                state.game.biome_name_at(p.x.floor() as i32, p.z.floor() as i32),
                state.game.loaded_chunk_count(),
                state.game.mesh_count(),
                state.game.entity_count(),
                state.player.flying,
                state.game.rtx_mode_name(),
            ))
        } else {
            None
        };
        let mut ui = overlay::build_ui(
            state.gpu.config.width,
            state.gpu.config.height,
            &state.inventory,
            state.player.health,
            state.player.hunger,
            !state.player.flying,
            debug_lines.as_deref(),
        );
        if state.screen == Screen::Inventory {
            ui.extend(overlay::build_inventory_screen(
                state.gpu.config.width,
                state.gpu.config.height,
                &state.inventory,
                state.cursor,
            ));
        }
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
        state.inventory.select(index);
    }
}

/// On screen close, drop any held cursor stack that no longer fits back into the world as items,
/// rather than leaving it stranded on the (now hidden) cursor.
fn drop_leftover(inventory: &mut Inventory, game: &mut Game, player_pos: Vec3) {
    if let Some(left) = inventory.return_held() {
        if let Some(b) = left.block() {
            let pos = player_pos + Vec3::new(0.0, 1.0, 0.0);
            for _ in 0..left.count {
                game.spawn_item(pos, b);
            }
        }
    }
}

/// Assemble the F3 debug overlay text lines.
#[allow(clippy::too_many_arguments)]
fn build_debug_lines(
    fps: f32,
    pos: Vec3,
    forward: Vec3,
    yaw: f32,
    pitch: f32,
    biome: &str,
    chunks: usize,
    meshes: usize,
    entities: usize,
    flying: bool,
    rtx: &str,
) -> Vec<String> {
    let (cx, cy, cz) = (
        (pos.x.floor() as i32).div_euclid(32),
        (pos.y.floor() as i32).div_euclid(32),
        (pos.z.floor() as i32).div_euclid(32),
    );
    let dir = if forward.x.abs() > forward.z.abs() {
        if forward.x > 0.0 { "East +X" } else { "West -X" }
    } else if forward.z > 0.0 {
        "South +Z"
    } else {
        "North -Z"
    };
    vec![
        format!("Voxelcraft  {fps:.0} fps"),
        format!("XYZ {:.1} {:.1} {:.1}", pos.x, pos.y, pos.z),
        format!("Chunk {cx} {cy} {cz}"),
        format!("Facing {dir}  yaw {:.0} pitch {:.0}", yaw.to_degrees(), pitch.to_degrees()),
        format!("Biome {biome}"),
        format!("Chunks {chunks}  Meshes {meshes}  Entities {entities}"),
        format!("Mode {}  RTX {rtx}", if flying { "fly" } else { "walk" }),
    ]
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
            // Debug knob: VOXELCRAFT_WSMOOTH overrides the water depth-clarity smoothing radius
            // in blocks (0 = single-tap, the pre-fix stepped look) for ad-hoc tuning/verification.
            if let Ok(s) = std::env::var("VOXELCRAFT_WSMOOTH") {
                if let Ok(r) = s.trim().parse::<f32>() {
                    game.set_water_smooth(r);
                }
            }
            // A long, uniformly shallow pool that runs PAST the voxel volume's ~128-block edge, to
            // verify the volume-edge fade at a grazing angle (like the reported screenshot): near
            // water (inside the volume) shows the sandy bottom, and as the columns cross the volume
            // boundary — where the depth march has no floor data — clarity fades smoothly to opaque
            // instead of hard-cutting to a right-angle seam.
            // A natural lake vista (no built structures): with the volume now covering the render
            // distance, the water's depth-driven clarity is consistent all the way out — no
            // rectangular seam where the old 256-block volume used to end.
            let environment = Environment::new(0.30);
            // Default vista; VOXELCRAFT_CAM="x,y,z,yaw,pitch" overrides for ad-hoc view sweeps
            // (water-seam debugging). Player is placed at the same spot so chunks load around it.
            let (cam_xyz, cam_yaw, cam_pitch) = std::env::var("VOXELCRAFT_CAM")
                .ok()
                .and_then(|s| {
                    let v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                    if v.len() == 5 {
                        Some((Vec3::new(v[0], v[1], v[2]), v[3], v[4]))
                    } else {
                        None
                    }
                })
                .unwrap_or((Vec3::new(8.0, 96.0, 24.0), -std::f32::consts::FRAC_PI_2, -0.30));
            let player = Player::new(cam_xyz, false);
            let camera = Camera::new(player.eye(), cam_yaw, cam_pitch);
            let mut camera_uniform = CameraUniform::new();

            game.load_all_blocking(&gpu, &renderer, player.position);
            let highlight: Option<IVec3> = None;

            // Debug knob: VOXELCRAFT_PLACE="x,y,z,id;x,y,z,id" places blocks before the shot
            // (milestone verification — e.g. a glowstone to check emissive lighting).
            if let Ok(s) = std::env::var("VOXELCRAFT_PLACE") {
                for spec in s.split(';') {
                    let v: Vec<i32> = spec.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                    if v.len() == 4 {
                        game.set_block(&gpu, &renderer, IVec3::new(v[0], v[1], v[2]), v[3] as u16);
                    }
                }
            }

            // Debug: VOXELCRAFT_ROOM carves a sealed underground room lit by one glowstone — the
            // M14 "dark cave" verification (pair with VOXELCRAFT_CAM to look inside).
            if let Ok(room) = std::env::var("VOXELCRAFT_ROOM") {
                let (cx, cy, cz) = (8, 70, 24);
                for y in cy - 2..=cy + 2 {
                    for z in cz - 3..=cz + 3 {
                        for x in cx - 1..=cx + 6 {
                            game.set_block(&gpu, &renderer, IVec3::new(x, y, z), block::AIR);
                        }
                    }
                }
                // ROOM=2 leaves it unlit (sealed + sky=0 => pitch black); else light it.
                if room != "2" {
                    game.set_block(&gpu, &renderer, IVec3::new(cx + 5, cy, cz), block::GLOWSTONE);
                }
            }

            // Populate the voxel volume so shadows / GI / water depth trace across the full vista.
            game.prime_volume(&gpu, player.position);

            camera_uniform.update(&camera, gpu.aspect());
            camera_uniform.set_environment(&environment, FOG_START, FOG_END);
            let shot_time = std::env::var("VOXELCRAFT_TIME")
                .ok()
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0);
            camera_uniform.set_time(shot_time, environment.time);
            renderer.set_sky(environment.wgpu_clear());
            renderer.update_camera(&gpu, &camera_uniform);

            let frustum = Frustum::from_view_proj(camera.view_proj(gpu.aspect()));
            let entity_mesh = game.build_entity_mesh(&gpu, &renderer);
            let mut visible = game.visible_meshes(&frustum);
            if let Some(em) = &entity_mesh {
                visible.push(em);
            }
            let volume_bg = game.volume_bind_group();
            let dbg = build_debug_lines(
                60.0,
                player.position,
                camera.forward(),
                cam_yaw,
                cam_pitch,
                game.biome_name_at(player.position.x.floor() as i32, player.position.z.floor() as i32),
                game.loaded_chunk_count(),
                game.mesh_count(),
                game.entity_count(),
                false,
                game.rtx_mode_name(),
            );
            // A few stacks so the headless shot exercises count + inventory-screen rendering.
            let mut shot_inv = Inventory::new(true);
            shot_inv.slots[2] = Some(item::ItemStack::new(item::item_of_block(block::STONE), 32));
            shot_inv.slots[4] = Some(item::ItemStack::new(item::item_of_block(block::WOOD), 7));
            shot_inv.slots[9] = Some(item::ItemStack::new(item::item_of_block(block::COAL_ORE), 12));
            shot_inv.slots[11] = Some(item::ItemStack::new(item::item_of_block(block::DIRT), 64));
            shot_inv.slots[19] = Some(item::ItemStack::new(item::item_of_block(block::IRON_ORE), 5));
            let mut ui = overlay::build_ui(
                gpu.config.width,
                gpu.config.height,
                &shot_inv,
                20.0,
                20.0,
                false,
                Some(&dbg),
            );
            if std::env::var("VOXELCRAFT_SCREEN").map(|s| s == "inv").unwrap_or(false) {
                // Cursor hovering the first main slot (coal ore) to show its tooltip.
                let cursor = (600.0, 343.0);
                ui.extend(overlay::build_inventory_screen(
                    gpu.config.width,
                    gpu.config.height,
                    &shot_inv,
                    cursor,
                ));
            }
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
        let inventory = persistence::load_inventory(&dir, flying);
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
            inventory,
            environment,
            camera_uniform,
            last_frame: Instant::now(),
            fps_accum: 0.0,
            fps_frames: 0,
            fps_smooth: 0.0,
            debug_f3: false,
            elapsed: 0.0,
            screen: Screen::None,
            cursor: (0.0, 0.0),
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
                    let screen_open = self.state.as_ref().is_some_and(|s| s.screen != Screen::None);
                    if screen_open {
                        self.inventory_click(button);
                    } else if !self.grabbed {
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
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(state) = &mut self.state {
                    if state.screen != Screen::None {
                        state.cursor = (position.x as f32, position.y as f32);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(state) = &mut self.state {
                    let dir = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -y.signum() as i32,
                        MouseScrollDelta::PixelDelta(p) => -(p.y.signum() as i32),
                    };
                    if dir != 0 && state.screen == Screen::None {
                        state.inventory.scroll(dir);
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
