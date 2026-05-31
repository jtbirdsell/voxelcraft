//! Application layer: the winit `App`/`State`, event routing, per-frame update + render, and the
//! headless screenshot path. `main.rs` is just the module tree + the event loop entry point.

use std::sync::Arc;
use std::time::Instant;

use glam::{IVec3, Vec3};
use rustc_hash::FxHashMap;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::camera::{Camera, CameraUniform};
use crate::environment::Environment;
use crate::frustum::Frustum;
use crate::game::Game;
use crate::gpu::Gpu;
use crate::item::Inventory;
use crate::persistence::Level;
use crate::player::{Input, Player};
use crate::renderer::ChunkRenderer;
use crate::{block, capture, crafting, item, overlay, persistence, raycast, smelting};

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
    Crafting,
    /// A furnace's GUI, tagged with the furnace block position (its contents live in `Game`).
    Furnace(IVec3),
}

impl Screen {
    /// Craft-grid size for this screen (2x2 in the inventory, 3x3 at a table).
    fn craft_size(self) -> usize {
        if self == Screen::Crafting {
            3
        } else {
            2
        }
    }
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
    /// Items placed in the craft grid (top-left of the 9-cell array; 2x2 inventory or 3x3 table).
    craft: [Option<item::ItemStack>; 9],
    /// A screen the frame asked to open (processed after the frame's state borrow ends).
    pending_open: Option<Screen>,
    /// Block currently being mined + accumulated progress (0..1) for hold-to-break.
    mine_target: Option<IVec3>,
    mine_progress: f32,
    /// Cooldown between melee swings (seconds).
    melee_cd: f32,
}

pub struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
    grabbed: bool,
}

impl App {
    pub fn new() -> Self {
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
                health: state.player.health,
                hunger: state.player.hunger,
                air: state.player.air,
                saturation: state.player.saturation(),
                xp: state.player.xp,
                level: state.player.level,
            };
            state.game.save(&level);
            let dir = persistence::save_dir();
            // Fold any cursor-held stack back into slots so it isn't lost when saving mid-screen
            // (P / window-close don't go through close_screen's return_held path).
            let mut inv = state.inventory.clone();
            if let Some(h) = inv.held.take() {
                let _ = inv.insert(h);
            }
            if let Err(e) = persistence::save_state(&dir, &inv, &state.game.furnaces_to_save()) {
                log::error!("failed to save state: {e}");
            }
        }
    }

    fn toggle_inventory(&mut self) {
        let open = {
            let Some(state) = &mut self.state else { return };
            if state.screen != Screen::None {
                return_craft_to_inventory(state);
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

    fn open_crafting(&mut self) {
        if let Some(state) = &mut self.state {
            state.screen = Screen::Crafting;
            state.input = Input::default();
            state.cursor = (
                state.gpu.config.width as f32 * 0.5,
                state.gpu.config.height as f32 * 0.5,
            );
        }
        self.set_grab(false);
    }

    fn open_furnace(&mut self, pos: IVec3) {
        if let Some(state) = &mut self.state {
            state.game.furnace_mut(pos); // ensure a state exists to render/tick
            state.screen = Screen::Furnace(pos);
            state.input = Input::default();
            state.cursor = (
                state.gpu.config.width as f32 * 0.5,
                state.gpu.config.height as f32 * 0.5,
            );
        }
        self.set_grab(false);
    }

    fn close_screen(&mut self) {
        if let Some(state) = &mut self.state {
            if state.screen != Screen::None {
                return_craft_to_inventory(state);
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
        let hit = |x: f32, y: f32| {
            cx >= x && cx < x + overlay::INV_SLOT && cy >= y && cy < y + overlay::INV_SLOT
        };
        for (slot_i, x, y) in overlay::inventory_slot_rects(w, h) {
            if hit(x, y) {
                state.inventory.click_slot(slot_i, right);
                return;
            }
        }
        // Furnace screen: move items between held and the input/fuel slots; output is take-only.
        if let Screen::Furnace(pos) = state.screen {
            let f = state.game.furnace_mut(pos);
            for (kind, x, y) in overlay::furnace_slot_rects(w, h) {
                if hit(x, y) {
                    let mut held = state.inventory.held;
                    match kind {
                        overlay::FurnaceSlot::Input => {
                            let mut s = f.input;
                            item::slot_click(&mut held, &mut s, right);
                            f.input = s;
                        }
                        overlay::FurnaceSlot::Fuel => {
                            let mut s = f.fuel;
                            item::slot_click(&mut held, &mut s, right);
                            f.fuel = s;
                        }
                        overlay::FurnaceSlot::Output => take_furnace_output(&mut held, &mut f.output),
                    }
                    state.inventory.held = held;
                    return;
                }
            }
            return;
        }
        // Equipped-armor slots: only the matching piece may be placed; taking a piece out is allowed.
        for (slot_i, x, y) in overlay::armor_slot_rects(w, h) {
            if hit(x, y) {
                let allowed = match state.inventory.held {
                    None => true,
                    Some(h) => item::is_armor(h.item) && item::armor_slot(h.item) == slot_i,
                };
                if allowed {
                    let mut held = state.inventory.held;
                    let mut s = state.inventory.slots[slot_i];
                    item::slot_click(&mut held, &mut s, right);
                    state.inventory.held = held;
                    state.inventory.slots[slot_i] = s;
                }
                return;
            }
        }
        let size = state.screen.craft_size();
        for (cell, x, y) in overlay::craft_cell_rects(w, h, size) {
            if hit(x, y) {
                let mut held = state.inventory.held;
                let mut c = state.craft[cell];
                item::slot_click(&mut held, &mut c, right);
                state.inventory.held = held;
                state.craft[cell] = c;
                return;
            }
        }
        let (ox, oy) = overlay::craft_output_rect(w, h, size);
        if hit(ox, oy) {
            craft_take_output(state);
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
            KeyCode::ShiftLeft => {
                state.input.down = pressed; // fly: descend
                state.input.sneak = pressed; // walk: sneak (ledge-stop)
            }
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
        let armor = state.inventory.equipped_armor();
        let game_ref = &state.game;
        state.player.update(
            dt,
            yaw,
            &state.input,
            |p| game_ref.block_at(p),
            armor,
        );
        state.camera.position = state.player.eye();

        // Sprinting widens the FOV slightly (and swimming narrows it); ease toward the target.
        let target_fov = if state.player.flying {
            70_f32
        } else if state.player.submerged {
            66.0
        } else if state.input.sprint && (state.input.forward || state.input.back) {
            78.0
        } else {
            70.0
        }
        .to_radians();
        state.camera.fovy += (target_fov - state.camera.fovy) * (10.0 * dt).min(1.0);

        // Survival: a hard fall or starvation can kill; respawn at spawn.
        if !state.player.flying && state.player.is_dead() {
            log::info!("You died — respawning at spawn.");
            // Drop the whole inventory at the death site, then respawn empty.
            let death_pos = state.player.position + Vec3::new(0.0, 1.0, 0.0);
            // One entity per stack (carries count + tool durability) — tools drop too, not vanish.
            for stack in state.inventory.drain_all() {
                state.game.spawn_item(death_pos, stack);
            }
            state.player.respawn();
            state.camera.position = state.player.eye();
        }

        // Stream chunks, advance fluids/entities; collect any picked-up item drops into inventory.
        let collected = state.game.update(
            &state.gpu,
            &state.renderer,
            state.player.position,
            dt,
            state.environment.day_factor(),
        );
        for stack in collected.items {
            if let Some(leftover) = state.inventory.insert(stack) {
                // Inventory full — drop the remainder back so items aren't vacuum-deleted.
                state
                    .game
                    .spawn_item(state.player.position + Vec3::new(0.0, 1.0, 0.0), leftover);
            }
        }
        if collected.xp > 0 {
            state.player.add_xp(collected.xp);
        }

        // Mob contact damage accumulated by the entity tick this frame (reduced by armor).
        if collected.player_damage > 0.0 {
            state.player.take_hit(collected.player_damage);
        }

        // Block targeting.
        let eye = state.camera.position;
        let fwd = state.camera.forward();
        let mut target = raycast::cast(eye, fwd, REACH, |p| state.game.is_solid_at(p));

        // Melee: if a mob is nearer than the targeted block FACE, the left-click hits it (not the
        // block). Using the ray's face-hit distance (not the block center) means a mob standing
        // flush behind a block can't be struck through it.
        state.melee_cd = (state.melee_cd - dt).max(0.0);
        let block_dist = target.as_ref().map_or(REACH, |h| h.dist);
        let mob_in_way = state.screen == Screen::None
            && state.input.break_held
            && state
                .game
                .nearest_mob_hit(eye, fwd, REACH)
                .is_some_and(|md| md <= block_dist);
        if mob_in_way {
            if state.melee_cd <= 0.0 {
                let dmg = item::attack_damage(state.inventory.selected_item());
                state.game.attack_nearest(eye, fwd, REACH, dmg);
                state.melee_cd = 0.5; // ~2 swings/sec
                if !state.inventory.creative {
                    state.inventory.damage_selected(1); // weapons wear from hitting
                }
            }
            state.mine_target = None;
            state.mine_progress = 0.0;
            target = None; // swinging at a mob, not mining — no block highlight
        }

        // Mining: hold LMB to break the targeted block, timed by hardness (instant in creative).
        if state.screen == Screen::None && state.input.break_held && !mob_in_way {
            if let Some(hit) = target.as_ref().map(|h| h.block) {
                let id = state.game.block_at(hit);
                if block::breakable(id) {
                    if state.mine_target != Some(hit) {
                        state.mine_target = Some(hit);
                        state.mine_progress = 0.0;
                    }
                    let sel = state.inventory.selected_item();
                    let block_tool = block::tool_class(id);
                    // A matching tool divides the break time by its tier speed.
                    let mut time = block::hardness(id);
                    if item::is_tool(sel)
                        && item::tool_class(sel) == block_tool
                        && block_tool != block::ToolClass::None
                    {
                        time /= item::tool_speed(item::tool_tier(sel));
                    }
                    if state.inventory.creative {
                        time = 0.0;
                    }
                    state.mine_progress += if time <= 0.0 { 1.0 } else { dt / time };
                    if state.mine_progress >= 1.0 {
                        let removed =
                            state.game.set_block(&state.gpu, &state.renderer, hit, block::AIR);
                        if removed && !state.inventory.creative {
                            // Pickaxe blocks need a matching tool of sufficient harvest level to drop.
                            let harvest_ok = if block::requires_tool(id) {
                                item::is_tool(sel)
                                    && item::tool_class(sel) == block_tool
                                    && item::harvest_level(item::tool_tier(sel))
                                        >= block::required_harvest(id)
                            } else {
                                true
                            };
                            if harvest_ok {
                                let center = hit.as_vec3() + Vec3::splat(0.5);
                                if let Some(drop) = block::drops(id) {
                                    let stack = item::ItemStack::new(item::item_of_block(drop), 1);
                                    state.game.spawn_item(center, stack);
                                }
                                // Some ores release experience orbs when mined.
                                state.game.spawn_xp(center, block::mining_xp(id));
                            }
                            state.inventory.damage_selected(1);
                        }
                        state.mine_target = None;
                        state.mine_progress = 0.0;
                        target = None; // don't flash a highlight on the now-air block this frame
                    }
                } else {
                    state.mine_target = None;
                }
            } else {
                state.mine_target = None;
            }
        } else {
            state.mine_target = None;
            state.mine_progress = 0.0;
        }
        if state.input.place_pressed {
            state.input.place_pressed = false;
            if let Some(hit) = &target {
                let targeted = state.game.block_at(hit.block);
                if targeted == block::CRAFTING_TABLE {
                    // Right-clicking a crafting table opens the 3x3 crafting screen instead.
                    state.pending_open = Some(Screen::Crafting);
                } else if targeted == block::FURNACE {
                    // Right-clicking a furnace opens its smelting screen.
                    state.pending_open = Some(Screen::Furnace(hit.block));
                } else {
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
        }

        // Q: drop one of the selected item ahead of the camera.
        if state.input.drop_pressed {
            state.input.drop_pressed = false;
            if let Some(dropped) = state.inventory.drop_one_selected() {
                let pos = state.camera.position + state.camera.forward() * 1.2;
                state.game.spawn_item(pos, dropped);
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
        let highlight = target.as_ref().map(|h| {
            let prog = if state.mine_target == Some(h.block) {
                state.mine_progress
            } else {
                0.0
            };
            (h.block, prog)
        });
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
            state.inventory.equipped_armor(),
            state.player.air_fraction(),
            state.player.submerged,
            state.player.level,
            state.player.xp_fraction(),
            debug_lines.as_deref(),
        );
        if let Screen::Furnace(pos) = state.screen {
            let f = state.game.furnace(pos);
            let burn_frac = f.map_or(0.0, |f| if f.burn_max > 0.0 { f.burn_remaining / f.burn_max } else { 0.0 });
            let cook_frac = f.map_or(0.0, |f| f.cook_progress / smelting::SMELT_TIME);
            ui.extend(overlay::build_furnace_screen(
                state.gpu.config.width,
                state.gpu.config.height,
                &state.inventory,
                f.and_then(|f| f.input),
                f.and_then(|f| f.fuel),
                f.and_then(|f| f.output),
                burn_frac,
                cook_frac,
                state.cursor,
            ));
        } else if state.screen != Screen::None {
            ui.extend(overlay::build_inventory_screen(
                state.gpu.config.width,
                state.gpu.config.height,
                &state.inventory,
                &state.craft,
                state.screen.craft_size(),
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
        game.spawn_item(player_pos + Vec3::new(0.0, 1.0, 0.0), left);
    }
}

/// Click the craft output: if the grid matches a recipe and the held stack can take the result,
/// consume one of each ingredient and add the output to the cursor.
fn craft_take_output(state: &mut State) {
    let ids: [u16; 9] = std::array::from_fn(|i| state.craft[i].map(|s| s.item).unwrap_or(0));
    let Some((out_item, out_count)) = crafting::match_grid(&ids) else {
        return;
    };
    let held_ok = match state.inventory.held {
        None => true,
        Some(h) => {
            h.item == out_item
                && !item::is_tool(out_item)
                && h.count as u16 + out_count as u16 <= item::max_stack(out_item) as u16
        }
    };
    if !held_ok {
        return;
    }
    for cell in state.craft.iter_mut() {
        if let Some(s) = cell {
            s.count -= 1;
            if s.count == 0 {
                *cell = None;
            }
        }
    }
    state.inventory.held = Some(match state.inventory.held {
        Some(mut h) => {
            h.count += out_count;
            h
        }
        None => item::ItemStack::new(out_item, out_count),
    });
}

/// Take a furnace's output into the held cursor stack (output is take-only): pick it up if the hand
/// is empty, or merge if holding the same item with room; holding a different item does nothing.
fn take_furnace_output(held: &mut Option<item::ItemStack>, output: &mut Option<item::ItemStack>) {
    let Some(out) = *output else { return };
    match held {
        None => {
            *held = Some(out);
            *output = None;
        }
        Some(h) if h.item == out.item && !item::is_tool(out.item) => {
            let space = item::max_stack(h.item).saturating_sub(h.count);
            let take = space.min(out.count);
            h.count += take;
            let rem = out.count - take;
            *output = if rem > 0 {
                Some(item::ItemStack::new(out.item, rem))
            } else {
                None
            };
        }
        Some(_) => {}
    }
}

/// Headless self-test (`VOXELCRAFT_PERSIST_TEST=1`): round-trips a sample inventory + armor +
/// furnace + survival state through the real on-disk save/load and logs whether it all survived.
pub fn persist_selftest() {
    let dir = std::env::temp_dir().join("voxelcraft_persist_selftest");
    let _ = std::fs::remove_dir_all(&dir);

    let mut inv = Inventory::new(false);
    inv.slots[0] = Some(item::ItemStack::new(item::item_of_block(block::COBBLESTONE), 40));
    let mut pick = item::ItemStack::new(item::DIAMOND_PICKAXE, 1);
    pick.durability = 777;
    inv.slots[5] = Some(pick);
    inv.slots[36] = Some(item::ItemStack::new(item::armor_id(3, 0), 1)); // diamond helmet equipped
    let furnaces = vec![persistence::FurnaceSave {
        pos: IVec3::new(10, 64, -3),
        input: Some(item::ItemStack::new(block::IRON_ORE, 4)),
        fuel: Some(item::ItemStack::new(block::PLANKS, 2)),
        output: Some(item::ItemStack::new(item::IRON_INGOT, 1)),
        burn_remaining: 3.0,
        burn_max: 9.0,
        cook_progress: 2.0,
        cook_item: block::IRON_ORE,
    }];
    let level = Level {
        seed: 42,
        spawn: [1.0, 2.0, 3.0],
        yaw: 0.1,
        pitch: 0.2,
        time: 0.5,
        flying: false,
        health: 11.0,
        hunger: 8.0,
        air: 5.0,
        saturation: 2.0,
        xp: 7.0,
        level: 3,
    };
    persistence::save_state(&dir, &inv, &furnaces).unwrap();
    persistence::save_level(&dir, &level).unwrap();

    let (linv, lf) = persistence::load_state(&dir, true);
    let ll = persistence::load_level(&dir).unwrap();
    let ok = linv.slots[5].map(|s| s.durability) == Some(777)
        && linv.slots[36].map(|s| s.item) == Some(item::armor_id(3, 0))
        && lf.len() == 1
        && lf[0].output.map(|s| s.item) == Some(item::IRON_INGOT)
        && (lf[0].cook_progress - 2.0).abs() < 1e-6
        && lf[0].cook_item == block::IRON_ORE
        && (ll.health - 11.0).abs() < 1e-6
        && (ll.air - 5.0).abs() < 1e-6
        && ll.level == 3;
    let _ = std::fs::remove_dir_all(&dir);
    if ok {
        log::info!("PERSIST_TEST: PASS — inventory + armor + furnace + survival all round-tripped on disk");
    } else {
        log::error!("PERSIST_TEST: FAIL — state did not survive save/load");
    }
}

/// On screen close, return any items left in the craft grid to the inventory (or drop them).
fn return_craft_to_inventory(state: &mut State) {
    let pos = state.player.position + Vec3::new(0.0, 1.0, 0.0);
    for cell in state.craft.iter_mut() {
        if let Some(stack) = cell.take() {
            if let Some(left) = state.inventory.insert(stack) {
                state.game.spawn_item(pos, left);
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
            // Debug: VOXELCRAFT_CRACK="x,y,z,progress" shows a mid-break crack overlay (verification).
            let highlight: Option<(IVec3, f32)> = std::env::var("VOXELCRAFT_CRACK").ok().and_then(|s| {
                let v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                (v.len() == 4).then(|| (IVec3::new(v[0] as i32, v[1] as i32, v[2] as i32), v[3]))
            });

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

            // Debug: VOXELCRAFT_MOBS=1 spawns one of each species on a flat stage and ticks them a
            // moment to settle (M27 mob verification). Note: with the player within ~14 blocks and
            // line-of-sight, hostile species will already be in Chase/Attack in the logged tally —
            // that exercises the M28 AI rather than showing a neutral idle state.
            if std::env::var("VOXELCRAFT_MOBS").is_ok() {
                for z in 20..=28 {
                    for x in 2..=20 {
                        game.set_block(&gpu, &renderer, IVec3::new(x, 84, z), block::STONE);
                    }
                }
                for (i, &sp) in crate::entity::Species::ALL.iter().enumerate() {
                    game.spawn_mob(Vec3::new(4.5 + i as f32 * 1.7, 88.0, 24.0), sp);
                }
                for _ in 0..90 {
                    let _ = game.update(&gpu, &renderer, player.position, 1.0 / 60.0, 1.0);
                }
                log::info!("M28 AI after settle: {}", game.mob_ai_summary());
                // M29: swing toward the mob stage (logs the hit), then flash them all red for the shot.
                let aim = (Vec3::new(10.0, 85.5, 24.0) - camera.position).normalize_or_zero();
                let hit = game.attack_nearest(camera.position, aim, 40.0, 6.0);
                log::info!("M29 melee: hit a mob = {hit}");
                game.flash_mobs(0.45);
                // Loot color check: a row of mob-drop items (should render in distinct colors).
                let loot = [
                    item::BEEF,
                    item::LEATHER,
                    item::BONE,
                    item::GUNPOWDER,
                    item::STRING,
                    item::FEATHER,
                ];
                for (i, &it) in loot.iter().enumerate() {
                    game.spawn_item(
                        Vec3::new(6.0 + i as f32 * 1.4, 86.3, 21.0),
                        item::ItemStack::new(it, 1),
                    );
                }
            }

            // Debug: VOXELCRAFT_EXPLODE=1 builds a stone platform, detonates a creeper-sized blast in
            // its top, and spawns a couple of arrows (M30 verification; pair with VOXELCRAFT_CAM).
            if std::env::var("VOXELCRAFT_EXPLODE").is_ok() {
                for x in 6..=14 {
                    for y in 78..=82 {
                        for z in 20..=28 {
                            game.set_block(&gpu, &renderer, IVec3::new(x, y, z), block::STONE);
                        }
                    }
                }
                game.spawn_arrow(Vec3::new(7.0, 83.5, 24.0), Vec3::new(7.0, 0.5, 0.0));
                game.spawn_arrow(Vec3::new(7.0, 84.0, 25.0), Vec3::new(7.0, 0.2, 0.0));
                let dmg = game.debug_explode(
                    &gpu,
                    &renderer,
                    Vec3::new(10.0, 82.5, 24.0),
                    3.0,
                    player.position,
                );
                log::info!("M30 explosion: radial player damage {dmg:.1}");
            }

            // Debug: VOXELCRAFT_SPAWN=1 exercises M31 natural spawning rules + cap + despawn.
            if std::env::var("VOXELCRAFT_SPAWN").is_ok() {
                for _ in 0..80 {
                    game.try_spawn(1.0, player.position); // day → passives on grass only
                }
                log::info!("M31 day spawns (passive on grass): {}", game.mob_species_summary());
                game.despawn_all_mobs();
                for _ in 0..80 {
                    game.try_spawn(0.0, player.position); // night → hostiles
                }
                log::info!("M31 night spawns (hostile): {}", game.mob_species_summary());
                log::info!("M31 cap held at {} mobs (cap 16)", game.mob_count());
                // Despawn-radius: a tick with the player teleported far culls the distant mobs.
                let _ = game.update(&gpu, &renderer, Vec3::splat(5000.0), 0.1, 0.0);
                log::info!("M31 after far tick (despawn): {}", game.mob_species_summary());
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
            // A partly-worn diamond pickaxe in the selected slot (tool swatch + durability bar + name).
            let mut pick = item::ItemStack::new(item::DIAMOND_PICKAXE, 1);
            pick.durability = (item::tool_max_durability(item::DIAMOND_PICKAXE) as f32 * 0.55) as u16;
            shot_inv.slots[5] = Some(pick);
            shot_inv.selected = 5;
            shot_inv.slots[9] = Some(item::ItemStack::new(item::item_of_block(block::COAL_ORE), 12));
            shot_inv.slots[11] = Some(item::ItemStack::new(item::item_of_block(block::DIRT), 64));
            shot_inv.slots[19] = Some(item::ItemStack::new(item::item_of_block(block::IRON_ORE), 5));
            // VOXELCRAFT_SURVIVAL=1 flips the HUD to survival mode with demo air/XP for verification.
            let survival_demo = std::env::var("VOXELCRAFT_SURVIVAL").is_ok();
            let mut ui = overlay::build_ui(
                gpu.config.width,
                gpu.config.height,
                &shot_inv,
                20.0,
                if survival_demo { 14.0 } else { 20.0 },
                survival_demo,
                if survival_demo { shot_inv.equipped_armor() } else { 0 },
                if survival_demo { 0.45 } else { 1.0 },
                survival_demo,
                if survival_demo { 7 } else { 0 },
                if survival_demo { 0.6 } else { 0.0 },
                Some(&dbg),
            );
            let screen_env = std::env::var("VOXELCRAFT_SCREEN").unwrap_or_default();
            if screen_env == "inv" || screen_env == "craft" {
                let size = if screen_env == "craft" { 3 } else { 2 };
                let mut shot_craft: [Option<item::ItemStack>; 9] = [None; 9];
                if size == 3 {
                    // A stone-pickaxe recipe so the output preview shows a result.
                    let cobble = item::item_of_block(block::COBBLESTONE);
                    shot_craft[0] = Some(item::ItemStack::new(cobble, 1));
                    shot_craft[1] = Some(item::ItemStack::new(cobble, 1));
                    shot_craft[2] = Some(item::ItemStack::new(cobble, 1));
                    shot_craft[4] = Some(item::ItemStack::new(item::STICK, 1));
                    shot_craft[7] = Some(item::ItemStack::new(item::STICK, 1));
                }
                let cursor = (600.0, 343.0);
                ui.extend(overlay::build_inventory_screen(
                    gpu.config.width,
                    gpu.config.height,
                    &shot_inv,
                    &shot_craft,
                    size,
                    cursor,
                ));
            } else if screen_env == "furnace" {
                // A mid-smelt furnace: iron ore cooking over planks into iron ingots.
                ui.extend(overlay::build_furnace_screen(
                    gpu.config.width,
                    gpu.config.height,
                    &shot_inv,
                    Some(item::ItemStack::new(item::item_of_block(block::IRON_ORE), 5)),
                    Some(item::ItemStack::new(item::item_of_block(block::PLANKS), 8)),
                    Some(item::ItemStack::new(item::IRON_INGOT, 3)),
                    0.6,
                    0.45,
                    (700.0, 360.0),
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
        let (inventory, saved_furnaces) = persistence::load_state(&dir, flying);
        let mut game = Game::new(&gpu, renderer.volume_bgl(), seed, RENDER_DISTANCE, saved);
        game.restore_furnaces(saved_furnaces);
        // One of each species near spawn; they fall onto terrain as it streams in.
        let ring = [(-4, -6), (4, -6), (7, 0), (4, 6), (-4, 6), (-7, 0), (0, 8), (0, -8)];
        for (species, &(dx, dz)) in crate::entity::Species::ALL.iter().zip(ring.iter()) {
            game.spawn_mob(
                Vec3::new(spawn.x + dx as f32, spawn.y, spawn.z + dz as f32),
                *species,
            );
        }
        let environment = Environment::new(time);
        let mut player = Player::new(spawn, flying);
        // Restore persisted survival state (from a loaded level; a new world keeps fresh defaults).
        if let Some(l) = &level {
            player.restore_state(l.health, l.hunger, l.air, l.saturation, l.xp, l.level);
        }
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
            craft: [None; 9],
            pending_open: None,
            mine_target: None,
            mine_progress: 0.0,
            melee_cd: 0.0,
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
                let pressed = btn_state == ElementState::Pressed;
                let screen_open = self.state.as_ref().is_some_and(|s| s.screen != Screen::None);
                if pressed && screen_open {
                    self.inventory_click(button);
                } else if pressed && !self.grabbed {
                    self.set_grab(true);
                } else if let Some(state) = &mut self.state {
                    match button {
                        MouseButton::Left => state.input.break_held = pressed, // hold to break
                        MouseButton::Right => {
                            if pressed {
                                state.input.place_pressed = true;
                            }
                        }
                        _ => {}
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
            WindowEvent::RedrawRequested => {
                self.frame();
                // Process a screen-open requested during the frame (after the state borrow ends).
                if let Some(sc) = self.state.as_mut().and_then(|s| s.pending_open.take()) {
                    match sc {
                        Screen::Crafting => self.open_crafting(),
                        Screen::Furnace(pos) => self.open_furnace(pos),
                        _ => {}
                    }
                }
            }
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
