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
use crate::gfx::graph::RenderTargets;
use crate::gfx::rt::RtScene;
use crate::renderer::ChunkRenderer;
use crate::{bench, block, capture, crafting, item, overlay, persistence, raycast, smelting};

const SEED: u64 = 0x5EED_C0FFEE;
// P21: render distance + fog are tier-driven now (`Quality::render_distance`, `fog_start/end()`;
// maxed tier == the old RENDER_DISTANCE=12 / FOG formulas, anchored by quality.rs tests).
const REACH: f32 = 6.0;
const EAT_TIME: f32 = 1.6; // seconds to eat a food (hold right-click)
const SHIELD_RAISE_DELAY: f32 = 0.25; // seconds a shield must be up before it blocks (MC ~5 ticks)
const SENSITIVITY: f32 = 0.0022;

/// Active GUI screen. Gameplay input is suppressed while any screen is open (M15b; grows later).
#[derive(PartialEq, Clone, Copy)]
enum Screen {
    None,
    Inventory,
    Crafting,
    /// A furnace's GUI, tagged with the furnace block position (its contents live in `Game`).
    Furnace(IVec3),
    /// A chest's GUI, tagged with the chest block position (its contents live in `Game`).
    Chest(IVec3),
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
    /// Platform quality tier (P21), resolved once at startup from the backend + env overrides.
    quality: crate::quality::Quality,
    renderer: ChunkRenderer,
    targets: RenderTargets,
    rt_scene: RtScene,
    /// M33-G8 DLSS Ray Reconstruction render state (render-res scene → upscaled output). `None` =>
    /// native resolution. Declared before `dlss` so the RR feature context drops before the SDK.
    dlss_render: Option<crate::dlss::DlssRender>,
    dlss: Option<crate::dlss::Dlss>,
    /// DLSS Frame Generation (DLSS-G) context, driven in the present path. `None` => no FG. (M33-G8-FG)
    frame_gen: Option<crate::frame_gen::FrameGen>,
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
    /// Remaining recharge before the held weapon is at full attack strength (seconds, P12).
    melee_cd: f32,
    /// Last frame's LMB-held state, for click-edge swing detection (P12).
    melee_was_held: bool,
    /// Last frame's selected hotbar slot, to reset the cooldown on a weapon swap (P12).
    melee_prev_sel: usize,
    /// Eat-hold progress (seconds) and the food id being eaten (right-click-hold on a food).
    eat_progress: f32,
    eat_item: Option<item::ItemId>,
    /// Bow-draw progress (seconds) and the bow id being drawn (right-click-hold with a bow, P13).
    draw_progress: f32,
    draw_item: Option<item::ItemId>,
    /// Shield-raise progress (seconds) and the shield id being raised; blocks once past
    /// SHIELD_RAISE_DELAY (right-click-hold with a shield, P14).
    shield_progress: f32,
    shield_item: Option<item::ItemId>,
    /// World difficulty (P6); mirrored to player + game, persisted in level.bin, cycled with G.
    difficulty: crate::rules::Difficulty,
    /// Fail-safe for GPU/driver wedges (2026-06-03 Metal incident): when submitted frames stop
    /// completing, save the world and exit instead of piling more work on a dead queue.
    gpu_watchdog: crate::gpu::GpuWatchdog,
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
                difficulty: state.difficulty.as_u8(),
            };
            state.game.save(&level);
            let dir = persistence::save_dir();
            // Fold any cursor-held stack back into slots so it isn't lost when saving mid-screen
            // (P / window-close don't go through close_screen's return_held path).
            let mut inv = state.inventory.clone();
            if let Some(h) = inv.held.take() {
                let _ = inv.insert(h);
            }
            if let Err(e) = persistence::save_state(
                &dir,
                &inv,
                &state.game.furnaces_to_save(),
                &state.game.chests_to_save(),
            ) {
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

    fn open_chest(&mut self, pos: IVec3) {
        if let Some(state) = &mut self.state {
            state.game.chest_mut(pos); // ensure a container exists to render
            state.screen = Screen::Chest(pos);
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
        // Chest screen: move items between the held cursor stack and the 27 container slots.
        if let Screen::Chest(pos) = state.screen {
            let mut held = state.inventory.held;
            let c = state.game.chest_mut(pos);
            for (i, x, y) in overlay::chest_slot_rects(w, h) {
                if hit(x, y) {
                    c.click(i, &mut held, right);
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
            KeyCode::KeyG if pressed && !repeat => {
                // Cycle difficulty Peaceful→Easy→Normal→Hard→… (suppressed while a screen is open).
                state.difficulty = state.difficulty.next();
                state.player.difficulty = state.difficulty;
                state.game.set_difficulty(state.difficulty);
                if !state.difficulty.spawns_hostiles() {
                    state.game.despawn_hostiles();
                }
                log::info!("Difficulty: {}", state.difficulty.name());
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
            |p| game_ref.block_state_at(p),
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
        let mut collected = state.game.update(
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

        // P14 shield raise-timer: hold right-click with a SHIELD (no screen open) to raise it. Updated
        // HERE (above the damage loop) so shield_ready reflects this frame. Pre-empts eat/place below.
        // `sel_item` is computed once and reused by the bow/eat blocks (selection can't change mid-tick).
        let sel_item = state.inventory.selected_item();
        if state.screen == Screen::None && state.input.place_held && item::is_shield(sel_item) {
            if state.shield_item != Some(sel_item) {
                state.shield_item = Some(sel_item);
                state.shield_progress = 0.0;
            }
            state.shield_progress += dt;
        } else {
            // Released, swapped off the shield, or a screen opened → lower instantly.
            state.shield_progress = 0.0;
            state.shield_item = None;
        }

        // Combat damage from the entity tick. Each hit carries its source position; a RAISED, ready
        // shield fully blocks hits whose source is in the front arc (melee, arrows, blasts alike — MC
        // blocks the frontal source). Summed into ONE take_hit so the 0.5s i-frame applies as before
        // (a per-hit call would drop all but the first simultaneous source). Environmental damage
        // (fall/lava/drown/starve) bypasses Collected entirely, so a shield never blocks it.
        let shield_up = state.shield_item.is_some() && state.shield_progress >= SHIELD_RAISE_DELAY;
        let fwd = state.camera.forward();
        let mut incoming = 0.0;
        for (amount, src) in std::mem::take(&mut collected.player_damage) {
            if shield_up && crate::player::shield_blocks(fwd, state.player.position, src) {
                continue; // fully blocked
            }
            incoming += amount;
        }
        if incoming > 0.0 {
            state.player.take_hit(incoming);
        }

        // Block targeting.
        let eye = state.camera.position;
        let fwd = state.camera.forward();
        let mut target = raycast::cast(eye, fwd, REACH, |p| state.game.is_solid_at(p));

        // Melee: if a mob is nearer than the targeted block FACE, the left-click hits it (not the
        // block). Using the ray's face-hit distance (not the block center) means a mob standing
        // flush behind a block can't be struck through it.
        // P12 (1.9): the recharge is per-weapon (item::attack_speed). A swing always lands but is weak
        // until the meter refills — tap-spamming does ~20% damage, a charged hit does full.
        state.melee_cd = (state.melee_cd - dt).max(0.0);
        // Switching the held weapon resets the cooldown (vanilla gives a fresh full-power swing).
        if state.inventory.selected != state.melee_prev_sel {
            state.melee_prev_sel = state.inventory.selected;
            state.melee_cd = 0.0;
        }
        let block_dist = target.as_ref().map_or(REACH, |h| h.dist);
        let mob_in_way = state.screen == Screen::None
            && state.input.break_held
            && state
                .game
                .nearest_mob_hit(eye, fwd, REACH)
                .is_some_and(|md| md <= block_dist);
        // A swing fires on the click EDGE (so spamming yields partial-charge hits) or as an auto-swing
        // when LMB is held and the meter is already full.
        let pressed_now = state.input.break_held && !state.melee_was_held;
        let auto_full = state.input.break_held && state.melee_cd <= 0.0;
        if mob_in_way && (pressed_now || auto_full) {
            let sel = state.inventory.selected_item();
            let cd_max = 1.0 / item::attack_speed(sel);
            let charge = (1.0 - state.melee_cd / cd_max).clamp(0.0, 1.0);
            let mut dmg = item::charged_damage(item::attack_damage(sel), charge);
            // Critical hit (+50%): a falling, non-sprinting, grounded-feet-off attack.
            let crit = crate::player::is_critical_hit(
                state.player.on_ground,
                state.player.velocity.y,
                state.input.sprint,
                state.player.submerged,
                state.player.flying,
            );
            if crit {
                dmg *= 1.5;
            }
            // Sweep: a full-charge sword swing standing on the ground (un-enchanted sweep = 1).
            let sweep = if charge >= 0.999
                && item::is_sword(sel)
                && state.player.on_ground
                && !state.input.sprint
            {
                Some(1.0)
            } else {
                None
            };
            // Sprint-attack: extra horizontal knockback (and it can't crit — see is_critical_hit).
            let kb_mult = if state.input.sprint && state.player.on_ground { 1.5 } else { 1.0 };
            state.game.attack_nearest(eye, fwd, REACH, dmg, crit, sweep, kb_mult);
            state.melee_cd = cd_max;
            if !state.inventory.creative {
                state.inventory.damage_selected(1); // weapons wear from hitting
            }
        }
        if mob_in_way {
            state.mine_target = None;
            state.mine_progress = 0.0;
            target = None; // swinging at a mob, not mining — no block highlight
        }
        // Edge tracker for the click-to-swing logic; MUST update every frame (not just when a mob is
        // in reach) so a fresh press is detected correctly even across screen opens / focus loss.
        state.melee_was_held = state.input.break_held;

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
                        // Capture the block state before clearing it (a double slab drops two).
                        let bstate = state.game.block_state_at(hit).1;
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
                                if let Some((drop_item, count)) = block::drops(id, bstate) {
                                    let stack = item::ItemStack::new(drop_item, count);
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

        // Bow (P13): hold right-click with a bow + ammo to DRAW (charge over BOW_DRAW_TIME); release
        // looses a gravity-arced arrow whose speed/damage scale with the draw. This pre-empts eating
        // and block placement. Combined into one block so the RELEASE (place_held just went false) is
        // detected and fires BEFORE the draw state is reset.
        let have_ammo = state.inventory.creative || state.inventory.has_item(item::ARROW);
        let bow_drawing = state.screen == Screen::None
            && state.input.place_held
            && item::is_bow(sel_item)
            && have_ammo;
        if state.draw_item.is_some() && !bow_drawing {
            // We were drawing and now aren't: fire on a genuine in-world release past the min draw;
            // a screen-open or weapon-swap cancels (those leave place_held/screen != the fire case).
            if state.screen == Screen::None
                && !state.input.place_held
                && state.draw_progress >= item::BOW_MIN_DRAW
            {
                let (speed, damage) = item::bow_shot(state.draw_progress / item::BOW_DRAW_TIME);
                let dir = state.camera.forward();
                let pos = state.camera.position + dir * 1.2; // muzzle clear of the player AABB
                state.game.spawn_player_arrow(pos, dir * speed, damage);
                if !state.inventory.creative {
                    state.inventory.consume_item(item::ARROW);
                }
            }
            state.draw_progress = 0.0;
            state.draw_item = None;
        } else if bow_drawing {
            if state.draw_item != Some(sel_item) {
                state.draw_item = Some(sel_item);
                state.draw_progress = 0.0;
            }
            state.draw_progress = (state.draw_progress + dt).min(item::BOW_DRAW_TIME);
        }

        // Eating: hold right-click while a food item is selected to eat it over EAT_TIME, then
        // restore hunger/saturation and consume one. (Foods place nothing, so the place edge below
        // is a no-op for them.) Gameplay-only — suppressed while any screen is open or drawing a bow.
        let edible = state.screen == Screen::None
            && state.input.place_held
            && !item::is_bow(sel_item)
            && !item::is_shield(sel_item)
            && crate::food::food(sel_item).is_some_and(|f| state.player.can_eat(f.always_edible));
        if edible {
            if state.eat_item != Some(sel_item) {
                state.eat_item = Some(sel_item);
                state.eat_progress = 0.0;
            }
            state.eat_progress += dt;
            if state.eat_progress >= EAT_TIME {
                if let Some(f) = crate::food::food(sel_item) {
                    state.player.eat(f.hunger, f.saturation);
                    state.inventory.consume_selected();
                }
                state.eat_progress = 0.0;
                state.eat_item = None;
            }
        } else {
            state.eat_progress = 0.0;
            state.eat_item = None;
        }

        if state.input.place_pressed {
            state.input.place_pressed = false;
            if item::is_bow(state.inventory.selected_item()) {
                // A bow press only starts the draw (handled above) — never place/interact.
            } else if item::is_shield(state.inventory.selected_item()) {
                // A shield press only starts the raise (handled above) — never place/interact.
            } else if state.game.try_feed_mob(eye, fwd, REACH, state.inventory.selected_item()) {
                // P17: fed a breedable animal its food → love-mode; consume one (takes priority over
                // placing, like melee's mob-precedence).
                state.inventory.consume_selected();
            } else if let Some(hit) = &target {
                let targeted = state.game.block_at(hit.block);
                if targeted == block::WOODEN_DOOR {
                    // Right-click toggles the door open/closed — both halves together.
                    let (_, st) = state.game.block_state_at(hit.block);
                    let half = block::door_half(st);
                    let (f, h, no) = (block::door_facing(st), block::door_hinge(st), !block::door_open(st));
                    let partner = if half == block::DOOR_LOWER { hit.block + IVec3::Y } else { hit.block - IVec3::Y };
                    let p_half = if half == block::DOOR_LOWER { block::DOOR_UPPER } else { block::DOOR_LOWER };
                    state.game.set_block_state(&state.gpu, &state.renderer, hit.block, block::WOODEN_DOOR, block::door_state(f, no, h, half));
                    state.game.set_block_state(&state.gpu, &state.renderer, partner, block::WOODEN_DOOR, block::door_state(f, no, h, p_half));
                } else if targeted == block::WOODEN_TRAPDOOR {
                    // Right-click toggles the trapdoor open/closed.
                    let (_, st) = state.game.block_state_at(hit.block);
                    let ns = block::trapdoor_state(block::trapdoor_facing(st), block::trapdoor_half(st), !block::trapdoor_open(st));
                    state.game.set_block_state(&state.gpu, &state.renderer, hit.block, block::WOODEN_TRAPDOOR, ns);
                } else if targeted == block::CRAFTING_TABLE {
                    // Right-clicking a crafting table opens the 3x3 crafting screen instead.
                    state.pending_open = Some(Screen::Crafting);
                } else if targeted == block::FURNACE {
                    // Right-clicking a furnace opens its smelting screen.
                    state.pending_open = Some(Screen::Furnace(hit.block));
                } else if targeted == block::CHEST {
                    // Right-clicking a chest opens its 27-slot storage screen.
                    state.pending_open = Some(Screen::Chest(hit.block));
                } else if targeted == block::LEVER || targeted == block::BUTTON {
                    // Right-click flips a lever / presses a button. Cosmetic in P11 (it re-meshes the
                    // on/pressed state); the bit drives redstone in P31. The attach face is preserved.
                    let (_, st) = state.game.block_state_at(hit.block);
                    let ns = block::attach_state(block::attach_face(st), !block::attach_on(st));
                    state.game.set_block_state(&state.gpu, &state.renderer, hit.block, targeted, ns);
                } else {
                    let id = state.inventory.selected_block();
                    if id == block::WOODEN_DOOR {
                        // 2-tall door: needs a solid block below + air in both cells. Facing from the
                        // camera; hinge defaults left (double-door auto-pairing deferred).
                        let lower = hit.block + hit.normal;
                        let upper = lower + IVec3::Y;
                        let support = block::is_solid(state.game.block_at(lower - IVec3::Y));
                        // Both cells must be empty AND not overlap the player — a closed door is solid,
                        // so without this you could seal a door panel inside yourself (aim at your feet).
                        let space = state.game.block_at(lower) == block::AIR
                            && state.game.block_at(upper) == block::AIR
                            && !state.player.intersects_block(lower)
                            && !state.player.intersects_block(upper);
                        if support && space {
                            let f = state.camera.forward();
                            let facing = if f.x.abs() > f.z.abs() {
                                if f.x > 0.0 { 1 } else { 3 }
                            } else if f.z > 0.0 {
                                0
                            } else {
                                2
                            };
                            state.game.set_block_state(&state.gpu, &state.renderer, lower, block::WOODEN_DOOR, block::door_state(facing, false, 0, block::DOOR_LOWER));
                            state.game.set_block_state(&state.gpu, &state.renderer, upper, block::WOODEN_DOOR, block::door_state(facing, false, 0, block::DOOR_UPPER));
                            state.inventory.consume_selected();
                        }
                    } else if id == block::WOODEN_TRAPDOOR {
                        let place = hit.block + hit.normal;
                        // half: top face → bottom flap, bottom face → top flap, side → by click height.
                        let half = if hit.normal.y == 1 {
                            0
                        } else if hit.normal.y == -1 {
                            1
                        } else if hit.hit_point.y.rem_euclid(1.0) > 0.5 {
                            1
                        } else {
                            0
                        };
                        let f = state.camera.forward();
                        let facing = if f.x.abs() > f.z.abs() {
                            if f.x > 0.0 { 1 } else { 3 }
                        } else if f.z > 0.0 {
                            0
                        } else {
                            2
                        };
                        // A closed trapdoor is solid — don't seal it inside the player (matches the
                        // generic placement guard).
                        if !state.player.intersects_block(place)
                            && state.game.set_block_state(&state.gpu, &state.renderer, place, block::WOODEN_TRAPDOOR, block::trapdoor_state(facing, half, false)) {
                            state.inventory.consume_selected();
                        }
                    } else {
                    let is_slab = matches!(id, block::STONE_SLAB | block::WOOD_SLAB);
                    // Double-slab merge: clicking the empty-half face of a matching single slab with
                    // the same slab item fills it into a full (double) block — no new neighbor block.
                    // hit.normal is the face normal pointing back at the camera: +Y = clicked the top
                    // face (merges a bottom slab), -Y = clicked the bottom face (merges a top slab).
                    let (tgt_id, tgt_state) = state.game.block_state_at(hit.block);
                    let merge = is_slab
                        && tgt_id == id
                        && ((block::slab_half(tgt_state) == block::SLAB_BOTTOM && hit.normal.y == 1)
                            || (block::slab_half(tgt_state) == block::SLAB_TOP && hit.normal.y == -1));
                    if merge {
                        if state.game.set_block_state(
                            &state.gpu,
                            &state.renderer,
                            hit.block,
                            id,
                            block::slab_state(block::SLAB_DOUBLE),
                        ) {
                            state.inventory.consume_selected();
                        }
                    } else {
                        let place = hit.block + hit.normal;
                        // Orientation/half state for the placed block.
                        let place_state = if id == block::STONE_STAIRS {
                            // Stairs orient by where the player faces (the high step rises away).
                            let f = state.camera.forward();
                            let facing = if f.x.abs() > f.z.abs() {
                                if f.x > 0.0 { 1 } else { 3 }
                            } else if f.z > 0.0 {
                                0
                            } else {
                                2
                            };
                            block::stair_state(facing)
                        } else if is_slab {
                            // Bottom when set on a top face, top when set under a face, else by where
                            // on a side face you clicked (upper half of the cell → top slab).
                            let half = if hit.normal.y == 1 {
                                block::SLAB_BOTTOM
                            } else if hit.normal.y == -1 {
                                block::SLAB_TOP
                            } else if hit.hit_point.y.rem_euclid(1.0) > 0.5 {
                                block::SLAB_TOP
                            } else {
                                block::SLAB_BOTTOM
                            };
                            block::slab_state(half)
                        } else if id == block::WOOD {
                            // A log's axis is the axis of the clicked face's normal: a top/bottom
                            // face gives the upright default (Y), side faces give X or Z.
                            let n = hit.normal;
                            let axis = if n.x != 0 {
                                block::AXIS_X
                            } else if n.z != 0 {
                                block::AXIS_Z
                            } else {
                                block::AXIS_Y
                            };
                            block::log_state(axis)
                        } else if matches!(id, block::TORCH | block::LEVER | block::BUTTON) {
                            // Attach face = the direction from the placed cell toward the support you
                            // clicked (= -hit.normal). A top/bottom face mounts on the floor; a side
                            // face mounts on that wall. (Clicking a block guarantees a support there.)
                            let n = hit.normal;
                            let face = if n.y != 0 {
                                block::ATTACH_FLOOR
                            } else if n.x > 0 {
                                block::ATTACH_NX
                            } else if n.x < 0 {
                                block::ATTACH_PX
                            } else if n.z > 0 {
                                block::ATTACH_NZ
                            } else {
                                block::ATTACH_PZ
                            };
                            block::attach_state(face, false)
                        } else {
                            0
                        };
                        // Attach fixtures (torch/lever/button) are walk-through, so never reject their
                        // placement for overlapping the player even though is_solid (targetable) is true.
                        let blocks_player = block::is_solid(id)
                            && !block::is_attach(id)
                            && state.player.intersects_block(place);
                        // Torches/levers/buttons can't mount on a ceiling (no ATTACH_CEILING variant) —
                        // reject an underside (bottom-face) click instead of spawning a floating fixture.
                        let attach_invalid = block::is_attach(id) && hit.normal.y == -1;
                        if id != block::AIR
                            && !blocks_player
                            && !attach_invalid
                            && state.game.set_block_state(
                                &state.gpu,
                                &state.renderer,
                                place,
                                id,
                                place_state,
                            )
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
        // FG self-test (VOXELCRAFT_FG_TEST=N): auto-rotate for consistent motion — DLSS-G needs
        // genuine motion to generate frames — and auto-exit after N frames logging the max generated
        // count, so Frame Generation can be verified without a human watching the window.
        let fg_test: Option<u32> = std::env::var("VOXELCRAFT_FG_TEST")
            .ok()
            .and_then(|s| s.parse().ok());
        if fg_test.is_some() {
            state.camera.yaw += 0.012;
        }
        let aspect = state.gpu.aspect();
        state.camera_uniform.update(&state.camera, aspect);
        state
            .camera_uniform
            .set_environment(&state.environment, state.quality.fog_start(), state.quality.fog_end());
        state.camera_uniform.set_time(state.elapsed, state.environment.time);
        // M33-G8: jitter the projection to match the jitter handed to NGX (DLSS temporal sampling).
        if let Some(dr) = &state.dlss_render {
            let (rw, rh) = dr.render_dims();
            state.camera_uniform.apply_jitter(dr.jitter(), rw, rh);
        }
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
                state.difficulty.name(),
            ))
        } else {
            None
        };
        let attack_charge = {
            let cd_max = 1.0 / item::attack_speed(state.inventory.selected_item());
            if cd_max > 0.0 {
                (1.0 - state.melee_cd / cd_max).clamp(0.0, 1.0)
            } else {
                1.0
            }
        };
        let draw_charge = if item::is_bow(state.inventory.selected_item()) {
            (state.draw_progress / item::BOW_DRAW_TIME).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let shield_charge = if state.shield_item.is_some() {
            (state.shield_progress / SHIELD_RAISE_DELAY).clamp(0.0, 1.0)
        } else {
            0.0
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
            attack_charge,
            draw_charge,
            shield_charge,
            debug_lines.as_deref(),
        );
        if let Screen::Chest(pos) = state.screen {
            let slots = state
                .game
                .chest(pos)
                .map(|c| c.slots.clone())
                .unwrap_or_else(|| vec![None; crate::container::CHEST_SLOTS]);
            ui.extend(overlay::build_chest_screen(
                state.gpu.config.width,
                state.gpu.config.height,
                &state.inventory,
                &slots,
                state.cursor,
            ));
        } else if let Screen::Furnace(pos) = state.screen {
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
        let as_bg = if state.renderer.use_hw_rt() {
            let all = state.game.all_meshes();
            state
                .rt_scene
                .rebuild(&state.gpu, &all)
                .map(|t| state.renderer.make_as_bind_group(&state.gpu.device, t))
        } else {
            None
        };
        let frame_submitted = state.renderer.render_frame(
            &state.gpu,
            &state.targets,
            &visible,
            volume_bg,
            as_bg.as_ref(),
            highlight,
            &ui,
            state.dlss_render.as_mut(),
            state.frame_gen.as_mut(),
        );
        // GPU wedge fail-safe (2026-06-03 Metal incident): track per-frame completion; if submitted
        // work stops signaling, the end of this fn saves the world and exits without touching the
        // GPU again (a wedged AGX driver took WindowServer — and the whole machine — down with it).
        if frame_submitted {
            state.gpu_watchdog.arm(&state.gpu.queue);
        }
        let gpu_stall = state.gpu_watchdog.check(&state.gpu.device);

        // FG self-test: after N frames, log the max generated-frame count and exit (DLSS-G generates
        // an interpolated frame between each rendered one => max presented == 2). Driven by a local
        // frame counter, NOT frame_gen.is_some(), so it still terminates + reports when FG is
        // unavailable (no SDK / non-Ada GPU) instead of spinning forever (review #1/#6). The throwaway
        // run exits WITHOUT save_world so the auto-panned camera yaw is never persisted (review #2/#6).
        if let Some(n) = fg_test {
            use std::sync::atomic::{AtomicU32, Ordering};
            static TEST_FRAMES: AtomicU32 = AtomicU32::new(0);
            let count = TEST_FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
            if count >= n {
                match &state.frame_gen {
                    Some(fg) => log::info!(
                        "FG SELF-TEST: max presented = {} over {count} frames => generating = {}",
                        fg.max_presented(),
                        fg.max_presented() >= 2
                    ),
                    None => log::warn!(
                        "FG SELF-TEST: Frame Generation unavailable (no DLSS-G context) — ran {count} \
                         frames, cannot verify generation"
                    ),
                }
                std::process::exit(0);
            }
        }

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

        // GPU wedge fail-safe trigger (see gpu_watchdog above). `state`'s borrow has ended, so the
        // world can be saved through `&self`; exit via process::exit — GPU teardown (surface/device
        // drops) can block forever against a wedged driver, and skipping it loses nothing.
        if let Some(stalled) = gpu_stall {
            log::error!(
                "GPU watchdog: submitted frames have not completed for {stalled:.1}s — the GPU/driver \
                 appears wedged. Saving world and exiting before the stall takes the system down \
                 (VOXELCRAFT_GPU_WATCHDOG=0 disables; default 5s)."
            );
            self.save_world();
            std::process::exit(70);
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
        difficulty: 3, // Hard
    };
    let mut chest_slots = vec![None; crate::container::CHEST_SLOTS];
    chest_slots[2] = Some(item::ItemStack::new(block::DIAMOND_ORE, 9));
    let chests = vec![persistence::ChestSave {
        pos: IVec3::new(-5, 70, 8),
        slots: chest_slots,
    }];
    persistence::save_state(&dir, &inv, &furnaces, &chests).unwrap();
    persistence::save_level(&dir, &level).unwrap();

    let (linv, lf, lc) = persistence::load_state(&dir, true);
    let ll = persistence::load_level(&dir).unwrap();
    let ok = linv.slots[5].map(|s| s.durability) == Some(777)
        && linv.slots[36].map(|s| s.item) == Some(item::armor_id(3, 0))
        && lf.len() == 1
        && lf[0].output.map(|s| s.item) == Some(item::IRON_INGOT)
        && (lf[0].cook_progress - 2.0).abs() < 1e-6
        && lf[0].cook_item == block::IRON_ORE
        && lc.len() == 1
        && lc[0].pos == IVec3::new(-5, 70, 8)
        && lc[0].slots[2].map(|s| s.item) == Some(block::DIAMOND_ORE)
        && (ll.health - 11.0).abs() < 1e-6
        && (ll.air - 5.0).abs() < 1e-6
        && ll.level == 3
        && ll.difficulty == 3;
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
    difficulty: &str,
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
        format!("Difficulty {difficulty}"),
    ]
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Voxelcraft")
            .with_inner_size(LogicalSize::new(1600.0, 900.0))
            // DLSS-G only presents generated frames to a foreground/composited window, so request
            // an active window and pull it to the foreground when FG is on. (M33-G8-FG)
            .with_active(true);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        if crate::frame_gen::requested() {
            window.focus_window();
        }
        self.window = Some(window.clone());

        let (gpu, frame_gen) = pollster::block_on(Gpu::new(window.clone()));
        // Surface the active backend + hardware-RT status in the title — at-a-glance confirmation
        // during the Vulkan/RT migration (M33-G0).
        window.set_title(&format!(
            "Voxelcraft  [{:?}{}]",
            gpu.backend,
            if gpu.rt_enabled { " · RT cores" } else { "" }
        ));
        // P21: resolve the platform quality tier (maxed off-Metal == pre-P21 behavior; tuned-down
        // on macOS/Metal) + its env overrides, then thread it through renderer/game construction.
        let quality = crate::quality::Quality::resolve(gpu.backend);
        let renderer = ChunkRenderer::new(&gpu, &quality);

        // M33-G8: bring up the DLSS SDK (Tensor-core denoise + upscale). `None` => native-resolution
        // rendering (off / non-DX12 / unsupported). Used by both the headless and interactive paths.
        let dlss = crate::dlss::Dlss::init(&gpu);

        // Headless path: a fresh generated world with a few verification edits, rendered offscreen.
        // VOXELCRAFT_SHOT=path.png saves one frame; VOXELCRAFT_BENCH=N times N frames (P21) — both
        // may be set (bench, then save the PNG as the run's visual artifact).
        let shot_path = std::env::var("VOXELCRAFT_SHOT").ok();
        let bench_frames = std::env::var("VOXELCRAFT_BENCH")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok());
        if shot_path.is_some() || bench_frames.is_some() {
            let mut game = Game::new(
                &gpu,
                renderer.volume_bgl(),
                SEED,
                &quality,
                FxHashMap::default(),
            );
            // Headless difficulty (P6): VOXELCRAFT_DIFFICULTY gates hostile spawning + the F3 line.
            let headless_difficulty = std::env::var("VOXELCRAFT_DIFFICULTY")
                .ok()
                .and_then(|s| crate::rules::Difficulty::from_env(&s))
                .unwrap_or_default();
            game.set_difficulty(headless_difficulty);
            // Offscreen screenshots are one-shot, so trace many more GI rays for a clean image; a
            // bench run instead defaults to the tier's INTERACTIVE ray count so its numbers reflect
            // gameplay cost. VOXELCRAFT_GI_RAYS overrides either (e.g. for A/B sweeps).
            let shot_rays = std::env::var("VOXELCRAFT_GI_RAYS")
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(if bench_frames.is_some() { quality.gi_rays } else { 64 });
            game.set_rtx_quality(shot_rays.max(1));
            // P21: VOXELCRAFT_RTX=0|1|2 forces the lighting mode (off/shadows/shadows+GI) so bench
            // runs can isolate the primary-shadow-ray and GI costs (interactively the R key cycles).
            if let Some(mode) = std::env::var("VOXELCRAFT_RTX")
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
            {
                game.set_rtx_mode(mode);
            }
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

            // Debug knob: VOXELCRAFT_PLACE="x,y,z,id[,state];..." places blocks before the shot
            // (milestone verification — e.g. a glowstone to check emissive lighting, or an oriented
            // stair/slab/door via the optional 5th block-state field).
            if let Ok(s) = std::env::var("VOXELCRAFT_PLACE") {
                for spec in s.split(';') {
                    let v: Vec<i32> = spec.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                    if v.len() >= 4 {
                        let bs = if v.len() >= 5 { v[4] as u8 } else { 0 };
                        game.set_block_state(&gpu, &renderer, IVec3::new(v[0], v[1], v[2]), v[3] as u16, bs);
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
                // P17: a baby cow beside the adult cow to show the juvenile half-scale render.
                game.spawn_baby(Vec3::new(4.5, 88.0, 26.5), crate::entity::Species::Cow);
                for _ in 0..90 {
                    let _ = game.update(&gpu, &renderer, player.position, 1.0 / 60.0, 1.0);
                }
                log::info!("M28 AI after settle: {}", game.mob_ai_summary());
                // M29: swing toward the mob stage (logs the hit), then flash them all red for the shot.
                let aim = (Vec3::new(10.0, 85.5, 24.0) - camera.position).normalize_or_zero();
                let hit = game.attack_nearest(camera.position, aim, 40.0, 6.0, false, None, 1.0);
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

            // Debug: VOXELCRAFT_SPAWN=1 exercises P16 light/biome-gated spawning + per-category caps.
            if std::env::var("VOXELCRAFT_SPAWN").is_ok() {
                for _ in 0..160 {
                    game.try_spawn(1.0, player.position); // day → passive packs on lit grass
                }
                log::info!(
                    "P16 day spawns: {} (passive_count={})",
                    game.mob_species_summary(),
                    game.passive_count()
                );
                assert!(game.passive_count() <= 8, "passive cap holds");
                game.despawn_all_mobs();
                for _ in 0..160 {
                    game.try_spawn(0.0, player.position); // night → hostiles in the dark
                }
                log::info!(
                    "P16 night spawns: {} (hostile_count={})",
                    game.mob_species_summary(),
                    game.hostile_count()
                );
                assert!(game.hostile_count() <= 12, "hostile cap holds");
                // Despawn-radius: a tick with the player teleported far culls the distant mobs.
                let _ = game.update(&gpu, &renderer, Vec3::splat(5000.0), 0.1, 0.0);
                log::info!("P16 after far tick (despawn): {}", game.mob_species_summary());
            }

            // P21: scripted edits above queue boundary-neighbor remeshes on the worker pool —
            // wait for them so the capture can't show a stale chunk seam.
            game.flush_meshes(&gpu, &renderer);

            // Populate the voxel volume so shadows / GI / water depth trace across the full vista.
            game.prime_volume(&gpu, player.position);

            camera_uniform.update(&camera, gpu.aspect());
            camera_uniform.set_environment(&environment, quality.fog_start(), quality.fog_end());
            let shot_time = std::env::var("VOXELCRAFT_TIME")
                .ok()
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0);
            camera_uniform.set_time(shot_time, environment.time);
            // M33-G8: build the Ray Reconstruction context (scene at render res → upscale to output).
            // `None` => native resolution. The per-frame camera jitter is applied inside `screenshot`
            // (it must advance with NGX's jitter across the warm-up frames).
            let mut dlss_render = dlss.as_ref().and_then(|d| {
                crate::dlss::DlssRender::new(
                    d,
                    &renderer,
                    &gpu,
                    (gpu.config.width, gpu.config.height),
                    crate::dlss::supersample_from_env(),
                )
            });
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
                headless_difficulty.name(),
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
                if survival_demo { 0.5 } else { 1.0 }, // attack_charge: show the cooldown bar in the demo
                if survival_demo { 0.7 } else { 0.0 }, // draw_charge: show the bow-draw bar in the demo
                if survival_demo { 1.0 } else { 0.0 }, // shield_charge: show the READY shield cue in the demo
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
            } else if screen_env == "chest" {
                // A chest holding a few sample stacks.
                let mut cslots = vec![None; crate::container::CHEST_SLOTS];
                cslots[0] = Some(item::ItemStack::new(item::item_of_block(block::DIAMOND_ORE), 12));
                cslots[1] = Some(item::ItemStack::new(item::IRON_INGOT, 30));
                cslots[4] = Some(item::ItemStack::new(item::item_of_block(block::OBSIDIAN), 64));
                cslots[10] = Some(item::ItemStack::new(item::DIAMOND_PICKAXE, 1));
                cslots[18] = Some(item::ItemStack::new(item::item_of_block(block::GLOWSTONE), 5));
                ui.extend(overlay::build_chest_screen(
                    gpu.config.width,
                    gpu.config.height,
                    &shot_inv,
                    &cslots,
                    (640.0, 250.0),
                ));
            }
            log::info!(
                "Headless: {} chunks, {} meshes, {} visible",
                game.loaded_chunk_count(),
                game.mesh_count(),
                visible.len()
            );
            if std::env::var("VOXELCRAFT_AS_STATS").is_ok() {
                crate::gfx::rt::log_as_stats(&gpu, &visible);
            }
            let all = game.all_meshes();
            // P21: timed multi-frame benchmark (native path only — DLSS does internal submits that
            // would skew per-pass attribution; on the Mac dlss_render is None anyway).
            if let Some(n) = bench_frames {
                if dlss_render.is_some() {
                    log::warn!("bench: DLSS is active but the bench loop times the native path");
                }
                bench::run(
                    &gpu,
                    &renderer,
                    &all,
                    volume_bg,
                    highlight,
                    &ui,
                    gpu.config.width,
                    gpu.config.height,
                    &camera_uniform,
                    n,
                );
            }
            if let Some(path) = &shot_path {
                capture::screenshot(
                    &gpu,
                    &renderer,
                    &all,
                    volume_bg,
                    highlight,
                    &ui,
                    gpu.config.width,
                    gpu.config.height,
                    &camera_uniform,
                    dlss_render.as_mut(),
                    path,
                );
            }
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
        let (inventory, saved_furnaces, saved_chests) = persistence::load_state(&dir, flying);
        let mut game = Game::new(&gpu, renderer.volume_bgl(), seed, &quality, saved);
        game.restore_furnaces(saved_furnaces);
        game.restore_chests(saved_chests);
        // World difficulty (P6): VOXELCRAFT_DIFFICULTY overrides; else the saved value; else Normal.
        let difficulty = std::env::var("VOXELCRAFT_DIFFICULTY")
            .ok()
            .and_then(|s| crate::rules::Difficulty::from_env(&s))
            .or_else(|| level.as_ref().map(|l| crate::rules::Difficulty::from_u8(l.difficulty)))
            .unwrap_or_default();
        game.set_difficulty(difficulty);
        // VOXELCRAFT_GI_RAYS overrides the hemisphere GI sample count (default 8) — more samples =
        // less grain for DLSS-RR to denoise; the GPU has headroom. (M33-G9)
        if let Ok(n) = std::env::var("VOXELCRAFT_GI_RAYS").map(|s| s.trim().parse::<u32>()) {
            if let Ok(rays) = n {
                game.set_rtx_quality(rays.max(1));
            }
        }
        // One of each species near spawn; they fall onto terrain as it streams in. 12 ring slots so
        // the four P18 species (wolf/enderman/slime/villager) appear too (zip truncates otherwise).
        let ring = [
            (-4, -6), (4, -6), (7, 0), (4, 6), (-4, 6), (-7, 0), (0, 8), (0, -8),
            (8, 8), (-8, 8), (8, -8), (-8, -8),
        ];
        for (species, &(dx, dz)) in crate::entity::Species::ALL.iter().zip(ring.iter()) {
            game.spawn_mob(
                Vec3::new(spawn.x + dx as f32, spawn.y, spawn.z + dz as f32),
                *species,
            );
        }
        // Peaceful clears the spawn ring's hostiles immediately (animals stay).
        if !difficulty.spawns_hostiles() {
            game.despawn_hostiles();
        }
        let environment = Environment::new(time);
        let mut player = Player::new(spawn, flying);
        player.difficulty = difficulty;
        // Restore persisted survival state (from a loaded level; a new world keeps fresh defaults).
        if let Some(l) = &level {
            player.restore_state(l.health, l.hunger, l.air, l.saturation, l.xp, l.level);
        }
        let camera = Camera::new(player.eye(), yaw, pitch);
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(&camera, gpu.aspect());
        camera_uniform.set_environment(&environment, quality.fog_start(), quality.fog_end());
        renderer.update_camera(&gpu, &camera_uniform);
        renderer.set_sky(environment.wgpu_clear());

        // M33-G8: build the DLSS Ray Reconstruction context (if available); scene targets are then
        // sized to its render resolution, else to the output resolution. M33-G8-FG Phase 3: RR and
        // Frame Generation now stack — RR denoises + upscales the scene, FG generates intermediate
        // frames from the (point-upscaled) output-res guides. RR off (VOXELCRAFT_DLSS=off) → FG runs
        // standalone over the native-res frame.
        let dlss_render = dlss.as_ref().and_then(|d| {
            crate::dlss::DlssRender::new(
                d,
                &renderer,
                &gpu,
                (gpu.config.width, gpu.config.height),
                crate::dlss::supersample_from_env(),
            )
        });
        let targets = match &dlss_render {
            Some(dr) => dr.make_render_targets(&renderer, &gpu.device),
            None => renderer.make_targets(&gpu.device, gpu.config.width, gpu.config.height),
        };
        let rt_scene = RtScene::new();
        self.state = Some(State {
            gpu,
            quality,
            renderer,
            targets,
            rt_scene,
            dlss_render,
            dlss,
            frame_gen,
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
            melee_was_held: false,
            melee_prev_sel: 0,
            eat_progress: 0.0,
            eat_item: None,
            draw_progress: 0.0,
            draw_item: None,
            shield_progress: 0.0,
            shield_item: None,
            difficulty,
            gpu_watchdog: crate::gpu::GpuWatchdog::new(),
        });
        self.set_grab(true);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.save_world();
                event_loop.exit();
            }
            // Moving to a display with a different DPI scale re-resolves the render scale
            // (Metal-only sub-native rendering); the Resized that follows applies it.
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(state) = &mut self.state {
                    state.gpu.rescale(scale_factor);
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.gpu.resize(size);
                    // M33-G8-FG (Phase 4): the DLSS-G recomposition textures are swapchain-sized.
                    if let Some(fg) = state.frame_gen.as_mut() {
                        fg.resize(&state.gpu.device, state.gpu.config.width, state.gpu.config.height);
                    }
                    // M33-G8: the DLSS context is output-resolution-bound — recreate it (which
                    // re-derives the render resolution), then size the scene targets to match.
                    state.dlss_render = state.dlss.as_ref().and_then(|d| {
                        crate::dlss::DlssRender::new(
                            d,
                            &state.renderer,
                            &state.gpu,
                            (state.gpu.config.width, state.gpu.config.height),
                            crate::dlss::supersample_from_env(),
                        )
                    });
                    state.targets = match &state.dlss_render {
                        Some(dr) => dr.make_render_targets(&state.renderer, &state.gpu.device),
                        None => state.renderer.make_targets(
                            &state.gpu.device,
                            state.gpu.config.width,
                            state.gpu.config.height,
                        ),
                    };
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
                            state.input.place_held = pressed; // hold to eat / use
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
                        // `state.cursor` is always swapchain (config) space — the UI rects in
                        // ui/overlay.rs are built from gpu.config dims. With a sub-native render
                        // scale (Metal) the window is larger than the drawable, so map the
                        // window-physical position per-axis (the floored config dims can round
                        // the two ratios differently).
                        let sx = state.gpu.config.width as f64 / state.gpu.size.width.max(1) as f64;
                        let sy =
                            state.gpu.config.height as f64 / state.gpu.size.height.max(1) as f64;
                        state.cursor = ((position.x * sx) as f32, (position.y * sy) as f32);
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
                        Screen::Chest(pos) => self.open_chest(pos),
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
