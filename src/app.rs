//! Application layer: the winit `App`/`State`, event routing, per-frame update + render, and the
//! headless screenshot path. `main.rs` is just the module tree + the event loop entry point.

use std::path::{Path, PathBuf};
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
const AUTOSAVE_SECS: f32 = 300.0; // S3: autosave cadence (vanilla saves every 5 min)
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

/// Top-level app/UI state machine (M35). The world-bearing `Session` exists only in InGame/Paused;
/// menus run with `session == None`. N1 introduces the split but only `InGame` is reachable (the game
/// still boots straight into a world); the menu scenes are wired in N3.
#[derive(Clone)]
#[allow(dead_code)] // menu scenes are constructed in N3; the enum lands now so the split is stable.
enum Scene {
    MainMenu,
    /// S10: the death screen (world frozen; Respawn / Save & Quit).
    Dead,
    WorldSelect,
    CreateWorld,
    Settings { return_to: Box<Scene> },
    InGame,
    Paused,
}

/// The always-present render stack + frame pacing + the top-level scene. Created once the GPU is up
/// and lives for both menus and gameplay; the world lives in `session`.
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
    last_frame: Instant,
    fps_accum: f32,
    fps_frames: u32,
    fps_smooth: f32,
    /// M35: the active top-level scene.
    scene: Scene,
    /// M35: the loaded world, or `None` while in a menu.
    session: Option<Session>,
    /// S9: the audio engine (`None` = silent: no device / VOXELCRAFT_MUTE / FG self-test).
    audio: Option<crate::audio::Audio>,
    /// Fail-safe for GPU/driver wedges (P21, 2026-06-03 Metal incident): when submitted frames stop
    /// completing, save the world and exit instead of piling more work on a dead queue.
    gpu_watchdog: crate::gpu::GpuWatchdog,
    // ── M35 menu state (used while `session` is None / in a menu scene) ──
    /// Pointer / hover / focus / caret for the active menu.
    menu: crate::menu::UiState,
    /// Persisted player settings (the source of truth; env vars override at startup).
    settings: crate::settings::Settings,
    /// The worlds list shown in WorldSelect (refreshed when the screen opens).
    worlds: Vec<crate::persistence::WorldInfo>,
    /// Selected world row in WorldSelect (cosmetic highlight only — play/delete carry their own index).
    world_sel: usize,
    /// Row whose delete control is "armed" (showing the red "Delete?" confirm); a second click on the
    /// same row commits the `remove_dir_all`. Any other click / scene change disarms it.
    world_pending_delete: Option<usize>,
    /// Create-world text-field buffers (name + seed) + the chosen game mode (S1).
    create_name: String,
    create_seed: String,
    create_mode: crate::rules::GameMode,
    /// Active settings tab (the "Back" target rides in `Scene::Settings { return_to }`).
    settings_tab: crate::menu::SettingsTab,
}

/// Everything that exists only while a world is loaded — created on Play/Create, dropped on Quit to
/// Menu. Accessed as `session.<field>` from the per-frame body (disjoint from `State`'s render stack).
struct Session {
    /// The world's on-disk directory (`saves/worlds/<slug>/`) — where this session saves.
    world_dir: PathBuf,
    game: Game,
    camera: Camera,
    player: Player,
    input: Input,
    inventory: Inventory,
    environment: Environment,
    camera_uniform: CameraUniform,
    debug_f3: bool,
    elapsed: f32,
    screen: Screen,
    cursor: (f32, f32),
    /// Items placed in the craft grid (top-left of the 9-cell array; 2x2 inventory or 3x3 table).
    craft: [Option<item::ItemStack>; 9],
    /// A screen the frame asked to open (processed after the frame's state borrow ends).
    pending_open: Option<Screen>,
    /// S2: whether the loaded position has been occupancy-validated against real chunks (the
    /// deferred buried check). Physics stays paused until it passes.
    spawn_checked: bool,
    /// S3: seconds of play since the last save; at AUTOSAVE_SECS the world autosaves.
    autosave: f32,
    // ── S9 audio edge trackers ──
    /// Health last frame (env damage has no event — hurt sounds fire on the drop edge).
    prev_health: f32,
    /// Highest y since leaving the ground (the player's own tracker resets mid-substep).
    fall_top: f32,
    was_on_ground: bool,
    was_wet: bool,
    /// Stride accumulator (blocks walked) toward the next footstep.
    step_accum: f32,
    /// S10: red damage-flash strength (decays); set on any health-drop edge.
    hurt_flash: f32,
    /// S10: directional damage cue — (world angle of the source, remaining seconds).
    hurt_dir: Option<(f32, f32)>,
    /// S10: transient HUD message line ("Autosaved", errors...) + remaining seconds.
    toast: Option<(String, f32)>,
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
    /// F1 creative menu: the full reachable item list (built once) + the current page (scroll to flip).
    creative_palette: Vec<item::ItemId>,
    creative_page: usize,
    /// M34-VM first-person view-model animation state (held item + swing/use/bob).
    view_model: crate::viewmodel::ViewModel,
    /// M34-VM5: whether the world camera also head-bobs when walking (VOXELCRAFT_VIEWBOB=0 disables).
    viewbob: bool,
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

    /// Save the world; returns whether EVERYTHING was written (S3). With no session there is
    /// nothing to save — that is success, not failure (Quit from the main menu).
    fn save_world(&self) -> bool {
        let Some(session) = self.state.as_ref().and_then(|s| s.session.as_ref()) else {
            return true;
        };
        let level = Level {
            seed: session.game.seed(),
            spawn: session.player.position.to_array(),
            yaw: session.camera.yaw,
            pitch: session.camera.pitch,
            time: session.environment.time,
            flying: session.player.flying,
            health: session.player.health,
            hunger: session.player.hunger,
            air: session.player.air,
            saturation: session.player.saturation(),
            xp: session.player.xp,
            level: session.player.level,
            difficulty: session.difficulty.as_u8(),
            // Runtime source of truth for mode is the inventory's creative flag (S1).
            mode: session.inventory.creative as u8,
            bed: session.player.bed_respawn.map(|v| v.to_array()),
        };
        let dir = &session.world_dir;
        let mut ok = true;
        if let Err(e) = session.game.save(dir, &level) {
            log::error!("SAVE FAILED (world chunks/level): {e} — the previous save is intact");
            ok = false;
        }
        // Fold any cursor-held stack back into slots so it isn't lost when saving mid-screen
        // (P / window-close don't go through close_screen's return_held path).
        let mut inv = session.inventory.clone();
        if let Some(h) = inv.held.take() {
            let _ = inv.insert(h);
        }
        if let Err(e) = persistence::save_state(
            dir,
            &inv,
            &session.game.furnaces_to_save(),
            &session.game.chests_to_save(),
            &session.game.entities_to_save(),
        ) {
            log::error!("SAVE FAILED (inventory/containers/entities): {e} — the previous save is intact");
            ok = false;
        }
        ok
    }

    /// Center of the window in pixels — for placing the cursor when a GUI screen opens.
    fn screen_center(&self) -> (f32, f32) {
        match &self.state {
            Some(s) => (s.gpu.config.width as f32 * 0.5, s.gpu.config.height as f32 * 0.5),
            None => (0.0, 0.0),
        }
    }

    fn toggle_inventory(&mut self) {
        let center = self.screen_center();
        let open = {
            let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) else { return };
            if session.screen != Screen::None {
                return_craft_to_inventory(session);
                drop_leftover(&mut session.inventory, &mut session.game, session.player.position);
                session.screen = Screen::None;
                false
            } else {
                session.screen = Screen::Inventory;
                session.input = Input::default();
                session.cursor = center;
                true
            }
        };
        self.set_grab(!open);
    }

    fn open_crafting(&mut self) {
        let center = self.screen_center();
        if let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
            session.screen = Screen::Crafting;
            session.input = Input::default();
            session.cursor = center;
        }
        self.set_grab(false);
    }

    fn open_furnace(&mut self, pos: IVec3) {
        let center = self.screen_center();
        if let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
            session.game.furnace_mut(pos); // ensure a state exists to render/tick
            session.screen = Screen::Furnace(pos);
            session.input = Input::default();
            session.cursor = center;
        }
        self.set_grab(false);
    }

    fn open_chest(&mut self, pos: IVec3) {
        let center = self.screen_center();
        if let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
            session.game.chest_mut(pos); // ensure a container exists to render
            session.screen = Screen::Chest(pos);
            session.input = Input::default();
            session.cursor = center;
        }
        self.set_grab(false);
    }

    fn close_screen(&mut self) {
        if let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
            if session.screen != Screen::None {
                return_craft_to_inventory(session);
                drop_leftover(&mut session.inventory, &mut session.game, session.player.position);
                session.screen = Screen::None;
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
        let Some(session) = state.session.as_mut() else { return };
        let (cx, cy) = session.cursor;
        let hit = |x: f32, y: f32| {
            cx >= x && cx < x + overlay::INV_SLOT && cy >= y && cy < y + overlay::INV_SLOT
        };
        for (slot_i, x, y) in overlay::inventory_slot_rects(w, h) {
            if hit(x, y) {
                session.inventory.click_slot(slot_i, right);
                return;
            }
        }
        // F1: clicking a creative-palette cell grabs a stack of that item onto the cursor (left = full
        // stack, right = one). The old held stack is simply discarded — creative items are infinite.
        if session.screen == Screen::Inventory && session.inventory.creative {
            for (i, x, y) in overlay::creative_palette_rects(w, h) {
                if hit(x, y) {
                    let idx = session.creative_page * overlay::PAL_PER_PAGE + i;
                    if let Some(&it) = session.creative_palette.get(idx) {
                        let count = if right { 1 } else { item::max_stack(it) };
                        session.inventory.held = Some(item::ItemStack::new(it, count));
                    } else {
                        session.inventory.held = None; // clicked an empty palette cell = trash the cursor
                    }
                    return;
                }
            }
        }
        // Furnace screen: move items between held and the input/fuel slots; output is take-only.
        if let Screen::Furnace(pos) = session.screen {
            let f = session.game.furnace_mut(pos);
            for (kind, x, y) in overlay::furnace_slot_rects(w, h) {
                if hit(x, y) {
                    let mut held = session.inventory.held;
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
                    session.inventory.held = held;
                    return;
                }
            }
            return;
        }
        // Chest screen: move items between the held cursor stack and the 27 container slots.
        if let Screen::Chest(pos) = session.screen {
            let mut held = session.inventory.held;
            let c = session.game.chest_mut(pos);
            for (i, x, y) in overlay::chest_slot_rects(w, h) {
                if hit(x, y) {
                    c.click(i, &mut held, right);
                    session.inventory.held = held;
                    return;
                }
            }
            return;
        }
        // Equipped-armor slots: only the matching piece may be placed; taking a piece out is allowed.
        for (slot_i, x, y) in overlay::armor_slot_rects(w, h) {
            if hit(x, y) {
                let allowed = match session.inventory.held {
                    None => true,
                    Some(h) => item::is_armor(h.item) && item::armor_slot(h.item) == slot_i,
                };
                if allowed {
                    let mut held = session.inventory.held;
                    let mut s = session.inventory.slots[slot_i];
                    item::slot_click(&mut held, &mut s, right);
                    session.inventory.held = held;
                    session.inventory.slots[slot_i] = s;
                }
                return;
            }
        }
        let size = session.screen.craft_size();
        for (cell, x, y) in overlay::craft_cell_rects(w, h, size) {
            if hit(x, y) {
                let mut held = session.inventory.held;
                let mut c = session.craft[cell];
                item::slot_click(&mut held, &mut c, right);
                session.inventory.held = held;
                session.craft[cell] = c;
                return;
            }
        }
        let (ox, oy) = overlay::craft_output_rect(w, h, size);
        if hit(ox, oy) {
            craft_take_output(session);
        }
    }

    fn handle_key(&mut self, code: KeyCode, pressed: bool, repeat: bool, _event_loop: &ActiveEventLoop) {
        if code == KeyCode::Escape && pressed {
            if self.state.as_ref().and_then(|s| s.session.as_ref()).is_some_and(|s| s.screen != Screen::None) {
                self.close_screen();
                return;
            }
            // M35: Esc in-game opens the pause menu (was: quit the process).
            self.pause_game();
            return;
        }
        if code == KeyCode::KeyP && pressed && !repeat {
            let ok = self.save_world();
            if let Some(se) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
                se.toast = Some((
                    if ok { "Saved".into() } else { "SAVE FAILED — see log".into() },
                    2.0,
                ));
            }
            return;
        }
        if code == KeyCode::KeyE && pressed && !repeat {
            self.toggle_inventory();
            return;
        }
        // While a GUI screen is open, swallow gameplay key PRESSES but let RELEASES through, so a
        // movement key held across the open/close edge can't get stuck on.
        if pressed && self.state.as_ref().and_then(|s| s.session.as_ref()).is_some_and(|s| s.screen != Screen::None) {
            return;
        }
        let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) else { return };
        match code {
            KeyCode::KeyW => session.input.forward = pressed,
            KeyCode::KeyS => session.input.back = pressed,
            KeyCode::KeyA => session.input.left = pressed,
            KeyCode::KeyD => session.input.right = pressed,
            KeyCode::Space => session.input.up = pressed,
            KeyCode::ShiftLeft => {
                session.input.down = pressed; // fly: descend
                session.input.sneak = pressed; // walk: sneak (ledge-stop)
            }
            KeyCode::ControlLeft => session.input.sprint = pressed,
            KeyCode::KeyF if pressed && !repeat => {
                // Flying is a creative-mode ability (S1); survival worlds always walk.
                if session.inventory.creative {
                    session.player.flying = !session.player.flying;
                    session.player.velocity = Vec3::ZERO;
                } else {
                    session.toast = Some(("Flying requires a Creative world".into(), 2.0));
                }
            }
            KeyCode::KeyR if pressed && !repeat => {
                let mode = session.game.cycle_rtx();
                log::info!("RTX lighting: {mode}");
            }
            KeyCode::F3 if pressed && !repeat => {
                session.debug_f3 = !session.debug_f3;
            }
            KeyCode::KeyG if pressed && !repeat => {
                // Cycle difficulty Peaceful→Easy→Normal→Hard→… (suppressed while a screen is open).
                session.difficulty = session.difficulty.next();
                session.player.difficulty = session.difficulty;
                session.game.set_difficulty(session.difficulty);
                if !session.difficulty.spawns_hostiles() {
                    session.game.despawn_hostiles();
                }
                log::info!("Difficulty: {}", session.difficulty.name());
            }
            KeyCode::KeyQ if pressed && !repeat => session.input.drop_pressed = true,
            KeyCode::Digit1 => maybe_select(session, pressed, 0),
            KeyCode::Digit2 => maybe_select(session, pressed, 1),
            KeyCode::Digit3 => maybe_select(session, pressed, 2),
            KeyCode::Digit4 => maybe_select(session, pressed, 3),
            KeyCode::Digit5 => maybe_select(session, pressed, 4),
            KeyCode::Digit6 => maybe_select(session, pressed, 5),
            KeyCode::Digit7 => maybe_select(session, pressed, 6),
            KeyCode::Digit8 => maybe_select(session, pressed, 7),
            KeyCode::Digit9 => maybe_select(session, pressed, 8),
            _ => {}
        }
    }

    /// Per-frame dispatcher (M35): the in-game world only ticks + renders in `Scene::InGame`; every
    /// menu/paused scene renders a 2D menu (the world session, if any, is kept but frozen — not ticked).
    fn frame(&mut self) {
        // Take a Copy-able decision so the `&self.state` borrow ends before the `&mut self` frame call.
        match self.state.as_ref().map(|s| matches!(s.scene, Scene::InGame)) {
            Some(true) => self.frame_ingame(),
            Some(false) => self.frame_menu(),
            None => {}
        }
    }

    // ── M35-N3: menu scene flow ────────────────────────────────────────────────────────────────────

    /// Build the active scene's menu vertices + hit-rects — the single source of truth shared by the
    /// renderer (`frame_menu`), hover tracking (CursorMoved), and click routing (`menu_click`).
    /// `None` for `Scene::InGame` (no menu).
    fn build_menu(state: &State, w: f32, h: f32) -> Option<crate::menu::Built> {
        use crate::menu;
        let ui = &state.menu;
        let s = &state.settings;
        Some(match &state.scene {
            Scene::MainMenu => menu::build_main(w, h, ui),
            Scene::WorldSelect => {
                menu::build_world_select(w, h, &state.worlds, state.world_sel, state.world_pending_delete, ui)
            }
            Scene::CreateWorld => {
                menu::build_create(w, h, &state.create_name, &state.create_seed, state.create_mode, ui)
            }
            Scene::Settings { .. } => {
                let diff = state
                    .session
                    .as_ref()
                    .map(|se| se.difficulty)
                    .unwrap_or(crate::rules::Difficulty::Normal);
                menu::build_settings(w, h, s, diff, state.settings_tab, ui)
            }
            Scene::Paused => menu::build_pause(w, h, ui),
            Scene::Dead => menu::build_death(w, h, ui),
            Scene::InGame => return None,
        })
    }

    /// Render the active menu screen (no world; world-independent render targets). A frozen session is
    /// left untouched behind the menu.
    fn frame_menu(&mut self) {
        let Some(state) = self.state.as_mut() else { return };
        // S9: keep the ambient pads fed/faded while in menus (a paused session keeps its palette).
        let day = state.session.as_ref().map(|s| s.environment.day_factor()).unwrap_or(1.0);
        if let Some(audio) = state.audio.as_mut() {
            audio.tick_music(day, 1.0 / 60.0);
        }
        let (w, h) = (state.gpu.config.width as f32, state.gpu.config.height as f32);
        let Some((verts, _)) = Self::build_menu(state, w, h) else { return };
        state.renderer.present_menu(&state.gpu, &verts);
    }

    /// True while a menu/paused scene is showing (cursor freed, gameplay input suspended).
    fn in_menu(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|s| !matches!(s.scene, Scene::InGame))
    }

    /// Cursor is grabbed iff we're actually playing (InGame with no GUI screen open).
    fn apply_grab_for_scene(&mut self) {
        let grab = self.state.as_ref().is_some_and(|s| {
            matches!(s.scene, Scene::InGame)
                && s.session.as_ref().is_some_and(|se| se.screen == Screen::None)
        });
        self.set_grab(grab);
    }

    /// Left-click in a menu: hit-test the active screen's rects, dispatch the hit widget.
    fn menu_click(&mut self, event_loop: &ActiveEventLoop) {
        let hit = {
            let Some(state) = self.state.as_ref() else { return };
            let (w, h) = (state.gpu.config.width as f32, state.gpu.config.height as f32);
            let cursor = state.menu.cursor;
            Self::build_menu(state, w, h).and_then(|(_, rects)| {
                rects
                    .iter()
                    .find(|(_, r)| r.contains(cursor))
                    .map(|(id, r)| (*id, *r))
            })
        };
        if let Some((id, rect)) = hit {
            self.activate(id, rect, event_loop);
        }
    }

    /// The single click-dispatch point for every menu widget.
    fn activate(&mut self, id: crate::menu::WidgetId, rect: crate::menu::Rect, event_loop: &ActiveEventLoop) {
        use crate::menu::WidgetId as W;
        if let Some(audio) = self.state.as_ref().and_then(|s| s.audio.as_ref()) {
            audio.play(crate::audio::Sfx::Click, 0.5); // S9
        }
        // Any click that isn't on a delete control disarms a pending-delete confirm (so it never lingers).
        if !matches!(id, W::DeleteWorld(_)) {
            if let Some(state) = self.state.as_mut() {
                state.world_pending_delete = None;
            }
        }
        match id {
            // Main menu.
            W::Singleplayer => self.open_world_select(),
            W::MainSettings => self.open_settings(Scene::MainMenu),
            W::Quit => {
                self.save_world();
                event_loop.exit();
            }
            // Death screen (S10).
            W::DeadRespawn => {
                self.respawn_player();
                self.set_scene(Scene::InGame);
                if let Some(state) = self.state.as_mut() {
                    state.last_frame = Instant::now();
                }
            }
            W::DeadSaveQuit => {
                // Respawn FIRST so the save never contains a dead player (and the drops land at
                // the death site before the world writes out).
                self.respawn_player();
                self.to_main_menu();
            }
            // Pause.
            W::Resume => self.resume_game(),
            W::PauseSettings => self.open_settings(Scene::Paused),
            W::SaveQuit => self.to_main_menu(),
            // Settings nav.
            W::GeneralTab => self.set_settings_tab(crate::menu::SettingsTab::General),
            W::GraphicsTab => self.set_settings_tab(crate::menu::SettingsTab::Graphics),
            W::SettingsBack => self.close_settings(),
            // Settings controls (mutate + persist; live-apply lands in N5, difficulty is live now).
            W::ViewBobToggle => self.mutate_settings(|s| s.view_bob = !s.view_bob),
            W::FrameGenToggle => self.mutate_settings(|s| s.frame_gen = !s.frame_gen),
            W::DlssModeCycle => self.mutate_settings(|s| s.dlss_mode = s.dlss_mode.next()),
            W::DlssQualityCycle => self.mutate_settings(|s| s.dlss_quality = s.dlss_quality.next()),
            W::GiModeCycle => self.mutate_settings(|s| s.gi_mode = s.gi_mode.next()),
            W::TracerCycle => self.mutate_settings(|s| s.tracer = s.tracer.next()),
            W::BackendCycle => self.mutate_settings(|s| s.backend = s.backend.next()),
            W::DifficultyCycle => self.cycle_difficulty(),
            W::FovSlider => self.slider_set(id, rect, 50.0, 110.0),
            W::SensSlider => self.slider_set(id, rect, 0.0005, 0.006),
            W::RenderDistSlider => self.slider_set(id, rect, 4.0, 24.0),
            W::SupersampleSlider => self.slider_set(id, rect, 1.0, 4.0),
            W::GiRaysSlider => self.slider_set(id, rect, 1.0, 16.0),
            // World select / create.
            W::CreateNew => self.set_scene(Scene::CreateWorld),
            W::BackFromWorlds => self.set_scene(Scene::MainMenu),
            W::CreateCancel => self.set_scene(Scene::WorldSelect),
            W::NameField => self.focus_field(W::NameField),
            W::SeedField => self.focus_field(W::SeedField),
            W::GameModeCycle => {
                if let Some(state) = self.state.as_mut() {
                    state.create_mode = state.create_mode.next();
                }
            }
            W::WorldRow(i) => self.play_world(i),
            W::DeleteWorld(i) => self.delete_world(i),
            W::CreateConfirm => self.create_world(),
        }
    }

    /// Open the world-select screen, refreshing the list from disk.
    fn open_world_select(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.worlds = persistence::list_worlds();
            state.world_sel = 0;
            state.world_pending_delete = None;
        }
        self.set_scene(Scene::WorldSelect);
    }

    /// Click a world row → load and enter that world.
    fn play_world(&mut self, i: usize) {
        // Clone the dir out before the &mut borrow in enter_world (the index could be stale vs a
        // refreshed-shorter list, so the bound check is load-bearing).
        let dir = self
            .state
            .as_ref()
            .and_then(|s| s.worlds.get(i))
            .map(|w| w.dir.clone());
        if let Some(dir) = dir {
            if let Some(state) = self.state.as_mut() {
                state.world_sel = i;
            }
            self.enter_world(dir, WorldSource::Load);
        }
    }

    /// Click a row's delete control: first click arms the red "Delete?" confirm; a second click on the
    /// same row commits the irreversible `remove_dir_all`.
    fn delete_world(&mut self, i: usize) {
        let Some(state) = self.state.as_mut() else { return };
        if state.worlds.get(i).is_none() {
            return;
        }
        if state.world_pending_delete != Some(i) {
            state.world_pending_delete = Some(i); // arm (or re-arm onto a different row)
            return;
        }
        // Confirmed. Never delete the world backing the (frozen) active session.
        let dir = state.worlds[i].dir.clone();
        let is_active = state.session.as_ref().is_some_and(|se| se.world_dir == dir);
        state.world_pending_delete = None;
        if is_active {
            log::warn!("refusing to delete the currently-loaded world");
            return;
        }
        if let Err(e) = persistence::delete_world(&dir) {
            log::error!("failed to delete world {}: {e}", dir.display());
        }
        state.worlds = persistence::list_worlds();
        state.world_sel = state.world_sel.min(state.worlds.len().saturating_sub(1));
    }

    /// Create World confirm: build a fresh world from the name + seed fields and enter it. The seed is
    /// parsed exactly once and threaded to both the saved header and the session, so they can't diverge.
    fn create_world(&mut self) {
        let (name, seed_text) = {
            let Some(state) = self.state.as_ref() else { return };
            (state.create_name.trim().to_string(), state.create_seed.clone())
        };
        let name = if name.is_empty() { "New World".to_string() } else { name };
        let seed = parse_seed(&seed_text);
        let mode = self.state.as_ref().map(|s| s.create_mode).unwrap_or_default();
        let info = match persistence::create_world(&name, seed, mode) {
            Ok(info) => info,
            Err(e) => {
                log::error!("failed to create world {name:?}: {e}");
                return; // stay on the Create screen
            }
        };
        if let Some(state) = self.state.as_mut() {
            state.create_name.clear();
            state.create_seed.clear();
            state.create_mode = crate::rules::GameMode::default();
            state.menu.focused = None;
            state.menu.caret = 0;
        }
        self.enter_world(info.dir, WorldSource::New { seed: info.seed, mode });
    }

    /// S10: drop the inventory at the death site and revive at the spawn point — the Respawn
    /// button's body (also runs before a Save & Quit from the death screen).
    fn respawn_player(&mut self) {
        let Some(state) = self.state.as_mut() else { return };
        let Some(se) = state.session.as_mut() else { return };
        let death_pos = se.player.position + Vec3::new(0.0, 1.0, 0.0);
        // One entity per stack (carries count + tool durability) — tools drop too, not vanish.
        for stack in se.inventory.drain_all() {
            se.game.spawn_item(death_pos, stack);
        }
        // S11: wake at the bed while it stands. An UNLOADED bed chunk is trusted (dying far from
        // home is the whole point of a bed); only a resident cell that is no longer a bed
        // invalidates the anchor.
        let bed_target = se.player.bed_respawn.and_then(|bp| {
            let cell = bp.floor().as_ivec3();
            let standing = bp + Vec3::new(0.5, 0.55, 0.5);
            if !se.game.chunk_loaded_at(cell) || se.game.block_at(cell) == block::BED {
                Some(standing)
            } else {
                None
            }
        });
        match bed_target {
            Some(t) => se.player.respawn_to(t),
            None => {
                if se.player.bed_respawn.take().is_some() {
                    se.toast = Some(("Your bed was destroyed".into(), 2.5));
                }
                se.player.respawn();
            }
        }
        se.camera.position = se.player.eye();
        se.prev_health = se.player.health;
        se.fall_top = se.player.position.y;
        se.hurt_flash = 0.0;
        se.hurt_dir = None;
    }

    /// Replace the scene and re-evaluate cursor grab.
    fn set_scene(&mut self, scene: Scene) {
        if let Some(state) = self.state.as_mut() {
            state.scene = scene;
            state.menu.focused = None;
            state.world_pending_delete = None;
        }
        self.apply_grab_for_scene();
    }

    fn set_settings_tab(&mut self, tab: crate::menu::SettingsTab) {
        if let Some(state) = self.state.as_mut() {
            state.settings_tab = tab;
        }
    }

    fn open_settings(&mut self, return_to: Scene) {
        self.set_scene(Scene::Settings {
            return_to: Box::new(return_to),
        });
    }

    fn close_settings(&mut self) {
        let back = match self.state.as_ref().map(|s| &s.scene) {
            Some(Scene::Settings { return_to }) => (**return_to).clone(),
            _ => Scene::MainMenu,
        };
        self.set_scene(back);
    }

    /// Esc in-game → pause (freeze the world, free the cursor).
    fn pause_game(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.scene = Scene::Paused;
        }
        self.set_grab(false);
    }

    fn resume_game(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.scene = Scene::InGame;
            state.last_frame = Instant::now(); // avoid a big dt step after a long pause
        }
        self.apply_grab_for_scene();
    }

    /// Build the world session in `dir` and enter gameplay.
    fn enter_world(&mut self, dir: PathBuf, source: WorldSource) {
        let session = {
            let Some(state) = self.state.as_ref() else { return };
            create_session(&state.gpu, &state.renderer, &state.quality, &dir, source)
        };
        if let Some(state) = self.state.as_mut() {
            state.session = Some(session);
            state.scene = Scene::InGame;
            state.last_frame = Instant::now();
        }
        self.apply_grab_for_scene();
    }

    /// Save & Quit to Menu: persist, drop the world (workers exit as their channels close), show the menu.
    fn to_main_menu(&mut self) {
        self.save_world();
        if let Some(state) = self.state.as_mut() {
            state.session = None;
            state.scene = Scene::MainMenu;
            state.menu.hovered = None;
            state.menu.focused = None;
        }
        self.set_grab(false);
    }

    fn mutate_settings(&mut self, f: impl FnOnce(&mut crate::settings::Settings)) {
        if let Some(state) = self.state.as_mut() {
            f(&mut state.settings);
            state.settings.save();
        }
    }

    /// Difficulty has no persisted-settings home (it rides the world): cycle it live on the session.
    fn cycle_difficulty(&mut self) {
        if let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
            session.difficulty = session.difficulty.next();
            session.player.difficulty = session.difficulty;
            session.game.set_difficulty(session.difficulty);
            if !session.difficulty.spawns_hostiles() {
                session.game.despawn_hostiles();
            }
        }
    }

    /// Click-to-set a slider: map the cursor X within the track (matching the builder's 14px pad) to
    /// `lo..hi`, snap integer-valued sliders, persist.
    fn slider_set(&mut self, id: crate::menu::WidgetId, rect: crate::menu::Rect, lo: f32, hi: f32) {
        use crate::menu::WidgetId as W;
        let Some(state) = self.state.as_mut() else { return };
        let pad = 14.0;
        let frac = ((state.menu.cursor.0 - (rect.x + pad)) / (rect.w - 2.0 * pad)).clamp(0.0, 1.0);
        let val = lo + frac * (hi - lo);
        let s = &mut state.settings;
        match id {
            W::FovSlider => s.fov = val.round(),
            W::SensSlider => s.sensitivity = val,
            W::RenderDistSlider => s.render_distance = val.round() as i32,
            W::SupersampleSlider => s.supersample = (val * 20.0).round() / 20.0, // 0.05 steps
            W::GiRaysSlider => s.gi_rays = val.round().max(1.0) as u32,
            _ => {}
        }
        s.save();
    }

    fn focus_field(&mut self, id: crate::menu::WidgetId) {
        use crate::menu::WidgetId as W;
        if let Some(state) = self.state.as_mut() {
            state.menu.focused = Some(id);
            state.menu.caret = match id {
                W::NameField => state.create_name.len(),
                W::SeedField => state.create_seed.len(),
                _ => 0,
            };
        }
    }

    /// Keyboard while a menu is showing: Esc backs out; otherwise feed a focused text field.
    fn menu_key(&mut self, event: &winit::event::KeyEvent, _event_loop: &ActiveEventLoop) {
        let code = match event.physical_key {
            PhysicalKey::Code(c) => Some(c),
            _ => None,
        };
        if code == Some(KeyCode::Escape) {
            self.menu_escape();
            return;
        }
        let focused = self.state.as_ref().and_then(|s| s.menu.focused).is_some();
        if !focused {
            return;
        }
        match code {
            Some(KeyCode::Backspace) => self.field_backspace(),
            Some(KeyCode::Tab) => self.field_focus_next(),
            Some(KeyCode::Enter) | Some(KeyCode::NumpadEnter) => {
                if let Some(state) = self.state.as_mut() {
                    state.menu.focused = None;
                }
            }
            _ => {
                if let Some(txt) = event.text.clone() {
                    self.field_insert(txt.as_str());
                }
            }
        }
    }

    /// Esc out of the current menu, back toward gameplay / the previous screen.
    fn menu_escape(&mut self) {
        let next = match self.state.as_ref().map(|s| &s.scene) {
            Some(Scene::Settings { return_to }) => (**return_to).clone(),
            Some(Scene::WorldSelect) => Scene::MainMenu,
            Some(Scene::CreateWorld) => Scene::WorldSelect,
            Some(Scene::Paused) => Scene::InGame,
            _ => return, // MainMenu / InGame: nothing to back out to
        };
        if matches!(next, Scene::InGame) {
            self.resume_game();
        } else {
            self.set_scene(next);
        }
    }

    fn field_insert(&mut self, txt: &str) {
        use crate::menu::WidgetId as W;
        let Some(state) = self.state.as_mut() else { return };
        let Some(field) = state.menu.focused else { return };
        let buf = match field {
            W::NameField => &mut state.create_name,
            W::SeedField => &mut state.create_seed,
            _ => return,
        };
        for ch in txt.chars() {
            let ok = match field {
                // ASCII only — the bitmap font renders 0x20..0x80 and would drop the rest while still
                // advancing the caret. Name: filename-ish (slugified on create); Seed: any printable
                // (parse_seed parses an int or hashes the bytes).
                W::NameField => buf.len() < 32 && (ch.is_ascii_alphanumeric() || " -_".contains(ch)),
                W::SeedField => buf.len() < 32 && ch.is_ascii_graphic(),
                _ => false,
            };
            if ok {
                buf.push(ch);
            }
        }
        state.menu.caret = buf.len();
    }

    fn field_backspace(&mut self) {
        use crate::menu::WidgetId as W;
        let Some(state) = self.state.as_mut() else { return };
        let Some(field) = state.menu.focused else { return };
        let buf = match field {
            W::NameField => &mut state.create_name,
            W::SeedField => &mut state.create_seed,
            _ => return,
        };
        buf.pop();
        state.menu.caret = buf.len();
    }

    fn field_focus_next(&mut self) {
        use crate::menu::WidgetId as W;
        let next = match self.state.as_ref().and_then(|s| s.menu.focused) {
            Some(W::NameField) => W::SeedField,
            _ => W::NameField,
        };
        self.focus_field(next);
    }

    fn frame_ingame(&mut self) {
        // S3 autosave: fires every AUTOSAVE_SECS of play. Peek immutably, save (save_world takes
        // &self), then reset the timer — all before the frame's mutable state borrow below.
        let autosave_due = self
            .state
            .as_ref()
            .and_then(|s| s.session.as_ref())
            .is_some_and(|se| se.autosave >= AUTOSAVE_SECS);
        if autosave_due {
            let ok = self.save_world();
            if ok {
                log::info!("autosaved");
            }
            if let Some(se) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
                se.autosave = 0.0;
                se.toast = Some((
                    if ok { "Autosaved".into() } else { "AUTOSAVE FAILED — see log".into() },
                    2.0,
                ));
            }
        }
        // S10: death → the death screen (pre-borrow peek, like autosave). The world freezes in
        // Scene::Dead; the drain/respawn moved into the Respawn button.
        let died = self
            .state
            .as_ref()
            .and_then(|s| s.session.as_ref())
            .is_some_and(|se| !se.player.creative && se.player.is_dead());
        if died {
            if let Some(state) = self.state.as_mut() {
                if let Some(audio) = &state.audio {
                    audio.play(crate::audio::Sfx::PlayerHurt, 1.0);
                }
                if let Some(se) = state.session.as_mut() {
                    se.input = Input::default(); // dying mid-sprint must not auto-run on respawn
                }
                state.scene = Scene::Dead;
            }
            self.apply_grab_for_scene();
            return;
        }
        let Some(state) = &mut self.state else { return };
        let now = Instant::now();
        let dt = (now - state.last_frame).as_secs_f32().min(0.1);
        state.last_frame = now;
        // The render stack stays `state.<field>`; the world is `session.<field>` (disjoint borrows).
        let Some(session) = state.session.as_mut() else { return };
        session.environment.update(dt);
        session.elapsed += dt;
        session.autosave += dt;
        // S9: ambient pads track day/night (crossfaded; a segment is always queued).
        if let Some(audio) = state.audio.as_mut() {
            audio.tick_music(session.environment.day_factor(), dt);
        }

        // Apply mouse look.
        let (yaw_d, pitch_d) = session.input.take_look();
        session.camera.yaw += yaw_d * SENSITIVITY;
        session.camera.pitch = (session.camera.pitch - pitch_d * SENSITIVITY).clamp(-1.5533, 1.5533);

        // Player physics (disjoint field borrows: player mut, game/input shared).
        let yaw = session.camera.yaw;
        let armor = session.inventory.equipped_armor();
        let game_ref = &session.game;
        // S2: deferred spawn validation. The eager on-load "buried" lift teleported every
        // underground survival save to the surface; the real test needs actual blocks (including
        // player edits), so it waits until every chunk the player AABB touches is resident, then
        // lifts only if the player is genuinely inside solid.
        if !session.spawn_checked {
            let p = session.player.position;
            let corners_loaded = [-0.3f32, 0.3].iter().all(|&dx| {
                [-0.3f32, 0.3].iter().all(|&dz| {
                    [0.0f32, 1.8].iter().all(|&dy| {
                        game_ref.chunk_loaded_at(
                            Vec3::new(p.x + dx, p.y + dy, p.z + dz).floor().as_ivec3(),
                        )
                    })
                })
            });
            if corners_loaded {
                if session.player.is_inside_solid(&|p| game_ref.block_state_at(p)) {
                    let surf = game_ref.spawn_surface_y(p.x as i32, p.z as i32);
                    log::warn!(
                        "loaded position ({:.0},{:.1},{:.0}) is inside solid blocks — lifting to the surface ({surf:.1})",
                        p.x, p.y, p.z
                    );
                    session.player.position.y = surf;
                    session.player.velocity = Vec3::ZERO;
                }
                session.spawn_checked = true;
            }
        }
        // S2: physics runs only once the chunks the player occupies AND will reach this frame are
        // resident — an unloaded chunk must not collide as air (fall-through-terrain). Flying
        // (creative) is exempt: it can outrun generation and has no fall-through consequence.
        let phys_ready = session.spawn_checked && {
            let feet = session.player.position;
            let dest = feet + session.player.velocity * dt;
            game_ref.chunk_loaded_at(feet.floor().as_ivec3())
                && game_ref.chunk_loaded_at(Vec3::new(feet.x, feet.y - 1.0, feet.z).floor().as_ivec3())
                && game_ref.chunk_loaded_at(dest.floor().as_ivec3())
        };
        if session.player.flying || phys_ready {
            // S2: substep the integration (max ~1/120 s per step) so jump/fall arcs are
            // frame-rate-independent; collision itself is swept inside move_axis regardless.
            let n = (dt * 120.0).ceil().max(1.0) as u32;
            let sub = dt / n as f32;
            for _ in 0..n {
                session.player.update(
                    sub,
                    yaw,
                    &session.input,
                    |p| game_ref.block_state_at(p),
                    armor,
                );
            }
        }
        // ── S9 body-feel audio: landings, footsteps, splashes, and the env-damage hurt edge ──
        {
            let on_ground = session.player.on_ground;
            let flying = session.player.flying;
            let pos = session.player.position;
            let vel = session.player.velocity;
            let health = session.player.health;
            if !on_ground {
                session.fall_top = session.fall_top.max(pos.y);
            }
            if let Some(audio) = &state.audio {
                if on_ground && !session.was_on_ground && !flying {
                    let fall = session.fall_top - pos.y;
                    if fall > 3.0 {
                        audio.play(crate::audio::Sfx::Land, (fall / 12.0).clamp(0.3, 1.0));
                    }
                }
                let hspeed = Vec3::new(vel.x, 0.0, vel.z).length();
                if on_ground && !flying && hspeed > 0.5 {
                    session.step_accum += hspeed * dt;
                    if session.step_accum >= 2.2 {
                        session.step_accum = 0.0;
                        let under =
                            session.game.block_at((pos - Vec3::Y * 0.5).floor().as_ivec3());
                        let (step, _) = crate::audio::block_sfx(under);
                        audio.play(step, 0.45);
                    }
                } else {
                    session.step_accum = 0.0;
                }
                let wet = block::is_fluid(session.game.block_at(pos.floor().as_ivec3()));
                if wet && !session.was_wet {
                    audio.play(crate::audio::Sfx::Splash, 0.8);
                }
                session.was_wet = wet;
                if health < session.prev_health - 0.05 {
                    audio.play(crate::audio::Sfx::PlayerHurt, 0.9);
                }
            }
            if health < session.prev_health - 0.05 {
                session.hurt_flash = 0.35; // S10 (fires with the hurt sound, env damage included)
            }
            if on_ground {
                session.fall_top = pos.y;
            }
            session.was_on_ground = on_ground;
            session.prev_health = health;
        }
        // S10: decay the transient feedback.
        session.hurt_flash = (session.hurt_flash - dt).max(0.0);
        if let Some((_, t)) = session.hurt_dir.as_mut() {
            *t -= dt;
            if *t <= 0.0 {
                session.hurt_dir = None;
            }
        }
        if let Some((_, t)) = session.toast.as_mut() {
            *t -= dt;
            if *t <= 0.0 {
                session.toast = None;
            }
        }
        session.camera.position = session.player.eye();
        // M34-VM5: optional camera head-bob, folded into the real camera transform BEFORE the camera
        // uniform rolls prev_view_proj — so DLSS motion vectors include it and stay correct. Uses the
        // previous frame's smoothed walk-bob phase (a one-frame lag is imperceptible).
        if session.viewbob {
            let b = session.view_model.camera_bob();
            let right = session.camera.forward().cross(Vec3::Y).normalize_or_zero();
            session.camera.position += right * b.x + Vec3::Y * b.y;
        }

        // Sprinting widens the FOV slightly (and swimming narrows it); ease toward the target.
        let target_fov = if session.player.flying {
            70_f32
        } else if session.player.submerged {
            66.0
        } else if session.input.sprint
            && session.player.can_sprint()
            && (session.input.forward || session.input.back)
        {
            78.0
        } else {
            70.0
        }
        .to_radians();
        session.camera.fovy += (target_fov - session.camera.fovy) * (10.0 * dt).min(1.0);

        // Stream chunks, advance fluids/entities; collect any picked-up item drops into inventory.
        let mut collected = session.game.update(
            &state.gpu,
            &state.renderer,
            session.player.position,
            dt,
            session.environment.day_factor(),
        );
        // U11: a sculk shriek this frame pulses Darkness onto the player (render effect deferred).
        let dark = session.game.take_pending_darkness();
        if dark > 0.0 {
            session.player.apply_darkness(dark);
        }
        let picked_any = !collected.items.is_empty();
        for stack in std::mem::take(&mut collected.items) {
            if let Some(leftover) = session.inventory.insert(stack) {
                // Inventory full — drop the remainder back so items aren't vacuum-deleted.
                session
                    .game
                    .spawn_item(session.player.position + Vec3::new(0.0, 1.0, 0.0), leftover);
            }
        }
        if collected.xp > 0 {
            session.player.add_xp(collected.xp);
        }
        // S9: pickup blips + the world-positioned sound feed from the entity/game tick.
        if let Some(audio) = &state.audio {
            if picked_any {
                audio.play(crate::audio::Sfx::ItemPop, 0.5);
            }
            if collected.xp > 0 {
                audio.play(crate::audio::Sfx::XpDing, 0.5);
            }
            let (cp, cf) = (session.camera.position, session.camera.forward());
            for (sfx, pos) in std::mem::take(&mut collected.sounds) {
                audio.play_at(sfx, 1.0, pos, cp, cf);
            }
        }
        // S10: particle bursts raised inside the entity tick (hearts...).
        for (b, pos) in std::mem::take(&mut collected.bursts) {
            session.game.spawn_burst(b, pos);
        }

        // P14 shield raise-timer: hold right-click with a SHIELD (no screen open) to raise it. Updated
        // HERE (above the damage loop) so shield_ready reflects this frame. Pre-empts eat/place below.
        // `sel_item` is computed once and reused by the bow/eat blocks (selection can't change mid-tick).
        let sel_item = session.inventory.selected_item();
        if session.screen == Screen::None && session.input.place_held && item::is_shield(sel_item) {
            if session.shield_item != Some(sel_item) {
                session.shield_item = Some(sel_item);
                session.shield_progress = 0.0;
            }
            session.shield_progress += dt;
        } else {
            // Released, swapped off the shield, or a screen opened → lower instantly.
            session.shield_progress = 0.0;
            session.shield_item = None;
        }

        // Combat damage from the entity tick. Each hit carries its source position; a RAISED, ready
        // shield fully blocks hits whose source is in the front arc (melee, arrows, blasts alike — MC
        // blocks the frontal source). Summed into ONE take_hit so the 0.5s i-frame applies as before
        // (a per-hit call would drop all but the first simultaneous source). Environmental damage
        // (fall/lava/drown/starve) bypasses Collected entirely, so a shield never blocks it.
        let shield_up = session.shield_item.is_some() && session.shield_progress >= SHIELD_RAISE_DELAY;
        let fwd = session.camera.forward();
        let mut incoming = 0.0;
        let mut strongest: Option<(f32, Vec3)> = None;
        for (amount, src) in std::mem::take(&mut collected.player_damage) {
            if shield_up && crate::player::shield_blocks(fwd, session.player.position, src) {
                continue; // fully blocked
            }
            if strongest.is_none_or(|(a, _)| amount > a) {
                strongest = Some((amount, src));
            }
            incoming += amount;
        }
        if incoming > 0.0 {
            let before = session.player.health;
            session.player.take_hit(incoming);
            if session.player.health < before - 0.05 {
                if let Some(audio) = &state.audio {
                    audio.play(crate::audio::Sfx::PlayerHurt, 0.9);
                }
                session.hurt_flash = 0.35; // S10
                if let Some((_, src)) = strongest {
                    // World angle of the attacker (yaw 0 = +X, toward +Z), shown 0.8 s.
                    let d = src - session.player.position;
                    session.hurt_dir = Some((d.z.atan2(d.x), 0.8));
                }
            }
            session.prev_health = session.player.health;
        }

        // Block targeting.
        let eye = session.camera.position;
        let fwd = session.camera.forward();
        let mut target = raycast::cast(eye, fwd, REACH, |p| session.game.is_targetable_at(p));

        // Melee: if a mob is nearer than the targeted block FACE, the left-click hits it (not the
        // block). Using the ray's face-hit distance (not the block center) means a mob standing
        // flush behind a block can't be struck through it.
        // P12 (1.9): the recharge is per-weapon (item::attack_speed). A swing always lands but is weak
        // until the meter refills — tap-spamming does ~20% damage, a charged hit does full.
        session.melee_cd = (session.melee_cd - dt).max(0.0);
        // Switching the held weapon resets the cooldown (vanilla gives a fresh full-power swing).
        if session.inventory.selected != session.melee_prev_sel {
            session.melee_prev_sel = session.inventory.selected;
            session.melee_cd = 0.0;
        }
        let block_dist = target.as_ref().map_or(REACH, |h| h.dist);
        let mob_dist = session.game.nearest_mob_hit(eye, fwd, REACH);
        let mob_in_way = session.screen == Screen::None
            && session.input.break_held
            && mob_dist.is_some_and(|md| md <= block_dist);
        // A swing fires on the click EDGE (so spamming yields partial-charge hits) or as an auto-swing
        // when LMB is held and the meter is already full.
        let pressed_now = session.input.break_held && !session.melee_was_held;
        let auto_full = session.input.break_held && session.melee_cd <= 0.0;
        if mob_in_way && (pressed_now || auto_full) {
            let sel = session.inventory.selected_item();
            let cd_max = 1.0 / item::attack_speed(sel);
            let charge = (1.0 - session.melee_cd / cd_max).clamp(0.0, 1.0);
            let mut dmg = item::charged_damage(item::attack_damage(sel), charge);
            // Critical hit (+50%): a falling, non-sprinting, grounded-feet-off attack.
            // S8: "sprinting" for combat rules means EFFECTIVE sprint (key + hunger gate) —
            // a starving player holding Ctrl is walking, so crits/sweeps behave accordingly.
            let sprinting = session.input.sprint && session.player.can_sprint();
            let crit = crate::player::is_critical_hit(
                session.player.on_ground,
                session.player.velocity.y,
                sprinting,
                session.player.submerged,
                session.player.flying,
            );
            if crit {
                dmg *= 1.5;
            }
            // Sweep: a full-charge sword swing standing on the ground (un-enchanted sweep = 1).
            let sweep = if charge >= 0.999
                && item::is_sword(sel)
                && session.player.on_ground
                && !sprinting
            {
                Some(1.0)
            } else {
                None
            };
            // Sprint-attack: extra horizontal knockback (and it can't crit — see is_critical_hit).
            let kb_mult = if sprinting && session.player.on_ground { 1.5 } else { 1.0 };
            let landed = session.game.attack_nearest(eye, fwd, REACH, dmg, crit, sweep, kb_mult);
            if landed && crit {
                // S10: crit stars where the hit connected.
                session
                    .game
                    .spawn_burst(crate::game::Burst::Crit, eye + fwd * mob_dist.unwrap_or(1.5));
            }
            if let Some(audio) = &state.audio {
                let at = eye + fwd * mob_dist.unwrap_or(1.5);
                audio.play_at(crate::audio::Sfx::MobHit, 0.9, at, eye, fwd); // S9
            }
            session.melee_cd = cd_max;
            session.player.add_exhaustion(crate::player::EXH_ATTACK); // S8 (no-op in creative)
            if !session.inventory.creative {
                session.inventory.damage_selected(1); // weapons wear from hitting
            }
        }
        if mob_in_way {
            session.mine_target = None;
            session.mine_progress = 0.0;
            target = None; // swinging at a mob, not mining — no block highlight
        }
        // Edge tracker for the click-to-swing logic; MUST update every frame (not just when a mob is
        // in reach) so a fresh press is detected correctly even across screen opens / focus loss.
        session.melee_was_held = session.input.break_held;

        // Mining: hold LMB to break the targeted block, timed by hardness (instant in creative).
        if session.screen == Screen::None && session.input.break_held && !mob_in_way {
            if let Some(hit) = target.as_ref().map(|h| h.block) {
                let id = session.game.block_at(hit);
                if block::breakable(id) {
                    if session.mine_target != Some(hit) {
                        session.mine_target = Some(hit);
                        session.mine_progress = 0.0;
                    }
                    let sel = session.inventory.selected_item();
                    let block_tool = block::tool_class(id);
                    // S8 vanilla formula: hardness is in vanilla units. canHarvest (right class AND
                    // sufficient tier, or no tool required) gives base 1.5x; a tool-requiring block
                    // without its proper tool takes the 5x penalty (stone by hand = 1.5*5 = 7.5 s).
                    // A class-matching tool divides by its tier speed either way (a wooden pick on
                    // diamond ore is still 5x-penalized AND dropless — the tier half of canHarvest).
                    let class_match = item::is_tool(sel)
                        && item::tool_class(sel) == block_tool
                        && block_tool != block::ToolClass::None;
                    let can_harvest = !block::requires_tool(id)
                        || (class_match
                            && item::harvest_level(item::tool_tier(sel))
                                >= block::required_harvest(id));
                    let mut time = block::hardness(id) * if can_harvest { 1.5 } else { 5.0 };
                    if class_match {
                        time /= item::tool_speed(item::tool_tier(sel));
                    }
                    if session.inventory.creative {
                        time = 0.0;
                    }
                    session.mine_progress += if time <= 0.0 { 1.0 } else { dt / time };
                    if session.mine_progress >= 1.0 {
                        // Capture the block state before clearing it (a double slab drops two).
                        let bstate = session.game.block_state_at(hit).1;
                        let removed =
                            session.game.set_block(&state.gpu, &state.renderer, hit, block::AIR);
                        if removed {
                            if let Some(audio) = &state.audio {
                                let (_, brk) = crate::audio::block_sfx(id);
                                let (cp, cf) = (session.camera.position, session.camera.forward());
                                audio.play_at(brk, 0.9, hit.as_vec3() + Vec3::splat(0.5), cp, cf);
                            }
                            // S10: textured shards of the broken block.
                            session
                                .game
                                .spawn_burst(crate::game::Burst::Break(id), hit.as_vec3() + Vec3::splat(0.5));
                        }
                        if removed && !session.inventory.creative {
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
                                // S6: random extras (bonus crop yield, tall-grass seeds).
                                let bonus = block::bonus_drops(id, bstate, session.game.next_rand());
                                // S13: gravel's flint REPLACES the gravel drop (vanilla semantics);
                                // every other bonus is additive to the base drop.
                                let replaced = bonus.is_some() && id == block::GRAVEL;
                                if !replaced {
                                    if let Some((drop_item, count)) = block::drops(id, bstate) {
                                        let stack = item::ItemStack::new(drop_item, count);
                                        session.game.spawn_item(center, stack);
                                    }
                                }
                                if let Some((bi, bc)) = bonus {
                                    session.game.spawn_item(center, item::ItemStack::new(bi, bc));
                                }
                                // Some ores release experience orbs when mined.
                                session.game.spawn_xp(center, block::mining_xp(id));
                            }
                            session.inventory.damage_selected(1);
                            session.player.add_exhaustion(crate::player::EXH_MINE); // S8
                        }
                        // S6: a crop can't float — breaking its support pops it as a drop.
                        if removed {
                            let above = hit + IVec3::Y;
                            let (aid, ast) = session.game.block_state_at(above);
                            if block::is_crop(aid) || aid == block::SAPLING {
                                session.game.set_block(&state.gpu, &state.renderer, above, block::AIR);
                                if !session.inventory.creative {
                                    if let Some((di, dc)) = block::drops(aid, ast) {
                                        session.game.spawn_item(
                                            above.as_vec3() + Vec3::splat(0.5),
                                            item::ItemStack::new(di, dc),
                                        );
                                    }
                                }
                            }
                        }
                        session.mine_target = None;
                        session.mine_progress = 0.0;
                        target = None; // don't flash a highlight on the now-air block this frame
                    }
                } else {
                    session.mine_target = None;
                }
            } else {
                session.mine_target = None;
            }
        } else {
            session.mine_target = None;
            session.mine_progress = 0.0;
        }

        // Bow (P13): hold right-click with a bow + ammo to DRAW (charge over BOW_DRAW_TIME); release
        // looses a gravity-arced arrow whose speed/damage scale with the draw. This pre-empts eating
        // and block placement. Combined into one block so the RELEASE (place_held just went false) is
        // detected and fires BEFORE the draw state is reset.
        let have_ammo = session.inventory.creative || session.inventory.has_item(item::ARROW);
        let bow_drawing = session.screen == Screen::None
            && session.input.place_held
            && item::is_bow(sel_item)
            && have_ammo;
        if session.draw_item.is_some() && !bow_drawing {
            // We were drawing and now aren't: fire on a genuine in-world release past the min draw;
            // a screen-open or weapon-swap cancels (those leave place_held/screen != the fire case).
            if session.screen == Screen::None
                && !session.input.place_held
                && session.draw_progress >= item::BOW_MIN_DRAW
            {
                let (speed, damage) = item::bow_shot(session.draw_progress / item::BOW_DRAW_TIME);
                let dir = session.camera.forward();
                let pos = session.camera.position + dir * 1.2; // muzzle clear of the player AABB
                session.game.spawn_player_arrow(pos, dir * speed, damage);
                if let Some(audio) = &state.audio {
                    audio.play(crate::audio::Sfx::BowShoot, 0.8); // S9
                }
                if !session.inventory.creative {
                    session.inventory.consume_item(item::ARROW);
                }
            }
            session.draw_progress = 0.0;
            session.draw_item = None;
        } else if bow_drawing {
            if session.draw_item != Some(sel_item) {
                session.draw_item = Some(sel_item);
                session.draw_progress = 0.0;
            }
            session.draw_progress = (session.draw_progress + dt).min(item::BOW_DRAW_TIME);
        }

        // Eating: hold right-click while a food item is selected to eat it over EAT_TIME, then
        // restore hunger/saturation and consume one. (Foods place nothing, so the place edge below
        // is a no-op for them.) Gameplay-only — suppressed while any screen is open or drawing a bow.
        let edible = session.screen == Screen::None
            && session.input.place_held
            && !item::is_bow(sel_item)
            && !item::is_shield(sel_item)
            && crate::food::food(sel_item).is_some_and(|f| session.player.can_eat(f.always_edible));
        if edible {
            if session.eat_item != Some(sel_item) {
                session.eat_item = Some(sel_item);
                session.eat_progress = 0.0;
            }
            let before_crunch = (session.eat_progress / 0.35) as i32;
            session.eat_progress += dt;
            if let Some(audio) = &state.audio {
                if (session.eat_progress / 0.35) as i32 > before_crunch {
                    audio.play(crate::audio::Sfx::Eat, 0.6); // S9: chewing
                }
            }
            if session.eat_progress >= EAT_TIME {
                if let Some(f) = crate::food::food(sel_item) {
                    session.player.eat(f.hunger, f.saturation);
                    session.inventory.consume_selected();
                    // S13: stew hands its bowl back (vanilla); spill it if the inventory is full.
                    if sel_item == item::MUSHROOM_STEW {
                        if let Some(left) =
                            session.inventory.insert(item::ItemStack::new(item::BOWL, 1))
                        {
                            session
                                .game
                                .spawn_item(session.player.position + Vec3::new(0.0, 1.0, 0.0), left);
                        }
                    }
                }
                if let Some(audio) = &state.audio {
                    audio.play(crate::audio::Sfx::Burp, 0.5); // S9
                }
                session.eat_progress = 0.0;
                session.eat_item = None;
            }
        } else {
            session.eat_progress = 0.0;
            session.eat_item = None;
        }

        // M34-VM3: a use/place click pulses one view-model swing (captured before the flag is consumed).
        // Bow/shield/food are excluded — they have their own use poses (VM4), not a swing.
        let vm_use_click = session.input.place_pressed
            && session.screen == Screen::None
            && !item::is_bow(sel_item)
            && !item::is_shield(sel_item)
            && crate::food::food(sel_item).is_none();
        if session.input.place_pressed {
            session.input.place_pressed = false;
            if item::is_bow(session.inventory.selected_item()) {
                // A bow press only starts the draw (handled above) — never place/interact.
            } else if item::is_shield(session.inventory.selected_item()) {
                // A shield press only starts the raise (handled above) — never place/interact.
            } else if session.game.try_feed_mob(eye, fwd, REACH, session.inventory.selected_item()) {
                // P17: fed a breedable animal its food → love-mode; consume one (takes priority over
                // placing, like melee's mob-precedence).
                session.inventory.consume_selected();
            } else if session.inventory.selected_item() == item::BUCKET
                && let Some(fcell) = {
                    // S12 bucket fill: a fluid-aware ray (the standard target ignores fluids, so
                    // open water yields no target at all). A solid hit first → None → fall through.
                    raycast::cast(eye, fwd, REACH, |p| {
                        session.game.is_targetable_at(p)
                            || block::is_fluid(session.game.block_at(p))
                    })
                    .filter(|h| block::is_fluid(session.game.block_at(h.block)))
                    .map(|h| h.block)
                }
            {
                let fid = session.game.block_at(fcell);
                session.game.set_block(&state.gpu, &state.renderer, fcell, block::AIR);
                if !session.inventory.creative {
                    let filled =
                        if fid == block::WATER { item::WATER_BUCKET } else { item::LAVA_BUCKET };
                    session.inventory.slots[session.inventory.selected] =
                        Some(item::ItemStack::new(filled, 1));
                }
                if fid == block::WATER {
                    if let Some(audio) = &state.audio {
                        audio.play_at(crate::audio::Sfx::Splash, 0.7, fcell.as_vec3() + Vec3::splat(0.5), session.camera.position, fwd);
                    }
                }
            } else if let Some(hit) = &target {
                let targeted = session.game.block_at(hit.block);
                if targeted == block::WOODEN_DOOR {
                    // Right-click toggles the door open/closed — both halves together.
                    let (_, st) = session.game.block_state_at(hit.block);
                    let half = block::door_half(st);
                    let (f, h, no) = (block::door_facing(st), block::door_hinge(st), !block::door_open(st));
                    let partner = if half == block::DOOR_LOWER { hit.block + IVec3::Y } else { hit.block - IVec3::Y };
                    let p_half = if half == block::DOOR_LOWER { block::DOOR_UPPER } else { block::DOOR_LOWER };
                    session.game.set_block_state(&state.gpu, &state.renderer, hit.block, block::WOODEN_DOOR, block::door_state(f, no, h, half));
                    session.game.set_block_state(&state.gpu, &state.renderer, partner, block::WOODEN_DOOR, block::door_state(f, no, h, p_half));
                    if let Some(audio) = &state.audio {
                        let sfx = if no { crate::audio::Sfx::DoorOpen } else { crate::audio::Sfx::DoorClose };
                        audio.play_at(sfx, 0.9, hit.block.as_vec3() + Vec3::splat(0.5), session.camera.position, session.camera.forward());
                    }
                } else if targeted == block::WOODEN_TRAPDOOR {
                    // Right-click toggles the trapdoor open/closed.
                    let (_, st) = session.game.block_state_at(hit.block);
                    let opening = !block::trapdoor_open(st);
                    let ns = block::trapdoor_state(block::trapdoor_facing(st), block::trapdoor_half(st), opening);
                    session.game.set_block_state(&state.gpu, &state.renderer, hit.block, block::WOODEN_TRAPDOOR, ns);
                    if let Some(audio) = &state.audio {
                        let sfx = if opening { crate::audio::Sfx::DoorOpen } else { crate::audio::Sfx::DoorClose };
                        audio.play_at(sfx, 0.8, hit.block.as_vec3() + Vec3::splat(0.5), session.camera.position, session.camera.forward());
                    }
                } else if targeted == block::BED {
                    // S11 sleep: night only, no monsters nearby; sets the respawn anchor and
                    // skips to just after sunrise (0.25 = sunrise; 0.30 clears the night gate).
                    if session.environment.day_factor() > 0.35 {
                        session.toast = Some(("You can only sleep at night".into(), 2.0));
                    } else if session.game.hostile_near(session.player.position, 12.0) {
                        session.toast =
                            Some(("You may not rest now — monsters are nearby".into(), 2.5));
                    } else {
                        session.environment.time = 0.30;
                        session.player.bed_respawn = Some(hit.block.as_vec3());
                        session.toast = Some(("Respawn point set".into(), 2.5));
                    }
                } else if targeted == block::CRAFTING_TABLE {
                    // Right-clicking a crafting table opens the 3x3 crafting screen instead.
                    session.pending_open = Some(Screen::Crafting);
                } else if targeted == block::FURNACE {
                    // Right-clicking a furnace opens its smelting screen.
                    session.pending_open = Some(Screen::Furnace(hit.block));
                } else if targeted == block::CHEST {
                    // Right-clicking a chest opens its 27-slot storage screen.
                    session.pending_open = Some(Screen::Chest(hit.block));
                } else if targeted == block::LEVER || targeted == block::BUTTON {
                    // Right-click flips a lever / presses a button. Cosmetic in P11 (it re-meshes the
                    // on/pressed state); the bit drives redstone in P31. The attach face is preserved.
                    let (_, st) = session.game.block_state_at(hit.block);
                    let ns = block::attach_state(block::attach_face(st), !block::attach_on(st));
                    session.game.set_block_state(&state.gpu, &state.renderer, hit.block, targeted, ns);
                    if let Some(audio) = &state.audio {
                        audio.play_at(crate::audio::Sfx::LeverClick, 0.7, hit.block.as_vec3() + Vec3::splat(0.5), session.camera.position, session.camera.forward());
                    }
                } else if matches!(
                    session.inventory.selected_item(),
                    item::WATER_BUCKET | item::LAVA_BUCKET
                ) && session.game.block_at(hit.block + hit.normal) == block::AIR
                {
                    // S12 pour: place the fluid as a live source; the bucket empties (survival).
                    let place = hit.block + hit.normal;
                    let fluid = if session.inventory.selected_item() == item::WATER_BUCKET {
                        block::WATER
                    } else {
                        block::LAVA
                    };
                    if session.game.set_block(&state.gpu, &state.renderer, place, fluid) {
                        session.game.add_fluid_source(place, fluid);
                        if !session.inventory.creative {
                            session.inventory.slots[session.inventory.selected] =
                                Some(item::ItemStack::new(item::BUCKET, 1));
                        }
                        if fluid == block::WATER {
                            if let Some(audio) = &state.audio {
                                audio.play_at(crate::audio::Sfx::Splash, 0.7, place.as_vec3() + Vec3::splat(0.5), session.camera.position, session.camera.forward());
                            }
                        }
                    }
                } else if session.inventory.selected_item() == item::BONE
                    && session.game.bonemeal(&state.gpu, &state.renderer, hit.block)
                {
                    // U9: bonemeal a growable block (budding amethyst / cave vine / moss); consume one.
                    session.inventory.consume_selected();
                } else if session.inventory.selected_item() == item::GLOW_BERRIES
                    && session.game.block_at(hit.block + hit.normal) == block::AIR
                    && block::is_opaque(session.game.block_at(hit.block + hit.normal + IVec3::Y))
                {
                    // U9: place glow berries hanging under a block (a berried cave-vine tip).
                    let place = hit.block + hit.normal;
                    session.game.set_block(&state.gpu, &state.renderer, place, block::CAVE_VINE_BERRIES);
                    session.inventory.consume_selected();
                } else if item::is_tool(session.inventory.selected_item())
                    && item::tool_class(session.inventory.selected_item()) == block::ToolClass::Hoe
                    && matches!(targeted, block::GRASS | block::DIRT)
                    && session.game.block_at(hit.block + IVec3::Y) == block::AIR
                {
                    // S6: the hoe tills grass/dirt (open above) into farmland; wears the tool.
                    if session.game.set_block(&state.gpu, &state.renderer, hit.block, block::FARMLAND) {
                        if let Some(audio) = &state.audio {
                            audio.play_at(crate::audio::Sfx::StepDirt, 0.8, hit.block.as_vec3() + Vec3::splat(0.5), session.camera.position, session.camera.forward());
                        }
                        if !session.inventory.creative {
                            session.inventory.damage_selected(1);
                        }
                    }
                } else if matches!(
                    session.inventory.selected_item(),
                    item::SEEDS | item::CARROT | item::POTATO
                ) && targeted == block::FARMLAND
                    && session.game.block_at(hit.block + IVec3::Y) == block::AIR
                {
                    // S6: plant on farmland — seeds grow wheat; carrots/potatoes plant themselves.
                    let crop = match session.inventory.selected_item() {
                        item::SEEDS => block::WHEAT_CROP,
                        item::CARROT => block::CARROT_CROP,
                        _ => block::POTATO_CROP,
                    };
                    if session.game.set_block(&state.gpu, &state.renderer, hit.block + IVec3::Y, crop) {
                        session.inventory.consume_selected();
                        if let Some(audio) = &state.audio {
                            audio.play_at(crate::audio::Sfx::StepLeaves, 0.6, hit.block.as_vec3() + Vec3::splat(0.5), session.camera.position, session.camera.forward());
                        }
                    }
                } else {
                    let id = session.inventory.selected_block();
                    if id == block::WOODEN_DOOR {
                        // 2-tall door: needs a solid block below + air in both cells. Facing from the
                        // camera; hinge defaults left (double-door auto-pairing deferred).
                        let lower = hit.block + hit.normal;
                        let upper = lower + IVec3::Y;
                        let support = block::is_solid(session.game.block_at(lower - IVec3::Y));
                        // Both cells must be empty AND not overlap the player — a closed door is solid,
                        // so without this you could seal a door panel inside yourself (aim at your feet).
                        let space = session.game.block_at(lower) == block::AIR
                            && session.game.block_at(upper) == block::AIR
                            && !session.player.intersects_block(lower)
                            && !session.player.intersects_block(upper);
                        if support && space {
                            let f = session.camera.forward();
                            let facing = if f.x.abs() > f.z.abs() {
                                if f.x > 0.0 { 1 } else { 3 }
                            } else if f.z > 0.0 {
                                0
                            } else {
                                2
                            };
                            session.game.set_block_state(&state.gpu, &state.renderer, lower, block::WOODEN_DOOR, block::door_state(facing, false, 0, block::DOOR_LOWER));
                            session.game.set_block_state(&state.gpu, &state.renderer, upper, block::WOODEN_DOOR, block::door_state(facing, false, 0, block::DOOR_UPPER));
                            session.inventory.consume_selected();
                        }
                    } else if id == block::BED {
                        // S11: 2-cell bed — foot at the placed cell, head one cell along the
                        // camera facing; both need support + air (door-pattern guards).
                        let foot = hit.block + hit.normal;
                        let f = session.camera.forward();
                        let facing: u8 = if f.x.abs() > f.z.abs() {
                            if f.x > 0.0 { 1 } else { 3 }
                        } else if f.z > 0.0 {
                            0
                        } else {
                            2
                        };
                        let (dx, dz) = match facing {
                            0 => (0, 1),
                            1 => (1, 0),
                            2 => (0, -1),
                            _ => (-1, 0),
                        };
                        let head = foot + IVec3::new(dx, 0, dz);
                        let support = block::is_solid(session.game.block_at(foot - IVec3::Y))
                            && block::is_solid(session.game.block_at(head - IVec3::Y));
                        let space = session.game.block_at(foot) == block::AIR
                            && session.game.block_at(head) == block::AIR
                            && !session.player.intersects_block(foot)
                            && !session.player.intersects_block(head);
                        if support && space {
                            session.game.set_block_state(&state.gpu, &state.renderer, foot, block::BED, block::bed_state(facing, false));
                            session.game.set_block_state(&state.gpu, &state.renderer, head, block::BED, block::bed_state(facing, true));
                            session.inventory.consume_selected();
                            if let Some(audio) = &state.audio {
                                audio.play_at(crate::audio::Sfx::Place, 0.7, foot.as_vec3() + Vec3::splat(0.5), session.camera.position, session.camera.forward());
                            }
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
                        let f = session.camera.forward();
                        let facing = if f.x.abs() > f.z.abs() {
                            if f.x > 0.0 { 1 } else { 3 }
                        } else if f.z > 0.0 {
                            0
                        } else {
                            2
                        };
                        // A closed trapdoor is solid — don't seal it inside the player (matches the
                        // generic placement guard).
                        if !session.player.intersects_block(place)
                            && session.game.set_block_state(&state.gpu, &state.renderer, place, block::WOODEN_TRAPDOOR, block::trapdoor_state(facing, half, false)) {
                            session.inventory.consume_selected();
                        }
                    } else {
                    let is_slab = matches!(id, block::STONE_SLAB | block::WOOD_SLAB);
                    // Double-slab merge: clicking the empty-half face of a matching single slab with
                    // the same slab item fills it into a full (double) block — no new neighbor block.
                    // hit.normal is the face normal pointing back at the camera: +Y = clicked the top
                    // face (merges a bottom slab), -Y = clicked the bottom face (merges a top slab).
                    let (tgt_id, tgt_state) = session.game.block_state_at(hit.block);
                    let merge = is_slab
                        && tgt_id == id
                        && ((block::slab_half(tgt_state) == block::SLAB_BOTTOM && hit.normal.y == 1)
                            || (block::slab_half(tgt_state) == block::SLAB_TOP && hit.normal.y == -1));
                    if merge {
                        if session.game.set_block_state(
                            &state.gpu,
                            &state.renderer,
                            hit.block,
                            id,
                            block::slab_state(block::SLAB_DOUBLE),
                        ) {
                            session.inventory.consume_selected();
                        }
                    } else {
                        let place = hit.block + hit.normal;
                        // Orientation/half state for the placed block.
                        let place_state = if id == block::STONE_STAIRS {
                            // Stairs orient by where the player faces (the high step rises away).
                            let f = session.camera.forward();
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
                        } else if matches!(id, block::LEAVES | block::AZALEA_LEAVES) {
                            1 // S7: player-placed leaves are persistent (never decay)
                        } else {
                            0
                        };
                        // Attach fixtures (torch/lever/button) are walk-through, so never reject their
                        // placement for overlapping the player even though is_solid (targetable) is true.
                        let blocks_player = block::is_solid(id)
                            && !block::is_attach(id)
                            && session.player.intersects_block(place);
                        // Torches/levers/buttons can't mount on a ceiling (no ATTACH_CEILING variant) —
                        // reject an underside (bottom-face) click instead of spawning a floating fixture.
                        let attach_invalid = block::is_attach(id) && hit.normal.y == -1;
                        // S6: crops only sit on farmland; S7: saplings only on grass/dirt.
                        let below_id = session.game.block_at(place - IVec3::Y);
                        let crop_unsupported = (block::is_crop(id) && below_id != block::FARMLAND)
                            || (id == block::SAPLING
                                && !matches!(
                                    below_id,
                                    block::GRASS | block::DIRT | block::ROOTED_DIRT
                                ));
                        if id != block::AIR
                            && !blocks_player
                            && !attach_invalid
                            && !crop_unsupported
                            && session.game.set_block_state(
                                &state.gpu,
                                &state.renderer,
                                place,
                                id,
                                place_state,
                            )
                        {
                            session.inventory.consume_selected();
                            if let Some(audio) = &state.audio {
                                audio.play_at(crate::audio::Sfx::Place, 0.7, place.as_vec3() + Vec3::splat(0.5), session.camera.position, session.camera.forward());
                            }
                            if block::is_fluid(id) {
                                // Placed water/lava becomes a flowing source.
                                session.game.add_fluid_source(place, id);
                            }
                        }
                    }
                    }
                }
            }
        }

        // Q: drop one of the selected item ahead of the camera.
        if session.input.drop_pressed {
            session.input.drop_pressed = false;
            if let Some(dropped) = session.inventory.drop_one_selected() {
                let pos = session.camera.position + session.camera.forward() * 1.2;
                session.game.spawn_item(pos, dropped);
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
            session.camera.yaw += 0.012;
        }
        let aspect = state.gpu.aspect();
        session.camera_uniform.update(&session.camera, aspect);
        session
            .camera_uniform
            .set_environment(&session.environment, state.quality.fog_start(), state.quality.fog_end());
        session.camera_uniform.set_time(session.elapsed, session.environment.time);
        // M33-G8: jitter the projection to match the jitter handed to NGX (DLSS temporal sampling).
        if let Some(dr) = &state.dlss_render {
            let (rw, rh) = dr.render_dims();
            session.camera_uniform.apply_jitter(dr.jitter(), rw, rh);
        }
        state.renderer.update_camera(&state.gpu, &session.camera_uniform);
        state.renderer.set_sky(session.environment.wgpu_clear());

        let frustum = Frustum::from_view_proj(session.camera.view_proj(aspect));
        let entity_mesh = session.game.build_entity_mesh(&state.gpu, &state.renderer);
        let particle_mesh =
            session.game.build_particle_mesh(&state.gpu, &state.renderer, session.camera.forward());
        let mut visible = session.game.visible_meshes(&frustum);
        if let Some(em) = &entity_mesh {
            visible.push(em);
        }
        if let Some(pm) = &particle_mesh {
            visible.push(pm); // S10 (no BLAS — the TLAS ignores dynamic meshes)
        }
        let highlight = target.as_ref().map(|h| {
            let prog = if session.mine_target == Some(h.block) {
                session.mine_progress
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
        let debug_lines = if session.debug_f3 {
            let p = session.player.position;
            Some(build_debug_lines(
                state.fps_smooth,
                p,
                session.camera.forward(),
                session.camera.yaw,
                session.camera.pitch,
                session.game.biome_name_at(p.x.floor() as i32, p.z.floor() as i32),
                session.game.loaded_chunk_count(),
                session.game.mesh_count(),
                session.game.entity_count(),
                session.player.flying,
                session.game.rtx_mode_name(),
                session.difficulty.name(),
            ))
        } else {
            None
        };
        let attack_charge = {
            let cd_max = 1.0 / item::attack_speed(session.inventory.selected_item());
            if cd_max > 0.0 {
                (1.0 - session.melee_cd / cd_max).clamp(0.0, 1.0)
            } else {
                1.0
            }
        };
        let draw_charge = if item::is_bow(session.inventory.selected_item()) {
            (session.draw_progress / item::BOW_DRAW_TIME).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let shield_charge = if session.shield_item.is_some() {
            (session.shield_progress / SHIELD_RAISE_DELAY).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut ui = overlay::build_ui(
            state.gpu.config.width,
            state.gpu.config.height,
            &session.inventory,
            session.player.health,
            session.player.hunger,
            !session.player.creative,
            session.inventory.equipped_armor(),
            session.player.air_fraction(),
            session.player.submerged,
            session.player.level,
            session.player.xp_fraction(),
            attack_charge,
            draw_charge,
            shield_charge,
            {
                // F3: a pulsing Darkness dim while the Warden's effect is on the player (eases out
                // over the final second; throbs like Minecraft's Darkness).
                let d = session.player.darkness;
                if d > 0.0 {
                    d.min(1.0) * (0.45 + 0.4 * (0.5 - 0.5 * (session.elapsed * 3.5).cos()))
                } else {
                    0.0
                }
            },
            session.hurt_flash,
            // Camera-relative attacker angle (yaw 0 = +X): rel 0 = ahead of the crosshair.
            session.hurt_dir.map(|(world, t)| (world - session.camera.yaw, (t / 0.8).clamp(0.0, 1.0))),
            session.toast.as_ref().map(|(m, _)| m.as_str()),
            debug_lines.as_deref(),
        );
        if let Screen::Chest(pos) = session.screen {
            let slots = session
                .game
                .chest(pos)
                .map(|c| c.slots.clone())
                .unwrap_or_else(|| vec![None; crate::container::CHEST_SLOTS]);
            ui.extend(overlay::build_chest_screen(
                state.gpu.config.width,
                state.gpu.config.height,
                &session.inventory,
                &slots,
                session.cursor,
            ));
        } else if let Screen::Furnace(pos) = session.screen {
            let f = session.game.furnace(pos);
            let burn_frac = f.map_or(0.0, |f| if f.burn_max > 0.0 { f.burn_remaining / f.burn_max } else { 0.0 });
            let cook_frac = f.map_or(0.0, |f| f.cook_progress / smelting::SMELT_TIME);
            ui.extend(overlay::build_furnace_screen(
                state.gpu.config.width,
                state.gpu.config.height,
                &session.inventory,
                f.and_then(|f| f.input),
                f.and_then(|f| f.fuel),
                f.and_then(|f| f.output),
                burn_frac,
                cook_frac,
                session.cursor,
            ));
        } else if session.screen != Screen::None {
            // F1: in the creative inventory screen, show the paged block/tool palette instead of craft.
            let palette = (session.screen == Screen::Inventory && session.inventory.creative)
                .then_some((session.creative_palette.as_slice(), session.creative_page));
            ui.extend(overlay::build_inventory_screen(
                state.gpu.config.width,
                state.gpu.config.height,
                &session.inventory,
                &session.craft,
                session.screen.craft_size(),
                session.cursor,
                palette,
            ));
        }
        let volume_bg = session.game.volume_bind_group();
        let as_bg = if state.renderer.use_hw_rt() {
            let all = session.game.all_meshes();
            state
                .rt_scene
                .rebuild(&state.gpu, &all)
                .map(|t| state.renderer.make_as_bind_group(&state.gpu.device, t))
        } else {
            None
        };
        // M34-VM3/VM4: advance the view-model — swing while mining/attacking/using, plus the active
        // eat/draw/shield use pose from the existing action timers.
        let swing_loop = session.input.break_held && (session.mine_target.is_some() || mob_in_way);
        let use_pose = crate::viewmodel::UsePose {
            eat: (session.eat_progress / EAT_TIME).clamp(0.0, 1.0),
            draw: if item::is_bow(session.inventory.selected_item()) {
                (session.draw_progress / item::BOW_DRAW_TIME).clamp(0.0, 1.0)
            } else {
                0.0
            },
            shield: if session.shield_item.is_some() {
                (session.shield_progress / SHIELD_RAISE_DELAY).clamp(0.0, 1.0)
            } else {
                0.0
            },
        };
        let horiz_speed =
            Vec3::new(session.player.velocity.x, 0.0, session.player.velocity.z).length();
        session.view_model.update(
            dt,
            session.inventory.selected_item(),
            horiz_speed,
            session.player.on_ground,
            yaw_d,
            pitch_d,
            vm_use_click,
            swing_loop,
            use_pose,
        );
        // M34-VM: build the first-person held-item geometry (view space) + its projection/brightness
        // uniform. Brightness samples the local block light so the item dims in caves.
        let vm_light = session.game.held_item_light(
            session.camera.position.floor().as_ivec3(),
            session.environment.day_factor(),
        );
        let vm_geom = session
            .view_model
            .build_geometry(session.inventory.selected_item(), vm_light);
        let vm_part = vm_geom
            .as_ref()
            .and_then(|g| state.renderer.upload_viewmodel(&state.gpu.device, g));
        let vm_uniform = crate::viewmodel::ViewModelUniform::new(aspect, vm_light);
        let viewmodel = vm_part.as_ref().map(|p| (p, vm_uniform));
        let frame_submitted = state.renderer.render_frame(
            &state.gpu,
            &state.targets,
            &visible,
            volume_bg,
            as_bg.as_ref(),
            highlight,
            &ui,
            viewmodel,
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
                session.game.loaded_chunk_count(),
                session.game.mesh_count(),
                visible.len(),
                session.game.entity_count(),
                if session.player.flying { "fly" } else { "walk" }
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

/// Resolve the Create-World seed field to a `u64` — called exactly once per create so the saved header
/// and the session seed can't diverge. Order: a signed integer (handles a leading `-`), then an
/// unsigned integer (the `> i64::MAX` range), else a **stable inline FNV-1a** hash of the bytes (chosen
/// over a dependency hasher so a crate bump can never silently change terrain from a typed text seed).
/// An empty field yields a clock-derived nonzero seed; overlong digit strings (`> u64::MAX`) hash.
fn parse_seed(s: &str) -> u64 {
    let t = s.trim();
    if t.is_empty() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        return (nanos ^ (nanos >> 32)).wrapping_add(0x9E37_79B9_7F4A_7C15) | 1; // never 0
    }
    if let Ok(n) = t.parse::<i64>() {
        return n as u64;
    }
    if let Ok(n) = t.parse::<u64>() {
        return n;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in t.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// What world a `Session` is created from (M35).
enum WorldSource {
    /// Load the saved world (or a fresh default world if there's no save yet).
    Load,
    /// A brand-new world with this seed + game mode — a fresh inventory, ignoring any existing save.
    New { seed: u64, mode: crate::rules::GameMode },
}

/// Build the in-game `Session` (world + player + camera + everything that exists only while playing).
/// Called on Play / Create New; the render stack (gpu/renderer/quality/dlss) already lives on `State`.
fn create_session(
    gpu: &Gpu,
    renderer: &ChunkRenderer,
    quality: &crate::quality::Quality,
    dir: &Path,
    source: WorldSource,
) -> Session {
    let default_spawn = Vec3::new(8.0, 96.0, 24.0);
    let new_mode = match source {
        WorldSource::New { mode, .. } => Some(mode),
        WorldSource::Load => None,
    };
    let new_seed = match source {
        WorldSource::New { seed, .. } => Some(seed),
        WorldSource::Load => None,
    };
    // A fresh world ignores any existing save; Load reads level.bin (or a fresh default if absent).
    let level = if new_seed.is_some() { None } else { persistence::load_level(dir) };
    let (seed, mut spawn, yaw, pitch, time, flying) = match (new_seed, &level) {
        (Some(seed), _) => {
            let fly = new_mode.is_some_and(|m| m.is_creative());
            (seed, default_spawn, -std::f32::consts::FRAC_PI_2, -0.30, 0.34, fly)
        }
        (None, Some(l)) => (l.seed, Vec3::from(l.spawn), l.yaw, l.pitch, l.time, l.flying),
        (None, None) => (SEED, default_spawn, -std::f32::consts::FRAC_PI_2, -0.30, 0.34, true),
    };
    let saved = if new_seed.is_some() {
        FxHashMap::default()
    } else {
        persistence::load_chunks(dir)
    };
    let (inventory, furnaces, chests, entities) = if let Some(mode) = new_mode {
        (Inventory::new(mode.is_creative()), Vec::new(), Vec::new(), Vec::new())
    } else {
        // Pre-first-save worlds have no save_state.bin: seed the creative flag from the level
        // header's mode byte (S1; legacy headers default to creative). save_state's own byte wins
        // once it exists.
        let creative_default = level
            .as_ref()
            .map(|l| crate::rules::GameMode::from_u8(l.mode).is_creative())
            .unwrap_or(flying);
        persistence::load_state(dir, creative_default)
    };
    let mut game = Game::new(gpu, renderer.volume_bgl(), seed, quality, saved);
    game.restore_furnaces(furnaces);
    game.restore_chests(chests);
    // S3: restored entities stay frozen (no physics/AI) until their chunks stream in.
    game.restore_entities(entities);
    // Spawn validation: only a below-world / non-finite position lifts eagerly. "Below the
    // worldgen surface" is NOT buried — an underground save (a cave home) is legitimate; the S2
    // deferred occupancy check in frame_ingame judges against real loaded blocks and lifts only
    // if the player is genuinely inside solid.
    let surf = game.spawn_surface_y(spawn.x as i32, spawn.z as i32);
    let below_world = !spawn.y.is_finite() || spawn.y < 2.0;
    if level.is_none() {
        // Fresh world: spawn on dry land (S1 — a non-flying spawn on an ocean column would start
        // the game on the seabed with a drowning timer).
        let (sx, sz) = game.find_dry_spawn(spawn.x as i32, spawn.z as i32);
        spawn = Vec3::new(sx as f32 + 0.5, game.spawn_surface_y(sx, sz), sz as f32 + 0.5);
    } else if below_world {
        log::warn!(
            "loaded spawn y={:.1} at ({:.0},{:.0}) is out-of-bounds — lifting to the surface ({surf:.1})",
            spawn.y, spawn.x, spawn.z
        );
        spawn.y = surf;
    }
    let difficulty = std::env::var("VOXELCRAFT_DIFFICULTY")
        .ok()
        .and_then(|s| crate::rules::Difficulty::from_env(&s))
        .or_else(|| level.as_ref().map(|l| crate::rules::Difficulty::from_u8(l.difficulty)))
        .unwrap_or_default();
    game.set_difficulty(difficulty);
    if level.is_none() {
        // Commit the validated spawn (+ mode) immediately, so a crash before the first manual save
        // reloads exactly here instead of at the canned header position (mid-air or underwater).
        let _ = persistence::save_level(dir, &Level {
            seed,
            spawn: spawn.to_array(),
            yaw,
            pitch,
            time,
            flying,
            health: 20.0,
            hunger: 20.0,
            air: 11.0,
            saturation: 5.0,
            xp: 0.0,
            level: 0,
            difficulty: difficulty.as_u8(),
            mode: inventory.creative as u8,
            bed: None,
        });
    }
    if let Ok(Ok(rays)) = std::env::var("VOXELCRAFT_GI_RAYS").map(|s| s.trim().parse::<u32>()) {
        game.set_rtx_quality(rays.max(1));
    }
    let environment = Environment::new(time);
    let mut player = Player::new(spawn, flying);
    player.creative = inventory.creative;
    player.difficulty = difficulty;
    if let Some(l) = &level {
        // S10: never load INTO death — a save written at 0 HP (crash on the death screen) revives
        // with fresh stats at the saved position instead of instantly re-dying on frame one.
        let health = if l.health <= 0.0 { 20.0 } else { l.health };
        player.restore_state(health, l.hunger, l.air, l.saturation, l.xp, l.level);
        player.bed_respawn = l.bed.map(Vec3::from); // S11
    }
    let spawn_health = player.health;
    let camera = Camera::new(player.eye(), yaw, pitch);
    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update(&camera, gpu.aspect());
    camera_uniform.set_environment(&environment, quality.fog_start(), quality.fog_end());
    renderer.update_camera(gpu, &camera_uniform);
    renderer.set_sky(environment.wgpu_clear());
    Session {
        world_dir: dir.to_path_buf(),
        game,
        camera,
        player,
        input: Input::default(),
        inventory,
        environment,
        camera_uniform,
        debug_f3: false,
        elapsed: 0.0,
        screen: Screen::None,
        cursor: (0.0, 0.0),
        craft: [None; 9],
        pending_open: None,
        mine_target: None,
        mine_progress: 0.0,
        spawn_checked: false,
        autosave: 0.0,
        prev_health: spawn_health,
        fall_top: spawn.y,
        was_on_ground: false,
        was_wet: false,
        step_accum: 0.0,
        hurt_flash: 0.0,
        hurt_dir: None,
        toast: None,
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
        creative_palette: item::creative_palette(),
        creative_page: 0,
        view_model: crate::viewmodel::ViewModel::default(),
        viewbob: std::env::var("VOXELCRAFT_VIEWBOB").map_or(true, |v| v != "0" && v != "off"),
    }
}

fn maybe_select(session: &mut Session, pressed: bool, index: usize) {
    if pressed {
        session.inventory.select(index);
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
fn craft_take_output(session: &mut Session) {
    let ids: [u16; 9] = std::array::from_fn(|i| session.craft[i].map(|s| s.item).unwrap_or(0));
    let Some((out_item, out_count)) = crafting::match_grid(&ids) else {
        return;
    };
    let held_ok = match session.inventory.held {
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
    for cell in session.craft.iter_mut() {
        if let Some(s) = cell {
            s.count -= 1;
            if s.count == 0 {
                *cell = None;
            }
        }
    }
    session.inventory.held = Some(match session.inventory.held {
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
        mode: 0,       // Survival
        bed: Some([7.0, 99.0, 7.0]),
    };
    let mut chest_slots = vec![None; crate::container::CHEST_SLOTS];
    chest_slots[2] = Some(item::ItemStack::new(block::DIAMOND_ORE, 9));
    let chests = vec![persistence::ChestSave {
        pos: IVec3::new(-5, 70, 8),
        slots: chest_slots,
    }];
    // S3: a mob + a dropped item + an XP orb ride the new trailing entity section.
    let entities = vec![
        persistence::EntitySave::Mob {
            species: crate::entity::Species::Cow.to_u8(),
            pos: [3.0, 100.0, 4.0],
            health: 7.5,
            baby: true,
            growth: 42.0,
            angry: false,
            slime_tier: 0,
        },
        persistence::EntitySave::Item {
            pos: [1.0, 99.0, 1.0],
            stack: item::ItemStack::new(item::item_of_block(block::COBBLESTONE), 17),
            age: 12.0,
        },
        persistence::EntitySave::Xp { pos: [2.0, 99.0, 2.0], amount: 9 },
    ];
    persistence::save_state(&dir, &inv, &furnaces, &chests, &entities).unwrap();
    persistence::save_level(&dir, &level).unwrap();

    let (linv, lf, lc, le) = persistence::load_state(&dir, true);
    let ll = persistence::load_level(&dir).unwrap();
    let ents_ok = le.len() == 3
        && matches!(le[0], persistence::EntitySave::Mob { health, baby: true, .. } if (health - 7.5).abs() < 1e-6)
        && matches!(le[1], persistence::EntitySave::Item { stack, .. } if stack.count == 17)
        && matches!(le[2], persistence::EntitySave::Xp { amount: 9, .. });
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
        && ll.difficulty == 3
        && ll.bed == Some([7.0, 99.0, 7.0])
        && ents_ok;
    let _ = std::fs::remove_dir_all(&dir);
    if ok {
        log::info!("PERSIST_TEST: PASS — inventory + armor + furnace + survival + entities all round-tripped on disk");
    } else {
        log::error!("PERSIST_TEST: FAIL — state did not survive save/load (entities ok: {ents_ok})");
    }
}

/// On screen close, return any items left in the craft grid to the inventory (or drop them).
fn return_craft_to_inventory(session: &mut Session) {
    let pos = session.player.position + Vec3::new(0.0, 1.0, 0.0);
    for cell in session.craft.iter_mut() {
        if let Some(stack) = cell.take() {
            if let Some(left) = session.inventory.insert(stack) {
                session.game.spawn_item(pos, left);
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

        // S13: VOXELCRAFT_ATLAS=path.png dumps the procedural texture atlas (all 256 tiles at
        // native 16px) and exits — the human-free way to verify new tile art without staging items.
        if let Ok(path) = std::env::var("VOXELCRAFT_ATLAS") {
            let rgba = crate::texture::build_atlas();
            image::save_buffer(
                &path,
                &rgba,
                crate::texture::ATLAS_W,
                crate::texture::ATLAS_H,
                image::ColorType::Rgba8,
            )
            .expect("failed to save atlas PNG");
            log::info!("Saved atlas: {path}");
            event_loop.exit();
            return;
        }

        // M35: VOXELCRAFT_MENU=main|worlds|create|settings|pause renders one menu screen to a PNG (no
        // world) so menu layouts can be verified headlessly. Path = VOXELCRAFT_SHOT or "menu.png".
        if let Ok(which) = std::env::var("VOXELCRAFT_MENU") {
            let (w, h) = (gpu.config.width as f32, gpu.config.height as f32);
            let ui = crate::menu::UiState { cursor: (-1.0, -1.0), ..Default::default() };
            let settings = crate::settings::Settings::default();
            // Inline dummies (no disk I/O / no list_worlds → no migration side-effect during a screenshot).
            let worlds = vec![
                crate::persistence::WorldInfo { dir: "saves/worlds/demo".into(), name: "My World".into(), seed: SEED },
                crate::persistence::WorldInfo { dir: "saves/worlds/flat".into(), name: "Flatlands".into(), seed: 42 },
            ];
            let (verts, _rects) = match which.as_str() {
                "worlds" => crate::menu::build_world_select(w, h, &worlds, 0, None, &ui),
                "create" => {
                    crate::menu::build_create(w, h, "New World", "", crate::rules::GameMode::default(), &ui)
                }
                "settings" => crate::menu::build_settings(
                    w,
                    h,
                    &settings,
                    crate::rules::Difficulty::Normal,
                    crate::menu::SettingsTab::Graphics,
                    &ui,
                ),
                "pause" => crate::menu::build_pause(w, h, &ui),
                "dead" => crate::menu::build_death(w, h, &ui), // S10
                _ => crate::menu::build_main(w, h, &ui),
            };
            let path = std::env::var("VOXELCRAFT_SHOT").unwrap_or_else(|_| "menu.png".into());
            capture::screenshot_ui(&gpu, &renderer, &verts, gpu.config.width, gpu.config.height, &path);
            event_loop.exit();
            return;
        }

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
                .unwrap_or_else(|| {
                    // Spawn on the (deepened) surface of the spawn column, not a fixed y96.
                    let sy = game.spawn_surface_y(8, 24);
                    (Vec3::new(8.0, sy, 24.0), -std::f32::consts::FRAC_PI_2, -0.30)
                });
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

            // Debug: VOXELCRAFT_SCULK="x,y,z,amount" seeds a sculk-spread charge (U10 verification);
            // seeded BEFORE the ticks so VOXELCRAFT_TICKS drives the spread over the exposed rock.
            if let Ok(s) = std::env::var("VOXELCRAFT_SCULK") {
                let v: Vec<i32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                if v.len() == 4 {
                    game.debug_seed_sculk(IVec3::new(v[0], v[1], v[2]), v[3] as u16);
                }
            }

            // Debug: VOXELCRAFT_TICKS=N advances the game N update steps before the shot — verifies
            // time-based features (U9 amethyst/dripstone/cave-vine growth, fluids). Pair with PLACE.
            if let Ok(Ok(n)) = std::env::var("VOXELCRAFT_TICKS").map(|s| s.trim().parse::<u32>()) {
                for _ in 0..n {
                    let _ = game.update(&gpu, &renderer, player.position, 1.0 / 60.0, 1.0);
                }
            }

            // Debug: VOXELCRAFT_BURST="x,y,z,kind;..." stages particle bursts for the shot
            // (kind: break:<block_id> | smoke | hearts | crit), advanced a few sim ticks so the
            // spray has opened up. Plain shots never run game.update, so step here.
            if let Ok(spec) = std::env::var("VOXELCRAFT_BURST") {
                for part in spec.split(';') {
                    let f: Vec<&str> = part.split(',').map(|t| t.trim()).collect();
                    if f.len() == 4 {
                        let (Ok(x), Ok(y), Ok(z)) =
                            (f[0].parse::<f32>(), f[1].parse::<f32>(), f[2].parse::<f32>())
                        else {
                            continue;
                        };
                        let kind = match f[3] {
                            "smoke" => Some(crate::game::Burst::Smoke),
                            "hearts" => Some(crate::game::Burst::Hearts),
                            "crit" => Some(crate::game::Burst::Crit),
                            k => k
                                .strip_prefix("break:")
                                .and_then(|id| id.parse().ok())
                                .map(crate::game::Burst::Break),
                        };
                        if let Some(k) = kind {
                            game.spawn_burst(k, Vec3::new(x, y, z));
                        }
                    }
                }
                for _ in 0..6 {
                    game.step_particles_debug(0.03);
                }
            }

            // Debug: VOXELCRAFT_BONEMEAL="x,y,z;..." bonemeals each position (and its neighbors) a few
            // rounds — U9 growth verification: budding amethyst sprouts buds that mature into clusters.
            if let Ok(s) = std::env::var("VOXELCRAFT_BONEMEAL") {
                for spec in s.split(';') {
                    let v: Vec<i32> = spec.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                    if v.len() == 3 {
                        let p = IVec3::new(v[0], v[1], v[2]);
                        for _ in 0..5 {
                            for d in [
                                IVec3::ZERO,
                                IVec3::X,
                                IVec3::NEG_X,
                                IVec3::Y,
                                IVec3::NEG_Y,
                                IVec3::Z,
                                IVec3::NEG_Z,
                            ] {
                                game.bonemeal(&gpu, &renderer, p + d);
                            }
                        }
                    }
                }
            }

            // Debug: VOXELCRAFT_MOBS=1 spawns one of each species on a flat stage and ticks them a
            // moment to settle (M27 mob verification). Note: with the player within ~14 blocks and
            // line-of-sight, hostile species will already be in Chase/Attack in the logged tally —
            // that exercises the M28 AI rather than showing a neutral idle state.
            // Debug: VOXELCRAFT_WARDEN=1 builds a small daylit platform high above the (deepened)
            // surface and spawns a single Warden for a clean portrait (U11 verification).
            if std::env::var("VOXELCRAFT_WARDEN").is_ok() {
                for z in 20..=30 {
                    for x in 2..=14 {
                        game.set_block(&gpu, &renderer, IVec3::new(x, 159, z), block::STONE);
                    }
                }
                game.spawn_mob(Vec3::new(8.0, 160.0, 24.0), crate::entity::Species::Warden);
                // A couple of ticks to settle onto the platform (emerge keeps the Warden immobile).
                for _ in 0..2 {
                    let _ = game.update(&gpu, &renderer, player.position, 1.0 / 60.0, 1.0);
                }
            }

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

            // Debug: VOXELCRAFT_SPAWN=1 exercises the spawn rules (P16 light/biome gates + S5
            // underground sampling): daytime hostiles must ALL be underground; caps hold.
            if std::env::var("VOXELCRAFT_SPAWN").is_ok() {
                for _ in 0..160 {
                    game.try_spawn(1.0, player.position); // day → passive packs + CAVE hostiles only
                }
                let underground = game.hostile_count() - game.hostiles_at_surface();
                log::info!(
                    "S5 day spawns: {} (passive_count={}, cave hostiles={underground})",
                    game.mob_species_summary(),
                    game.passive_count()
                );
                assert!(game.passive_count() <= 8, "passive cap holds");
                assert_eq!(
                    game.hostiles_at_surface(),
                    0,
                    "daytime hostiles must spawn underground only (S5)"
                );
                game.despawn_all_mobs();
                for _ in 0..160 {
                    game.try_spawn(0.0, player.position); // night → surface AND cave hostiles
                }
                log::info!(
                    "S5 night spawns: {} (hostile_count={}, at surface={})",
                    game.mob_species_summary(),
                    game.hostile_count(),
                    game.hostiles_at_surface()
                );
                assert!(game.hostile_count() <= crate::game::HOSTILE_CAP, "hostile cap holds");
                // Despawn-radius: a tick with the player teleported far culls the distant mobs.
                let _ = game.update(&gpu, &renderer, Vec3::splat(5000.0), 0.1, 0.0);
                log::info!("S5 after far tick (despawn): {}", game.mob_species_summary());
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
            // M34-VM: VOXELCRAFT_HELD=<item_id> forces the held item so the view-model is screenshot-
            // verifiable (default selected slot holds a diamond pickaxe — a non-block, no VM1 geometry).
            if let Some(id) = std::env::var("VOXELCRAFT_HELD")
                .ok()
                .and_then(|s| s.trim().parse::<item::ItemId>().ok())
            {
                shot_inv.slots[shot_inv.selected] = Some(item::ItemStack::new(id, 1));
            }
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
                if std::env::var("VOXELCRAFT_DARK").is_ok() { 0.7 } else { 0.0 }, // F3 Darkness demo
                if survival_demo { 0.2 } else { 0.0 }, // S10: hurt flash visible in the demo HUD
                if survival_demo { Some((0.8, 0.8)) } else { None }, // S10: attacker cue demo
                survival_demo.then_some("Autosaved"), // S10: toast demo
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
                // F1: show the creative palette in the headless inventory shot (shot_inv is creative).
                let shot_pal = item::creative_palette();
                ui.extend(overlay::build_inventory_screen(
                    gpu.config.width,
                    gpu.config.height,
                    &shot_inv,
                    &shot_craft,
                    size,
                    cursor,
                    Some((&shot_pal, 0)),
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
            let mut all = game.all_meshes();
            // Include mobs/items/xp in the headless shot (the entity mesh was built above but the
            // screenshot previously only drew chunk meshes — so mobs never appeared in shots).
            if let Some(em) = &entity_mesh {
                all.push(em);
            }
            // S10: staged particle bursts render in shots too.
            let particle_mesh = game.build_particle_mesh(&gpu, &renderer, camera.forward());
            if let Some(pm) = &particle_mesh {
                all.push(pm);
            }
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
                // M34-VM: build the held-item view-model for the shot (forced via VOXELCRAFT_HELD).
                // VOXELCRAFT_VM_POSE=swing|eat|draw|shield|equip|idle + VOXELCRAFT_VM_T=<0..1> force a pose.
                let vm_light = game.held_item_light(
                    camera.position.floor().as_ivec3(),
                    environment.day_factor(),
                );
                let mut vm_state = crate::viewmodel::ViewModel::default();
                let vm_pose = std::env::var("VOXELCRAFT_VM_POSE").unwrap_or_default();
                let vm_t: f32 = std::env::var("VOXELCRAFT_VM_T")
                    .ok()
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                match vm_pose.as_str() {
                    "swing" | "place" => {
                        vm_state.swing = vm_t;
                        vm_state.swinging = true;
                    }
                    "eat" => vm_state.use_pose.eat = vm_t,
                    "draw" => vm_state.use_pose.draw = vm_t,
                    "shield" => vm_state.use_pose.shield = vm_t,
                    "equip" => vm_state.equip = vm_t,
                    "idle" => {
                        vm_state.bob_strength = 1.0;
                        vm_state.bob_phase = vm_t * std::f32::consts::TAU;
                    }
                    _ => {}
                }
                let vm_geom = vm_state.build_geometry(shot_inv.selected_item(), vm_light);
                let vm_part = vm_geom
                    .as_ref()
                    .and_then(|g| renderer.upload_viewmodel(&gpu.device, g));
                let vm_uniform = crate::viewmodel::ViewModelUniform::new(gpu.aspect(), vm_light);
                let viewmodel = vm_part.as_ref().map(|p| (p, vm_uniform));
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
                    viewmodel,
                    dlss_render.as_mut(),
                    path,
                );
            }
            event_loop.exit();
            return;
        }

        // Interactive: build the world-independent render targets once (resized on window resize),
        // then either land on the main menu (M35) or — VOXELCRAFT_SKIPMENU — boot straight into the
        // saved world (dev fast-path + the pre-menu behaviour for scripts).
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
        // S4b: FG rides on RR. Standalone FG tagged the raw LINEAR view depth as DLSS-G's [0,1]
        // NDC depth guide (the linear→NDC conversion lives only in the RR guide-resize passes),
        // corrupting every interpolated frame. Until a standalone conversion pass exists, FG
        // without RR degrades to off — loudly.
        let frame_gen = if frame_gen.is_some() && dlss_render.is_none() {
            log::warn!(
                "VOXELCRAFT_FG=1 without DLSS Ray Reconstruction: Frame Generation disabled — \
                 DLSS-G needs the RR guide passes' NDC depth (set VOXELCRAFT_DLSS=rr)"
            );
            None
        } else {
            frame_gen
        };
        let rt_scene = RtScene::new();
        // N4: one-time migration of the legacy single world into saves/worlds/ — done here (interactive
        // entry only; the headless SHOT/BENCH/MENU paths returned above, so screenshots never trigger it).
        persistence::migrate_legacy_world();
        let skip_menu = std::env::var("VOXELCRAFT_SKIPMENU").is_ok();
        let (scene, session) = if skip_menu {
            // Read-only resolution: load the most-recently-played world if any, else fall back to the
            // legacy save_dir() WITHOUT creating one (create_session tolerates an absent level.bin →
            // fresh default world). create_world only ever runs on an explicit user Create.
            let dir = persistence::list_worlds()
                .into_iter()
                .next()
                .map(|w| w.dir)
                .unwrap_or_else(persistence::save_dir);
            (Scene::InGame, Some(create_session(&gpu, &renderer, &quality, &dir, WorldSource::Load)))
        } else {
            (Scene::MainMenu, None)
        };
        self.state = Some(State {
            gpu,
            renderer,
            targets,
            rt_scene,
            dlss_render,
            dlss,
            frame_gen,
            last_frame: Instant::now(),
            fps_accum: 0.0,
            fps_frames: 0,
            fps_smooth: 0.0,
            scene,
            session,
            audio: crate::audio::Audio::new(),
            quality,
            gpu_watchdog: crate::gpu::GpuWatchdog::new(),
            menu: crate::menu::UiState::default(),
            settings: crate::settings::Settings::load(),
            worlds: Vec::new(),
            world_sel: 0,
            world_pending_delete: None,
            create_name: String::new(),
            create_seed: String::new(),
            create_mode: crate::rules::GameMode::default(),
            settings_tab: crate::menu::SettingsTab::General,
        });
        self.set_grab(skip_menu);
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
                    // M34-VM: the view-model's private depth is output-resolution-bound.
                    state.renderer.resize(&state.gpu.device, state.gpu.config.width, state.gpu.config.height);
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
                // M35: menu scenes swallow the mouse — a left press activates the hovered widget.
                if self.in_menu() {
                    if pressed && button == MouseButton::Left {
                        self.menu_click(event_loop);
                    }
                    return;
                }
                let screen_open = self.state.as_ref().and_then(|s| s.session.as_ref()).is_some_and(|s| s.screen != Screen::None);
                if pressed && screen_open {
                    self.inventory_click(button);
                } else if pressed && !self.grabbed {
                    self.set_grab(true);
                } else if let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
                    match button {
                        MouseButton::Left => session.input.break_held = pressed, // hold to break
                        MouseButton::Right => {
                            session.input.place_held = pressed; // hold to eat / use
                            if pressed {
                                session.input.place_pressed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(state) = &mut self.state {
                    // `cursor` is swapchain (config) space — the UI rects in ui/overlay.rs are built
                    // from gpu.config dims. With a sub-native render scale (Metal) the window is larger
                    // than the drawable, so map the window-physical position per-axis (the floored
                    // config dims can round the two ratios differently).
                    let sx = state.gpu.config.width as f64 / state.gpu.size.width.max(1) as f64;
                    let sy = state.gpu.config.height as f64 / state.gpu.size.height.max(1) as f64;
                    let p = ((position.x * sx) as f32, (position.y * sy) as f32);
                    if matches!(state.scene, Scene::InGame) {
                        if let Some(session) = state.session.as_mut() {
                            if session.screen != Screen::None {
                                session.cursor = p;
                            }
                        }
                    } else {
                        // M35: in a menu — track the pointer + recompute the hovered widget from the
                        // active screen's rect list (the same list `menu_click` hit-tests).
                        state.menu.cursor = p;
                        let (w, h) = (state.gpu.config.width as f32, state.gpu.config.height as f32);
                        state.menu.hovered = Self::build_menu(state, w, h).and_then(|(_, rects)| {
                            rects.iter().find(|(_, r)| r.contains(p)).map(|(id, _)| *id)
                        });
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !self.in_menu() => {
                if let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
                    let dir = match delta {
                        MouseScrollDelta::LineDelta(_, y) => -y.signum() as i32,
                        MouseScrollDelta::PixelDelta(p) => -(p.y.signum() as i32),
                    };
                    if dir != 0 {
                        if session.screen == Screen::None {
                            session.inventory.scroll(dir);
                        } else if session.screen == Screen::Inventory && session.inventory.creative {
                            // F1: scroll pages the creative palette.
                            let pages =
                                session.creative_palette.len().div_ceil(overlay::PAL_PER_PAGE).max(1);
                            let p = session.creative_page as i32 + dir;
                            session.creative_page = p.clamp(0, pages as i32 - 1) as usize;
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                // M35: menu scenes route keys to menu navigation / text entry, not gameplay.
                if self.in_menu() {
                    if pressed {
                        self.menu_key(&event, event_loop);
                    }
                    return;
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.handle_key(code, pressed, event.repeat, event_loop);
                }
            }
            WindowEvent::RedrawRequested => {
                self.frame();
                // Process a screen-open requested during the frame (after the state borrow ends).
                if let Some(sc) = self
                    .state
                    .as_mut()
                    .and_then(|s| s.session.as_mut())
                    .and_then(|sess| sess.pending_open.take())
                {
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
                if let Some(session) = self.state.as_mut().and_then(|s| s.session.as_mut()) {
                    session.input.yaw_delta += delta.0 as f32;
                    session.input.pitch_delta += delta.1 as f32;
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

#[cfg(test)]
mod tests {
    use super::parse_seed;

    #[test]
    fn parse_seed_numbers_then_hash() {
        // Plain numbers map to themselves (Minecraft expectation).
        assert_eq!(parse_seed("123"), 123);
        assert_eq!(parse_seed(" 42 "), 42); // trimmed
        assert_eq!(parse_seed("0"), 0); // '0' is a valid seed, not special-cased
        // Signed wraps via `as u64`; the full u64 range parses too.
        assert_eq!(parse_seed("-1"), u64::MAX);
        assert_eq!(parse_seed("18446744073709551615"), u64::MAX);
        // Overlong digit strings (> u64::MAX) fall through to the hash (deterministic, not the digits).
        assert_eq!(parse_seed("99999999999999999999"), parse_seed("99999999999999999999"));
        // Text hashes stably and is order-sensitive.
        assert_eq!(parse_seed("hello"), parse_seed("hello"));
        assert_ne!(parse_seed("hello"), parse_seed("olleh"));
        // Empty → clock-derived, but never 0.
        assert_ne!(parse_seed(""), 0);
    }
}
