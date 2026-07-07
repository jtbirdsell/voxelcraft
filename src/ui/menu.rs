//! Full-screen menus (M35): main menu, pause, settings (general + graphics), world select, and create
//! world — built on the existing `UiVertex` system (`overlay::push_px_rect` / `push_text`). Each
//! builder is **pure**: `(screen_w, screen_h, …, &UiState) -> (Vec<UiVertex>, Vec<(WidgetId, Rect)>)`.
//! The same rect list drives both rendering (hover tint) and click hit-testing (mirrors how
//! `overlay::inventory_slot_rects` feeds both the renderer and `app::inventory_click`).
#![allow(dead_code)] // widget layer + builders — wired into the scene flow / input in N3–N5.

use crate::overlay::{push_px_rect, push_text, text_width, UiVertex};
use crate::rules::{Difficulty, GameMode};
use crate::settings::Settings;

/// A pixel-space rectangle (top-left origin), for layout + hit-testing.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn contains(&self, p: (f32, f32)) -> bool {
        p.0 >= self.x && p.0 < self.x + self.w && p.1 >= self.y && p.1 < self.y + self.h
    }
}

/// Stable identity for every clickable / focusable widget across the menus. One enum keeps click
/// routing a single match (in `app`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WidgetId {
    // Main menu
    Singleplayer,
    MainSettings,
    Quit,
    // World select
    WorldRow(usize),
    DeleteWorld(usize),
    CreateNew,
    BackFromWorlds,
    // Create world
    NameField,
    SeedField,
    GameModeCycle,
    CreateConfirm,
    CreateCancel,
    // Settings — general tab
    FovSlider,
    SensSlider,
    RenderDistSlider,
    ViewBobToggle,
    DifficultyCycle,
    // Settings — graphics tab
    DlssModeCycle,
    DlssQualityCycle,
    SupersampleSlider,
    GiModeCycle,
    GiRaysSlider,
    TracerCycle,
    BackendCycle,
    FrameGenToggle,
    // Settings nav
    GeneralTab,
    GraphicsTab,
    SettingsBack,
    // Pause
    Resume,
    PauseSettings,
    SaveQuit,
    // Death screen (S10)
    DeadRespawn,
    DeadSaveQuit,
}

/// Which settings tab is showing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SettingsTab {
    General,
    Graphics,
}

pub use crate::persistence::WorldInfo;

/// Transient interaction state owned by `App` (not by the pure builders): the pointer, the hovered
/// widget (recomputed each frame from the rect list), and the focused text field + caret.
#[derive(Default)]
pub struct UiState {
    pub cursor: (f32, f32),
    pub hovered: Option<WidgetId>,
    pub focused: Option<WidgetId>,
    pub caret: usize,
}

// ── palette ──────────────────────────────────────────────────────────────────────────────────────
const BACKDROP: [f32; 4] = [0.055, 0.065, 0.085, 1.0];
const TITLE: [f32; 4] = [0.93, 0.95, 0.99, 1.0];
const TEXT: [f32; 4] = [0.86, 0.89, 0.94, 1.0];
const DIM: [f32; 4] = [0.55, 0.58, 0.64, 1.0];
const BTN: [f32; 4] = [0.17, 0.19, 0.24, 0.98];
const BTN_HOVER: [f32; 4] = [0.29, 0.34, 0.44, 1.0];
const BTN_SEL: [f32; 4] = [0.22, 0.30, 0.46, 1.0];
const TRACK: [f32; 4] = [0.12, 0.13, 0.16, 1.0];
const FILL: [f32; 4] = [0.36, 0.52, 0.85, 1.0];
const KNOB: [f32; 4] = [0.84, 0.88, 0.96, 1.0];
const ON: [f32; 4] = [0.27, 0.62, 0.40, 1.0];
const OFF: [f32; 4] = [0.42, 0.30, 0.32, 1.0];
const FIELD: [f32; 4] = [0.09, 0.10, 0.13, 1.0];
const FIELD_FOCUS: [f32; 4] = [0.36, 0.52, 0.85, 1.0];
const CONFIRM: [f32; 4] = [0.62, 0.22, 0.24, 1.0]; // armed-delete "Delete?" button
const CONFIRM_HOVER: [f32; 4] = [0.74, 0.28, 0.30, 1.0];

const BTN_W: f32 = 360.0;
/// Visible rows in the world-select list (S15: the wheel scrolls longer lists).
pub const WORLD_ROWS: usize = 6;
const ROW_H: f32 = 46.0;
const GAP: f32 = 12.0;
const LABEL_SCALE: f32 = 2.0;

/// A vertically-stacked, horizontally-centered column of fixed-height rows.
struct Column {
    x: f32,
    y: f32,
    w: f32,
    row_h: f32,
    gap: f32,
}

impl Column {
    fn centered(sw: f32, top: f32, w: f32, row_h: f32, gap: f32) -> Self {
        Self { x: (sw - w) * 0.5, y: top, w, row_h, gap }
    }
    fn row(&self, i: usize) -> Rect {
        Rect { x: self.x, y: self.y + i as f32 * (self.row_h + self.gap), w: self.w, h: self.row_h }
    }
}

// ── widget emitters ──────────────────────────────────────────────────────────────────────────────

fn label_left(out: &mut Vec<UiVertex>, sw: f32, sh: f32, x: f32, y_center: f32, scale: f32, s: &str, c: [f32; 4]) {
    let cell = crate::font::GLYPH as f32 * scale;
    push_text(out, sw, sh, x, y_center - cell * 0.5, scale, s, c);
}

fn label_centered(out: &mut Vec<UiVertex>, sw: f32, sh: f32, r: Rect, scale: f32, s: &str, c: [f32; 4]) {
    let tw = text_width(s, scale);
    let cell = crate::font::GLYPH as f32 * scale;
    push_text(out, sw, sh, r.x + (r.w - tw) * 0.5, r.y + (r.h - cell) * 0.5, scale, s, c);
}

/// A title centered horizontally at pixel `y`.
fn title(out: &mut Vec<UiVertex>, sw: f32, sh: f32, y: f32, scale: f32, s: &str) {
    let tw = text_width(s, scale);
    push_text(out, sw, sh, (sw - tw) * 0.5, y, scale, s, TITLE);
}

fn button(out: &mut Vec<UiVertex>, sw: f32, sh: f32, r: Rect, s: &str, hovered: bool, selected: bool) {
    let bg = if hovered {
        BTN_HOVER
    } else if selected {
        BTN_SEL
    } else {
        BTN
    };
    push_px_rect(out, sw, sh, r.x, r.y, r.w, r.h, bg);
    label_centered(out, sw, sh, r, LABEL_SCALE, s, TEXT);
}

/// A labeled slider: "Label  value" on the left/right + a track with a fill + knob. `frac` is the
/// normalized 0..1 position. Returns the track rect (the caller already has `r`).
fn slider(out: &mut Vec<UiVertex>, sw: f32, sh: f32, r: Rect, label: &str, value: &str, frac: f32, hovered: bool) {
    // Row background (subtle) so it reads as a control.
    push_px_rect(out, sw, sh, r.x, r.y, r.w, r.h, if hovered { BTN_HOVER } else { BTN });
    let pad = 14.0;
    label_left(out, sw, sh, r.x + pad, r.y + r.h * 0.32, 1.6, label, TEXT);
    let vw = text_width(value, 1.6);
    label_left(out, sw, sh, r.x + r.w - pad - vw, r.y + r.h * 0.32, 1.6, value, DIM);
    // Track along the bottom third.
    let tx = r.x + pad;
    let tw = r.w - 2.0 * pad;
    let ty = r.y + r.h * 0.66;
    let th = 5.0;
    push_px_rect(out, sw, sh, tx, ty, tw, th, TRACK);
    let f = frac.clamp(0.0, 1.0);
    push_px_rect(out, sw, sh, tx, ty, tw * f, th, FILL);
    push_px_rect(out, sw, sh, tx + tw * f - 4.0, ty - 4.0, 8.0, th + 8.0, KNOB);
}

fn toggle(out: &mut Vec<UiVertex>, sw: f32, sh: f32, r: Rect, label: &str, on: bool, hovered: bool) {
    push_px_rect(out, sw, sh, r.x, r.y, r.w, r.h, if hovered { BTN_HOVER } else { BTN });
    let pad = 14.0;
    label_left(out, sw, sh, r.x + pad, r.y + r.h * 0.5, 1.6, label, TEXT);
    // A pill on the right.
    let pw = 56.0;
    let ph = 22.0;
    let px = r.x + r.w - pad - pw;
    let py = r.y + (r.h - ph) * 0.5;
    push_px_rect(out, sw, sh, px, py, pw, ph, if on { ON } else { OFF });
    let knob = pw * 0.5 - 4.0;
    let kx = if on { px + pw - knob - 4.0 } else { px + 4.0 };
    push_px_rect(out, sw, sh, kx, py + 4.0, knob, ph - 8.0, KNOB);
    label_left(out, sw, sh, px - 8.0 - text_width(if on { "ON" } else { "OFF" }, 1.4), r.y + r.h * 0.5, 1.4, if on { "ON" } else { "OFF" }, DIM);
}

/// A "< value >" cycler control; `restart` appends a dim "(restart)" tag for apply-on-restart options.
fn cycler(out: &mut Vec<UiVertex>, sw: f32, sh: f32, r: Rect, label: &str, value: &str, hovered: bool, restart: bool) {
    push_px_rect(out, sw, sh, r.x, r.y, r.w, r.h, if hovered { BTN_HOVER } else { BTN });
    let pad = 14.0;
    label_left(out, sw, sh, r.x + pad, r.y + r.h * 0.5, 1.6, label, TEXT);
    let shown = format!("< {value} >");
    let vw = text_width(&shown, 1.6);
    label_left(out, sw, sh, r.x + r.w - pad - vw, r.y + r.h * 0.5, 1.6, &shown, FILL);
    if restart {
        let tag = "(restart)";
        label_left(out, sw, sh, r.x + r.w - pad - vw - 12.0 - text_width(tag, 1.2), r.y + r.h * 0.5, 1.2, tag, DIM);
    }
}

/// A text-entry field showing `text` (or a dim placeholder) + a blinking-free caret when focused.
fn text_field(out: &mut Vec<UiVertex>, sw: f32, sh: f32, r: Rect, label: &str, text: &str, placeholder: &str, focused: bool, caret: usize) {
    label_left(out, sw, sh, r.x, r.y + r.h * 0.5, 1.5, label, DIM);
    let fx = r.x + 130.0;
    let fw = r.w - 130.0;
    push_px_rect(out, sw, sh, fx, r.y, fw, r.h, FIELD);
    // Focus border (a 2px frame).
    if focused {
        for (bx, by, bw, bh) in [
            (fx, r.y, fw, 2.0),
            (fx, r.y + r.h - 2.0, fw, 2.0),
            (fx, r.y, 2.0, r.h),
            (fx + fw - 2.0, r.y, 2.0, r.h),
        ] {
            push_px_rect(out, sw, sh, bx, by, bw, bh, FIELD_FOCUS);
        }
    }
    let scale = 1.6;
    let cell = crate::font::GLYPH as f32 * scale;
    let ty = r.y + (r.h - cell) * 0.5;
    if text.is_empty() && !focused {
        push_text(out, sw, sh, fx + 8.0, ty, scale, placeholder, DIM);
    } else {
        push_text(out, sw, sh, fx + 8.0, ty, scale, text, TEXT);
        if focused {
            let cx = fx + 8.0 + text_width(&text[..caret.min(text.len())], scale);
            push_px_rect(out, sw, sh, cx, ty, 2.0, cell, KNOB);
        }
    }
}

/// Full-screen dark backdrop, drawn first.
fn backdrop(out: &mut Vec<UiVertex>, sw: f32, sh: f32) {
    push_px_rect(out, sw, sh, 0.0, 0.0, sw, sh, BACKDROP);
}

// ── builders ─────────────────────────────────────────────────────────────────────────────────────

pub type Built = (Vec<UiVertex>, Vec<(WidgetId, Rect)>);

fn emit(out: &mut Vec<UiVertex>, rects: &mut Vec<(WidgetId, Rect)>, id: WidgetId, r: Rect, sw: f32, sh: f32, label: &str, ui: &UiState) {
    button(out, sw, sh, r, label, ui.hovered == Some(id), false);
    rects.push((id, r));
}

pub fn build_main(sw: f32, sh: f32, ui: &UiState) -> Built {
    let (mut v, mut rects) = (Vec::new(), Vec::new());
    backdrop(&mut v, sw, sh);
    title(&mut v, sw, sh, sh * 0.22, 5.0, "VOXELCRAFT");
    let col = Column::centered(sw, sh * 0.42, BTN_W, ROW_H, GAP);
    emit(&mut v, &mut rects, WidgetId::Singleplayer, col.row(0), sw, sh, "Singleplayer", ui);
    emit(&mut v, &mut rects, WidgetId::MainSettings, col.row(1), sw, sh, "Settings", ui);
    emit(&mut v, &mut rects, WidgetId::Quit, col.row(2), sw, sh, "Quit", ui);
    (v, rects)
}

pub fn build_pause(sw: f32, sh: f32, ui: &UiState) -> Built {
    let (mut v, mut rects) = (Vec::new(), Vec::new());
    // Pause is drawn over the frozen world, so a translucent dim instead of the opaque backdrop.
    push_px_rect(&mut v, sw, sh, 0.0, 0.0, sw, sh, [0.0, 0.0, 0.0, 0.55]);
    title(&mut v, sw, sh, sh * 0.24, 4.0, "Paused");
    let col = Column::centered(sw, sh * 0.40, BTN_W, ROW_H, GAP);
    emit(&mut v, &mut rects, WidgetId::Resume, col.row(0), sw, sh, "Resume", ui);
    emit(&mut v, &mut rects, WidgetId::PauseSettings, col.row(1), sw, sh, "Settings", ui);
    emit(&mut v, &mut rects, WidgetId::SaveQuit, col.row(2), sw, sh, "Save & Quit to Menu", ui);
    (v, rects)
}

/// `pending_delete` arms one row's delete control into a red "Delete?" confirm (a second click on the
/// same row commits) — the two-step guard against a one-misclick `remove_dir_all`.
/// S10: the death screen — same translucent-dim treatment as Pause (the world behind is frozen).
pub fn build_death(sw: f32, sh: f32, ui: &UiState) -> Built {
    let (mut v, mut rects) = (Vec::new(), Vec::new());
    push_px_rect(&mut v, sw, sh, 0.0, 0.0, sw, sh, [0.25, 0.0, 0.0, 0.65]);
    title(&mut v, sw, sh, sh * 0.24, 5.0, "You Died!");
    label_centered(
        &mut v,
        sw,
        sh,
        Rect { x: 0.0, y: sh * 0.36, w: sw, h: 24.0 },
        1.6,
        "Your items dropped where you fell.",
        DIM,
    );
    let col = Column::centered(sw, sh * 0.46, BTN_W, ROW_H, GAP);
    emit(&mut v, &mut rects, WidgetId::DeadRespawn, col.row(0), sw, sh, "Respawn", ui);
    emit(&mut v, &mut rects, WidgetId::DeadSaveQuit, col.row(1), sw, sh, "Save & Quit to Menu", ui);
    (v, rects)
}

pub fn build_world_select(
    sw: f32,
    sh: f32,
    worlds: &[WorldInfo],
    selected: usize,
    pending_delete: Option<usize>,
    scroll: usize,
    ui: &UiState,
) -> Built {
    let (mut v, mut rects) = (Vec::new(), Vec::new());
    backdrop(&mut v, sw, sh);
    title(&mut v, sw, sh, sh * 0.10, 4.0, "Select World");
    let list_w = 520.0;
    let col = Column::centered(sw, sh * 0.22, list_w, 56.0, 10.0);
    // S15: a window of WORLD_ROWS rows starting at `scroll` (wheel-scrolled, clamped); the widget
    // ids keep the ABSOLUTE index so play/delete stay correct on any page.
    let scroll = scroll.min(worlds.len().saturating_sub(WORLD_ROWS));
    if scroll > 0 {
        let r0 = col.row(0);
        let hint = Rect { x: r0.x, y: r0.y - 26.0, w: r0.w, h: 18.0 };
        label_centered(&mut v, sw, sh, hint, 1.2, &format!("{scroll} more above - scroll"), DIM);
    }
    for (row, (i, w)) in worlds.iter().enumerate().skip(scroll).take(WORLD_ROWS).enumerate() {
        let r = col.row(row);
        let armed = pending_delete == Some(i);
        let del_hover = ui.hovered == Some(WidgetId::DeleteWorld(i));
        // The delete control sits inside the full-width row rect; size it wider when armed for "Delete?".
        let del = if armed {
            Rect { x: r.x + r.w - 8.0 - 120.0, y: r.y + 8.0, w: 120.0, h: r.h - 16.0 }
        } else {
            Rect { x: r.x + r.w - 44.0, y: r.y + 8.0, w: 36.0, h: r.h - 16.0 }
        };
        // Row button (drawn first, visually behind the delete control).
        let hovered = ui.hovered == Some(WidgetId::WorldRow(i));
        button(&mut v, sw, sh, r, "", hovered, i == selected);
        label_left(&mut v, sw, sh, r.x + 16.0, r.y + r.h * 0.36, 1.8, &w.name, TEXT);
        label_left(&mut v, sw, sh, r.x + 16.0, r.y + r.h * 0.72, 1.2, &format!("Seed: {}", w.seed), DIM);
        // Delete control drawn on top.
        if armed {
            push_px_rect(&mut v, sw, sh, del.x, del.y, del.w, del.h, if del_hover { CONFIRM_HOVER } else { CONFIRM });
            label_centered(&mut v, sw, sh, del, 1.4, "Delete?", TEXT);
        } else {
            button(&mut v, sw, sh, del, "X", del_hover, false);
        }
        // Push the delete rect BEFORE the row rect: hit-testing is first-match, so a click on the (small,
        // inset) delete control resolves to DeleteWorld, not the full-width WorldRow underneath it.
        rects.push((WidgetId::DeleteWorld(i), del));
        rects.push((WidgetId::WorldRow(i), r));
    }
    let below = worlds.len().saturating_sub(scroll + WORLD_ROWS);
    if below > 0 {
        let rl = col.row(WORLD_ROWS - 1);
        let hint = Rect { x: rl.x, y: rl.y + rl.h + 8.0, w: rl.w, h: 18.0 };
        label_centered(&mut v, sw, sh, hint, 1.2, &format!("{below} more below - scroll"), DIM);
    }
    if worlds.is_empty() {
        label_centered(&mut v, sw, sh, col.row(0), 1.8, "No worlds yet — create one.", DIM);
    }
    // Footer: Create New / Back.
    let foot = Column::centered(sw, sh * 0.82, 520.0, ROW_H, GAP);
    let create = Rect { x: foot.x, y: foot.y, w: 250.0, h: ROW_H };
    let back = Rect { x: foot.x + 270.0, y: foot.y, w: 250.0, h: ROW_H };
    emit(&mut v, &mut rects, WidgetId::CreateNew, create, sw, sh, "Create New World", ui);
    emit(&mut v, &mut rects, WidgetId::BackFromWorlds, back, sw, sh, "Back", ui);
    (v, rects)
}

pub fn build_create(sw: f32, sh: f32, name: &str, seed: &str, mode: GameMode, ui: &UiState) -> Built {
    let (mut v, mut rects) = (Vec::new(), Vec::new());
    backdrop(&mut v, sw, sh);
    title(&mut v, sw, sh, sh * 0.18, 4.0, "Create World");
    let col = Column::centered(sw, sh * 0.38, 520.0, ROW_H, 22.0);
    let nr = col.row(0);
    text_field(&mut v, sw, sh, nr, "Name", name, "New World", ui.focused == Some(WidgetId::NameField), ui.caret);
    rects.push((WidgetId::NameField, nr));
    let sr = col.row(1);
    text_field(&mut v, sw, sh, sr, "Seed", seed, "(random)", ui.focused == Some(WidgetId::SeedField), ui.caret);
    rects.push((WidgetId::SeedField, sr));
    // Game mode (S1): Survival (default) <-> Creative.
    let mr = col.row(2);
    cycler(&mut v, sw, sh, mr, "Game Mode", mode.name(), ui.hovered == Some(WidgetId::GameModeCycle), false);
    rects.push((WidgetId::GameModeCycle, mr));
    // Buttons.
    let br = col.row(4);
    let confirm = Rect { x: br.x, y: br.y, w: 250.0, h: ROW_H };
    let cancel = Rect { x: br.x + 270.0, y: br.y, w: 250.0, h: ROW_H };
    emit(&mut v, &mut rects, WidgetId::CreateConfirm, confirm, sw, sh, "Create", ui);
    emit(&mut v, &mut rects, WidgetId::CreateCancel, cancel, sw, sh, "Cancel", ui);
    (v, rects)
}

#[allow(clippy::too_many_arguments)]
pub fn build_settings(sw: f32, sh: f32, s: &Settings, difficulty: Difficulty, tab: SettingsTab, ui: &UiState) -> Built {
    let (mut v, mut rects) = (Vec::new(), Vec::new());
    backdrop(&mut v, sw, sh);
    title(&mut v, sw, sh, sh * 0.08, 4.0, "Settings");
    // Tab selector.
    let tabs = Column::centered(sw, sh * 0.18, 360.0, 36.0, 0.0);
    let tr = tabs.row(0);
    let gen_tab = Rect { x: tr.x, y: tr.y, w: 178.0, h: 36.0 };
    let gfx_tab = Rect { x: tr.x + 182.0, y: tr.y, w: 178.0, h: 36.0 };
    button(&mut v, sw, sh, gen_tab, "General", ui.hovered == Some(WidgetId::GeneralTab), tab == SettingsTab::General);
    button(&mut v, sw, sh, gfx_tab, "Graphics", ui.hovered == Some(WidgetId::GraphicsTab), tab == SettingsTab::Graphics);
    rects.push((WidgetId::GeneralTab, gen_tab));
    rects.push((WidgetId::GraphicsTab, gfx_tab));

    let col = Column::centered(sw, sh * 0.30, 520.0, 44.0, 10.0);
    let h = |id: WidgetId| ui.hovered == Some(id);
    match tab {
        SettingsTab::General => {
            let r = col.row(0);
            slider(&mut v, sw, sh, r, "FOV", &format!("{:.0}", s.fov), (s.fov - 50.0) / (110.0 - 50.0), h(WidgetId::FovSlider));
            rects.push((WidgetId::FovSlider, r));
            let r = col.row(1);
            slider(&mut v, sw, sh, r, "Sensitivity", &format!("{:.4}", s.sensitivity), (s.sensitivity - 0.0005) / (0.006 - 0.0005), h(WidgetId::SensSlider));
            rects.push((WidgetId::SensSlider, r));
            let r = col.row(2);
            slider(&mut v, sw, sh, r, "Render Distance", &format!("{}", s.render_distance), (s.render_distance as f32 - 4.0) / (24.0 - 4.0), h(WidgetId::RenderDistSlider));
            rects.push((WidgetId::RenderDistSlider, r));
            let r = col.row(3);
            toggle(&mut v, sw, sh, r, "View Bob", s.view_bob, h(WidgetId::ViewBobToggle));
            rects.push((WidgetId::ViewBobToggle, r));
            let r = col.row(4);
            cycler(&mut v, sw, sh, r, "Difficulty", difficulty.name(), h(WidgetId::DifficultyCycle), false);
            rects.push((WidgetId::DifficultyCycle, r));
        }
        SettingsTab::Graphics => {
            let r = col.row(0);
            cycler(&mut v, sw, sh, r, "DLSS", s.dlss_mode.as_str(), h(WidgetId::DlssModeCycle), true);
            rects.push((WidgetId::DlssModeCycle, r));
            let r = col.row(1);
            cycler(&mut v, sw, sh, r, "DLSS Quality", s.dlss_quality.as_str(), h(WidgetId::DlssQualityCycle), true);
            rects.push((WidgetId::DlssQualityCycle, r));
            let r = col.row(2);
            slider(&mut v, sw, sh, r, "Supersample", &format!("{:.2}x", s.supersample), (s.supersample - 1.0) / (4.0 - 1.0), h(WidgetId::SupersampleSlider));
            rects.push((WidgetId::SupersampleSlider, r));
            let r = col.row(3);
            cycler(&mut v, sw, sh, r, "GI", s.gi_mode.as_str(), h(WidgetId::GiModeCycle), true);
            rects.push((WidgetId::GiModeCycle, r));
            let r = col.row(4);
            slider(&mut v, sw, sh, r, "GI Rays", &format!("{}", s.gi_rays), (s.gi_rays as f32 - 1.0) / (16.0 - 1.0), h(WidgetId::GiRaysSlider));
            rects.push((WidgetId::GiRaysSlider, r));
            let r = col.row(5);
            cycler(&mut v, sw, sh, r, "Tracer", s.tracer.as_str(), h(WidgetId::TracerCycle), true);
            rects.push((WidgetId::TracerCycle, r));
            let r = col.row(6);
            cycler(&mut v, sw, sh, r, "Backend", s.backend.as_str(), h(WidgetId::BackendCycle), true);
            rects.push((WidgetId::BackendCycle, r));
            let r = col.row(7);
            toggle(&mut v, sw, sh, r, "Frame Generation", s.frame_gen, h(WidgetId::FrameGenToggle));
            rects.push((WidgetId::FrameGenToggle, r));
        }
    }
    // Back.
    let back = Rect { x: (sw - 240.0) * 0.5, y: sh * 0.90, w: 240.0, h: ROW_H };
    emit(&mut v, &mut rects, WidgetId::SettingsBack, back, sw, sh, "Back", ui);
    (v, rects)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ui() -> UiState {
        UiState { cursor: (-1.0, -1.0), ..Default::default() }
    }

    #[test]
    fn main_menu_has_three_buttons_and_hit_tests() {
        let (verts, rects) = build_main(1600.0, 900.0, &ui());
        assert!(!verts.is_empty());
        let ids: Vec<_> = rects.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&WidgetId::Singleplayer));
        assert!(ids.contains(&WidgetId::MainSettings));
        assert!(ids.contains(&WidgetId::Quit));
        // A click at the center of the Singleplayer rect hits it.
        let (_, r) = rects.iter().find(|(id, _)| *id == WidgetId::Singleplayer).unwrap();
        let center = (r.x + r.w * 0.5, r.y + r.h * 0.5);
        let hit = rects.iter().find(|(_, rr)| rr.contains(center)).map(|(id, _)| *id);
        assert_eq!(hit, Some(WidgetId::Singleplayer));
    }

    #[test]
    fn settings_tabs_differ_and_create_has_fields() {
        let s = Settings::default();
        let (_, gen_r) = build_settings(1600.0, 900.0, &s, Difficulty::Normal, SettingsTab::General, &ui());
        let (_, gfx_r) = build_settings(1600.0, 900.0, &s, Difficulty::Normal, SettingsTab::Graphics, &ui());
        assert!(gen_r.iter().any(|(id, _)| *id == WidgetId::FovSlider));
        assert!(gfx_r.iter().any(|(id, _)| *id == WidgetId::DlssModeCycle));
        // Both tabs keep the nav + back.
        for set in [&gen_r, &gfx_r] {
            assert!(set.iter().any(|(id, _)| *id == WidgetId::SettingsBack));
            assert!(set.iter().any(|(id, _)| *id == WidgetId::GraphicsTab));
        }
        let (_, cr) = build_create(1600.0, 900.0, "", "", GameMode::Survival, &ui());
        assert!(cr.iter().any(|(id, _)| *id == WidgetId::NameField));
        assert!(cr.iter().any(|(id, _)| *id == WidgetId::SeedField));
        assert!(cr.iter().any(|(id, _)| *id == WidgetId::GameModeCycle));
    }

    fn worlds() -> Vec<WorldInfo> {
        vec![
            WorldInfo { dir: "saves/worlds/alpha".into(), name: "Alpha".into(), seed: 1 },
            WorldInfo { dir: "saves/worlds/beta".into(), name: "Beta".into(), seed: 2 },
        ]
    }

    /// First-match hit-testing must resolve a click on the delete "X" to DeleteWorld (not the
    /// full-width WorldRow it sits inside), and a click elsewhere on the row to WorldRow.
    fn hit(rects: &[(WidgetId, Rect)], p: (f32, f32)) -> Option<WidgetId> {
        rects.iter().find(|(_, r)| r.contains(p)).map(|(id, _)| *id)
    }

    #[test]
    fn death_screen_has_respawn_and_quit() {
        let (verts, rects) = build_death(1600.0, 900.0, &ui());
        assert!(!verts.is_empty());
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::DeadRespawn));
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::DeadSaveQuit));
    }

    #[test]
    fn s15_world_select_scrolls_past_six() {
        // 10 worlds, scrolled by 3: rows 3..9 visible with ABSOLUTE ids; 0..3 and 9 clipped.
        let many: Vec<WorldInfo> = (0..10)
            .map(|i| WorldInfo { dir: format!("saves/worlds/w{i}").into(), name: format!("W{i}"), seed: i })
            .collect();
        let (_, rects) = build_world_select(1600.0, 900.0, &many, 0, None, 3, &ui());
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::WorldRow(3)));
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::WorldRow(8)));
        assert!(!rects.iter().any(|(id, _)| *id == WidgetId::WorldRow(2)));
        assert!(!rects.iter().any(|(id, _)| *id == WidgetId::WorldRow(9)));
        // An oversized scroll clamps to the last window instead of blanking the list.
        let (_, rects) = build_world_select(1600.0, 900.0, &many, 0, None, 99, &ui());
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::WorldRow(9)));
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::WorldRow(4)));
    }

    #[test]
    fn world_select_lists_rows() {
        let (_, rects) = build_world_select(1600.0, 900.0, &worlds(), 0, None, 0, &ui());
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::WorldRow(0)));
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::WorldRow(1)));
        assert!(rects.iter().any(|(id, _)| *id == WidgetId::CreateNew));
    }

    #[test]
    fn delete_x_wins_over_row_then_row_elsewhere() {
        let (_, rects) = build_world_select(1600.0, 900.0, &worlds(), 0, None, 0, &ui());
        let row = rects.iter().find(|(id, _)| *id == WidgetId::WorldRow(0)).unwrap().1;
        let del = rects.iter().find(|(id, _)| *id == WidgetId::DeleteWorld(0)).unwrap().1;
        // Center of the X resolves to delete...
        assert_eq!(hit(&rects, (del.x + del.w * 0.5, del.y + del.h * 0.5)), Some(WidgetId::DeleteWorld(0)));
        // ...the left of the row (away from the X) resolves to the row.
        assert_eq!(hit(&rects, (row.x + 20.0, row.y + row.h * 0.5)), Some(WidgetId::WorldRow(0)));
    }

    #[test]
    fn armed_delete_renders_wider_and_still_hit_tests_to_delete() {
        let (_, rects) = build_world_select(1600.0, 900.0, &worlds(), 0, Some(1), 0, &ui());
        let del = rects.iter().find(|(id, _)| *id == WidgetId::DeleteWorld(1)).unwrap().1;
        assert!(del.w > 100.0, "armed delete widens for the 'Delete?' label");
        assert_eq!(hit(&rects, (del.x + del.w * 0.5, del.y + del.h * 0.5)), Some(WidgetId::DeleteWorld(1)));
    }
}
