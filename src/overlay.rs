//! Overlay geometry: the targeted-block wireframe (world space) and the 2D HUD
//! (crosshair + hotbar) in normalized device coordinates.

use glam::{IVec3, Vec3};

use crate::font;
use crate::item;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
}

impl LineVertex {
    pub const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    /// 0 = flat color (rects), 1 = atlas sample * color (icons), 2 = font coverage (alpha = tex.r).
    pub mode: f32,
}

impl UiVertex {
    pub const ATTRS: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32];
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

/// The 12 edges (24 vertices) of the cube occupying `block`, slightly inflated to avoid z-fight.
pub fn highlight_lines(block: IVec3) -> Vec<LineVertex> {
    let min = block.as_vec3() - Vec3::splat(0.003);
    let max = block.as_vec3() + Vec3::splat(1.003);
    let c = |x: f32, y: f32, z: f32| LineVertex { pos: [x, y, z] };
    let corners = [
        c(min.x, min.y, min.z),
        c(max.x, min.y, min.z),
        c(max.x, min.y, max.z),
        c(min.x, min.y, max.z),
        c(min.x, max.y, min.z),
        c(max.x, max.y, min.z),
        c(max.x, max.y, max.z),
        c(min.x, max.y, max.z),
    ];
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // bottom
        (4, 5), (5, 6), (6, 7), (7, 4), // top
        (0, 4), (1, 5), (2, 6), (3, 7), // verticals
    ];
    let mut v = Vec::with_capacity(24);
    for (a, b) in edges {
        v.push(corners[a]);
        v.push(corners[b]);
    }
    v
}

/// Crack segments across the faces of `block`, revealed proportionally to mining `progress` (0..1).
/// Drawn with the same line pipeline as the highlight, slightly inflated to sit on the surface.
pub fn crack_lines(block: IVec3, progress: f32) -> Vec<LineVertex> {
    let o = block.as_vec3();
    let p = |x: f32, y: f32, z: f32| LineVertex { pos: [o.x + x, o.y + y, o.z + z] };
    const E: f32 = 1.004; // sit just outside the +face
    let segs: [[(f32, f32, f32); 2]; 9] = [
        [(0.5, E, 0.1), (0.6, E, 0.5)],
        [(0.6, E, 0.5), (0.3, E, 0.9)],
        [(0.05, E, 0.45), (0.5, E, 0.5)],
        [(E, 0.1, 0.5), (E, 0.6, 0.4)],
        [(E, 0.6, 0.4), (E, 0.95, 0.65)],
        [(0.4, 0.05, E), (0.5, 0.6, E)],
        [(0.5, 0.6, E), (0.2, 0.95, E)],
        [(E, 0.5, 0.5), (E, 0.3, 0.15)],
        [(0.5, E, 0.5), (0.75, E, 0.2)],
    ];
    let n = ((progress * segs.len() as f32).ceil() as usize).min(segs.len());
    let mut v = Vec::with_capacity(n * 2);
    for s in segs.iter().take(n) {
        v.push(p(s[0].0, s[0].1, s[0].2));
        v.push(p(s[1].0, s[1].1, s[1].2));
    }
    v
}

fn push_px_rect(
    out: &mut Vec<UiVertex>,
    sw: f32,
    sh: f32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: [f32; 4],
) {
    // Pixel space (top-left origin) -> NDC.
    let to_ndc = |px: f32, py: f32| [px / sw * 2.0 - 1.0, 1.0 - py / sh * 2.0];
    let p0 = to_ndc(x, y);
    let p1 = to_ndc(x + w, y);
    let p2 = to_ndc(x + w, y + h);
    let p3 = to_ndc(x, y + h);
    for p in [p0, p1, p2, p0, p2, p3] {
        out.push(UiVertex { pos: p, uv: [0.0, 0.0], color, mode: 0.0 });
    }
}

/// One textured glyph quad (mode 2 = font coverage) at pixel rect (x,y,size,size), atlas `uv`.
#[allow(clippy::too_many_arguments)]
fn push_glyph(
    out: &mut Vec<UiVertex>,
    sw: f32,
    sh: f32,
    x: f32,
    y: f32,
    size: f32,
    uv: [f32; 4],
    color: [f32; 4],
) {
    let to_ndc = |px: f32, py: f32| [px / sw * 2.0 - 1.0, 1.0 - py / sh * 2.0];
    let p = [
        to_ndc(x, y),
        to_ndc(x + size, y),
        to_ndc(x + size, y + size),
        to_ndc(x, y + size),
    ];
    let t = [
        [uv[0], uv[1]],
        [uv[2], uv[1]],
        [uv[2], uv[3]],
        [uv[0], uv[3]],
    ];
    for &i in &[0usize, 1, 2, 0, 2, 3] {
        out.push(UiVertex { pos: p[i], uv: t[i], color, mode: 2.0 });
    }
}

/// Width in pixels that `push_text` would advance for `text` at `scale`.
pub fn text_width(text: &str, scale: f32) -> f32 {
    text.len() as f32 * 6.0 * scale
}

/// Draw an ASCII line at pixel (x,y) top-left; each glyph cell is `8*scale` px, tracking `6*scale`.
pub fn push_text(out: &mut Vec<UiVertex>, sw: f32, sh: f32, x: f32, y: f32, scale: f32, text: &str, color: [f32; 4]) {
    let cell = font::GLYPH as f32 * scale;
    let adv = 6.0 * scale;
    let mut cx = x;
    for &b in text.as_bytes() {
        if b >= 0x20 && b < 0x80 && b != b' ' {
            push_glyph(out, sw, sh, cx, y, cell, font::glyph_uv(b), color);
        }
        cx += adv;
    }
}

pub const INV_SLOT: f32 = 46.0;

/// Inventory-screen slot layout: (slot_index, x_px, y_px). 3 main rows (9..36) above the hotbar
/// row (0..9). Used for both rendering and click hit-testing so they can't drift.
pub fn inventory_slot_rects(width: u32, height: u32) -> Vec<(usize, f32, f32)> {
    let step = INV_SLOT + 4.0;
    let grid_w = 9.0 * step - 4.0;
    let ox = (width as f32 - grid_w) * 0.5;
    let oy = height as f32 * 0.5 - 130.0;
    let mut rects = Vec::with_capacity(36);
    for r in 0..3 {
        for c in 0..9 {
            rects.push((9 + r * 9 + c, ox + c as f32 * step, oy + r as f32 * step));
        }
    }
    let hy = oy + 3.0 * step + 14.0;
    for c in 0..9 {
        rects.push((c, ox + c as f32 * step, hy));
    }
    rects
}

/// Build the inventory screen overlay: dim backdrop, panel, slots, the held (cursor) stack, tooltip.
pub fn build_inventory_screen(
    width: u32,
    height: u32,
    inv: &item::Inventory,
    cursor: (f32, f32),
) -> Vec<UiVertex> {
    let sw = width as f32;
    let sh = height as f32;
    let mut v = Vec::new();
    push_px_rect(&mut v, sw, sh, 0.0, 0.0, sw, sh, [0.0, 0.0, 0.0, 0.55]);
    let rects = inventory_slot_rects(width, height);
    let step = INV_SLOT + 4.0;
    let (minx, miny) = (rects[0].1, rects[0].2);
    let panel_w = 9.0 * step - 4.0;
    let pad = 14.0;
    let panel_h = rects[35].2 + INV_SLOT - miny;
    push_px_rect(&mut v, sw, sh, minx - pad, miny - pad - 24.0, panel_w + 2.0 * pad, panel_h + 2.0 * pad + 24.0, [0.12, 0.12, 0.14, 0.97]);
    push_text(&mut v, sw, sh, minx, miny - pad - 18.0, 2.0, "Inventory", [0.95, 0.95, 0.95, 1.0]);

    let mut hovered: Option<usize> = None;
    for &(slot_i, x, y) in &rects {
        let hover = cursor.0 >= x && cursor.0 < x + INV_SLOT && cursor.1 >= y && cursor.1 < y + INV_SLOT;
        if hover {
            hovered = Some(slot_i);
        }
        let bg = if hover { [0.45, 0.45, 0.5, 1.0] } else { [0.28, 0.28, 0.32, 1.0] };
        push_px_rect(&mut v, sw, sh, x, y, INV_SLOT, INV_SLOT, bg);
        if let Some(stack) = inv.slots[slot_i] {
            let c = item::item_color(stack.item);
            push_px_rect(&mut v, sw, sh, x + 3.0, y + 3.0, INV_SLOT - 6.0, INV_SLOT - 6.0, [c[0], c[1], c[2], 1.0]);
            if stack.count > 1 {
                let label = format!("{}", stack.count);
                let tw = text_width(&label, 2.0);
                push_text(&mut v, sw, sh, x + INV_SLOT - tw - 3.0, y + INV_SLOT - 18.0, 2.0, &label, [1.0, 1.0, 1.0, 1.0]);
            }
            durability_bar(&mut v, sw, sh, x, y, INV_SLOT, stack);
        }
    }
    // Held stack follows the cursor.
    if let Some(held) = inv.held {
        let c = item::item_color(held.item);
        let sz = INV_SLOT - 6.0;
        let (hx, hy) = (cursor.0 - sz * 0.5, cursor.1 - sz * 0.5); // centered on the cursor
        push_px_rect(&mut v, sw, sh, hx, hy, sz, sz, [c[0], c[1], c[2], 1.0]);
        if held.count > 1 {
            let label = format!("{}", held.count);
            let tw = text_width(&label, 2.0);
            push_text(&mut v, sw, sh, hx + sz - tw - 2.0, hy + sz - 18.0, 2.0, &label, [1.0, 1.0, 1.0, 1.0]);
        }
        durability_bar(&mut v, sw, sh, hx - 4.0, hy, sz + 8.0, held);
    } else if let Some(slot_i) = hovered {
        // Tooltip when not dragging.
        if let Some(stack) = inv.slots[slot_i] {
            let name = item::item_name(stack.item);
            let tw = text_width(name, 2.0);
            push_px_rect(&mut v, sw, sh, cursor.0 + 12.0, cursor.1 - 4.0, tw + 8.0, 22.0, [0.05, 0.05, 0.07, 0.92]);
            push_text(&mut v, sw, sh, cursor.0 + 16.0, cursor.1, 2.0, name, [0.95, 0.95, 0.8, 1.0]);
        }
    }
    v
}

/// A small wear bar at the bottom of a slot for a damaged tool (green → red); hidden when full.
fn durability_bar(out: &mut Vec<UiVertex>, sw: f32, sh: f32, sx: f32, y: f32, slot: f32, stack: item::ItemStack) {
    if !item::is_tool(stack.item) {
        return;
    }
    let max = item::tool_max_durability(stack.item).max(1) as f32;
    let frac = (stack.durability as f32 / max).clamp(0.0, 1.0);
    if frac >= 1.0 {
        return;
    }
    let bw = slot - 8.0;
    let (bx, by) = (sx + 4.0, y + slot - 8.0);
    push_px_rect(out, sw, sh, bx, by, bw, 4.0, [0.0, 0.0, 0.0, 0.85]);
    push_px_rect(out, sw, sh, bx, by, bw * frac, 4.0, [1.0 - frac, frac, 0.12, 1.0]);
}

/// One row of `pips` square cells, each worth 2 units, filled left-to-right to show `value`.
#[allow(clippy::too_many_arguments)]
fn stat_bar(
    out: &mut Vec<UiVertex>,
    sw: f32,
    sh: f32,
    x: f32,
    y: f32,
    pip: f32,
    gap: f32,
    pips: i32,
    value: f32,
    on: [f32; 4],
) {
    let off = [0.06, 0.06, 0.07, 0.6];
    for i in 0..pips {
        let px = x + i as f32 * (pip + gap);
        push_px_rect(out, sw, sh, px, y, pip, pip, off);
        let fill = (value * 0.5 - i as f32).clamp(0.0, 1.0);
        if fill > 0.0 {
            push_px_rect(out, sw, sh, px, y, pip * fill, pip, on);
        }
    }
}

/// Build the HUD (crosshair + hotbar, plus health/hunger bars in survival) for the framebuffer.
#[allow(clippy::too_many_arguments)]
pub fn build_ui(
    width: u32,
    height: u32,
    inv: &item::Inventory,
    health: f32,
    hunger: f32,
    survival: bool,
    debug: Option<&[String]>,
) -> Vec<UiVertex> {
    let sw = width as f32;
    let sh = height as f32;
    let mut v = Vec::new();

    // Crosshair.
    let white = [0.95, 0.95, 0.95, 0.85];
    let (cx, cy) = (sw * 0.5, sh * 0.5);
    push_px_rect(&mut v, sw, sh, cx - 9.0, cy - 1.5, 18.0, 3.0, white);
    push_px_rect(&mut v, sw, sh, cx - 1.5, cy - 9.0, 3.0, 18.0, white);

    // Hotbar (9 slots backed by the inventory; shows block swatch + stack count).
    let n = item::HOTBAR;
    let slot = 46.0;
    let pad = 5.0;
    let total = n as f32 * slot + (n as f32 - 1.0) * pad;
    let start_x = (sw - total) * 0.5;
    let y = sh - slot - 18.0;
    push_px_rect(&mut v, sw, sh, start_x - 6.0, y - 6.0, total + 12.0, slot + 12.0, [0.0, 0.0, 0.0, 0.35]);
    for i in 0..n {
        let sx = start_x + i as f32 * (slot + pad);
        if i == inv.selected {
            push_px_rect(&mut v, sw, sh, sx - 3.0, y - 3.0, slot + 6.0, slot + 6.0, [1.0, 1.0, 1.0, 0.95]);
        }
        match inv.slots[i] {
            Some(stack) => {
                let c = item::item_color(stack.item);
                push_px_rect(&mut v, sw, sh, sx, y, slot, slot, [c[0], c[1], c[2], 1.0]);
                if stack.count > 1 {
                    let label = format!("{}", stack.count);
                    let tw = text_width(&label, 2.0);
                    push_text(&mut v, sw, sh, sx + slot - tw - 3.0, y + slot - 18.0, 2.0, &label, [1.0, 1.0, 1.0, 1.0]);
                }
                durability_bar(&mut v, sw, sh, sx, y, slot, stack);
            }
            None => {
                push_px_rect(&mut v, sw, sh, sx, y, slot, slot, [0.1, 0.1, 0.12, 0.5]);
            }
        }
    }
    // Selected item name, centered above the hotbar (and above the stat bars).
    if let Some(stack) = inv.slots[inv.selected] {
        let name = item::item_name(stack.item);
        let tw = text_width(name, 2.0);
        push_text(&mut v, sw, sh, (sw - tw) * 0.5, y - 52.0, 2.0, name, [0.95, 0.95, 0.95, 1.0]);
    }

    // Health (red) and hunger (orange) pip bars sit just above the hotbar in survival mode.
    if survival {
        let pip = 16.0;
        let gap = 3.0;
        let pips = 10;
        let bar_w = pips as f32 * pip + (pips as f32 - 1.0) * gap;
        let bars_y = y - 28.0;
        let health_x = sw * 0.5 - bar_w - 12.0;
        let hunger_x = sw * 0.5 + 12.0;
        stat_bar(&mut v, sw, sh, health_x, bars_y, pip, gap, pips, health, [0.85, 0.13, 0.15, 1.0]);
        stat_bar(&mut v, sw, sh, hunger_x, bars_y, pip, gap, pips, hunger, [0.86, 0.55, 0.18, 1.0]);
    }

    // F3-style debug overlay (top-left): a translucent backing panel + one text line per entry.
    if let Some(lines) = debug {
        let scale = 2.0;
        let lh = font::GLYPH as f32 * scale + 2.0;
        let pad = 4.0;
        let (bx, by) = (8.0, 8.0);
        let maxw = lines
            .iter()
            .map(|l| text_width(l, scale))
            .fold(0.0_f32, f32::max);
        push_px_rect(
            &mut v,
            sw,
            sh,
            bx - pad,
            by - pad,
            maxw + pad * 2.0,
            lines.len() as f32 * lh + pad,
            [0.0, 0.0, 0.0, 0.5],
        );
        for (i, line) in lines.iter().enumerate() {
            push_text(&mut v, sw, sh, bx, by + i as f32 * lh, scale, line, [1.0, 1.0, 1.0, 1.0]);
        }
    }

    v
}
