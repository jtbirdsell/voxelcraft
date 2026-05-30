//! Overlay geometry: the targeted-block wireframe (world space) and the 2D HUD
//! (crosshair + hotbar) in normalized device coordinates.

use glam::{IVec3, Vec3};

use crate::crafting;
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

/// Craft-grid cell rects: (craft cell index 0..9, x, y) for a `size`x`size` grid above the inventory.
/// Cells map into the top-left of the 9-cell craft array (so size=2 uses 0,1,3,4).
pub fn craft_cell_rects(width: u32, height: u32, size: usize) -> Vec<(usize, f32, f32)> {
    let step = INV_SLOT + 4.0;
    let inv_top = inventory_slot_rects(width, height)[0].2;
    let total = size as f32 * step + 60.0 + INV_SLOT; // grid + gap/arrow + output
    let ox = width as f32 * 0.5 - total * 0.5;
    let oy = inv_top - size as f32 * step - 36.0;
    let mut rects = Vec::with_capacity(size * size);
    for r in 0..size {
        for c in 0..size {
            rects.push((r * 3 + c, ox + c as f32 * step, oy + r as f32 * step));
        }
    }
    rects
}

/// Output-slot rect (x, y) to the right of the craft grid.
pub fn craft_output_rect(width: u32, height: u32, size: usize) -> (f32, f32) {
    let step = INV_SLOT + 4.0;
    let cells = craft_cell_rects(width, height, size);
    let (ox, oy) = (cells[0].1, cells[0].2);
    (ox + size as f32 * step + 56.0, oy + (size as f32 * step - INV_SLOT) * 0.5)
}

fn slot_item(out: &mut Vec<UiVertex>, sw: f32, sh: f32, x: f32, y: f32, stack: Option<item::ItemStack>) {
    push_px_rect(out, sw, sh, x, y, INV_SLOT, INV_SLOT, [0.28, 0.28, 0.32, 1.0]);
    if let Some(s) = stack {
        let c = item::item_color(s.item);
        push_px_rect(out, sw, sh, x + 3.0, y + 3.0, INV_SLOT - 6.0, INV_SLOT - 6.0, [c[0], c[1], c[2], 1.0]);
        if s.count > 1 {
            let label = format!("{}", s.count);
            let tw = text_width(&label, 2.0);
            push_text(out, sw, sh, x + INV_SLOT - tw - 3.0, y + INV_SLOT - 18.0, 2.0, &label, [1.0, 1.0, 1.0, 1.0]);
        }
        durability_bar(out, sw, sh, x, y, INV_SLOT, s);
    }
}

/// Build the inventory / crafting screen: dim backdrop, panel, inventory slots, a `size`x`size`
/// craft grid + output preview, the held (cursor) stack, and a tooltip.
pub fn build_inventory_screen(
    width: u32,
    height: u32,
    inv: &item::Inventory,
    craft: &[Option<item::ItemStack>; 9],
    size: usize,
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
    let title = if size >= 3 { "Crafting Table" } else { "Inventory" };
    push_text(&mut v, sw, sh, minx, miny - pad - 18.0, 2.0, title, [0.95, 0.95, 0.95, 1.0]);

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
    // Craft grid + output preview, on its own backing panel above the inventory grid.
    let cells = craft_cell_rects(width, height, size);
    let (out_x, out_y) = craft_output_rect(width, height, size);
    let (cox, coy) = (cells[0].1, cells[0].2);
    let cpw = out_x + INV_SLOT + 8.0 - (cox - 8.0);
    let cph = size as f32 * step + 16.0;
    push_px_rect(&mut v, sw, sh, cox - 8.0, coy - 8.0, cpw, cph, [0.12, 0.12, 0.14, 0.97]);
    let mut grid_ids = [0u16; 9];
    for &(cell, cx, cy) in &cells {
        slot_item(&mut v, sw, sh, cx, cy, craft[cell]);
        grid_ids[cell] = craft[cell].map(|s| s.item).unwrap_or(0);
    }
    push_px_rect(&mut v, sw, sh, out_x - 40.0, out_y + INV_SLOT * 0.5 - 2.0, 28.0, 4.0, [0.8, 0.8, 0.8, 0.9]);
    let result = crafting::match_grid(&grid_ids).map(|(it, n)| item::ItemStack::new(it, n));
    slot_item(&mut v, sw, sh, out_x, out_y, result);

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

/// The three furnace slots (M21), used by both rendering and click hit-testing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FurnaceSlot {
    Input,
    Fuel,
    Output,
}

/// Furnace slot layout: input (top-left), fuel (below it, past the flame gauge), and output (right,
/// past the progress arrow). Returns (kind, x_px, y_px), centered above the inventory grid.
pub fn furnace_slot_rects(width: u32, height: u32) -> [(FurnaceSlot, f32, f32); 3] {
    let step = INV_SLOT + 4.0;
    let gap = 30.0; // vertical flame gap between input and fuel
    let group_w = 3.0 * step + INV_SLOT;
    let left_x = width as f32 * 0.5 - group_w * 0.5;
    let inv_top = inventory_slot_rects(width, height)[0].2;
    let top_y = inv_top - (2.0 * INV_SLOT + gap) - 56.0;
    let out_x = left_x + 3.0 * step;
    let out_y = top_y + (INV_SLOT + gap) * 0.5;
    [
        (FurnaceSlot::Input, left_x, top_y),
        (FurnaceSlot::Fuel, left_x, top_y + INV_SLOT + gap),
        (FurnaceSlot::Output, out_x, out_y),
    ]
}

/// Draw a stack's swatch + count + durability inside a slot (the slot background is drawn already).
fn draw_stack(out: &mut Vec<UiVertex>, sw: f32, sh: f32, x: f32, y: f32, stack: item::ItemStack) {
    let c = item::item_color(stack.item);
    push_px_rect(out, sw, sh, x + 3.0, y + 3.0, INV_SLOT - 6.0, INV_SLOT - 6.0, [c[0], c[1], c[2], 1.0]);
    if stack.count > 1 {
        let label = format!("{}", stack.count);
        let tw = text_width(&label, 2.0);
        push_text(out, sw, sh, x + INV_SLOT - tw - 3.0, y + INV_SLOT - 18.0, 2.0, &label, [1.0, 1.0, 1.0, 1.0]);
    }
    durability_bar(out, sw, sh, x, y, INV_SLOT, stack);
}

/// Build the furnace screen: input/fuel/output slots with a burning-fuel flame gauge and a smelt
/// progress arrow, over the usual inventory grid. `burn_frac`/`cook_frac` are 0..1.
#[allow(clippy::too_many_arguments)]
pub fn build_furnace_screen(
    width: u32,
    height: u32,
    inv: &item::Inventory,
    input: Option<item::ItemStack>,
    fuel: Option<item::ItemStack>,
    output: Option<item::ItemStack>,
    burn_frac: f32,
    cook_frac: f32,
    cursor: (f32, f32),
) -> Vec<UiVertex> {
    let sw = width as f32;
    let sh = height as f32;
    let mut v = Vec::new();
    push_px_rect(&mut v, sw, sh, 0.0, 0.0, sw, sh, [0.0, 0.0, 0.0, 0.55]);

    // Inventory grid (hover-highlighted), on its panel.
    let rects = inventory_slot_rects(width, height);
    let step = INV_SLOT + 4.0;
    let (minx, miny) = (rects[0].1, rects[0].2);
    let panel_w = 9.0 * step - 4.0;
    let pad = 14.0;
    let panel_h = rects[35].2 + INV_SLOT - miny;
    push_px_rect(&mut v, sw, sh, minx - pad, miny - pad - 24.0, panel_w + 2.0 * pad, panel_h + 2.0 * pad + 24.0, [0.12, 0.12, 0.14, 0.97]);
    push_text(&mut v, sw, sh, minx, miny - pad - 18.0, 2.0, "Furnace", [0.95, 0.95, 0.95, 1.0]);

    let mut hovered: Option<usize> = None;
    for &(slot_i, x, y) in &rects {
        let hover = cursor.0 >= x && cursor.0 < x + INV_SLOT && cursor.1 >= y && cursor.1 < y + INV_SLOT;
        if hover {
            hovered = Some(slot_i);
        }
        let bg = if hover { [0.45, 0.45, 0.5, 1.0] } else { [0.28, 0.28, 0.32, 1.0] };
        push_px_rect(&mut v, sw, sh, x, y, INV_SLOT, INV_SLOT, bg);
        if let Some(stack) = inv.slots[slot_i] {
            draw_stack(&mut v, sw, sh, x, y, stack);
        }
    }

    // Furnace slots + gauges on their own backing panel above the inventory.
    let slots = furnace_slot_rects(width, height);
    let p_x = slots[0].1 - 16.0;
    let p_y = slots[0].2 - 34.0;
    let p_w = (slots[2].1 + INV_SLOT) - slots[0].1 + 32.0;
    let p_h = (slots[1].2 + INV_SLOT) - slots[0].2 + 50.0;
    push_px_rect(&mut v, sw, sh, p_x, p_y, p_w, p_h, [0.12, 0.12, 0.14, 0.97]);
    push_text(&mut v, sw, sh, p_x + 8.0, p_y + 8.0, 2.0, "Smelting", [0.8, 0.8, 0.85, 1.0]);

    let mut furn_hover: Option<&str> = None;
    for &(kind, x, y) in &slots {
        let st = match kind {
            FurnaceSlot::Input => input,
            FurnaceSlot::Fuel => fuel,
            FurnaceSlot::Output => output,
        };
        let hover = cursor.0 >= x && cursor.0 < x + INV_SLOT && cursor.1 >= y && cursor.1 < y + INV_SLOT;
        let bg = if hover { [0.45, 0.45, 0.5, 1.0] } else { [0.28, 0.28, 0.32, 1.0] };
        push_px_rect(&mut v, sw, sh, x, y, INV_SLOT, INV_SLOT, bg);
        if let Some(stack) = st {
            draw_stack(&mut v, sw, sh, x, y, stack);
            if hover {
                furn_hover = Some(item::item_name(stack.item));
            }
        }
    }

    // Flame gauge in the gap between input (top) and fuel (bottom), filling upward as fuel burns.
    let (ix, iy) = (slots[0].1, slots[0].2);
    let (flame_w, flame_h) = (20.0, 24.0);
    let fx = ix + INV_SLOT * 0.5 - flame_w * 0.5;
    let fy = iy + INV_SLOT + 3.0;
    push_px_rect(&mut v, sw, sh, fx, fy, flame_w, flame_h, [0.16, 0.11, 0.05, 1.0]);
    let fh = flame_h * burn_frac.clamp(0.0, 1.0);
    if fh > 0.0 {
        push_px_rect(&mut v, sw, sh, fx, fy + (flame_h - fh), flame_w, fh, [1.0, 0.55, 0.12, 1.0]);
    }

    // Progress arrow from input toward output, filling left→right with the smelt progress.
    let arrow_x = ix + INV_SLOT + 12.0;
    let arrow_y = slots[2].2 + INV_SLOT * 0.5 - 3.0;
    let arrow_len = slots[2].1 - arrow_x - 14.0;
    push_px_rect(&mut v, sw, sh, arrow_x, arrow_y, arrow_len, 6.0, [0.22, 0.22, 0.25, 1.0]);
    let prog = arrow_len * cook_frac.clamp(0.0, 1.0);
    if prog > 0.0 {
        push_px_rect(&mut v, sw, sh, arrow_x, arrow_y, prog, 6.0, [0.95, 0.92, 0.55, 1.0]);
    }
    push_px_rect(&mut v, sw, sh, arrow_x + arrow_len, arrow_y - 4.0, 5.0, 14.0, [0.5, 0.5, 0.55, 1.0]);

    // Held stack follows the cursor; otherwise a hover tooltip (furnace slot beats inventory slot).
    if let Some(held) = inv.held {
        let c = item::item_color(held.item);
        let sz = INV_SLOT - 6.0;
        let (hx, hy) = (cursor.0 - sz * 0.5, cursor.1 - sz * 0.5);
        push_px_rect(&mut v, sw, sh, hx, hy, sz, sz, [c[0], c[1], c[2], 1.0]);
        if held.count > 1 {
            let label = format!("{}", held.count);
            let tw = text_width(&label, 2.0);
            push_text(&mut v, sw, sh, hx + sz - tw - 2.0, hy + sz - 18.0, 2.0, &label, [1.0, 1.0, 1.0, 1.0]);
        }
        durability_bar(&mut v, sw, sh, hx - 4.0, hy, sz + 8.0, held);
    } else {
        let name = furn_hover.or_else(|| hovered.and_then(|i| inv.slots[i]).map(|s| item::item_name(s.item)));
        if let Some(name) = name {
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
