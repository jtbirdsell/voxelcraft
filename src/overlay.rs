//! Overlay geometry: the targeted-block wireframe (world space) and the 2D HUD
//! (crosshair + hotbar) in normalized device coordinates.

use glam::{IVec3, Vec3};

use crate::block::{self, BlockId};
use crate::font;

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

pub struct Hotbar {
    pub slots: Vec<BlockId>,
    pub selected: usize,
}

impl Hotbar {
    pub fn new() -> Self {
        Self {
            slots: vec![
                block::GRASS,
                block::DIRT,
                block::STONE,
                block::SAND,
                block::WOOD,
                block::LEAVES,
                block::SNOW,
                block::WATER,
                block::LAVA,
                block::COAL_ORE,
            ],
            selected: 0,
        }
    }

    pub fn selected_block(&self) -> BlockId {
        self.slots[self.selected]
    }

    pub fn select(&mut self, index: usize) {
        if index < self.slots.len() {
            self.selected = index;
        }
    }

    pub fn scroll(&mut self, delta: i32) {
        let n = self.slots.len() as i32;
        self.selected = (((self.selected as i32 + delta) % n + n) % n) as usize;
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
    hotbar: &Hotbar,
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

    // Hotbar.
    let n = hotbar.slots.len();
    let slot = 46.0;
    let pad = 5.0;
    let total = n as f32 * slot + (n as f32 - 1.0) * pad;
    let start_x = (sw - total) * 0.5;
    let y = sh - slot - 18.0;
    let bg = [0.0, 0.0, 0.0, 0.35];
    push_px_rect(
        &mut v,
        sw,
        sh,
        start_x - 6.0,
        y - 6.0,
        total + 12.0,
        slot + 12.0,
        bg,
    );
    for (i, &id) in hotbar.slots.iter().enumerate() {
        let sx = start_x + i as f32 * (slot + pad);
        if i == hotbar.selected {
            push_px_rect(&mut v, sw, sh, sx - 3.0, y - 3.0, slot + 6.0, slot + 6.0, [1.0, 1.0, 1.0, 0.95]);
        }
        let c = block::face_color(id, [0, 1, 0]);
        push_px_rect(&mut v, sw, sh, sx, y, slot, slot, [c[0], c[1], c[2], 1.0]);
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
