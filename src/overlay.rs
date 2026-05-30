//! Overlay geometry: the targeted-block wireframe (world space) and the 2D HUD
//! (crosshair + hotbar) in normalized device coordinates.

use glam::{IVec3, Vec3};

use crate::block::{self, BlockId};

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
    pub color: [f32; 4],
}

impl UiVertex {
    pub const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];
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
        out.push(UiVertex { pos: p, color });
    }
}

/// Build the HUD (crosshair + hotbar) for the given framebuffer size.
pub fn build_ui(width: u32, height: u32, hotbar: &Hotbar) -> Vec<UiVertex> {
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
    v
}
