//! First-person view-model (M34-VM): the held item rendered in front of the camera, with swing/use
//! animations and idle bob/sway. This module owns the CPU side — the animation STATE and the per-frame
//! VIEW-SPACE geometry it bakes — which a dedicated LDR render pass (renderer.rs `record_viewmodel`)
//! draws after the DLSS resolve + tonemap, isolated from the HDR scene / G-buffer / motion guides.
//!
//! View space is `look_to_rh`: −Z forward (into the screen), +X right, +Y up. The held item sits in
//! the bottom-right, a short distance in front of the near plane, and is projected by a fixed
//! near-perspective (`projection`) so it never clips into the world and is immune to the world FOV.

use glam::{Mat4, Vec3};

use crate::block;
use crate::item::{self, ItemId};
use crate::mesher::{Geometry, Vertex};

/// The held item's rest anchor in view space (right, down, forward).
const REST: Vec3 = Vec3::new(0.55, -0.5, -0.85);
/// Small base orientation so the item is angled, not axis-aligned to the screen.
const BASE_YAW: f32 = -0.30; // toed inward toward the screen center
const BASE_PITCH: f32 = 0.20; // tipped up a touch
/// Held-block cube half-extent (≈0.4-unit cube).
const BLOCK_HALF: f32 = 0.2;

/// Per-frame uniform for the view-model pass: the fixed near-perspective projection + a brightness
/// (local-light) scalar. 80 bytes, std140-friendly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewModelUniform {
    pub proj: [[f32; 4]; 4],
    /// x = brightness; y/z/w reserved.
    pub params: [f32; 4],
}

impl ViewModelUniform {
    pub fn new(aspect: f32, brightness: f32) -> Self {
        // znear 0.01 so the hand never clips into a wall; independent of the world camera.
        let proj = Mat4::perspective_rh(70_f32.to_radians(), aspect.max(1e-4), 0.01, 10.0);
        Self {
            proj: proj.to_cols_array_2d(),
            params: [brightness, 0.0, 0.0, 0.0],
        }
    }
}

/// Animation state for the first-person view-model. Fields beyond the rest pose are filled in by the
/// later milestones (swing/use/bob/sway/equip); VM1 renders the static rest pose only.
#[allow(dead_code)] // swing/equip/bob/sway are consumed by VM3–VM5; staged here so the struct is stable.
pub struct ViewModel {
    /// Swing arc progress 0..1 (one-shot, retriggerable) and whether one is playing.
    pub swing: f32,
    pub swinging: bool,
    /// Equip lower→raise: 1 = fully raised, 0 = lowered off-screen; `prev_item` detects a swap.
    pub equip: f32,
    pub prev_item: ItemId,
    /// Idle walk bob: a phase accumulator + a smoothed strength from horizontal speed.
    pub bob_phase: f32,
    pub bob_strength: f32,
    /// Smoothed look-sway offset (view-space x,y).
    pub sway: glam::Vec2,
}

impl Default for ViewModel {
    fn default() -> Self {
        Self {
            swing: 0.0,
            swinging: false,
            equip: 1.0,
            prev_item: block::AIR,
            bob_phase: 0.0,
            bob_strength: 0.0,
            sway: glam::Vec2::ZERO,
        }
    }
}

impl ViewModel {
    /// The view-space pose transform applied to the held item this frame. VM1: the static rest pose.
    /// Later milestones compose equip / swing / use / bob / sway offsets here.
    fn transform(&self) -> Mat4 {
        Mat4::from_translation(REST)
            * Mat4::from_rotation_y(BASE_YAW)
            * Mat4::from_rotation_x(BASE_PITCH)
    }

    /// Build the view-space geometry for the currently-held item, or `None` when there is nothing to
    /// draw. `light` is the local brightness scalar baked into the vertices. VM1 handles block items
    /// (a textured mini-cube); item sprites + the empty hand arrive in VM2.
    pub fn build_geometry(&self, sel_item: ItemId, light: f32) -> Option<Geometry> {
        let block = item::block_of_item(sel_item)?;
        if block == block::AIR {
            return None;
        }
        let mut geom = Geometry::default();
        let m = self.transform();
        push_block_cube(&mut geom, block, &m, light);
        Some(geom)
    }
}

/// Emit a textured cube for a held block: per-face atlas tiles via `block::face_tile`, transformed by
/// `m` into view space.
fn push_block_cube(geom: &mut Geometry, block: block::BlockId, m: &Mat4, light: f32) {
    let half = Vec3::splat(BLOCK_HALF);
    // Faces ordered +X, −X, +Y, −Y, +Z, −Z (matches the `faces` table below); each takes the block's
    // tile for that facing so e.g. grass shows grass-top / dirt-side / dirt-bottom.
    let tiles = [
        block::face_tile(block, [1, 0, 0]),
        block::face_tile(block, [-1, 0, 0]),
        block::face_tile(block, [0, 1, 0]),
        block::face_tile(block, [0, -1, 0]),
        block::face_tile(block, [0, 0, 1]),
        block::face_tile(block, [0, 0, -1]),
    ];
    push_box_xform(geom, half, m, &tiles, light, block::emission(block));
}

/// Append a textured box, transforming each corner + normal by `m` (a full view-space transform — not
/// just a yaw like `entity::push_box`). `tiles` are the six face tiles in +X,−X,+Y,−Y,+Z,−Z order.
fn push_box_xform(geom: &mut Geometry, half: Vec3, m: &Mat4, tiles: &[u32; 6], light: f32, emission: f32) {
    let shade = [emission, 0.0];
    let face_uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let cor = [
        Vec3::new(-half.x, -half.y, -half.z),
        Vec3::new(half.x, -half.y, -half.z),
        Vec3::new(half.x, -half.y, half.z),
        Vec3::new(-half.x, -half.y, half.z),
        Vec3::new(-half.x, half.y, -half.z),
        Vec3::new(half.x, half.y, -half.z),
        Vec3::new(half.x, half.y, half.z),
        Vec3::new(-half.x, half.y, half.z),
    ];
    // (corner indices forming the quad, outward normal, tile slot) — same winding as entity::push_box.
    let faces: [([usize; 4], Vec3, usize); 6] = [
        ([1, 2, 6, 5], Vec3::X, 0),
        ([3, 0, 4, 7], Vec3::NEG_X, 1),
        ([4, 5, 6, 7], Vec3::Y, 2),
        ([0, 3, 2, 1], Vec3::NEG_Y, 3),
        ([2, 3, 7, 6], Vec3::Z, 4),
        ([0, 1, 5, 4], Vec3::NEG_Z, 5),
    ];
    for (idx, n, ti) in faces {
        let normal = m.transform_vector3(n).normalize_or_zero().to_array();
        let tile = tiles[ti];
        let base = geom.vertices.len() as u32;
        for (j, &k) in idx.iter().enumerate() {
            geom.vertices.push(Vertex {
                position: m.transform_point3(cor[k]).to_array(),
                normal,
                uv: face_uv[j],
                tile,
                light: [light, light],
                shade,
            });
        }
        geom.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_block_builds_a_cube_in_front_of_the_camera() {
        let vm = ViewModel::default();
        let grass = item::item_of_block(block::GRASS);
        let geom = vm.build_geometry(grass, 1.0).expect("a held block builds geometry");
        // A cube: 6 faces × 4 verts, 6 faces × 6 indices.
        assert_eq!(geom.vertices.len(), 24);
        assert_eq!(geom.indices.len(), 36);
        // Every vertex sits in front of the camera (view-space −Z) and to the lower-right.
        for v in &geom.vertices {
            assert!(v.position[2] < 0.0, "held item must be in front of the camera (z={})", v.position[2]);
        }
        // Holding nothing (AIR) draws nothing.
        assert!(vm.build_geometry(block::AIR, 1.0).is_none());
    }
}
