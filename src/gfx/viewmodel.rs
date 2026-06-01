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
/// Seconds for one swing arc (≈ Minecraft's 6-tick swing).
const SWING_TIME: f32 = 0.28;
/// Seconds to lower + raise the item on a hotbar swap.
const EQUIP_TIME: f32 = 0.18;
/// Walk-bob phase rate (rad/s) at full walk speed, and the speed that maps to full bob strength.
const BOB_CADENCE: f32 = 6.5;
const WALK_SPEED_REF: f32 = 4.3;
const BOB_SMOOTH: f32 = 8.0; // bob-strength low-pass rate
/// Look-sway: view-space offset per unit look delta, low-pass rate, and the max offset.
const SWAY_GAIN: f32 = 0.9;
const SWAY_SMOOTH: f32 = 10.0;
const SWAY_CLAMP: f32 = 0.05;

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

/// The active "use" animation this frame, as normalized 0..1 progress per kind (mutually exclusive in
/// the action system, so at most one is non-zero). Drives the eat / bow-draw / shield-raise poses.
#[derive(Clone, Copy, Default)]
pub struct UsePose {
    pub eat: f32,    // eat_progress / EAT_TIME
    pub draw: f32,   // draw_progress / BOW_DRAW_TIME
    pub shield: f32, // shield_progress / SHIELD_RAISE_DELAY
}

/// Animation state for the first-person view-model: swing arc, equip lower→raise, walk-bob, look-sway,
/// and the active use pose. Composed into the per-frame pose transform.
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
    /// The active use animation this frame (eat / bow-draw / shield-raise).
    pub use_pose: UsePose,
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
            use_pose: UsePose::default(),
        }
    }
}

impl ViewModel {
    /// Advance the animation. `sel_item` detects a hotbar swap (equip dip); `horiz_speed`/`on_ground`
    /// drive the walk-bob; `(yaw_d, pitch_d)` drive the look-sway; `swing_start`/`swing_loop` the swing
    /// arc; `use_pose` the eat/draw/shield pose.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dt: f32,
        sel_item: ItemId,
        horiz_speed: f32,
        on_ground: bool,
        yaw_d: f32,
        pitch_d: f32,
        swing_start: bool,
        swing_loop: bool,
        use_pose: UsePose,
    ) {
        self.use_pose = use_pose;
        // Equip: a hotbar swap drops the item, then it eases back up. While lowered, suppress swings.
        if sel_item != self.prev_item {
            self.equip = 0.0;
            self.prev_item = sel_item;
        }
        self.equip = (self.equip + dt / EQUIP_TIME).min(1.0);
        let equipped = self.equip >= 1.0;
        // Swing.
        if swing_start && equipped {
            self.swing = 0.0;
            self.swinging = true;
        }
        if self.swinging {
            self.swing += dt / SWING_TIME;
            if self.swing >= 1.0 {
                if swing_loop && equipped {
                    self.swing -= 1.0; // keep swinging while held
                } else {
                    self.swing = 0.0;
                    self.swinging = false;
                }
            }
        } else if swing_loop && equipped {
            self.swinging = true; // begin a held swing loop
            self.swing = 0.0;
        }
        // Walk-bob: strength low-passes toward (speed / ref) while grounded; phase advances with it.
        let target = if on_ground {
            (horiz_speed / WALK_SPEED_REF).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.bob_strength += (target - self.bob_strength) * (BOB_SMOOTH * dt).min(1.0);
        self.bob_phase =
            (self.bob_phase + BOB_CADENCE * self.bob_strength * dt) % std::f32::consts::TAU;
        // Look-sway: a smoothed, clamped offset opposite the turn direction.
        let target_sway =
            (glam::Vec2::new(-yaw_d, pitch_d) * SWAY_GAIN).clamp_length_max(SWAY_CLAMP);
        self.sway += (target_sway - self.sway) * (SWAY_SMOOTH * dt).min(1.0);
    }

    /// The swing arc contribution this frame: a (translation, extra pitch, extra roll) that dips the
    /// item down-forward and rotates it, peaking mid-swing. Zero when not swinging.
    fn swing_pose(&self) -> (Vec3, f32, f32) {
        if !self.swinging {
            return (Vec3::ZERO, 0.0, 0.0);
        }
        let p = self.swing.clamp(0.0, 1.0);
        let a = (p * std::f32::consts::PI).sin(); // 0 → 1 → 0 envelope
        let trans = Vec3::new(
            -0.10 * a,
            -0.18 * a + 0.10 * (p * std::f32::consts::TAU).sin(),
            -0.16 * a,
        );
        (trans, 1.1 * a, 0.5 * a)
    }

    /// The active use-pose contribution: (translation, extra pitch, extra yaw, weight). A use pose
    /// overrides the swing by its weight. Mutually exclusive (only one progress is non-zero).
    fn use_offset(&self) -> (Vec3, f32, f32, f32) {
        let u = self.use_pose;
        if u.eat > 0.0 {
            // Food rises toward the mouth (up + toward camera) with a chewing wobble.
            let w = smoothstep(0.0, 0.3, u.eat);
            let chew = (u.eat * 28.0).sin() * 0.015 * w;
            return (Vec3::new(-0.16 * w, 0.20 * w + chew, 0.12 * w), 0.55 * w, 0.0, w);
        }
        if u.draw > 0.0 {
            // The bow rises up toward the aiming centre as it charges, with a hold-shake at full draw.
            let t = u.draw;
            let shake = if t > 0.95 { (t * 70.0).sin() * 0.006 } else { 0.0 };
            return (Vec3::new(-0.20 * t + shake, 0.12 * t, 0.10 * t), -0.10 * t, 0.35 * t, 1.0);
        }
        if u.shield > 0.0 {
            // The shield slides toward the centre into a blocking pose.
            let t = u.shield;
            return (Vec3::new(-0.28 * t, 0.16 * t, 0.10 * t), 0.10 * t, 0.40 * t, t);
        }
        (Vec3::ZERO, 0.0, 0.0, 0.0)
    }

    /// Idle walk-bob + look-sway contribution: (translation, extra roll).
    fn idle_offset(&self) -> (Vec3, f32) {
        let amp = self.bob_strength;
        let bob = Vec3::new(
            self.bob_phase.cos() * 0.018 * amp,
            -(self.bob_phase * 2.0).sin().abs() * 0.020 * amp,
            0.0,
        );
        let roll = self.bob_phase.sin() * 0.03 * amp;
        (bob + Vec3::new(self.sway.x, self.sway.y, 0.0), roll)
    }

    /// Equip lower→raise contribution: (translation, extra pitch). 1 = fully raised.
    fn equip_offset(&self) -> (Vec3, f32) {
        let e = self.equip;
        let s = e * e * (3.0 - 2.0 * e); // smoothstep ease
        (Vec3::new(0.0, -0.9 * (1.0 - s), 0.0), 1.2 * (1.0 - s))
    }

    /// World-space-ish camera head-bob (local right.x, world up.y), driven by the same walk-bob phase.
    /// Subtle; the caller maps `.x` along the camera's right vector and `.y` to world up.
    pub fn camera_bob(&self) -> Vec3 {
        let amp = self.bob_strength;
        Vec3::new(
            self.bob_phase.cos() * 0.022 * amp,
            -(self.bob_phase * 2.0).sin().abs() * 0.028 * amp,
            0.0,
        )
    }

    /// Compose a pose from a base anchor + orientation plus the swing arc, use pose, equip, and idle
    /// motion (a use pose damps the swing + idle by its weight).
    fn pose(&self, base_t: Vec3, yaw: f32, base_pitch: f32, base_roll: f32) -> Mat4 {
        let (st, sp, sr) = self.swing_pose();
        let (ut, up, uy, uw) = self.use_offset();
        let (it, ir) = self.idle_offset();
        let (et, ep) = self.equip_offset();
        let damp = 1.0 - uw;
        Mat4::from_translation(base_t + st * damp + ut + it * damp + et)
            * Mat4::from_rotation_y(yaw + uy)
            * Mat4::from_rotation_x(base_pitch + sp * damp + up + ep)
            * Mat4::from_rotation_z(base_roll + sr * damp + ir * damp)
    }

    /// The view-space pose transform for a 3D held item (block cube / arm). A noticeable tilt shows
    /// the cube's faces.
    fn transform(&self) -> Mat4 {
        self.pose(REST, BASE_YAW, BASE_PITCH, 0.0)
    }

    /// Pose for a flat sprite card (tools/materials/food). Held nearly face-on — only a slight turn +
    /// tip — so the 2D sprite stays readable instead of going edge-on like the 3D pose would.
    fn card_transform(&self) -> Mat4 {
        self.pose(Vec3::new(0.42, -0.42, -0.78), -0.16, 0.10, 0.0)
    }

    /// Build the view-space geometry for the currently-held item. `light` is the local brightness
    /// scalar baked into the vertices. Block items → a textured mini-cube; tools/materials/food → a
    /// flat sprite card (alpha-cutout, via `item::item_tile`); empty hand → a skin-toned fist.
    pub fn build_geometry(&self, sel_item: ItemId, light: f32) -> Option<Geometry> {
        let mut geom = Geometry::default();
        let m = self.transform();
        if sel_item == block::AIR {
            push_arm(&mut geom, &m, light);
        } else if let Some(block) = item::block_of_item(sel_item) {
            push_block_cube(&mut geom, block, &m, light);
        } else {
            push_sprite_card(
                &mut geom,
                item::item_tile(sel_item),
                &self.card_transform(),
                light,
                item::item_emission(sel_item),
            );
        }
        Some(geom)
    }
}

/// Hermite smoothstep, for easing the use-pose weight in.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A flat, camera-facing sprite card textured with one atlas tile (an item sprite), upright (tile row
/// 0 = card top). Single quad — the view-model pipeline has culling off, so it shows from both sides.
fn push_sprite_card(geom: &mut Geometry, tile: u32, m: &Mat4, light: f32, emission: f32) {
    let (hw, hh) = (0.26, 0.28);
    let shade = [emission, 0.0];
    let quad = [
        (Vec3::new(-hw, hh, 0.0), [0.0, 0.0]),  // top-left
        (Vec3::new(hw, hh, 0.0), [1.0, 0.0]),   // top-right
        (Vec3::new(hw, -hh, 0.0), [1.0, 1.0]),  // bottom-right
        (Vec3::new(-hw, -hh, 0.0), [0.0, 1.0]), // bottom-left
    ];
    let normal = m.transform_vector3(Vec3::Z).normalize_or_zero().to_array();
    let base = geom.vertices.len() as u32;
    for (p, uv) in quad {
        geom.vertices.push(Vertex {
            position: m.transform_point3(p).to_array(),
            normal,
            uv,
            tile,
            light: [light, light],
            shade,
        });
    }
    geom.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// A skin-toned fist/forearm box for the empty hand (uses the pinkish mob tile — no skin sprite).
fn push_arm(geom: &mut Geometry, m: &Mat4, light: f32) {
    let half = Vec3::new(0.11, 0.2, 0.11);
    let arm = *m * Mat4::from_translation(Vec3::new(-0.05, -0.05, 0.0));
    push_box_xform(geom, half, &arm, &[block::tile::MOB; 6], light, 0.0);
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
        // A tool builds a flat sprite card (one quad).
        let card = vm
            .build_geometry(item::DIAMOND_PICKAXE, 1.0)
            .expect("a held tool builds a sprite card");
        assert_eq!(card.vertices.len(), 4);
        assert_eq!(card.indices.len(), 6);
        // Empty hand (AIR) still draws something (the fist), all in front of the camera.
        let hand = vm.build_geometry(block::AIR, 1.0).expect("empty hand draws a fist");
        assert!(!hand.vertices.is_empty());
        for v in &hand.vertices {
            assert!(v.position[2] < 0.0);
        }
    }
}
