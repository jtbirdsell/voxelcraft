//! Entities: wandering mobs and dropped items. Both are AABBs with the same swept, axis-separated
//! voxel collision the player uses, and both are drawn as boxes through the chunk pipeline, so they
//! pick up ray-traced shadows / AO / GI like the rest of the world. Mobs wander with a tiny random
//! AI; items fall, rest, bob/spin, and are collected when the player walks over them.

use std::f32::consts::TAU;

use glam::{IVec3, Vec3};

use crate::block::{self, BlockId};
use crate::mesher::{Geometry, MeshData, Vertex};

const GRAVITY: f32 = 28.0;
const MOB_SPEED: f32 = 1.8;
const MOB_W: f32 = 0.8;
const MOB_H: f32 = 0.9;
const ITEM_SIZE: f32 = 0.28;
const ITEM_LIFETIME: f32 = 90.0;
const COLLECT_RADIUS: f32 = 1.2;
const FALL_OUT_Y: f32 = -24.0;

#[inline]
fn comp(v: Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[inline]
fn set_comp(v: &mut Vec3, axis: usize, value: f32) {
    match axis {
        0 => v.x = value,
        1 => v.y = value,
        _ => v.z = value,
    }
}

/// xorshift64 — a tiny dependency-free RNG; each entity carries its own (nonzero) state.
#[inline]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn randf(state: &mut u64) -> f32 {
    (xorshift(state) >> 40) as f32 / 16_777_216.0
}

enum Kind {
    Mob,
    Item(BlockId),
}

struct Entity {
    kind: Kind,
    pos: Vec3, // feet / bottom-center
    vel: Vec3,
    on_ground: bool,
    age: f32,
    heading: f32,
    wander: f32,
    idle: bool,
    rng: u64,
    dead: bool,
}

#[derive(Default)]
pub struct Entities {
    list: Vec<Entity>,
    spawn_counter: u64,
}

impl Entities {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn count(&self) -> usize {
        self.list.len()
    }

    fn next_seed(&mut self) -> u64 {
        self.spawn_counter = self.spawn_counter.wrapping_add(1);
        // Mix the counter so seeds are well-spread and never zero.
        let mut h = self.spawn_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        h ^= h >> 29;
        h | 1
    }

    pub fn spawn_mob(&mut self, pos: Vec3) {
        let mut rng = self.next_seed();
        let heading = randf(&mut rng) * TAU;
        self.list.push(Entity {
            kind: Kind::Mob,
            pos,
            vel: Vec3::ZERO,
            on_ground: false,
            age: 0.0,
            heading,
            wander: 1.0 + randf(&mut rng) * 2.0,
            idle: false,
            rng,
            dead: false,
        });
    }

    pub fn spawn_item(&mut self, pos: Vec3, block: BlockId) {
        let mut rng = self.next_seed();
        // Pop out with a little random horizontal velocity and an upward kick.
        let a = randf(&mut rng) * TAU;
        let vel = Vec3::new(a.cos() * 1.5, 3.0, a.sin() * 1.5);
        self.list.push(Entity {
            kind: Kind::Item(block),
            pos,
            vel,
            on_ground: false,
            age: 0.0,
            heading: 0.0,
            wander: 0.0,
            idle: false,
            rng,
            dead: false,
        });
    }

    /// Advance AI + physics for all entities and drop the dead/collected ones.
    pub fn update(&mut self, dt: f32, player_pos: Vec3, is_solid: impl Fn(IVec3) -> bool) {
        for e in &mut self.list {
            e.age += dt;
            match e.kind {
                Kind::Mob => {
                    e.wander -= dt;
                    if e.wander <= 0.0 {
                        e.wander = 2.0 + randf(&mut e.rng) * 3.0;
                        e.idle = randf(&mut e.rng) < 0.3;
                        e.heading = randf(&mut e.rng) * TAU;
                    }
                    if e.idle {
                        e.vel.x = 0.0;
                        e.vel.z = 0.0;
                    } else {
                        e.vel.x = e.heading.cos() * MOB_SPEED;
                        e.vel.z = e.heading.sin() * MOB_SPEED;
                        // Occasional hop helps clear 1-block steps without real pathfinding.
                        if e.on_ground && randf(&mut e.rng) < 0.02 {
                            e.vel.y = 7.0;
                        }
                    }
                    e.vel.y -= GRAVITY * dt;
                    e.on_ground = collide_move(&mut e.pos, &mut e.vel, MOB_W, MOB_H, dt, &is_solid);
                    if e.pos.y < FALL_OUT_Y {
                        e.dead = true;
                    }
                }
                Kind::Item(_) => {
                    e.vel.y -= GRAVITY * dt;
                    e.on_ground = collide_move(&mut e.pos, &mut e.vel, ITEM_SIZE, ITEM_SIZE, dt, &is_solid);
                    if e.on_ground {
                        e.vel.x *= 0.7;
                        e.vel.z *= 0.7;
                    }
                    // Collect once it has settled a moment and the player is close.
                    if e.age > 0.4 && (player_pos - e.pos).length() < COLLECT_RADIUS {
                        e.dead = true;
                    }
                    if e.age > ITEM_LIFETIME || e.pos.y < FALL_OUT_Y {
                        e.dead = true;
                    }
                }
            }
        }
        self.list.retain(|e| !e.dead);
    }

    /// Box geometry for every entity (opaque layer), lit by the chunk pipeline.
    pub fn build_mesh(&self) -> MeshData {
        let mut mesh = MeshData::default();
        for e in &self.list {
            match e.kind {
                Kind::Mob => {
                    let half = MOB_W * 0.5;
                    let body = [0.86, 0.55, 0.58];
                    let body_min = e.pos + Vec3::new(-half, 0.0, -half);
                    let body_max = e.pos + Vec3::new(half, MOB_H * 0.78, half);
                    push_box(&mut mesh.opaque, body_min, body_max, 0.0, body, 0.0);
                    // A smaller head offset toward the heading gives the box a facing.
                    let hs = MOB_W * 0.3;
                    let hc = e.pos
                        + Vec3::new(
                            e.heading.cos() * half * 0.9,
                            MOB_H * 0.72,
                            e.heading.sin() * half * 0.9,
                        );
                    push_box(
                        &mut mesh.opaque,
                        hc - Vec3::new(hs, hs, hs),
                        hc + Vec3::new(hs, hs * 1.6, hs),
                        0.0,
                        [0.80, 0.50, 0.52],
                        0.0,
                    );
                }
                Kind::Item(b) => {
                    let c = block::face_color(b, [0, 1, 0]);
                    let em = block::emission(b);
                    let s = ITEM_SIZE * 0.5;
                    let bob = (e.age * 3.0).sin() * 0.06;
                    let center = e.pos + Vec3::new(0.0, ITEM_SIZE * 0.5 + 0.05 + bob, 0.0);
                    push_box(
                        &mut mesh.opaque,
                        center - Vec3::new(s, s, s),
                        center + Vec3::new(s, s, s),
                        e.age * 1.6,
                        c,
                        em,
                    );
                }
            }
        }
        mesh
    }
}

/// Move an AABB (width `w`, height `h`, feet at `pos`) by `vel*dt` with axis-separated voxel
/// collision; returns whether it ended up resting on ground.
fn collide_move(
    pos: &mut Vec3,
    vel: &mut Vec3,
    w: f32,
    h: f32,
    dt: f32,
    is_solid: &impl Fn(IVec3) -> bool,
) -> bool {
    let half = w * 0.5;
    let delta = *vel * dt;
    let mut on_ground = false;
    for &axis in &[0usize, 2, 1] {
        let amount = comp(delta, axis);
        if amount == 0.0 {
            continue;
        }
        set_comp(pos, axis, comp(*pos, axis) + amount);

        let min = Vec3::new(pos.x - half, pos.y, pos.z - half);
        let max = Vec3::new(pos.x + half, pos.y + h, pos.z + half);
        let x0 = min.x.floor() as i32;
        let x1 = (max.x - 1e-4).floor() as i32;
        let y0 = min.y.floor() as i32;
        let y1 = (max.y - 1e-4).floor() as i32;
        let z0 = min.z.floor() as i32;
        let z1 = (max.z - 1e-4).floor() as i32;

        let (lo, hi) = if axis == 1 { (0.0, h) } else { (-half, half) };
        let mut clamp: Option<f32> = None;
        for vx in x0..=x1 {
            for vy in y0..=y1 {
                for vz in z0..=z1 {
                    if is_solid(IVec3::new(vx, vy, vz)) {
                        let coord = match axis {
                            0 => vx,
                            1 => vy,
                            _ => vz,
                        } as f32;
                        if amount > 0.0 {
                            let p = coord - hi;
                            clamp = Some(clamp.map_or(p, |c| c.min(p)));
                        } else {
                            let p = (coord + 1.0) - lo;
                            clamp = Some(clamp.map_or(p, |c| c.max(p)));
                        }
                    }
                }
            }
        }
        if let Some(p) = clamp {
            set_comp(pos, axis, p);
            set_comp(vel, axis, 0.0);
            if axis == 1 && amount < 0.0 {
                on_ground = true;
            }
        }
    }
    on_ground
}

/// Append a (optionally yaw-rotated) colored box to `geom`. `color` is rgb, `emission` the alpha.
fn push_box(geom: &mut Geometry, min: Vec3, max: Vec3, yaw: f32, color: [f32; 3], emission: f32) {
    let center = (min + max) * 0.5;
    let (s, co) = yaw.sin_cos();
    let rot = |p: Vec3| -> [f32; 3] {
        let dx = p.x - center.x;
        let dz = p.z - center.z;
        [center.x + dx * co - dz * s, p.y, center.z + dx * s + dz * co]
    };
    let rotn = |n: [f32; 3]| -> [f32; 3] { [n[0] * co - n[2] * s, n[1], n[0] * s + n[2] * co] };
    let col = [color[0], color[1], color[2], emission];

    let cor = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    // (corner indices forming the quad, outward face normal)
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([1, 2, 6, 5], [1.0, 0.0, 0.0]),
        ([3, 0, 4, 7], [-1.0, 0.0, 0.0]),
        ([4, 5, 6, 7], [0.0, 1.0, 0.0]),
        ([0, 3, 2, 1], [0.0, -1.0, 0.0]),
        ([2, 3, 7, 6], [0.0, 0.0, 1.0]),
        ([0, 1, 5, 4], [0.0, 0.0, -1.0]),
    ];
    for (idx, n) in faces {
        let normal = rotn(n);
        let base = geom.vertices.len() as u32;
        for &k in &idx {
            geom.vertices.push(Vertex {
                position: rot(cor[k]),
                normal,
                color: col,
            });
        }
        geom.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}
