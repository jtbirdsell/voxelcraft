//! Entities: wandering mobs and dropped items. Both are AABBs with the same swept, axis-separated
//! voxel collision the player uses, and both are drawn as boxes through the chunk pipeline, so they
//! pick up ray-traced shadows / AO / GI like the rest of the world. Mobs wander with a tiny random
//! AI; items fall, rest, bob/spin, and are collected when the player walks over them.

use std::f32::consts::TAU;

use glam::{IVec3, Vec3};

use crate::block;
use crate::block::tile as T;
use crate::item::{self, ItemStack};
use crate::mesher::{Geometry, MeshData, Vertex};

const GRAVITY: f32 = 28.0;
const MOB_SPEED: f32 = 1.8;
const ITEM_SIZE: f32 = 0.28;
const ITEM_LIFETIME: f32 = 90.0;
const COLLECT_RADIUS: f32 = 1.2;
const FALL_OUT_Y: f32 = -24.0;
const XP_SIZE: f32 = 0.18;
const XP_HOMING_RADIUS: f32 = 4.0; // orbs start drifting toward the player within this range
const XP_HOMING_SPEED: f32 = 9.0;

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

/// A creature type (M27). Passive species wander and flee; hostile species (used by combat in M29)
/// will hunt the player. Each has a distinct size, health, and multi-box `model()`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Species {
    Cow,
    Pig,
    Sheep,
    Chicken,
    Zombie,
    Skeleton,
    Creeper,
    Spider,
}

impl Species {
    /// All species, for spawning a representative set.
    pub const ALL: [Species; 8] = [
        Species::Cow,
        Species::Pig,
        Species::Sheep,
        Species::Chicken,
        Species::Zombie,
        Species::Skeleton,
        Species::Creeper,
        Species::Spider,
    ];

    fn max_health(self) -> f32 {
        match self {
            Species::Chicken => 4.0,
            Species::Sheep => 8.0,
            Species::Cow | Species::Pig => 10.0,
            Species::Spider => 16.0,
            Species::Zombie | Species::Skeleton | Species::Creeper => 20.0,
        }
    }

    /// Collision AABB (width, height).
    fn size(self) -> (f32, f32) {
        match self {
            Species::Chicken => (0.4, 0.7),
            Species::Pig => (0.8, 0.9),
            Species::Sheep => (0.8, 1.3),
            Species::Cow => (0.9, 1.4),
            Species::Creeper => (0.6, 1.7),
            Species::Zombie | Species::Skeleton => (0.6, 1.9),
            Species::Spider => (1.1, 0.7),
        }
    }

    /// Hostile species hunt the player (wired up by combat/AI in M28-M29).
    #[allow(dead_code)]
    pub fn hostile(self) -> bool {
        matches!(
            self,
            Species::Zombie | Species::Skeleton | Species::Creeper | Species::Spider
        )
    }
}

/// One box of a mob's model, in local space (feet at origin, +x = forward/facing, +y = up).
struct Part {
    min: [f32; 3],
    max: [f32; 3],
    tile: u32,
}

#[inline]
fn part(min: [f32; 3], max: [f32; 3], tile: u32) -> Part {
    Part { min, max, tile }
}

/// Build a species' multi-box model. Distinct silhouettes/sizes/colors so types read apart.
fn model(s: Species) -> Vec<Part> {
    match s {
        Species::Cow => quadruped(T::MOB_COW, 0.9, 0.6, 0.6, 1.15, 0.5),
        Species::Pig => quadruped(T::MOB_PIG, 0.8, 0.55, 0.4, 0.85, 0.42),
        Species::Sheep => quadruped(T::MOB_SHEEP, 0.85, 0.7, 0.55, 1.2, 0.42),
        Species::Chicken => chicken(),
        Species::Zombie => biped(T::MOB_ZOMBIE),
        Species::Skeleton => biped(T::MOB_SKELETON),
        Species::Creeper => creeper(),
        Species::Spider => spider(),
    }
}

/// A body + forward head + four corner legs.
fn quadruped(tile: u32, len: f32, wid: f32, leg_h: f32, top: f32, hs: f32) -> Vec<Part> {
    let (l, w) = (len * 0.5, wid * 0.5);
    let mut v = vec![
        part([-l, leg_h, -w], [l, top, w], tile), // body
        part([l - 0.05, top - hs, -hs * 0.5], [l - 0.05 + hs, top - hs + hs * 1.2, hs * 0.5], tile), // head (front)
    ];
    let lw = 0.15;
    for (lx, lz) in [(l - lw, w - lw), (l - lw, -w), (-l, w - lw), (-l, -w)] {
        v.push(part([lx, 0.0, lz], [lx + lw, leg_h, lz + lw], tile));
    }
    v
}

/// A humanoid: two legs, torso, head, two forward arms.
fn biped(tile: u32) -> Vec<Part> {
    vec![
        part([-0.18, 0.0, -0.12], [-0.01, 0.72, 0.12], tile), // left leg
        part([0.01, 0.0, -0.12], [0.18, 0.72, 0.12], tile),   // right leg
        part([-0.2, 0.72, -0.14], [0.2, 1.42, 0.14], tile),   // torso
        part([-0.18, 1.42, -0.18], [0.18, 1.78, 0.18], tile), // head
        part([0.04, 0.8, -0.36], [0.34, 1.4, -0.16], tile),   // arm (reaching forward)
        part([0.04, 0.8, 0.16], [0.34, 1.4, 0.36], tile),     // arm
    ]
}

fn chicken() -> Vec<Part> {
    vec![
        part([-0.18, 0.25, -0.15], [0.18, 0.58, 0.15], T::MOB_CHICKEN), // body
        part([0.12, 0.48, -0.1], [0.3, 0.74, 0.1], T::MOB_CHICKEN),     // head/neck
        part([0.28, 0.56, -0.04], [0.4, 0.64, 0.04], T::MOB_PIG),       // beak
        part([-0.06, 0.0, -0.1], [0.04, 0.25, -0.02], T::MOB_PIG),      // legs
        part([-0.06, 0.0, 0.02], [0.04, 0.25, 0.1], T::MOB_PIG),
    ]
}

fn creeper() -> Vec<Part> {
    let mut v = vec![
        part([-0.2, 0.3, -0.13], [0.2, 1.3, 0.13], T::MOB_CREEPER), // tall body
        part([-0.22, 1.25, -0.22], [0.22, 1.65, 0.22], T::MOB_CREEPER), // head
    ];
    for (lx, lz) in [(0.04, 0.01), (0.04, -0.13), (-0.2, 0.01), (-0.2, -0.13)] {
        v.push(part([lx, 0.0, lz], [lx + 0.16, 0.3, lz + 0.12], T::MOB_CREEPER));
    }
    v
}

fn spider() -> Vec<Part> {
    let mut v = vec![
        part([-0.32, 0.18, -0.4], [0.3, 0.58, 0.4], T::MOB_SPIDER), // wide low abdomen
        part([0.25, 0.22, -0.2], [0.58, 0.52, 0.2], T::MOB_SPIDER), // head
    ];
    for (lx, lz) in [(0.05, 0.4), (0.05, -0.52), (-0.3, 0.4), (-0.3, -0.52)] {
        v.push(part([lx, 0.0, lz], [lx + 0.12, 0.36, lz + 0.12], T::MOB_SPIDER));
    }
    v
}

/// Per-mob state carried by `Kind::Mob`. `health`/`hurt` drive combat + the hurt-flash in M29.
#[derive(Clone, Copy)]
struct MobData {
    species: Species,
    health: f32,
    hurt: f32,
}

#[derive(Clone, Copy)]
enum Kind {
    Mob(MobData),
    Item(ItemStack),
    /// An experience orb worth `n` points; homes toward a nearby player and grants XP on pickup.
    Xp(u32),
}

/// What the player swept up this frame: item stacks (added to the inventory) and experience points.
#[derive(Default)]
pub struct Collected {
    pub items: Vec<ItemStack>,
    pub xp: u32,
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

    pub fn spawn_mob(&mut self, pos: Vec3, species: Species) {
        let mut rng = self.next_seed();
        let heading = randf(&mut rng) * TAU;
        self.list.push(Entity {
            kind: Kind::Mob(MobData {
                species,
                health: species.max_health(),
                hurt: 0.0,
            }),
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

    pub fn spawn_item(&mut self, pos: Vec3, stack: ItemStack) {
        let mut rng = self.next_seed();
        // Pop out with a little random horizontal velocity and an upward kick.
        let a = randf(&mut rng) * TAU;
        let vel = Vec3::new(a.cos() * 1.5, 3.0, a.sin() * 1.5);
        self.list.push(Entity {
            kind: Kind::Item(stack),
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

    /// Spawn an experience orb worth `amount` points (mining ores, smelting, killing mobs).
    pub fn spawn_xp(&mut self, pos: Vec3, amount: u32) {
        if amount == 0 {
            return;
        }
        let mut rng = self.next_seed();
        let a = randf(&mut rng) * TAU;
        let vel = Vec3::new(a.cos() * 1.2, 2.5 + randf(&mut rng), a.sin() * 1.2);
        self.list.push(Entity {
            kind: Kind::Xp(amount),
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

    /// Advance AI + physics for all entities and drop the dead/collected ones. Returns the blocks
    /// of any item drops the player walked over this frame (to add to their inventory).
    pub fn update(&mut self, dt: f32, player_pos: Vec3, is_solid: impl Fn(IVec3) -> bool) -> Collected {
        let mut collected = Collected::default();
        for e in &mut self.list {
            e.age += dt;
            match e.kind {
                Kind::Mob(mut m) => {
                    m.hurt = (m.hurt - dt).max(0.0); // hurt-flash decays (set by combat in M29)
                    let (mw, mh) = m.species.size();
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
                    e.on_ground = collide_move(&mut e.pos, &mut e.vel, mw, mh, dt, &is_solid);
                    if e.pos.y < FALL_OUT_Y || m.health <= 0.0 {
                        e.dead = true;
                    }
                    e.kind = Kind::Mob(m); // persist hurt decay
                }
                Kind::Item(stack) => {
                    e.vel.y -= GRAVITY * dt;
                    e.on_ground = collide_move(&mut e.pos, &mut e.vel, ITEM_SIZE, ITEM_SIZE, dt, &is_solid);
                    if e.on_ground {
                        e.vel.x *= 0.7;
                        e.vel.z *= 0.7;
                    }
                    // Collect once it has settled a moment and the player is close.
                    if e.age > 0.4 && (player_pos - e.pos).length() < COLLECT_RADIUS {
                        e.dead = true;
                        collected.items.push(stack);
                    } else if e.age > ITEM_LIFETIME || e.pos.y < FALL_OUT_Y {
                        e.dead = true;
                    }
                }
                Kind::Xp(amount) => {
                    e.vel.y -= GRAVITY * dt;
                    // Drift toward the player once close enough (the classic orb magnetism).
                    let to_player = player_pos + Vec3::new(0.0, 0.5, 0.0) - e.pos;
                    let dist = to_player.length();
                    if dist < XP_HOMING_RADIUS && dist > 1e-3 {
                        let pull = to_player / dist * XP_HOMING_SPEED;
                        e.vel.x = pull.x;
                        e.vel.z = pull.z;
                        e.vel.y = e.vel.y.max(pull.y);
                    }
                    e.on_ground = collide_move(&mut e.pos, &mut e.vel, XP_SIZE, XP_SIZE, dt, &is_solid);
                    if e.on_ground {
                        e.vel.x *= 0.7;
                        e.vel.z *= 0.7;
                    }
                    // Collect using the *post-move* distance so an orb a wall has stopped can't be
                    // vacuumed through it (matches the dropped-item arm; no straight-line shortcut).
                    let settled = (player_pos + Vec3::new(0.0, 0.5, 0.0) - e.pos).length();
                    if e.age > 0.2 && settled < COLLECT_RADIUS {
                        e.dead = true;
                        collected.xp += amount;
                    } else if e.age > ITEM_LIFETIME * 3.0 || e.pos.y < FALL_OUT_Y {
                        e.dead = true;
                    }
                }
            }
        }
        self.list.retain(|e| !e.dead);
        collected
    }

    /// Box geometry for every entity (opaque layer), lit by the chunk pipeline.
    pub fn build_mesh(&self) -> MeshData {
        let mut mesh = MeshData::default();
        for e in &self.list {
            match e.kind {
                Kind::Mob(m) => {
                    // Each species draws its own multi-box model, yaw-rotated to face its heading.
                    for p in model(m.species) {
                        push_part(&mut mesh.opaque, e.pos, e.heading, &p, m.hurt);
                    }
                }
                Kind::Item(stack) => {
                    let tile = item::item_tile(stack.item);
                    let em = item::item_emission(stack.item);
                    let s = ITEM_SIZE * 0.5;
                    let bob = (e.age * 3.0).sin() * 0.06;
                    let center = e.pos + Vec3::new(0.0, ITEM_SIZE * 0.5 + 0.05 + bob, 0.0);
                    push_box(
                        &mut mesh.opaque,
                        center - Vec3::new(s, s, s),
                        center + Vec3::new(s, s, s),
                        e.age * 1.6,
                        tile,
                        em,
                    );
                }
                Kind::Xp(_) => {
                    // A small glowing orb (glowstone tile + strong emission), bobbing and spinning.
                    let s = XP_SIZE * 0.5;
                    let bob = (e.age * 4.0).sin() * 0.05;
                    let center = e.pos + Vec3::new(0.0, XP_SIZE * 0.5 + 0.05 + bob, 0.0);
                    push_box(
                        &mut mesh.opaque,
                        center - Vec3::new(s, s, s),
                        center + Vec3::new(s, s, s),
                        e.age * 2.4,
                        block::tile::GLOWSTONE,
                        0.9,
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

/// Emit one mob model part: a box given in LOCAL space (feet at origin, +x forward), rotated by
/// `yaw` around the mob's vertical axis and placed at `feet`. `hurt` (0..1) goes in `shade.y` for
/// the M29 hurt-flash. Faces are never culled (mobs are small and self-contained).
fn push_part(geom: &mut Geometry, feet: Vec3, yaw: f32, p: &Part, hurt: f32) {
    let (s, co) = yaw.sin_cos();
    let rot = |lx: f32, ly: f32, lz: f32| -> [f32; 3] {
        [feet.x + lx * co - lz * s, feet.y + ly, feet.z + lx * s + lz * co]
    };
    let rotn = |nx: f32, nz: f32| -> [f32; 3] { [nx * co - nz * s, 0.0, nx * s + nz * co] };
    let (mn, mx) = (p.min, p.max);
    let shade = [0.0, hurt];
    let face_uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let cor = [
        (mn[0], mn[1], mn[2]),
        (mx[0], mn[1], mn[2]),
        (mx[0], mn[1], mx[2]),
        (mn[0], mn[1], mx[2]),
        (mn[0], mx[1], mn[2]),
        (mx[0], mx[1], mn[2]),
        (mx[0], mx[1], mx[2]),
        (mn[0], mx[1], mx[2]),
    ];
    let pw: [[f32; 3]; 8] = std::array::from_fn(|i| {
        let (x, y, z) = cor[i];
        rot(x, y, z)
    });
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([1, 2, 6, 5], [1.0, 0.0, 0.0]),
        ([3, 0, 4, 7], [-1.0, 0.0, 0.0]),
        ([4, 5, 6, 7], [0.0, 1.0, 0.0]),
        ([0, 3, 2, 1], [0.0, -1.0, 0.0]),
        ([2, 3, 7, 6], [0.0, 0.0, 1.0]),
        ([0, 1, 5, 4], [0.0, 0.0, -1.0]),
    ];
    for (idx, n) in faces {
        let normal = if n[1] != 0.0 { n } else { rotn(n[0], n[2]) };
        let base = geom.vertices.len() as u32;
        for (j, &k) in idx.iter().enumerate() {
            geom.vertices.push(Vertex {
                position: pw[k],
                normal,
                uv: face_uv[j],
                tile: p.tile,
                light: [1.0, 1.0],
                shade,
            });
        }
        geom.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Append a (optionally yaw-rotated) box textured with atlas `tile` to `geom`. `emission` is the
/// self-glow strength (lava items). Uses the unified Vertex; each face spans one tile (uv 0..1).
fn push_box(geom: &mut Geometry, min: Vec3, max: Vec3, yaw: f32, tile: u32, emission: f32) {
    let center = (min + max) * 0.5;
    let (s, co) = yaw.sin_cos();
    let rot = |p: Vec3| -> [f32; 3] {
        let dx = p.x - center.x;
        let dz = p.z - center.z;
        [center.x + dx * co - dz * s, p.y, center.z + dx * s + dz * co]
    };
    let rotn = |n: [f32; 3]| -> [f32; 3] { [n[0] * co - n[2] * s, n[1], n[0] * s + n[2] * co] };
    let shade = [emission, 0.0];
    let face_uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

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
        for (j, &k) in idx.iter().enumerate() {
            geom.vertices.push(Vertex {
                position: rot(cor[k]),
                normal,
                uv: face_uv[j],
                tile,
                light: [1.0, 1.0],
                shade,
            });
        }
        geom.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}
