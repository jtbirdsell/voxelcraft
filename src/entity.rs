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

// Mob AI (M28) tuning.
const DETECT_RADIUS: f32 = 14.0; // hostile mobs notice the player within this (needs line-of-sight)
const CALM_RADIUS: f32 = 22.0; // ...and give up the chase beyond this
const FLEE_RADIUS: f32 = 5.0; // passive mobs bolt when the player gets this close (or after a hit)
const ATTACK_RADIUS: f32 = 1.6; // hostile mobs lunge/attack when this close
const CHASE_SPEED: f32 = 3.2;
const FLEE_SPEED: f32 = 3.6;
const KNOCKBACK: f32 = 6.0; // horizontal impulse imparted by a melee hit
const MOB_ATTACK_COOLDOWN: f32 = 1.0; // seconds between a hostile mob's contact hits
// Projectiles + explosions (M30).
const ARROW_DAMAGE: f32 = 3.0;
const ARROW_SPEED: f32 = 22.0;
const ARROW_GRAVITY: f32 = 7.0;
const ARROW_LIFETIME: f32 = 4.0;
const SKELETON_SHOOT_CD: f32 = 1.6;
const CREEPER_FUSE: f32 = 1.5;
const CREEPER_RADIUS: f32 = 3.0;

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

    /// Hostile species hunt the player and deal contact damage.
    pub fn hostile(self) -> bool {
        matches!(
            self,
            Species::Zombie | Species::Skeleton | Species::Creeper | Species::Spider
        )
    }

    /// Contact damage a hostile mob deals to the player per hit (passives deal none).
    fn contact_damage(self) -> f32 {
        match self {
            Species::Creeper => 4.0,
            Species::Zombie => 3.0,
            Species::Skeleton | Species::Spider => 2.0,
            _ => 0.0,
        }
    }

    /// Experience dropped when this mob is killed.
    fn xp_drop(self) -> u32 {
        if self.hostile() {
            5
        } else if matches!(self, Species::Chicken) {
            1
        } else {
            2
        }
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

/// Mob AI state (M28). Passive species cycle Idle/Wander and Flee; hostile species Chase + Attack.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ai {
    Idle,
    Wander,
    Flee,
    Chase,
    Attack,
}

/// Per-mob state carried by `Kind::Mob`. `health`/`hurt` drive combat + the hurt-flash in M29.
#[derive(Clone, Copy)]
struct MobData {
    species: Species,
    health: f32,
    hurt: f32,
    ai: Ai,
    /// Cooldown between contact attacks / arrow shots (seconds).
    atk_cd: f32,
    /// Creeper detonation fuse: <0 = unprimed, else seconds left before it explodes.
    fuse: f32,
}

#[derive(Clone, Copy)]
enum Kind {
    Mob(MobData),
    Item(ItemStack),
    /// An experience orb worth `n` points; homes toward a nearby player and grants XP on pickup.
    Xp(u32),
    /// A skeleton arrow carrying `damage`; flies until it hits a block, the player, or expires.
    Arrow(f32),
}

/// What the entity tick produced this frame: item stacks + XP the player swept up, contact/projectile
/// damage dealt to the player (applied through armor by the caller), and any creeper explosions
/// `(center, radius)` for `game` to carve into the world + apply radial damage.
#[derive(Default)]
pub struct Collected {
    pub items: Vec<ItemStack>,
    pub xp: u32,
    pub player_damage: f32,
    pub explosions: Vec<(Vec3, f32)>,
}

struct Entity {
    kind: Kind,
    pos: Vec3, // feet / bottom-center
    vel: Vec3,
    on_ground: bool,
    age: f32,
    heading: f32,
    wander: f32,
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

    /// One-line tally of mob AI states (headless verification of the M28 state machine).
    pub fn ai_summary(&self) -> String {
        let (mut idle, mut wander, mut flee, mut chase, mut attack) = (0, 0, 0, 0, 0);
        for e in &self.list {
            if let Kind::Mob(m) = e.kind {
                match m.ai {
                    Ai::Idle => idle += 1,
                    Ai::Wander => wander += 1,
                    Ai::Flee => flee += 1,
                    Ai::Chase => chase += 1,
                    Ai::Attack => attack += 1,
                }
            }
        }
        format!("mobs idle:{idle} wander:{wander} flee:{flee} chase:{chase} attack:{attack}")
    }

    /// Distance to the nearest mob whose AABB the ray (origin,dir) enters within `reach`, else None.
    /// Used to decide whether a left-click hits a mob (melee) or the block behind it (mining).
    pub fn nearest_mob_hit(&self, origin: Vec3, dir: Vec3, reach: f32) -> Option<f32> {
        let dir = dir.normalize_or_zero();
        let mut best: Option<f32> = None;
        for e in &self.list {
            if let Kind::Mob(m) = e.kind {
                let (w, h) = m.species.size();
                let half = w * 0.5;
                let min = e.pos - Vec3::new(half, 0.0, half);
                let max = e.pos + Vec3::new(half, h, half);
                if let Some(t) = ray_aabb(origin, dir, min, max) {
                    if t <= reach && best.is_none_or(|b| t < b) {
                        best = Some(t);
                    }
                }
            }
        }
        best
    }

    /// Melee: damage the nearest mob the ray hits within `reach`, with knockback + a hurt-flash.
    /// Returns true if a mob was struck.
    pub fn attack(&mut self, origin: Vec3, dir: Vec3, reach: f32, damage: f32) -> bool {
        let dir = dir.normalize_or_zero();
        let mut best: Option<(usize, f32)> = None;
        for (i, e) in self.list.iter().enumerate() {
            if let Kind::Mob(m) = e.kind {
                let (w, h) = m.species.size();
                let half = w * 0.5;
                let min = e.pos - Vec3::new(half, 0.0, half);
                let max = e.pos + Vec3::new(half, h, half);
                if let Some(t) = ray_aabb(origin, dir, min, max) {
                    if t <= reach && best.is_none_or(|(_, b)| t < b) {
                        best = Some((i, t));
                    }
                }
            }
        }
        let Some((i, _)) = best else { return false };
        let e = &mut self.list[i];
        if let Kind::Mob(m) = &mut e.kind {
            m.health -= damage;
            m.hurt = 0.35;
        }
        let mut kb = e.pos - origin;
        kb.y = 0.0;
        let kb = kb.normalize_or_zero();
        e.vel.x += kb.x * KNOCKBACK;
        e.vel.z += kb.z * KNOCKBACK;
        e.vel.y = e.vel.y.max(0.0) + 3.0; // small pop-up
        true
    }

    /// Debug: flash every mob red (headless hurt-flash verification screenshot).
    pub fn flash_all(&mut self, amount: f32) {
        for e in &mut self.list {
            if let Kind::Mob(m) = &mut e.kind {
                m.hurt = amount;
            }
        }
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
                ai: Ai::Wander,
                atk_cd: 0.0,
                fuse: -1.0,
            }),
            pos,
            vel: Vec3::ZERO,
            on_ground: false,
            age: 0.0,
            heading,
            wander: 1.0 + randf(&mut rng) * 2.0,
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
            rng,
            dead: false,
        });
    }

    /// Spawn an arrow at `pos` with velocity `vel` (oriented along its flight).
    pub fn spawn_arrow(&mut self, pos: Vec3, vel: Vec3) {
        let rng = self.next_seed();
        self.list.push(Entity {
            kind: Kind::Arrow(ARROW_DAMAGE),
            pos,
            vel,
            on_ground: false,
            age: 0.0,
            heading: vel.z.atan2(vel.x),
            wander: 0.0,
            rng,
            dead: false,
        });
    }

    /// Advance AI + physics for all entities and drop the dead/collected ones. Returns the blocks
    /// of any item drops the player walked over this frame (to add to their inventory).
    pub fn update(&mut self, dt: f32, player_pos: Vec3, is_solid: impl Fn(IVec3) -> bool) -> Collected {
        let mut collected = Collected::default();
        let mut deaths: Vec<(Vec3, Species)> = Vec::new();
        let mut shots: Vec<(Vec3, Vec3)> = Vec::new(); // skeleton arrows (pos, vel), spawned post-loop
        for e in &mut self.list {
            e.age += dt;
            match e.kind {
                Kind::Mob(mut m) => {
                    m.hurt = (m.hurt - dt).max(0.0); // hurt-flash decays after a hit
                    m.atk_cd = (m.atk_cd - dt).max(0.0);
                    let (mw, mh) = m.species.size();

                    // Perception: distance to the player + (for hostiles) exact line-of-sight.
                    let to_player = player_pos - e.pos;
                    let dist = to_player.length();
                    let eye = Vec3::new(0.0, mh * 0.5, 0.0);
                    let target = player_pos + Vec3::new(0.0, 0.9, 0.0);

                    // LOS only matters within the give-up range; skip the DDA for far/passive mobs.
                    let los = m.species.hostile()
                        && dist < CALM_RADIUS
                        && line_of_sight(e.pos + eye, target, &is_solid);

                    // State transition.
                    if m.species.hostile() {
                        let sees = dist < DETECT_RADIUS && los;
                        // Hysteresis: stay in Attack a bit past ATTACK_RADIUS so it doesn't jitter.
                        let attack_keep = m.ai == Ai::Attack && dist <= ATTACK_RADIUS * 1.5;
                        m.ai = if (dist <= ATTACK_RADIUS || attack_keep) && los {
                            Ai::Attack
                        } else if sees || (matches!(m.ai, Ai::Chase | Ai::Attack) && los) {
                            Ai::Chase // keeps chasing only while it can still SEE the player
                        } else if matches!(m.ai, Ai::Chase | Ai::Attack) {
                            Ai::Wander // lost sight / out of range
                        } else {
                            m.ai
                        };
                    } else if dist < FLEE_RADIUS || m.hurt > 0.0 {
                        if m.ai != Ai::Flee {
                            e.wander = 1.5 + randf(&mut e.rng); // sidestep timer (escape corners)
                        }
                        m.ai = Ai::Flee;
                    } else if m.ai == Ai::Flee && dist > FLEE_RADIUS * 1.8 {
                        m.ai = Ai::Wander;
                    }

                    // Drive velocity from the state.
                    let heading_to = |v: Vec3| v.z.atan2(v.x);
                    let speed = match m.ai {
                        Ai::Flee => {
                            e.wander -= dt;
                            let away = heading_to(-to_player);
                            // Periodically deflect so a cornered mob sidesteps instead of running
                            // straight into the wall it's trapped against forever.
                            e.heading = if e.wander <= 0.0 {
                                e.wander = 1.5 + randf(&mut e.rng);
                                away + (randf(&mut e.rng) - 0.5) * 2.2
                            } else {
                                away
                            };
                            FLEE_SPEED
                        }
                        Ai::Chase => {
                            e.heading = heading_to(to_player);
                            CHASE_SPEED
                        }
                        Ai::Attack => {
                            e.heading = heading_to(to_player); // face the player, hold (damage in M29)
                            0.0
                        }
                        Ai::Idle | Ai::Wander => {
                            e.wander -= dt;
                            if e.wander <= 0.0 {
                                e.wander = 2.0 + randf(&mut e.rng) * 3.0;
                                m.ai = if randf(&mut e.rng) < 0.3 { Ai::Idle } else { Ai::Wander };
                                e.heading = randf(&mut e.rng) * TAU;
                            }
                            if m.ai == Ai::Idle { 0.0 } else { MOB_SPEED }
                        }
                    };
                    e.vel.x = e.heading.cos() * speed;
                    e.vel.z = e.heading.sin() * speed;
                    // Occasional hop helps clear 1-block steps without real pathfinding.
                    if speed > 0.0 && e.on_ground && randf(&mut e.rng) < 0.05 {
                        e.vel.y = 7.0;
                    }

                    e.vel.y -= GRAVITY * dt;
                    e.on_ground = collide_move(&mut e.pos, &mut e.vel, mw, mh, dt, &is_solid);

                    // Ranged / contact attacks, per species.
                    match m.species {
                        // Skeleton: looses an arrow at the player while it has a clear shot.
                        Species::Skeleton => {
                            if matches!(m.ai, Ai::Chase | Ai::Attack) && los && m.atk_cd <= 0.0 {
                                let from = e.pos + Vec3::new(0.0, mh * 0.7, 0.0);
                                let aim = (target - from).normalize_or_zero();
                                // Aim a touch high so gravity arcs the shot into the target.
                                let vel = aim * ARROW_SPEED + Vec3::new(0.0, dist * 0.05, 0.0);
                                shots.push((from, vel));
                                m.atk_cd = SKELETON_SHOOT_CD;
                            }
                        }
                        // Creeper: primes a fuse in attack range, then detonates (no melee, no XP).
                        Species::Creeper => {
                            if m.ai == Ai::Attack {
                                if m.fuse < 0.0 {
                                    m.fuse = CREEPER_FUSE;
                                }
                                m.fuse -= dt;
                                m.hurt = (1.0 - m.fuse / CREEPER_FUSE).clamp(0.0, 0.45); // blink
                                if m.fuse <= 0.0 {
                                    collected
                                        .explosions
                                        .push((e.pos + Vec3::new(0.0, mh * 0.5, 0.0), CREEPER_RADIUS));
                                    e.dead = true;
                                }
                            } else {
                                m.fuse = -1.0; // walked out of range — defuse
                            }
                        }
                        // Zombie / spider: melee contact damage on cooldown.
                        _ => {
                            if m.ai == Ai::Attack && m.atk_cd <= 0.0 {
                                let dmg = m.species.contact_damage();
                                if dmg > 0.0 {
                                    collected.player_damage += dmg;
                                    m.atk_cd = MOB_ATTACK_COOLDOWN;
                                }
                            }
                        }
                    }

                    if e.dead {
                        // already consumed (creeper detonated) — no loot
                    } else if m.health <= 0.0 {
                        e.dead = true;
                        deaths.push((e.pos, m.species)); // killed → drops loot
                    } else if e.pos.y < FALL_OUT_Y {
                        e.dead = true; // fell out of the world → no loot
                    }
                    e.kind = Kind::Mob(m); // persist hurt/atk decay + ai state
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
                Kind::Arrow(dmg) => {
                    e.vel.y -= ARROW_GRAVITY * dt;
                    e.heading = e.vel.z.atan2(e.vel.x);
                    // Point-like flight: move freely until the next cell is solid (then it sticks).
                    let next = e.pos + e.vel * dt;
                    let cell = IVec3::new(
                        next.x.floor() as i32,
                        next.y.floor() as i32,
                        next.z.floor() as i32,
                    );
                    if is_solid(cell) {
                        e.dead = true; // stuck in a block
                    } else {
                        e.pos = next;
                    }
                    let to_player = player_pos + Vec3::new(0.0, 1.0, 0.0) - e.pos;
                    if to_player.length() < 0.7 {
                        collected.player_damage += dmg;
                        e.dead = true;
                    } else if e.age > ARROW_LIFETIME || e.pos.y < FALL_OUT_Y {
                        e.dead = true;
                    }
                }
            }
        }
        self.list.retain(|e| !e.dead);
        // Killed mobs drop experience (item loot waits on a food/material milestone).
        for (pos, species) in deaths {
            self.spawn_xp(pos + Vec3::new(0.0, 0.3, 0.0), species.xp_drop());
        }
        // Arrows skeletons loosed this frame (spawned after the iteration to avoid aliasing the list).
        for (pos, vel) in shots {
            self.spawn_arrow(pos, vel);
        }
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
                Kind::Arrow(_) => {
                    // A thin dark dart, elongated along its (horizontal) heading.
                    let (l, w) = (0.5, 0.05);
                    push_box(
                        &mut mesh.opaque,
                        e.pos - Vec3::new(l, w, w),
                        e.pos + Vec3::new(l, w, w),
                        e.heading,
                        block::tile::DEEPSLATE,
                        0.0,
                    );
                }
            }
        }
        mesh
    }
}

/// Exact voxel line-of-sight via an Amanatides–Woo DDA: walks every voxel the segment crosses and
/// returns false on the first solid one (the mob's own start cell and the player's end cell are not
/// tested). Used so hostile mobs only chase a player they can actually see — no thin-wall leaks.
fn line_of_sight(from: Vec3, to: Vec3, is_solid: &impl Fn(IVec3) -> bool) -> bool {
    let d = to - from;
    let dist = d.length();
    if dist < 1e-3 {
        return true;
    }
    let dir = d / dist;
    let (mut vx, mut vy, mut vz) = (
        from.x.floor() as i32,
        from.y.floor() as i32,
        from.z.floor() as i32,
    );
    let end = IVec3::new(to.x.floor() as i32, to.y.floor() as i32, to.z.floor() as i32);
    let step = |c: f32| if c >= 0.0 { 1 } else { -1 };
    let (sx, sy, sz) = (step(dir.x), step(dir.y), step(dir.z));
    let boundary = |o: f32, c: f32, v: i32, s: i32| -> f32 {
        if c == 0.0 {
            f32::INFINITY
        } else {
            let b = if s > 0 { (v + 1) as f32 } else { v as f32 };
            (b - o) / c
        }
    };
    let (mut tx, mut ty, mut tz) = (
        boundary(from.x, dir.x, vx, sx),
        boundary(from.y, dir.y, vy, sy),
        boundary(from.z, dir.z, vz, sz),
    );
    let inv = |c: f32| if c == 0.0 { f32::INFINITY } else { (1.0 / c).abs() };
    let (dx, dy, dz) = (inv(dir.x), inv(dir.y), inv(dir.z));
    loop {
        // Advance into the next voxel along the nearest axis boundary.
        if tx <= ty && tx <= tz {
            if tx > dist {
                break;
            }
            vx += sx;
            tx += dx;
        } else if ty <= tz {
            if ty > dist {
                break;
            }
            vy += sy;
            ty += dy;
        } else {
            if tz > dist {
                break;
            }
            vz += sz;
            tz += dz;
        }
        let cell = IVec3::new(vx, vy, vz);
        if cell == end {
            break; // reached the player's cell; don't test it
        }
        if is_solid(cell) {
            return false;
        }
    }
    true
}

/// Slab-method ray vs AABB; returns the near intersection distance along `dir` (clamped to ≥0), or
/// None if the ray misses. `dir` should be normalized so the result is in world units.
fn ray_aabb(o: Vec3, d: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let inv = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
    let t1 = (min - o) * inv;
    let t2 = (max - o) * inv;
    let tmin = t1.min(t2).max_element();
    let tmax = t1.max(t2).min_element();
    if tmax >= tmin.max(0.0) {
        Some(tmin.max(0.0))
    } else {
        None
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
/// `yaw` around the mob's vertical axis and placed at `feet`. `hurt` (0..1) is stashed in `shade.y`;
/// the chunk shader doesn't read it yet — M29 combat will both set it and add the red-flash tint.
/// Faces are never culled (mobs are small and self-contained).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn los_blocked_by_wall() {
        let wall = |p: IVec3| p.x == 2; // a wall plane between from and to
        assert!(!line_of_sight(Vec3::new(0.5, 1.0, 0.5), Vec3::new(4.5, 1.0, 0.5), &wall));
        let air = |_: IVec3| false;
        assert!(line_of_sight(Vec3::new(0.5, 1.0, 0.5), Vec3::new(4.5, 1.0, 0.5), &air));
    }

    #[test]
    fn los_dda_catches_single_cell_on_diagonal() {
        // A lone solid voxel the diagonal segment passes through — the DDA must visit it.
        let wall = |p: IVec3| p == IVec3::new(2, 1, 2);
        assert!(!line_of_sight(Vec3::new(0.5, 1.0, 0.5), Vec3::new(4.5, 1.0, 4.5), &wall));
        // Start and end cells are not tested (mob/player stand in solid-feeling cells freely).
        let at_end = |p: IVec3| p == IVec3::new(4, 1, 4);
        assert!(line_of_sight(Vec3::new(0.5, 1.0, 0.5), Vec3::new(4.5, 1.0, 4.5), &at_end));
    }

    #[test]
    fn hostility_split() {
        for s in [Species::Zombie, Species::Skeleton, Species::Creeper, Species::Spider] {
            assert!(s.hostile());
        }
        for s in [Species::Cow, Species::Pig, Species::Sheep, Species::Chicken] {
            assert!(!s.hostile());
        }
    }

    #[test]
    fn ray_aabb_hit_and_miss() {
        let (min, max) = (Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.0, 0.5));
        // Straight at the box from 5 blocks away along +z.
        let t = ray_aabb(Vec3::new(0.0, 0.5, -5.0), Vec3::new(0.0, 0.0, 1.0), min, max);
        assert!(t.is_some_and(|t| (t - 4.5).abs() < 1e-3));
        // Pointing away, and off to the side, both miss.
        assert!(ray_aabb(Vec3::new(0.0, 0.5, -5.0), Vec3::new(0.0, 0.0, -1.0), min, max).is_none());
        assert!(ray_aabb(Vec3::new(5.0, 0.5, -5.0), Vec3::new(0.0, 0.0, 1.0), min, max).is_none());
    }

    #[test]
    fn attack_damages_then_kills_mob() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::ZERO, Species::Chicken); // 4 hp
        let (o, d) = (Vec3::new(0.0, 0.4, -3.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(es.attack(o, d, 6.0, 2.0)); // 4 -> 2
        assert!(es.attack(o, d, 6.0, 2.0)); // 2 -> 0
        // The dead mob is culled on the next tick (and drops an XP orb in its place).
        let _ = es.update(0.016, o, |_| false);
        assert_eq!(es.ai_summary(), "mobs idle:0 wander:0 flee:0 chase:0 attack:0");
    }

    #[test]
    fn arrow_hits_player_and_despawns() {
        let mut es = Entities::new();
        es.spawn_arrow(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 12.0)); // flying +z
        let player = Vec3::new(0.0, 0.0, 3.0); // feet; chest (+1) sits at z=3
        let mut dmg = 0.0;
        for _ in 0..120 {
            dmg += es.update(1.0 / 60.0, player, |_| false).player_damage;
            if es.count() == 0 {
                break;
            }
        }
        assert!(dmg > 0.0, "the arrow should damage the player");
        assert_eq!(es.count(), 0, "and despawn on impact");
    }

    #[test]
    fn arrow_sticks_in_a_wall() {
        let mut es = Entities::new();
        es.spawn_arrow(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 12.0));
        let wall = |p: IVec3| p.z >= 2;
        for _ in 0..120 {
            es.update(1.0 / 60.0, Vec3::new(0.0, 200.0, 200.0), wall); // player far away
            if es.count() == 0 {
                break;
            }
        }
        assert_eq!(es.count(), 0, "the arrow stops at the wall and despawns");
    }
}
