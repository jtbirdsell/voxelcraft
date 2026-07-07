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
// S5: hostile despawn shell (was 48). Must exceed the max 3D spawn distance
// √(HOSTILE_MAX² + Y_RANGE²) ≈ 50, or fresh cave spawns would be culled the same frame.
// Hostile-only since S3 (passives persist).
const DESPAWN_RADIUS: f32 = 64.0;
const FLEE_RADIUS: f32 = 5.0; // passive mobs bolt when the player gets this close (or after a hit)
const ATTACK_RADIUS: f32 = 1.6; // hostile mobs lunge/attack when this close
const CHASE_SPEED: f32 = 3.2;
const FLEE_SPEED: f32 = 3.6;
const KNOCKBACK: f32 = 6.0; // horizontal impulse imparted by a melee hit
const SWEEP_RADIUS: f32 = 1.5; // a charged sword sweep hits OTHER mobs within this of the primary
const SWEEP_KNOCKBACK: f32 = 2.5; // lighter shove (away from the player) for swept mobs
const MOB_ATTACK_COOLDOWN: f32 = 1.0; // seconds between a hostile mob's contact hits
// P17 life-cycle tuning (timers shortened from MC for a clone's pace).
const LOVE_TIME: f32 = 30.0; // how long a fed animal stays ready to breed
const BREED_COOLDOWN: f32 = 60.0; // post-breed lockout before it can breed again
const GROW_TIME: f32 = 120.0; // baby → adult
const BREED_RANGE: f32 = 3.0; // two in-love adults within this (blocks) breed (must be penned close)
const BABY_SCALE: f32 = 0.5; // juvenile model scale
const BURN_DPS: f32 = 1.0; // undead daylight burn damage per second
// P15 local navigation tuning.
const NAV_MARGIN: f32 = 0.4; // look-ahead beyond the mob's leading face (probe = half_width + this)
const NAV_MAX_DROP: i32 = 3; // wander/flee refuse to step off a drop this many empty cells deep or more
const STEP_UP_VEL: f32 = 8.5; // upward impulse to clear a 1-block ledge: apex v²/2g ≈ 1.29 > 1.0 (g=28)
const DEFLECT_HOLD: f32 = 0.5; // seconds a mob commits to a chosen detour heading (kills stutter)
// Detour headings tried in order when the goal is blocked: ±30°, ±60°, ±120° (small turns first).
const DEFLECT_FAN: [f32; 6] = [0.5236, -0.5236, 1.0472, -1.0472, 2.0944, -2.0944];
// Projectiles + explosions (M30).
const ARROW_DAMAGE: f32 = 3.0;
const ARROW_SPEED: f32 = 22.0;
const ARROW_GRAVITY: f32 = 7.0;
const ARROW_LIFETIME: f32 = 4.0;
const SKELETON_SHOOT_CD: f32 = 1.6;
const CREEPER_FUSE: f32 = 1.5;
const CREEPER_RADIUS: f32 = 3.0;
// U11 Warden tuning (a blind, vibration-hunting boss).
const WARDEN_HEAR_RANGE: f32 = 24.0; // hears the most-recent vibration within this radius
const WARDEN_TARGET_STALE: f32 = 2.5; // seconds of silence after which a cold vibration trail is dropped (re-hunt the player)
const WARDEN_DIG_TIMEOUT: f32 = 12.0; // seconds of silence (and no nearby player) → it digs away
const WARDEN_SPEED: f32 = 3.2; // blind-pursuit walk speed
const WARDEN_EMERGE_TIME: f32 = 2.5; // rise-from-ground: immobile + invulnerable while >0
const WARDEN_SONIC_RANGE: f32 = 16.0; // ranged sonic-boom reach
const WARDEN_SONIC_DAMAGE: f32 = 10.0; // sonic-boom damage (ignores armor at the source like a blast)
const WARDEN_SONIC_CD: f32 = 3.0; // cooldown between sonic booms

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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Species {
    Cow,
    Pig,
    Sheep,
    Chicken,
    Zombie,
    Skeleton,
    Creeper,
    Spider,
    // P18
    Wolf,
    Enderman,
    Slime,
    Villager,
    // U11: the deep-dark boss. Blind; hunts vibrations, not sight.
    Warden,
}

impl Species {
    /// All species, for spawning a representative set. ALSO the persisted mob-id table (S3):
    /// `EntitySave.species` is an index into this array — NEVER reorder or remove entries, only
    /// append (the indices are baked into every save_state.bin on disk).
    pub const ALL: [Species; 13] = [
        Species::Cow,
        Species::Pig,
        Species::Sheep,
        Species::Chicken,
        Species::Zombie,
        Species::Skeleton,
        Species::Creeper,
        Species::Spider,
        Species::Wolf,
        Species::Enderman,
        Species::Slime,
        Species::Villager,
        Species::Warden,
    ];

    /// Persisted species id (S3): the index into `Species::ALL`.
    pub fn to_u8(self) -> u8 {
        Self::ALL.iter().position(|&s| s == self).unwrap_or(0) as u8
    }

    /// Persisted species id → species; None for out-of-range (a future/corrupt save).
    pub fn from_u8(v: u8) -> Option<Species> {
        Self::ALL.get(v as usize).copied()
    }

    fn max_health(self) -> f32 {
        match self {
            Species::Chicken => 4.0,
            Species::Sheep => 8.0,
            Species::Cow | Species::Pig => 10.0,
            Species::Spider => 16.0,
            Species::Zombie | Species::Skeleton | Species::Creeper => 20.0,
            Species::Wolf => 8.0,
            Species::Enderman => 20.0, // teleport deferred → a fair non-boss threat
            Species::Villager => 10.0,
            Species::Slime => 8.0, // LARGE default; a tier-spawned slime rewrites health (spawn_slime)
            Species::Warden => 500.0, // U11 boss: a huge HP pool (MC parity)
        }
    }

    /// Collision AABB (width, height). Slime returns its LARGE box; the live tier scales it (`dims`).
    fn size(self) -> (f32, f32) {
        match self {
            Species::Chicken => (0.4, 0.7),
            Species::Pig => (0.8, 0.9),
            Species::Sheep => (0.8, 1.3),
            Species::Cow => (0.9, 1.4),
            Species::Creeper => (0.6, 1.7),
            Species::Zombie | Species::Skeleton => (0.6, 1.9),
            Species::Spider => (1.1, 0.7),
            Species::Wolf => (0.7, 0.85),
            // Enderman: short-enough AABB to fit the 2-air spawn check; the model() draws it 2.9 tall.
            Species::Enderman => (0.6, 1.95),
            Species::Villager => (0.6, 1.9),
            Species::Slime => (1.0, 1.0),
            // Warden: a short-enough AABB to clear the 2-air spawn check (like the Enderman); the
            // model() draws it ~2.9 tall. Wide + heavy footprint.
            Species::Warden => (0.9, 2.0),
        }
    }

    /// Hostile species hunt the player and deal contact damage. (Wolf/Enderman are NEUTRAL — they
    /// turn hostile at runtime via MobData.angry once attacked, so they count as passive until then.)
    pub fn hostile(self) -> bool {
        matches!(
            self,
            Species::Zombie
                | Species::Skeleton
                | Species::Creeper
                | Species::Spider
                | Species::Slime
                | Species::Enderman
                | Species::Warden
        )
    }

    /// Contact damage a hostile mob deals to the player per hit (passives deal none).
    fn contact_damage(self) -> f32 {
        match self {
            Species::Warden => 16.0, // U11: a crushing melee hit
            Species::Creeper => 4.0,
            Species::Zombie | Species::Enderman => 3.0,
            Species::Skeleton | Species::Spider | Species::Wolf | Species::Slime => 2.0,
            _ => 0.0,
        }
    }

    /// The item that puts this animal into love-mode when fed (P17). Hostiles don't breed.
    fn breeding_food(self) -> Option<crate::item::ItemId> {
        use crate::item;
        match self {
            Species::Cow | Species::Sheep => Some(item::WHEAT),
            Species::Chicken => Some(item::SEEDS),
            Species::Pig => Some(item::CARROT),
            _ => None,
        }
    }

    /// Experience dropped when this mob is killed.
    fn xp_drop(self) -> u32 {
        match self {
            Species::Slime => 1, // each split body drops a little (a large→cascade isn't 50 XP)
            Species::Chicken => 1,
            Species::Warden => 0, // U11: the Warden gives no XP (MC parity)
            _ if self.hostile() => 5,
            _ => 2,
        }
    }

    /// Item loot dropped when this mob is killed (in addition to XP): (item, min, max) — the
    /// deaths loop rolls each entry an independent uniform count in [min, max] (min 0 = may skip).
    fn loot(self) -> &'static [(crate::item::ItemId, u8, u8)] {
        use crate::item::*;
        match self {
            Species::Cow => &[(BEEF, 1, 3), (LEATHER, 0, 2)],
            Species::Pig => &[(PORK, 1, 3)],
            Species::Sheep => &[(MUTTON, 1, 2), (crate::block::WOOL, 1, 1)], // S11: +wool (shearing lands S24)
            Species::Chicken => &[(CHICKEN_MEAT, 1, 1), (FEATHER, 0, 2)],
            Species::Zombie => &[(ROTTEN_FLESH, 0, 2)],
            Species::Skeleton => &[(BONE, 0, 2), (ARROW, 0, 2)],
            Species::Creeper => &[(GUNPOWDER, 0, 2)],
            Species::Spider => &[(STRING, 0, 2), (SPIDER_EYE, 0, 1)],
            // S13: slimeballs come from the smallest body ONLY (gated in the deaths loop, where
            // the split tier is known) — a large slime's full cascade would otherwise roll ~7x.
            Species::Slime => &[(SLIMEBALL, 0, 2)],
            Species::Enderman => &[(ENDER_PEARL, 0, 1)],
            // U11: the Warden drops nothing (MC: only a sculk catalyst, deferred); wolves too.
            Species::Wolf | Species::Villager | Species::Warden => &[],
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
        Species::Wolf => wolf(),
        Species::Enderman => enderman(),
        Species::Slime => slime(),
        Species::Villager => villager(),
        Species::Warden => warden(),
    }
}

/// Warden (U11): a hulking dark-teal humanoid ~2.9 tall — bigger and bulkier than the bipeds, with a
/// SCULK body and a glowing SCULK_SENSOR chest/face accent (the bioluminescent core). The collision
/// AABB (size()) is shorter so it fits the 2-air spawn check; the model towers over it.
fn warden() -> Vec<Part> {
    vec![
        part([-0.22, 0.0, -0.16], [-0.02, 1.1, 0.16], T::SCULK), // left leg (thick)
        part([0.02, 0.0, -0.16], [0.22, 1.1, 0.16], T::SCULK),   // right leg
        part([-0.42, 1.1, -0.26], [0.42, 2.3, 0.26], T::SCULK),  // huge torso
        part([-0.18, 1.35, -0.30], [0.18, 1.9, -0.27], T::SCULK_SENSOR), // glowing chest core
        part([-0.34, 2.3, -0.30], [0.34, 2.9, 0.30], T::SCULK),  // broad head
        part([-0.20, 2.5, -0.32], [0.20, 2.74, -0.30], T::SCULK_SENSOR), // glowing face
        part([0.10, 1.2, -0.62], [0.34, 2.25, -0.30], T::SCULK), // left arm (long, heavy)
        part([0.10, 1.2, 0.30], [0.34, 2.25, 0.62], T::SCULK),   // right arm
    ]
}

/// Wolf: a small grey quadruped with pointed ears, a snout, and a tail (distinct from the cow).
fn wolf() -> Vec<Part> {
    let mut v = quadruped(T::MOB_WOLF, 0.7, 0.4, 0.45, 0.78, 0.3);
    v.push(part([0.28, 0.78, -0.14], [0.34, 0.92, -0.06], T::MOB_WOLF)); // left ear
    v.push(part([0.28, 0.78, 0.06], [0.34, 0.92, 0.14], T::MOB_WOLF)); // right ear
    v.push(part([0.40, 0.50, -0.05], [0.54, 0.62, 0.05], T::MOB_WOLF)); // snout
    v.push(part([-0.40, 0.52, -0.04], [-0.54, 0.66, 0.04], T::MOB_WOLF)); // tail
    v
}

/// Enderman: a very tall, thin near-black biped — the AABB is short (size()) but the model is 2.9 high.
fn enderman() -> Vec<Part> {
    vec![
        part([-0.10, 0.0, -0.10], [-0.01, 1.5, 0.10], T::MOB_ENDERMAN), // left leg (long)
        part([0.01, 0.0, -0.10], [0.10, 1.5, 0.10], T::MOB_ENDERMAN),   // right leg
        part([-0.12, 1.5, -0.10], [0.12, 2.5, 0.10], T::MOB_ENDERMAN),  // slim torso
        part([-0.18, 2.5, -0.18], [0.18, 2.9, 0.18], T::MOB_ENDERMAN),  // oversized head
        part([0.02, 1.55, -0.26], [0.10, 2.5, -0.12], T::MOB_ENDERMAN), // left arm (long)
        part([0.02, 1.55, 0.12], [0.10, 2.5, 0.26], T::MOB_ENDERMAN),   // right arm
    ]
}

/// Slime: a single gel cube. Its size tier scales the whole model at render (see `mob_scale`).
fn slime() -> Vec<Part> {
    vec![part([-0.5, 0.0, -0.5], [0.5, 1.0, 0.5], T::MOB_SLIME)]
}

/// Villager: the humanoid biped plus a protruding nose (a distinct silhouette from the undead bipeds).
fn villager() -> Vec<Part> {
    let mut v = biped(T::MOB_VILLAGER);
    v.push(part([0.18, 1.50, -0.05], [0.32, 1.62, 0.05], T::MOB_VILLAGER)); // big nose
    v
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
    /// Local-nav detour commitment (P15): while >0 the mob holds `deflect_heading` (counts down).
    deflect: f32,
    deflect_heading: f32,
    /// Life-cycle (P17): a juvenile (scaled model, can't breed); seconds left until adult; love-mode
    /// timer (>0 = ready to breed with a nearby in-love same-species adult); post-breed lockout.
    baby: bool,
    growth: f32,
    love: f32,
    breed_cd: f32,
    /// P18: a NEUTRAL mob (wolf/enderman) flips permanently hostile toward the player once attacked.
    angry: bool,
    /// P18 slime size tier: 2 = large, 1 = medium, 0 = small (0 for every non-slime). Scales the
    /// model + collision/hit AABB via `mob_scale`/`dims`; the smallest tier does not split on death.
    slime_tier: u8,
    /// U11 Warden state (0 / None for every other species, harmless):
    /// `emerge` = rise-from-ground timer; while >0 it is immobile + invulnerable.
    emerge: f32,
    /// `no_vib` = seconds since it last heard a vibration; past a timeout (and with no nearby player)
    /// the Warden digs back underground (despawns, no loot).
    no_vib: f32,
    /// `sonic_cd` = sonic-boom cooldown (seconds).
    sonic_cd: f32,
    /// `warden_target` = world position of the last vibration it heard (its blind pursuit goal).
    warden_target: Option<Vec3>,
    /// S9: seconds until this mob next vocalizes (groan/moo/cluck...). Transient.
    voice: f32,
}

/// S9: a species' idle vocalization (None = a silent species).
fn voice_sfx(s: Species) -> Option<crate::audio::Sfx> {
    use crate::audio::Sfx;
    Some(match s {
        Species::Zombie => Sfx::ZombieGroan,
        Species::Skeleton => Sfx::SkeletonRattle,
        Species::Cow => Sfx::CowMoo,
        Species::Pig => Sfx::PigOink,
        Species::Sheep => Sfx::SheepBaa,
        Species::Chicken => Sfx::ChickenCluck,
        Species::Wolf => Sfx::WolfBark,
        _ => return None,
    })
}

/// Slime size-tier factor (small/medium/large): scales both the render model and the collision/hit
/// AABB so they stay locked together. Non-slimes are unscaled (1.0).
fn mob_scale(m: &MobData) -> f32 {
    if m.species == Species::Slime {
        [0.5, 0.75, 1.0][(m.slime_tier.min(2)) as usize]
    } else {
        1.0
    }
}

/// Collision/hit footprint (width, height): `species.size()` folded with the live slime tier scale.
fn dims(m: &MobData) -> (f32, f32) {
    let (w, h) = m.species.size();
    let s = mob_scale(m);
    (w * s, h * s)
}

/// Slime health per size tier (large = the species max).
fn slime_health(tier: u8) -> f32 {
    [1.0, 4.0, 8.0][(tier.min(2)) as usize]
}

/// Who loosed an arrow — decides what it can hit (player arrows hit mobs, mob arrows hit the player).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArrowOwner {
    Player,
    Mob,
}

#[derive(Clone, Copy)]
enum Kind {
    Mob(MobData),
    Item(ItemStack),
    /// An experience orb worth `n` points; homes toward a nearby player and grants XP on pickup.
    Xp(u32),
    /// An arrow carrying `damage`, tagged by who loosed it. Flies under gravity until it hits a block,
    /// its target (a mob for a player arrow; the player for a mob arrow), or expires.
    Arrow { damage: f32, owner: ArrowOwner },
}

/// What the entity tick produced this frame: item stacks + XP the player swept up, contact/projectile
/// damage dealt to the player (applied through armor by the caller), and any creeper explosions
/// `(center, radius)` for `game` to carve into the world + apply radial damage.
#[derive(Default)]
pub struct Collected {
    pub items: Vec<ItemStack>,
    pub xp: u32,
    /// Per-source combat damage to the player this frame: `(amount, source_pos)`. The source position
    /// lets the consumer test directional shield blocking (P14); each is pre-armor/pre-difficulty.
    /// Environmental damage (fall/lava/drown/starve) is applied directly in player.rs, never here.
    pub player_damage: Vec<(f32, Vec3)>,
    pub explosions: Vec<(Vec3, f32)>,
    /// World positions where a mob died this frame (U10): drives sculk-catalyst spread + a death
    /// vibration the Warden/sensors can hear.
    pub deaths: Vec<Vec3>,
    /// S9: world-positioned sound events raised inside the entity/game tick (creeper fuses,
    /// skeleton shots, mob voices, explosions...). Drained by the app each frame; also the feed
    /// S10's particles extend.
    pub sounds: Vec<(crate::audio::Sfx, Vec3)>,
    /// S10: particle bursts raised inside the tick (breeding hearts...), drained like `sounds`.
    pub bursts: Vec<(crate::game::Burst, Vec3)>,
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
    /// Knockback/stun timer: while >0 the mob AI doesn't drive velocity, so a melee impulse rides.
    stun: f32,
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

    /// Number of live mobs (excludes dropped items, XP orbs, arrows). Test-only since P16 replaced the
    /// single spawn cap with per-category counts (hostile_count/passive_count).
    #[cfg(test)]
    pub fn mob_count(&self) -> usize {
        self.list
            .iter()
            .filter(|e| matches!(e.kind, Kind::Mob(_)))
            .count()
    }

    /// Live hostile / passive mob counts (P16 per-category spawn caps).
    pub fn hostile_count(&self) -> usize {
        self.list.iter().filter(|e| matches!(e.kind, Kind::Mob(m) if m.species.hostile())).count()
    }
    pub fn passive_count(&self) -> usize {
        self.list.iter().filter(|e| matches!(e.kind, Kind::Mob(m) if !m.species.hostile())).count()
    }

    /// Feet positions of all live hostiles (S5 spawn selftest: surface-vs-underground tally).
    pub fn hostile_positions(&self) -> Vec<Vec3> {
        self.list
            .iter()
            .filter(|e| !e.dead && matches!(e.kind, Kind::Mob(m) if m.species.hostile()))
            .map(|e| e.pos)
            .collect()
    }

    /// Passive/neutral mobs within a horizontal `radius` of `pos` (S3): the natural-spawn cap
    /// counts only the LOCAL population — persisted pens far away must not starve fresh areas of
    /// animal spawns. XZ distance only (the reference point carries no meaningful altitude).
    pub fn passive_count_near_xz(&self, pos: Vec3, radius: f32) -> usize {
        self.list
            .iter()
            .filter(|e| {
                let dx = e.pos.x - pos.x;
                let dz = e.pos.z - pos.z;
                matches!(e.kind, Kind::Mob(m) if !m.species.hostile())
                    && (dx * dx + dz * dz) < radius * radius
            })
            .count()
    }

    /// Snapshot every persistable entity (S3): live mobs (except the Warden, which "digs away"),
    /// dropped item stacks, XP orbs. Arrows are transient; AI state / love timers / creeper fuses
    /// deliberately reset on load. Capped defensively so a pathological save can't balloon.
    pub fn to_save(&self) -> Vec<crate::persistence::EntitySave> {
        use crate::persistence::EntitySave as E;
        let mut out: Vec<E> = self
            .list
            .iter()
            .filter(|e| !e.dead)
            .filter_map(|e| match e.kind {
                Kind::Mob(m) if m.species != Species::Warden => Some(E::Mob {
                    species: m.species.to_u8(),
                    pos: e.pos.to_array(),
                    health: m.health,
                    baby: m.baby,
                    growth: m.growth,
                    angry: m.angry,
                    slime_tier: m.slime_tier,
                }),
                Kind::Item(stack) => Some(E::Item { pos: e.pos.to_array(), stack, age: e.age }),
                Kind::Xp(n) => Some(E::Xp { pos: e.pos.to_array(), amount: n }),
                _ => None,
            })
            .collect();
        if out.len() > 512 {
            log::warn!("entity save capped at 512 (had {})", out.len());
            out.truncate(512);
        }
        out
    }

    /// Rebuild entities from a save (S3). Semantics are clamped per species here (persistence only
    /// checks structure); restored bodies start at rest — a reload must not re-roll the random pop
    /// velocity and scatter resting drops toward lava/cliffs.
    pub fn restore(&mut self, saved: Vec<crate::persistence::EntitySave>) {
        use crate::persistence::EntitySave as E;
        for s in saved {
            match s {
                E::Mob { species, pos, health, baby, growth, angry, slime_tier } => {
                    let Some(sp) = Species::from_u8(species) else { continue };
                    if sp == Species::Warden {
                        continue;
                    }
                    let pos = Vec3::from(pos);
                    if sp == Species::Slime {
                        self.spawn_slime(pos, slime_tier);
                    } else if baby {
                        self.spawn_baby(pos, sp);
                    } else {
                        self.spawn_mob(pos, sp);
                    }
                    if let Some(e) = self.list.last_mut() {
                        e.vel = Vec3::ZERO;
                        if let Kind::Mob(ref mut m) = e.kind {
                            let cap = if sp == Species::Slime {
                                slime_health(slime_tier)
                            } else {
                                sp.max_health()
                            };
                            m.health = health.clamp(0.5, cap);
                            m.angry = angry;
                            if baby {
                                m.growth = growth.clamp(0.0, GROW_TIME);
                            }
                        }
                    }
                }
                E::Item { pos, stack, age } => {
                    self.spawn_item(Vec3::from(pos), stack);
                    if let Some(e) = self.list.last_mut() {
                        e.vel = Vec3::ZERO;
                        e.age = age;
                    }
                }
                E::Xp { pos, amount } => {
                    self.spawn_xp(Vec3::from(pos), amount);
                    if let Some(e) = self.list.last_mut() {
                        e.vel = Vec3::ZERO;
                    }
                }
            }
        }
    }

    /// Live Warden count (U11): the shrieker won't summon a second Warden while one already prowls.
    pub fn warden_count(&self) -> usize {
        self.list.iter().filter(|e| matches!(e.kind, Kind::Mob(m) if m.species == Species::Warden)).count()
    }

    /// Feet position of the i-th live mob (P15 navigation tests).
    #[cfg(test)]
    pub(crate) fn mob_pos(&self, i: usize) -> Vec3 {
        self.list
            .iter()
            .filter(|e| matches!(e.kind, Kind::Mob(_)))
            .nth(i)
            .unwrap()
            .pos
    }

    /// MobData of the i-th live mob (P17 life-cycle tests).
    #[cfg(test)]
    fn mob_data(&self, i: usize) -> MobData {
        self.list
            .iter()
            .filter_map(|e| if let Kind::Mob(m) = e.kind { Some(m) } else { None })
            .nth(i)
            .unwrap()
    }
    #[cfg(test)]
    pub(crate) fn mob_in_love(&self, i: usize) -> bool {
        self.mob_data(i).love > 0.0
    }
    #[cfg(test)]
    pub(crate) fn mob_is_baby(&self, i: usize) -> bool {
        self.mob_data(i).baby
    }
    #[cfg(test)]
    pub(crate) fn mob_health(&self, i: usize) -> f32 {
        self.mob_data(i).health
    }
    #[cfg(test)]
    pub(crate) fn mob_breed_cd(&self, i: usize) -> f32 {
        self.mob_data(i).breed_cd
    }

    /// Passive/hostile mob tally (headless spawn verification).
    pub fn species_summary(&self) -> String {
        let (mut passive, mut hostile) = (0, 0);
        for e in &self.list {
            if let Kind::Mob(m) = e.kind {
                if m.species.hostile() {
                    hostile += 1;
                } else {
                    passive += 1;
                }
            }
        }
        format!("passive:{passive} hostile:{hostile}")
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
                let (w, h) = dims(&m);
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

    /// P17 feed: if the ray (origin,dir) hits a breedable adult within `reach` and `food` is that
    /// animal's breeding food, put it into love-mode. Returns true (→ the caller spends one food).
    pub fn try_feed(&mut self, origin: Vec3, dir: Vec3, reach: f32, food: crate::item::ItemId) -> bool {
        let dir = dir.normalize_or_zero();
        let mut best: Option<(usize, f32)> = None;
        for (i, e) in self.list.iter().enumerate() {
            if let Kind::Mob(m) = e.kind {
                let (w, h) = dims(&m);
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
        if let Some((i, _)) = best {
            if let Kind::Mob(m) = &mut self.list[i].kind {
                if !m.baby && m.breed_cd <= 0.0 && m.species.breeding_food() == Some(food) {
                    m.love = LOVE_TIME;
                    return true;
                }
            }
        }
        false
    }

    /// Melee: damage the nearest mob the ray hits within `reach`, with knockback + a hurt-flash.
    /// `crit` lengthens the hurt-flash (the damage multiplier is applied by the caller). `kb_mult`
    /// scales the horizontal knockback (a 1.9 sprint-attack passes >1). `sweep = Some(dmg)` also deals
    /// that fixed light damage + a shove (away from the player) to OTHER mobs within `SWEEP_RADIUS` of
    /// the primary target — the 1.9 sword sweep. Returns true if the primary mob was struck.
    pub fn attack(
        &mut self,
        origin: Vec3,
        dir: Vec3,
        reach: f32,
        damage: f32,
        crit: bool,
        sweep: Option<f32>,
        kb_mult: f32,
    ) -> bool {
        let dir = dir.normalize_or_zero();
        let mut best: Option<(usize, f32)> = None;
        for (i, e) in self.list.iter().enumerate() {
            if let Kind::Mob(m) = e.kind {
                let (w, h) = dims(&m);
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
        let target_pos = self.list[i].pos;
        {
            let e = &mut self.list[i];
            if let Kind::Mob(m) = &mut e.kind {
                // U11: an emerging Warden is invulnerable (and absorbs the knockback) — bail early.
                if m.emerge > 0.0 {
                    return true;
                }
                m.health -= damage;
                m.hurt = if crit { 0.6 } else { 0.35 };
                if matches!(m.species, Species::Wolf | Species::Enderman) {
                    m.angry = true; // P18: a struck neutral turns hostile toward the player
                }
            }
            let mut kb = e.pos - origin;
            kb.y = 0.0;
            let kb = kb.normalize_or_zero();
            e.vel.x = kb.x * KNOCKBACK * kb_mult;
            e.vel.z = kb.z * KNOCKBACK * kb_mult;
            e.vel.y = e.vel.y.max(0.0) + 3.0; // small pop-up
            e.stun = 0.3; // suppress the AI velocity drive so the impulse actually integrates
        }
        // Sweep: a charged sword on the ground also lightly hits OTHER mobs near the primary (fixed
        // damage, no crit/charge scaling, no pop-up), shoved away from the player.
        if let Some(sweep_dmg) = sweep {
            for (j, e) in self.list.iter_mut().enumerate() {
                if j == i {
                    continue;
                }
                if let Kind::Mob(m) = &mut e.kind {
                    let mut to = e.pos - target_pos;
                    to.y = 0.0;
                    if to.length() <= SWEEP_RADIUS {
                        m.health -= sweep_dmg;
                        m.hurt = m.hurt.max(0.25);
                        let mut push = e.pos - origin;
                        push.y = 0.0;
                        let push = push.normalize_or_zero();
                        e.vel.x = push.x * SWEEP_KNOCKBACK;
                        e.vel.z = push.z * SWEEP_KNOCKBACK;
                        e.stun = e.stun.max(0.2);
                    }
                }
            }
        }
        true
    }

    /// Debug: remove all mobs (keeps items/XP/arrows) — headless spawn-rule verification.
    pub fn clear_mobs(&mut self) {
        self.list.retain(|e| !matches!(e.kind, Kind::Mob(_)));
    }

    /// Remove all HOSTILE mobs (keeps passive animals, items, XP). Called once when the world
    /// switches to Peaceful — Minecraft makes hostiles vanish but leaves the animals.
    pub fn despawn_hostiles(&mut self) {
        self.list
            .retain(|e| !matches!(e.kind, Kind::Mob(m) if m.species.hostile()));
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
        self.spawn_mob_kind(pos, species, false);
    }

    /// Spawn a juvenile (P17): scaled model, cannot breed, grows up after GROW_TIME.
    pub fn spawn_baby(&mut self, pos: Vec3, species: Species) {
        self.spawn_mob_kind(pos, species, true);
    }

    /// Spawn a slime of an explicit size tier (P18) — used for split children. (Plain `spawn_mob`
    /// makes a LARGE slime, so try_spawn / the demo need no special-case.)
    pub fn spawn_slime(&mut self, pos: Vec3, tier: u8) {
        self.spawn_mob_kind(pos, Species::Slime, false);
        if let Some(e) = self.list.last_mut() {
            if let Kind::Mob(m) = &mut e.kind {
                m.slime_tier = tier;
                m.health = slime_health(tier);
            }
        }
    }

    fn spawn_mob_kind(&mut self, pos: Vec3, species: Species, baby: bool) {
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
                deflect: 0.0,
                deflect_heading: 0.0,
                baby,
                growth: if baby { GROW_TIME } else { 0.0 },
                love: 0.0,
                breed_cd: 0.0,
                angry: false,
                // A slime spawned via spawn_mob defaults to the LARGE tier (health already = max).
                slime_tier: if species == Species::Slime { 2 } else { 0 },
                // U11 Warden: a fresh Warden emerges (immobile + invulnerable) for a moment; all other
                // species (and these fields generally) are inert.
                emerge: if species == Species::Warden { WARDEN_EMERGE_TIME } else { 0.0 },
                no_vib: 0.0,
                sonic_cd: 0.0,
                warden_target: None,
                voice: 3.0 + randf(&mut rng) * 8.0, // S9: first vocalization 3-11 s in
            }),
            pos,
            vel: Vec3::ZERO,
            on_ground: false,
            age: 0.0,
            heading,
            wander: 1.0 + randf(&mut rng) * 2.0,
            rng,
            dead: false,
            stun: 0.0,
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
            stun: 0.0,
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
            stun: 0.0,
        });
    }

    /// Spawn a MOB arrow (skeleton shot) at `pos` with velocity `vel`. Kept as `spawn_arrow` so the
    /// skeleton-shot loop + existing tests need no change.
    pub fn spawn_arrow(&mut self, pos: Vec3, vel: Vec3) {
        self.spawn_arrow_owned(pos, vel, ARROW_DAMAGE, ArrowOwner::Mob);
    }

    /// Spawn a PLAYER arrow (bow shot, P13): hits mobs, never the player.
    pub fn spawn_player_arrow(&mut self, pos: Vec3, vel: Vec3, damage: f32) {
        self.spawn_arrow_owned(pos, vel, damage, ArrowOwner::Player);
    }

    fn spawn_arrow_owned(&mut self, pos: Vec3, vel: Vec3, damage: f32, owner: ArrowOwner) {
        let rng = self.next_seed();
        self.list.push(Entity {
            kind: Kind::Arrow { damage, owner },
            pos,
            vel,
            on_ground: false,
            age: 0.0,
            heading: vel.z.atan2(vel.x),
            wander: 0.0,
            rng,
            dead: false,
            stun: 0.0,
        });
    }

    /// Advance AI + physics for all entities and drop the dead/collected ones. Returns the blocks
    /// of any item drops the player walked over this frame (to add to their inventory).
    /// `is_hazard(cell)` reports a contact-damage block (lava) for mob navigation (P15) — distinct
    /// from `is_solid` because lava is non-solid (a mob would otherwise walk straight into it).
    pub fn update(
        &mut self,
        dt: f32,
        player_pos: Vec3,
        is_solid: impl Fn(IVec3) -> bool,
        is_hazard: impl Fn(IVec3) -> bool,
        // P17: true at a cell where an undead should burn — i.e. it is daytime AND the cell is open to
        // the sky. Folds the day factor + the P16 sky-light query into one closure (supplied by game.rs)
        // so the entity tick stays self-contained and `update` keeps a single new parameter.
        sun_burn: impl Fn(IVec3) -> bool,
        // U11: the most-recent live vibration within `range` of a point (or None) — the Warden's only
        // sense. Supplied by game.rs over its vibration bus (footsteps / block edits / deaths).
        vibration: impl Fn(Vec3, f32) -> Option<Vec3>,
        // S3: whether the chunk at a world position is resident. An entity in an unloaded chunk is
        // FROZEN (no physics/AI/aging) — the world reads as AIR there, so ticking it would drop a
        // persisted pen or a distant no-despawn passive into the void.
        chunk_loaded: impl Fn(IVec3) -> bool,
    ) -> Collected {
        let mut collected = Collected::default();
        let mut deaths: Vec<(Vec3, Species, u8)> = Vec::new(); // (pos, species, slime tier)
        let mut shots: Vec<(Vec3, Vec3)> = Vec::new(); // skeleton arrows (pos, vel), spawned post-loop
        let mut arrow_hits: Vec<(usize, Vec3, f32)> = Vec::new(); // player arrows -> (mob idx, pos, dmg)
        let mut births: Vec<(Vec3, Species)> = Vec::new(); // P17 babies, spawned post-loop
        let mut slime_births: Vec<(Vec3, u8)> = Vec::new(); // P18 split children (pos, tier), post-loop
        // P17 breeding partner snapshot: in-love, breedable adults (list-idx, pos, species). Read-only
        // (same reason as mob_boxes — can't borrow siblings during the &mut iteration).
        let lovers: Vec<(usize, Vec3, Species)> = self
            .list
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e.kind {
                Kind::Mob(m) if !m.baby && m.love > 0.0 && m.breed_cd <= 0.0 => {
                    Some((i, e.pos, m.species))
                }
                _ => None,
            })
            .collect();
        // Mob hit-spheres for player-arrow tests — a read-only snapshot taken before the `&mut self.list`
        // iteration (we can't borrow other entities while one is mutably borrowed). Pre-move positions are
        // fine: a mob drifts <~0.06 blocks per frame, the same one-frame lag the deaths/knockback already
        // settle with. A forgiving sphere (like the skeleton->player test) so fast arrows don't slip past.
        let mob_boxes: Vec<(Vec3, f32, usize)> = self
            .list
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if let Kind::Mob(m) = e.kind {
                    let (w, h) = dims(&m);
                    let center = e.pos + Vec3::new(0.0, h * 0.5, 0.0);
                    let radius = (w * 0.5).max(h * 0.5) + 0.3;
                    Some((center, radius, i))
                } else {
                    None
                }
            })
            .collect();
        for (self_idx, e) in self.list.iter_mut().enumerate() {
            if !chunk_loaded(e.pos.floor().as_ivec3()) {
                // S3: frozen until its chunk streams in — but a frozen HOSTILE beyond the despawn
                // shell still despawns (S5): the player outrunning the streaming radius must not
                // leave far frozen hostiles pinning the population cap (no physics needed to die).
                if let Kind::Mob(m) = e.kind {
                    if m.species.hostile() && (e.pos - player_pos).length() > DESPAWN_RADIUS {
                        e.dead = true;
                    }
                }
                continue;
            }
            e.age += dt;
            match e.kind {
                Kind::Mob(mut m) => {
                    // U11 Warden: a dedicated branch — it is BLIND (ignores line-of-sight), hunts the
                    // most-recent vibration it can hear, melees on contact, and fires a ranged sonic
                    // boom. While emerging it is immobile + invulnerable. The generic mob AI below is
                    // skipped (`continue`).
                    if m.species == Species::Warden {
                        m.emerge = (m.emerge - dt).max(0.0);
                        m.sonic_cd = (m.sonic_cd - dt).max(0.0);
                        m.atk_cd = (m.atk_cd - dt).max(0.0);
                        m.hurt = (m.hurt - dt).max(0.0);
                        let (mw, mh) = dims(&m);
                        let to_player = player_pos - e.pos;
                        let dist = to_player.length();
                        // Hear the nearest recent vibration; else age the idle timer toward digging away.
                        if let Some(v) = vibration(e.pos, WARDEN_HEAR_RANGE) {
                            m.warden_target = Some(v);
                            m.no_vib = 0.0;
                        } else {
                            m.no_vib += dt;
                            // Trail went cold: drop the stale vibration so pursuit falls back to the
                            // player (otherwise it parks on the last-heard spot forever).
                            if m.no_vib > WARDEN_TARGET_STALE {
                                m.warden_target = None;
                            }
                        }
                        if m.emerge > 0.0 {
                            // Rising from the ground: hold position (gravity only), invulnerable.
                            e.vel.x = 0.0;
                            e.vel.z = 0.0;
                            e.vel.y -= GRAVITY * dt;
                            e.on_ground = collide_move(&mut e.pos, &mut e.vel, mw, mh, dt, &is_solid);
                        } else if m.no_vib > WARDEN_DIG_TIMEOUT && dist > WARDEN_HEAR_RANGE {
                            e.dead = true; // lost the trail → digs back underground (no loot)
                        } else {
                            // Blind pursuit: head toward the last vibration heard (or the player until
                            // it has heard one). No line-of-sight test — it walks through the dark.
                            let tgt = m.warden_target.unwrap_or(player_pos);
                            let to = tgt - e.pos;
                            let horiz = (to.x * to.x + to.z * to.z).sqrt();
                            if horiz > 1.8 {
                                e.heading = to.z.atan2(to.x);
                                let st = steer(
                                    e.pos,
                                    &mut e.heading,
                                    &mut m.deflect,
                                    &mut m.deflect_heading,
                                    dt,
                                    mw * 0.5,
                                    false, // edge_avoid off — relentless like a chase
                                    true,
                                    e.on_ground,
                                    &is_solid,
                                    &is_hazard,
                                );
                                let fwd = if st.stop { 0.0 } else { WARDEN_SPEED };
                                e.vel.x = e.heading.cos() * fwd;
                                e.vel.z = e.heading.sin() * fwd;
                                if let Some(vy) = st.vy {
                                    e.vel.y = vy;
                                }
                            } else {
                                e.heading = to.z.atan2(to.x);
                                e.vel.x = 0.0;
                                e.vel.z = 0.0;
                            }
                            e.vel.y -= GRAVITY * dt;
                            e.on_ground = collide_move(&mut e.pos, &mut e.vel, mw, mh, dt, &is_solid);
                            // Melee on genuine AABB contact with the player.
                            let dh = player_pos - e.pos;
                            let hd = (dh.x * dh.x + dh.z * dh.z).sqrt();
                            if hd <= mw * 0.5 + 0.9 && dh.y.abs() <= mh + 0.7 && m.atk_cd <= 0.0 {
                                collected.player_damage.push((m.species.contact_damage(), e.pos));
                                m.atk_cd = MOB_ATTACK_COOLDOWN;
                            }
                            // Sonic boom: ranged, when the player is within range and roughly ahead.
                            if dist < WARDEN_SONIC_RANGE && dist > 2.5 && m.sonic_cd <= 0.0 {
                                let aim = Vec3::new(e.heading.cos(), 0.0, e.heading.sin());
                                if aim.dot(to_player.normalize_or_zero()) > 0.35 {
                                    collected.player_damage.push((WARDEN_SONIC_DAMAGE, e.pos));
                                    m.sonic_cd = WARDEN_SONIC_CD;
                                }
                            }
                        }
                        // Despawn if it fell out / wandered absurdly far (mirror the generic guard, NO loot).
                        if e.pos.y < FALL_OUT_Y || dist > DESPAWN_RADIUS {
                            e.dead = true;
                        }
                        if m.health <= 0.0 {
                            e.dead = true;
                            deaths.push((e.pos, m.species, 0)); // killed (tables give no loot/xp)
                        }
                        e.kind = Kind::Mob(m);
                        continue; // skip the generic mob AI for this entity
                    }

                    m.hurt = (m.hurt - dt).max(0.0); // hurt-flash decays after a hit
                    // P17 life-cycle timers (persist via the e.kind write-back at the block's end).
                    m.love = (m.love - dt).max(0.0);
                    m.breed_cd = (m.breed_cd - dt).max(0.0);
                    if m.baby {
                        m.growth = (m.growth - dt).max(0.0);
                        if m.growth <= 0.0 {
                            m.baby = false; // matured into an adult
                        }
                    }
                    m.atk_cd = (m.atk_cd - dt).max(0.0);
                    // S9: idle vocalizations on a per-mob timer (species without a voice stay mute).
                    m.voice -= dt;
                    if m.voice <= 0.0 {
                        e.rng ^= e.rng << 13;
                        e.rng ^= e.rng >> 7;
                        e.rng ^= e.rng << 17;
                        m.voice = 6.0 + (e.rng % 10) as f32;
                        if let Some(v) = voice_sfx(m.species) {
                            collected.sounds.push((v, e.pos));
                        }
                    }
                    let (mw, mh) = dims(&m);

                    // Perception: distance to the player + (for hostiles) exact line-of-sight.
                    let to_player = player_pos - e.pos;
                    let dist = to_player.length();
                    let eye = Vec3::new(0.0, mh * 0.5, 0.0);
                    let target = player_pos + Vec3::new(0.0, 0.9, 0.0);

                    // P18: a NEUTRAL mob (wolf/enderman) acts hostile once it's been provoked (angry).
                    let effective_hostile = m.species.hostile() || m.angry;

                    // LOS only matters within the give-up range; skip the DDA for far/passive mobs.
                    let los = effective_hostile
                        && dist < CALM_RADIUS
                        && line_of_sight(e.pos + eye, target, &is_solid);

                    // State transition.
                    if effective_hostile {
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
                            // Run away from the player; `steer` (below) handles cornering/deflection now.
                            e.heading = heading_to(-to_player);
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
                    if e.stun > 0.0 {
                        // Recovering from a knockback hit: let the impulse ride (with friction),
                        // don't let the AI immediately re-drive velocity back toward the player.
                        e.stun -= dt;
                        e.vel.x *= 0.86;
                        e.vel.z *= 0.86;
                    } else {
                        // P15 local steering: deflect around walls/cliffs/lava + deterministic step-up
                        // (replaces the old blind 5% random hop). Chase/Attack pursue relentlessly
                        // (no edge-avoid) but still route around hazards; passive states avoid drops.
                        let relentless = matches!(m.ai, Ai::Chase | Ai::Attack);
                        let st = if speed > 0.0 {
                            steer(
                                e.pos,
                                &mut e.heading,
                                &mut m.deflect,
                                &mut m.deflect_heading,
                                dt,
                                mw * 0.5,
                                !relentless, // edge_avoid off for chase
                                relentless,
                                e.on_ground,
                                &is_solid,
                                &is_hazard,
                            )
                        } else {
                            Steered { vy: None, stop: false }
                        };
                        let fwd = if st.stop { 0.0 } else { speed };
                        e.vel.x = e.heading.cos() * fwd;
                        e.vel.z = e.heading.sin() * fwd;
                        if let Some(vy) = st.vy {
                            e.vel.y = vy;
                        }
                        // P18: slimes hop while grounded + active (overrides any step-up impulse).
                        if m.species == Species::Slime
                            && e.on_ground
                            && matches!(m.ai, Ai::Wander | Ai::Chase | Ai::Attack)
                        {
                            e.vel.y = 7.0;
                        }
                    }

                    // Snapshot the firing eye BEFORE the move so a skeleton's LOS check (taken from
                    // the pre-move position) and its arrow's spawn point agree.
                    let shot_from = e.pos + Vec3::new(0.0, mh * 0.7, 0.0);
                    e.vel.y -= GRAVITY * dt;
                    e.on_ground = collide_move(&mut e.pos, &mut e.vel, mw, mh, dt, &is_solid);

                    // Ranged / contact attacks, per species.
                    match m.species {
                        // Skeleton: looses an arrow at the player while it has a clear shot.
                        Species::Skeleton => {
                            if matches!(m.ai, Ai::Chase | Ai::Attack) && los && m.atk_cd <= 0.0 {
                                let aim = (target - shot_from).normalize_or_zero();
                                // Aim a touch high so gravity arcs the shot into the target.
                                let vel = aim * ARROW_SPEED + Vec3::new(0.0, dist * 0.05, 0.0);
                                shots.push((shot_from, vel));
                                collected.sounds.push((crate::audio::Sfx::BowShoot, e.pos)); // S9
                                m.atk_cd = SKELETON_SHOOT_CD;
                            }
                        }
                        // Creeper: primes a fuse in attack range, then detonates (no melee, no XP).
                        Species::Creeper => {
                            if m.ai == Ai::Attack {
                                if m.fuse < 0.0 {
                                    m.fuse = CREEPER_FUSE;
                                    // S9: THE warning cue — the hiss starts exactly with the fuse.
                                    collected.sounds.push((crate::audio::Sfx::CreeperHiss, e.pos));
                                }
                                m.fuse -= dt;
                                // Blink white as it primes, but never hide a fresh melee hit-flash.
                                let blink = (1.0 - m.fuse / CREEPER_FUSE).clamp(0.0, 0.45);
                                m.hurt = m.hurt.max(blink);
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
                        // Zombie / spider: melee contact damage, but only on genuine AABB contact
                        // (the hysteresis Attack state can hold ~2.4 blocks out — too far to bite).
                        _ => {
                            let dh = player_pos - e.pos;
                            let horiz = (dh.x * dh.x + dh.z * dh.z).sqrt();
                            let touching = horiz <= mw * 0.5 + 0.6 && dh.y.abs() <= mh + 0.5;
                            if m.ai == Ai::Attack && m.atk_cd <= 0.0 && touching {
                                let dmg = m.species.contact_damage();
                                if dmg > 0.0 {
                                    collected.player_damage.push((dmg, e.pos)); // source = mob feet
                                    m.atk_cd = MOB_ATTACK_COOLDOWN;
                                }
                            }
                        }
                    }

                    // P17B undead daytime burn: zombies/skeletons take fire damage in the open sun
                    // (reuses the M29 hurt-flash + the existing death/loot path below).
                    if matches!(m.species, Species::Zombie | Species::Skeleton) {
                        let feet = IVec3::new(
                            e.pos.x.floor() as i32,
                            e.pos.y.floor() as i32,
                            e.pos.z.floor() as i32,
                        );
                        if sun_burn(feet) {
                            m.health -= BURN_DPS * dt;
                            m.hurt = m.hurt.max(0.25);
                        }
                    }

                    // P17A breeding: an in-love adult with an in-love same-species partner in range
                    // produces one baby (the lower-indexed parent spawns it); both then go on cooldown.
                    if !m.baby && m.love > 0.0 && m.breed_cd <= 0.0 {
                        if let Some(&(jidx, jpos, _)) = lovers.iter().find(|&&(lidx, ppos, sp)| {
                            lidx != self_idx && sp == m.species && (ppos - e.pos).length() <= BREED_RANGE
                        }) {
                            if self_idx < jidx {
                                births.push(((e.pos + jpos) * 0.5, m.species));
                            }
                            m.love = 0.0;
                            m.breed_cd = BREED_COOLDOWN;
                        }
                    }

                    if e.dead {
                        // already consumed (creeper detonated) — no loot
                    } else if m.health <= 0.0 {
                        e.dead = true;
                        deaths.push((e.pos, m.species, m.slime_tier)); // killed → drops loot
                        // P18: a slime splits into smaller slimes (large→3, medium→2); the smallest
                        // (tier 0) does NOT split. Children spawn post-loop (after retain), no aliasing.
                        if m.species == Species::Slime && m.slime_tier > 0 {
                            let child = m.slime_tier - 1;
                            let n = if m.slime_tier == 2 { 3 } else { 2 };
                            for k in 0..n {
                                let a = k as f32 / n as f32 * TAU;
                                slime_births.push((e.pos + Vec3::new(a.cos() * 0.4, 0.1, a.sin() * 0.4), child));
                            }
                        }
                    } else if e.pos.y < FALL_OUT_Y || (m.species.hostile() && dist > DESPAWN_RADIUS) {
                        // S3: only HOSTILE mobs distance-despawn (vanilla). Passives + neutrals
                        // (cow/pig/sheep/chicken/wolf/villager) persist, so pens and pets survive
                        // the player roaming — they also persist across save/load now.
                        e.dead = true; // fell out of the world / roamed too far → no loot
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
                Kind::Arrow { damage: dmg, owner } => {
                    e.vel.y -= ARROW_GRAVITY * dt;
                    e.heading = e.vel.z.atan2(e.vel.x);
                    // Sub-step the flight so a fast arrow can't tunnel through a 1-block wall or skip
                    // past its target in a single frame: march ~0.4 blocks at a time, testing the cell
                    // ahead and the owner-appropriate target hit at each step.
                    let move_vec = e.vel * dt;
                    let steps = (move_vec.length() / 0.4).ceil().max(1.0) as i32;
                    let step = move_vec / steps as f32;
                    let chest = player_pos + Vec3::new(0.0, 1.0, 0.0);
                    for _ in 0..steps {
                        let next = e.pos + step;
                        let cell = IVec3::new(
                            next.x.floor() as i32,
                            next.y.floor() as i32,
                            next.z.floor() as i32,
                        );
                        if is_solid(cell) {
                            e.dead = true; // stuck in the wall — stop before entering it (no hit)
                            break;
                        }
                        e.pos = next;
                        match owner {
                            ArrowOwner::Mob => {
                                if (chest - e.pos).length() < 0.7 {
                                    // Source = a point back along the flight path (the impact point sits
                                    // on the player's chest, too degenerate for a direction test).
                                    let from = e.pos - e.vel.normalize_or_zero();
                                    collected.player_damage.push((dmg, from));
                                    e.dead = true;
                                    break;
                                }
                            }
                            ArrowOwner::Player => {
                                // The shooter is never in mob_boxes, and the muzzle spawns outside the
                                // player AABB, so a player arrow can't self-hit. Defer the mob mutation
                                // (can't touch other entities mid-&mut-iteration).
                                let ap = e.pos;
                                if let Some(&(_, _, mi)) =
                                    mob_boxes.iter().find(|&&(c, r, _)| (c - ap).length() < r)
                                {
                                    arrow_hits.push((mi, ap, dmg));
                                    e.dead = true;
                                    break;
                                }
                            }
                        }
                    }
                    if !e.dead && (e.age > ARROW_LIFETIME || e.pos.y < FALL_OUT_Y) {
                        e.dead = true;
                    }
                }
            }
        }
        // Apply player-arrow hits (deferred from the &mut iteration above; indices are still valid
        // pre-retain). Light knockback + hurt-flash; the mob's death/loot settles next tick like melee.
        for (mi, hit_pos, dmg) in arrow_hits {
            collected.sounds.push((crate::audio::Sfx::ArrowHit, hit_pos)); // S9
            if let Some(e) = self.list.get_mut(mi) {
                if let Kind::Mob(m) = &mut e.kind {
                    if m.emerge > 0.0 {
                        continue; // U11: an emerging Warden shrugs off arrows too
                    }
                    m.health -= dmg;
                    m.hurt = m.hurt.max(0.35);
                }
                let mut kb = e.pos - hit_pos;
                kb.y = 0.0;
                let kb = kb.normalize_or_zero();
                e.vel.x = kb.x * KNOCKBACK * 0.5;
                e.vel.z = kb.z * KNOCKBACK * 0.5;
                e.vel.y = e.vel.y.max(0.0) + 2.0;
                e.stun = e.stun.max(0.3);
            }
        }
        self.list.retain(|e| !e.dead);
        // Killed mobs drop their species loot + experience.
        for (pos, species, tier) in deaths {
            collected.sounds.push((crate::audio::Sfx::MobDeath, pos)); // S9
            collected.deaths.push(pos); // U10: sculk-catalyst spread + death vibration
            let drop_at = pos + Vec3::new(0.0, 0.3, 0.0);
            self.spawn_xp(drop_at, species.xp_drop());
            // S13: roll each table entry's count independently; only the smallest slime body
            // sheds slimeballs (its parents already dropped XP above, matching vanilla).
            if species != Species::Slime || tier == 0 {
                for &(item, min, max) in species.loot() {
                    let span = (max - min) as u64 + 1;
                    // next_seed() is ODD-ONLY (a seed guarantee): raw `% 2` would flip every
                    // 0-1 range into "always 1" (ender pearls at 100%). Remix, then roll.
                    let r = self.next_seed().wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33;
                    let count = min + (r % span) as u8;
                    if count > 0 {
                        self.spawn_item(drop_at, ItemStack::new(item, count));
                    }
                }
            }
            // S6: zombies rarely carry produce — the bootstrap source for carrots/potatoes
            // (vanilla's rare drop; farms take over once you have one).
            if species == Species::Zombie {
                let r = self.next_seed().wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33; // de-bias
                if r % 100 < 5 {
                    let item = if (r >> 8) % 2 == 0 { crate::item::CARROT } else { crate::item::POTATO };
                    self.spawn_item(drop_at, ItemStack::new(item, 1));
                }
            }
        }
        // Arrows skeletons loosed this frame (spawned after the iteration to avoid aliasing the list).
        for (pos, vel) in shots {
            self.spawn_arrow(pos, vel);
        }
        // P17 babies bred this frame (deferred — couldn't push into self.list mid-iteration).
        for (pos, species) in births {
            collected.bursts.push((crate::game::Burst::Hearts, pos)); // S10
            self.spawn_baby(pos, species);
        }
        for (pos, tier) in slime_births {
            self.spawn_slime(pos, tier);
        }
        collected
    }

    /// Box geometry for every entity (opaque layer), lit by the chunk pipeline.
    pub fn build_mesh(&self) -> MeshData {
        let mut mesh = MeshData::default();
        for e in &self.list {
            match e.kind {
                Kind::Mob(m) => {
                    // Each species draws its own multi-box model, yaw-rotated to face its heading;
                    // a baby renders at BABY_SCALE (P17), a slime by its size tier (P18).
                    let scale = (if m.baby { BABY_SCALE } else { 1.0 }) * mob_scale(&m);
                    for p in model(m.species) {
                        push_part(&mut mesh.opaque, e.pos, e.heading, &p, m.hurt, scale);
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
                Kind::Arrow { .. } => {
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

/// Result of a `steer` call: an optional step-up impulse, and whether the mob should hold position
/// this frame (a relentless chaser boxed in by a hazard it can't route around — mill, don't retreat).
struct Steered {
    vy: Option<f32>,
    stop: bool,
}

/// P15 local steering. Given the goal `heading` the AI state already chose, nudge it to a heading the
/// mob can actually walk — stepping up 1-block ledges, refusing cliffs (when `edge_avoid`) and hazards
/// (always), and deflecting around obstacles (committing to a detour for DEFLECT_HOLD so it doesn't
/// stutter). Mutates `heading`/`deflect`/`deflect_heading` in place. `half` = mob footprint half-width.
#[allow(clippy::too_many_arguments)]
fn steer(
    pos: Vec3,
    heading: &mut f32,
    deflect: &mut f32,
    deflect_heading: &mut f32,
    dt: f32,
    half: f32,
    edge_avoid: bool,
    relentless: bool,
    on_ground: bool,
    is_solid: &impl Fn(IVec3) -> bool,
    is_hazard: &impl Fn(IVec3) -> bool,
) -> Steered {
    let probe = half + NAV_MARGIN; // sample beyond the leading face, not inside the footprint
    let base = pos.y.floor() as i32; // the mob's body/feet cell row; dy=0 = body, +1 = head, -1 = ground
    let cell = |h: f32, dy: i32| {
        IVec3::new(
            (pos.x + h.cos() * probe).floor() as i32,
            base + dy,
            (pos.z + h.sin() * probe).floor() as i32,
        )
    };
    // Can the mob safely advance along `h`? (A 1-block ledge IS walkable — it steps up below.)
    let walkable = |h: f32| -> bool {
        let foot = cell(h, 0); // body-level cell ahead (a solid here is a wall or a step-up ledge)
        // Never advance toward a hazard at body level, or one we'd drop a step into.
        if is_hazard(foot) || is_hazard(cell(h, -1)) {
            return false;
        }
        // A 2-tall wall (solid at both body and head height) is not a passable step.
        if is_solid(foot) && is_solid(cell(h, 1)) {
            return false;
        }
        // Edge: only when grounded + edge-avoiding, refuse a drop NAV_MAX_DROP or more cells deep.
        // (Open ahead at body level AND no ground at the step-down cell = a drop to measure.)
        if edge_avoid && on_ground && !is_solid(foot) && !is_solid(cell(h, -1)) {
            let mut depth = 1;
            while depth < NAV_MAX_DROP && !is_solid(cell(h, -1 - depth)) {
                depth += 1;
            }
            if depth >= NAV_MAX_DROP {
                return false;
            }
        }
        true
    };

    if walkable(*heading) {
        *deflect = 0.0; // path to the goal is clear — abandon any detour
    } else if *deflect > 0.0 && walkable(*deflect_heading) {
        *heading = *deflect_heading; // hold the still-valid committed detour
    } else {
        let chosen = DEFLECT_FAN.iter().map(|&off| *heading + off).find(|&h| walkable(h));
        match chosen {
            Some(h) => {
                *heading = h;
                *deflect_heading = h;
                *deflect = DEFLECT_HOLD;
            }
            None if relentless => {
                // Boxed in while chasing: mill at the edge facing the player (don't moonwalk away).
                return Steered { vy: None, stop: true };
            }
            None => {
                *heading += std::f32::consts::PI; // dead-end: back out
                *deflect_heading = *heading;
                *deflect = DEFLECT_HOLD;
            }
        }
    }
    *deflect = (*deflect - dt).max(0.0);

    // Step-up: a 1-block ledge straight ahead (solid foot, clear head + the cell above), on the ground.
    let vy = if on_ground
        && is_solid(cell(*heading, 0))
        && !is_solid(cell(*heading, 1))
        && !is_hazard(cell(*heading, 1))
        && !is_solid(cell(*heading, 2))
    {
        Some(STEP_UP_VEL)
    } else {
        None
    };
    Steered { vy, stop: false }
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
    // S2: terminal velocity for everything that falls (mobs, items, XP orbs) — an uncapped fall
    // integrated over a 0.1 s frame used to cross many blocks and tunnel thin floors.
    const TERMINAL: f32 = 60.0;
    // S2: substep each axis. Cells are 1.0 thick, so any step below 1.0 keeps a crossed cell
    // overlapping the stepped AABB regardless of body size (the smallest is the 0.18 XP orb —
    // body extent only adds margin on top of the cell-thickness bound); 0.2 is comfortable.
    const MAX_STEP: f32 = 0.2;
    vel.y = vel.y.max(-TERMINAL);
    let half = w * 0.5;
    let delta = *vel * dt;
    let mut on_ground = false;
    for &axis in &[0usize, 2, 1] {
        let mut remaining = comp(delta, axis);
        while remaining != 0.0 {
            let amount = remaining.clamp(-MAX_STEP, MAX_STEP);
            remaining -= amount;
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
                break; // the axis is blocked; discard the remaining distance
            }
        }
    }
    on_ground
}

/// Emit one mob model part: a box given in LOCAL space (feet at origin, +x forward), rotated by
/// `yaw` around the mob's vertical axis and placed at `feet`. `hurt` (0..1) is stashed in `shade.y`;
/// the chunk shader doesn't read it yet — M29 combat will both set it and add the red-flash tint.
/// Faces are never culled (mobs are small and self-contained).
fn push_part(geom: &mut Geometry, feet: Vec3, yaw: f32, p: &Part, hurt: f32, scale: f32) {
    let (s, co) = yaw.sin_cos();
    let rot = |lx: f32, ly: f32, lz: f32| -> [f32; 3] {
        [feet.x + lx * co - lz * s, feet.y + ly, feet.z + lx * s + lz * co]
    };
    let rotn = |nx: f32, nz: f32| -> [f32; 3] { [nx * co - nz * s, 0.0, nx * s + nz * co] };
    // P17: `scale` shrinks the model around the feet (local origin) for a baby; 1.0 = adult.
    let (mn, mx) = ([p.min[0] * scale, p.min[1] * scale, p.min[2] * scale],
        [p.max[0] * scale, p.max[1] * scale, p.max[2] * scale]);
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

    /// S2: entity falls are terminal-velocity-capped and substepped — a long uncapped fall used
    /// to cross several blocks per frame and tunnel straight through a 1-block floor.
    #[test]
    fn entity_fall_is_capped_and_lands_on_thin_floor() {
        let floor = |p: IVec3| p.y == 10; // a single 1-block-thick floor
        let mut pos = Vec3::new(0.5, 400.0, 0.5);
        let mut vel = Vec3::ZERO;
        let mut grounded = false;
        for _ in 0..200 {
            vel.y -= GRAVITY * 0.1;
            grounded = collide_move(&mut pos, &mut vel, 0.6, 0.9, 0.1, &floor);
            assert!(vel.y >= -60.0 - 1e-3, "terminal velocity must cap the fall: {}", vel.y);
            if grounded {
                break;
            }
        }
        assert!(grounded, "entity should land, not tunnel (y = {})", pos.y);
        assert!((pos.y - 11.0).abs() < 0.05, "rest y = {}", pos.y);
    }

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
        assert!(es.attack(o, d, 6.0, 2.0, false, None, 1.0)); // 4 -> 2
        assert!(es.attack(o, d, 6.0, 2.0, false, None, 1.0)); // 2 -> 0
        // The dead mob is culled on the next tick (and drops an XP orb in its place).
        let _ = es.update(0.016, o, |_| false, |_| false, |_| false, |_, _| None, |_| true);
        assert_eq!(es.ai_summary(), "mobs idle:0 wander:0 flee:0 chase:0 attack:0");
    }

    #[test]
    fn sweep_hits_neighbors_within_radius() {
        // A charged-sword sweep (Some(dmg)) also damages OTHER mobs within SWEEP_RADIUS of the primary
        // target; mobs beyond it are untouched. Lethal primary + lethal sweep kills exactly the two.
        let mut es = Entities::new();
        es.spawn_mob(Vec3::ZERO, Species::Chicken); // primary (the ray hits this one)
        es.spawn_mob(Vec3::new(1.0, 0.0, 0.0), Species::Chicken); // 1.0 < 1.5 -> swept
        es.spawn_mob(Vec3::new(3.0, 0.0, 0.0), Species::Chicken); // 3.0 > 1.5 -> safe
        let (o, d) = (Vec3::new(0.0, 0.4, -3.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(es.attack(o, d, 6.0, 10.0, false, Some(10.0), 1.0));
        let _ = es.update(0.016, o, |_| false, |_| false, |_| false, |_, _| None, |_| true);
        assert_eq!(es.mob_count(), 1, "primary + the in-radius neighbor die; the far mob survives");
    }

    #[test]
    fn no_sweep_leaves_neighbors_untouched() {
        // Without a sweep (None), a lethal hit kills only the primary; the neighbor is untouched.
        let mut es = Entities::new();
        es.spawn_mob(Vec3::ZERO, Species::Chicken);
        es.spawn_mob(Vec3::new(1.0, 0.0, 0.0), Species::Chicken);
        let (o, d) = (Vec3::new(0.0, 0.4, -3.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(es.attack(o, d, 6.0, 10.0, false, None, 1.0));
        let _ = es.update(0.016, o, |_| false, |_| false, |_| false, |_, _| None, |_| true);
        assert_eq!(es.mob_count(), 1, "only the primary dies without a sweep");
    }

    #[test]
    fn arrow_hits_player_and_despawns() {
        let mut es = Entities::new();
        es.spawn_arrow(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 12.0)); // flying +z
        let player = Vec3::new(0.0, 0.0, 3.0); // feet; chest (+1) sits at z=3
        let mut dmg = 0.0;
        for _ in 0..120 {
            dmg += es.update(1.0 / 60.0, player, |_| false, |_| false, |_| false, |_, _| None, |_| true).player_damage.iter().map(|(a, _)| *a).sum::<f32>();
            if es.count() == 0 {
                break;
            }
        }
        assert!(dmg > 0.0, "the arrow should damage the player");
        assert_eq!(es.count(), 0, "and despawn on impact");
    }

    #[test]
    fn player_arrow_hits_mob_not_player() {
        // A player (bow) arrow damages MOBS and never the shooter. (The mob-arrow path is covered by
        // arrow_hits_player_and_despawns, which now routes through the spawn_arrow mob-owner shim.)
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 0.0, 5.0), Species::Cow); // 10 hp, ahead on +z
        es.spawn_player_arrow(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 18.0), 12.0); // lethal
        let player = Vec3::ZERO; // the shooter
        let mut dmg = 0.0;
        for _ in 0..240 {
            dmg += es.update(1.0 / 60.0, player, |_| false, |_| false, |_| false, |_, _| None, |_| true).player_damage.iter().map(|(a, _)| *a).sum::<f32>();
            if es.mob_count() == 0 {
                break;
            }
        }
        assert_eq!(dmg, 0.0, "a player arrow never damages the player");
        assert_eq!(es.mob_count(), 0, "the player arrow killed the cow");
    }

    #[test]
    fn player_arrow_does_not_hit_shooter() {
        // No mobs: a player arrow loosed from the player's chest must never register a player hit.
        let mut es = Entities::new();
        es.spawn_player_arrow(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 12.0), 5.0);
        let player = Vec3::ZERO;
        let mut dmg = 0.0;
        for _ in 0..300 {
            dmg += es.update(1.0 / 60.0, player, |_| false, |_| false, |_| false, |_, _| None, |_| true).player_damage.iter().map(|(a, _)| *a).sum::<f32>();
            if es.count() == 0 {
                break;
            }
        }
        assert_eq!(dmg, 0.0, "a player arrow never hurts the shooter");
        assert_eq!(es.count(), 0, "it despawns by lifetime / fall-out");
    }

    #[test]
    fn distant_mobs_despawn() {
        // S3: only HOSTILES distance-despawn; passives persist (pens must survive roaming).
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 1.0, 0.0), Species::Cow); // near passive
        es.spawn_mob(Vec3::new(100.0, 1.0, 0.0), Species::Pig); // far passive — STAYS
        es.spawn_mob(Vec3::new(100.0, 1.0, 8.0), Species::Zombie); // far hostile — despawns
        assert_eq!(es.mob_count(), 3);
        let _ = es.update(0.05, Vec3::new(0.0, 0.5, 0.0), |p: IVec3| p.y < 1, |_| false, |_| false, |_, _| None, |_| true); // floor at y<1
        assert_eq!(es.mob_count(), 2, "the far hostile despawns; both passives stay");
        assert_eq!(es.passive_count_near_xz(Vec3::ZERO, 48.0), 1, "only the near passive counts locally");
    }

    #[test]
    fn killed_mob_drops_loot_and_xp() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::ZERO, Species::Cow); // -> beef + leather + xp
        let (o, d) = (Vec3::new(0.0, 0.6, -3.0), Vec3::new(0.0, 0.0, 1.0));
        for _ in 0..3 {
            es.attack(o, d, 6.0, 5.0, false, None, 1.0); // 10 hp -> dead
        }
        let _ = es.update(0.016, o, |_| false, |_| false, |_| false, |_, _| None, |_| true);
        assert_eq!(es.species_summary(), "passive:0 hostile:0", "the cow died");
        // S13: counts are ranged (beef 1-3, leather 0-2) — only beef>=1 + the XP orb are certain.
        assert!(es.count() >= 2, "cow should drop loot + xp (count = {})", es.count());
        assert!(
            es.list.iter().any(|e| matches!(&e.kind, Kind::Item(s) if s.item == crate::item::BEEF)),
            "a beef drop must exist"
        );
    }

    #[test]
    fn s13_loot_tables_are_ranged_and_sane() {
        for (i, &s) in Species::ALL.iter().enumerate() {
            for &(item, min, max) in s.loot() {
                assert!(min <= max, "species #{i} loot {item}: min {min} > max {max}");
                assert!(max >= 1, "species #{i} loot {item}: a 0-max entry is dead weight");
                assert!(crate::item::is_known(item), "species #{i} loot {item} unregistered");
            }
        }
        // The S13 additions: slimes and endermen finally have tables; wolves stay empty.
        assert!(!Species::Slime.loot().is_empty());
        assert!(!Species::Enderman.loot().is_empty());
        assert!(Species::Wolf.loot().is_empty());
    }

    #[test]
    fn fast_arrow_does_not_tunnel_a_thin_wall() {
        let mut es = Entities::new();
        // 22 m/s arrow at z=0; a 1-cell wall at z=1. With a 0.1s frame the naive single-step would
        // jump from z=0 to z=2.2 (final cell z=2), never testing z=1 — the sub-step march must catch it.
        es.spawn_arrow(Vec3::new(0.5, 1.0, 0.0), Vec3::new(0.0, 0.0, 22.0));
        let wall = |p: IVec3| p.z == 1;
        es.update(0.1, Vec3::new(0.0, 500.0, 500.0), wall, |_| false, |_| false, |_, _| None, |_| true); // player far away
        assert_eq!(es.count(), 0, "the arrow stops at the wall instead of tunneling through it");
    }

    #[test]
    fn arrow_sticks_in_a_wall() {
        let mut es = Entities::new();
        es.spawn_arrow(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 12.0));
        let wall = |p: IVec3| p.z >= 2;
        for _ in 0..120 {
            es.update(1.0 / 60.0, Vec3::new(0.0, 200.0, 200.0), wall, |_| false, |_| false, |_, _| None, |_| true); // player far away
            if es.count() == 0 {
                break;
            }
        }
        assert_eq!(es.count(), 0, "the arrow stops at the wall and despawns");
    }

    // P15 local navigation. Passive (Cow) tests drive a deterministic heading via FLEE: a player just
    // inside FLEE_RADIUS on -x makes the cow flee +x toward the scripted obstacle.
    #[test]
    fn mob_steps_up_one_block_ledge() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.5, 1.0, 0.5), Species::Cow);
        // Floor at y<1, plus a raised plateau (solid at y==1) for x>=2 — a 1-block step up at x=2.
        let solid = |p: IVec3| p.y < 1 || (p.x >= 2 && p.y == 1);
        let player = Vec3::new(-1.0, 1.0, 0.5);
        for _ in 0..240 {
            es.update(1.0 / 60.0, player, solid, |_| false, |_| false, |_, _| None, |_| true);
        }
        assert!(es.mob_pos(0).x > 2.0, "cow crossed onto the ledge (x={})", es.mob_pos(0).x);
        assert!(es.mob_pos(0).y >= 1.9, "cow climbed the 1-block step (y={})", es.mob_pos(0).y);
    }

    #[test]
    fn wanderer_does_not_walk_off_a_tall_drop() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(1.0, 1.0, 0.5), Species::Cow);
        let solid = |p: IVec3| p.y < 1 && p.x < 3; // floor ends at x=3 → a deep void beyond
        let player = Vec3::new(-0.5, 1.0, 0.5);
        for _ in 0..300 {
            es.update(1.0 / 60.0, player, solid, |_| false, |_| false, |_, _| None, |_| true);
        }
        assert!(es.mob_pos(0).x < 3.0, "cow stayed back from the cliff (x={})", es.mob_pos(0).x);
        assert!(es.mob_pos(0).y > 0.5, "cow did not fall into the void (y={})", es.mob_pos(0).y);
    }

    #[test]
    fn mob_does_not_path_into_lava() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(1.0, 1.0, 0.5), Species::Cow);
        let solid = |p: IVec3| p.y < 1; // full floor
        let lava = |p: IVec3| p.x >= 3 && p.y == 1; // a lava body at body level for x>=3
        let player = Vec3::new(-0.5, 1.0, 0.5);
        for _ in 0..300 {
            es.update(1.0 / 60.0, player, solid, lava, |_| false, |_, _| None, |_| true);
        }
        assert!(es.mob_pos(0).x < 3.0, "cow avoided the lava (x={})", es.mob_pos(0).x);
    }

    #[test]
    fn deflection_picks_a_side_around_a_wall() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(1.0, 1.0, 0.5), Species::Cow);
        let solid = |p: IVec3| p.y < 1 || (p.x == 3 && p.y >= 1); // a 2-tall wall plane at x=3
        let player = Vec3::new(-0.5, 1.0, 0.5);
        for _ in 0..240 {
            es.update(1.0 / 60.0, player, solid, |_| false, |_| false, |_, _| None, |_| true);
        }
        assert!((es.mob_pos(0).z - 0.5).abs() > 0.5, "cow deflected sideways (z={})", es.mob_pos(0).z);
        assert!(es.mob_pos(0).x < 3.0, "cow did not tunnel the wall (x={})", es.mob_pos(0).x);
    }

    #[test]
    fn chase_steps_up_to_follow_the_player() {
        // A hostile zombie chases a player on a raised plateau; chase keeps step-up (only edge-avoid
        // is disabled), so it must climb the ledge to approach. (Hazard avoidance is shared with the
        // flee path proven above — it is not state-gated.)
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.5, 1.0, 0.5), Species::Zombie);
        let solid = |p: IVec3| p.y < 1 || (p.x >= 2 && p.y == 1);
        let player = Vec3::new(6.0, 2.0, 0.5);
        for _ in 0..400 {
            es.update(1.0 / 60.0, player, solid, |_| false, |_| false, |_, _| None, |_| true);
        }
        assert!(es.mob_pos(0).x > 3.0, "zombie advanced toward the player (x={})", es.mob_pos(0).x);
        assert!(es.mob_pos(0).y >= 1.9, "zombie climbed onto the plateau (y={})", es.mob_pos(0).y);
    }

    #[test]
    fn per_category_mob_counts() {
        // P16 spawn caps count hostiles and passives separately.
        let mut es = Entities::new();
        es.spawn_mob(Vec3::ZERO, Species::Cow);
        es.spawn_mob(Vec3::ZERO, Species::Pig);
        es.spawn_mob(Vec3::ZERO, Species::Zombie);
        assert_eq!(es.mob_count(), 3);
        assert_eq!(es.passive_count(), 2);
        assert_eq!(es.hostile_count(), 1);
    }

    #[test]
    fn feeding_flags_love_and_gates() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 1.0, 3.0), Species::Cow);
        let (o, d) = (Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(!es.try_feed(o, d, 6.0, crate::item::APPLE), "wrong food doesn't breed");
        assert!(!es.mob_in_love(0));
        assert!(es.try_feed(o, d, 6.0, crate::item::WHEAT), "wheat feeds a cow");
        assert!(es.mob_in_love(0));
        // A baby can't be fed into love.
        let mut es2 = Entities::new();
        es2.spawn_baby(Vec3::new(0.0, 1.0, 3.0), Species::Cow);
        assert!(!es2.try_feed(o, d, 6.0, crate::item::WHEAT));
        assert!(!es2.mob_in_love(0));
    }

    #[test]
    fn two_in_love_breed_one_baby_then_cooldown() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 1.0, 3.0), Species::Cow);
        es.spawn_mob(Vec3::new(1.2, 1.0, 3.0), Species::Cow); // within BREED_RANGE
        let d = Vec3::new(0.0, 0.0, 1.0);
        assert!(es.try_feed(Vec3::new(0.0, 1.5, 0.0), d, 6.0, crate::item::WHEAT));
        assert!(es.try_feed(Vec3::new(1.2, 1.5, 0.0), d, 6.0, crate::item::WHEAT));
        assert!(es.mob_in_love(0) && es.mob_in_love(1));
        let near = Vec3::new(10.0, 1.0, 3.0); // inside DESPAWN_RADIUS, outside FLEE_RADIUS
        let floor = |p: IVec3| p.y < 1;
        es.update(0.05, near, floor, |_| false, |_| false, |_, _| None, |_| true);
        assert_eq!(es.mob_count(), 3, "exactly one baby is born");
        assert!(!es.mob_in_love(0) && !es.mob_in_love(1), "both parents leave love-mode");
        assert!(es.mob_breed_cd(0) > 0.0 && es.mob_breed_cd(1) > 0.0, "both go on cooldown");
        es.update(0.05, near, floor, |_| false, |_| false, |_, _| None, |_| true);
        assert_eq!(es.mob_count(), 3, "no second baby while on cooldown");
    }

    #[test]
    fn baby_grows_into_adult() {
        // Small dt so collide_move keeps the mob on the floor (a big dt tunnels it out of the world).
        let dt = 0.05;
        let mut es = Entities::new();
        es.spawn_baby(Vec3::new(0.0, 1.0, 0.0), Species::Cow);
        assert!(es.mob_is_baby(0));
        // Track the baby with the player (dist≈0 → never despawns); floor at y<1 so it doesn't fall.
        for _ in 0..((GROW_TIME / dt) as i32 + 20) {
            let p = es.mob_pos(0);
            es.update(dt, p, |pp| pp.y < 1, |_| false, |_| false, |_, _| None, |_| true);
        }
        assert!(!es.mob_is_baby(0), "the baby matured after GROW_TIME");
    }

    #[test]
    fn exposed_undead_burns_in_daylight() {
        let dt = 0.05;
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 1.0, 0.0), Species::Zombie);
        let h0 = es.mob_health(0);
        for _ in 0..100 {
            let p = es.mob_pos(0); // player tracks the (milling) zombie: no despawn, floor → no fall
            es.update(dt, p, |pp| pp.y < 1, |_| false, |_| true, |_, _| None, |_| true); // sun_burn = exposed everywhere
        }
        assert!(es.mob_health(0) < h0, "the zombie took daylight burn damage");
    }

    #[test]
    fn sheltered_undead_does_not_burn() {
        let dt = 0.05;
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 1.0, 0.0), Species::Zombie);
        let h0 = es.mob_health(0);
        for _ in 0..100 {
            let p = es.mob_pos(0);
            es.update(dt, p, |pp| pp.y < 1, |_| false, |_| false, |_, _| None, |_| true); // never sky-exposed
        }
        assert!((es.mob_health(0) - h0).abs() < 1e-3, "sheltered/night undead don't burn");
    }

    #[test]
    fn hit_wolf_becomes_hostile_and_chases() {
        let mut es = Entities::new();
        es.spawn_mob(Vec3::ZERO, Species::Wolf);
        // Player beyond FLEE_RADIUS*1.8 (=9) → a CALM wolf wanders, never chases.
        let far = Vec3::new(0.0, 0.0, 12.0);
        let _ = es.update(0.016, far, |_| false, |_| false, |_| false, |_, _| None, |_| true);
        let s = es.ai_summary();
        assert!(!s.contains("chase:1") && !s.contains("attack:1"), "calm wolf doesn't chase ({s})");
        // Hit it (from the far side so it survives), then a close visible player → it chases.
        es.attack(Vec3::new(0.0, 0.4, -3.0), Vec3::new(0.0, 0.0, 1.0), 6.0, 1.0, false, None, 1.0);
        let _ = es.update(0.016, Vec3::new(0.0, 0.0, 4.0), |_| false, |_| false, |_| false, |_, _| None, |_| true);
        let s = es.ai_summary();
        assert!(s.contains("chase:1") || s.contains("attack:1"), "an angry wolf chases ({s})");
    }

    #[test]
    fn large_slime_splits_and_smallest_does_not() {
        let (o, d) = (Vec3::new(0.0, 0.4, -3.0), Vec3::new(0.0, 0.0, 1.0));
        let mut es = Entities::new();
        es.spawn_slime(Vec3::ZERO, 2); // large (8 hp)
        es.attack(o, d, 6.0, 10.0, false, None, 1.0); // lethal
        let _ = es.update(0.016, o, |_| false, |_| false, |_| false, |_, _| None, |_| true);
        assert_eq!(es.mob_count(), 3, "a large slime splits into 3 mediums");
        let mut es2 = Entities::new();
        es2.spawn_slime(Vec3::ZERO, 0); // smallest (1 hp)
        es2.attack(o, d, 6.0, 10.0, false, None, 1.0);
        let _ = es2.update(0.016, o, |_| false, |_| false, |_| false, |_, _| None, |_| true);
        assert_eq!(es2.mob_count(), 0, "the smallest slime leaves no children");
    }

    #[test]
    fn villager_is_passive() {
        assert!(!Species::Villager.hostile());
        let mut es = Entities::new();
        es.spawn_mob(Vec3::ZERO, Species::Villager);
        assert_eq!(es.species_summary(), "passive:1 hostile:0");
    }

    #[test]
    fn new_species_in_all_with_valid_tables() {
        assert_eq!(Species::ALL.len(), 13);
        for s in Species::ALL {
            assert!(s.max_health() > 0.0, "{s:?} has health");
            let (w, h) = s.size();
            assert!(w > 0.0 && h > 0.0, "{s:?} has a size");
            assert!(!model(s).is_empty(), "{s:?} has a model");
        }
        assert!(Species::Slime.hostile() && Species::Enderman.hostile());
        assert!(!Species::Wolf.hostile() && !Species::Villager.hostile());
    }

    #[test]
    fn warden_tables_are_sane() {
        // U11: the Warden is in ALL with the expected boss stats, a non-empty model, hostile, no XP/loot.
        assert!(Species::ALL.contains(&Species::Warden));
        assert_eq!(Species::Warden.max_health(), 500.0);
        assert!(Species::Warden.hostile());
        assert_eq!(Species::Warden.xp_drop(), 0);
        assert!(Species::Warden.loot().is_empty());
        assert_eq!(Species::Warden.contact_damage(), 16.0);
        assert!(!model(Species::Warden).is_empty());
        // The collision box stays short enough to clear the 2-air spawn check.
        let (_w, h) = Species::Warden.size();
        assert!(h <= 2.05, "warden AABB height {h} fits the spawn-air check");
        // Sane constants.
        assert!(WARDEN_SPEED > 0.0 && WARDEN_EMERGE_TIME > 0.0 && WARDEN_HEAR_RANGE > 0.0);
        assert!(WARDEN_SONIC_RANGE > 2.5 && WARDEN_SONIC_DAMAGE > 0.0);
    }

    #[test]
    fn warden_emerges_invulnerable_then_chases_a_vibration() {
        // A fresh Warden spends WARDEN_EMERGE_TIME immobile + invulnerable, then walks toward the most-
        // recent vibration it hears (no line-of-sight needed). The vibration closure feeds it a target.
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 1.0, 0.0), Species::Warden);
        let floor = |p: IVec3| p.y < 1;
        // During emerge: a melee swing does NOT reduce its health.
        let (o, d) = (Vec3::new(0.0, 1.4, -3.0), Vec3::new(0.0, 0.0, 1.0));
        es.attack(o, d, 6.0, 50.0, false, None, 1.0);
        assert_eq!(es.mob_health(0), 500.0, "an emerging Warden is invulnerable");
        // Run past the emerge timer with NO vibration nearby; it must still be alive (player is close,
        // so the dig-away guard doesn't fire).
        let player = Vec3::new(2.0, 1.0, 0.0);
        for _ in 0..((WARDEN_EMERGE_TIME / (1.0 / 60.0)) as i32 + 5) {
            es.update(1.0 / 60.0, player, floor, |_| false, |_| false, |_, _| None, |_| true);
        }
        assert_eq!(es.mob_count(), 1, "the Warden is still present after emerging");
        // Now feed a vibration to the NORTH (+x) and step — it should move toward it.
        let start_x = es.mob_pos(0).x;
        let vib = Vec3::new(15.0, 1.0, 0.0);
        for _ in 0..60 {
            es.update(1.0 / 60.0, player, floor, |_| false, |_| false, move |_, _| Some(vib), |_| true);
        }
        assert!(es.mob_pos(0).x > start_x, "the Warden walks toward the vibration it heard");
    }

    #[test]
    fn warden_digs_away_when_it_loses_the_trail() {
        // No vibration AND the player stays out of hear range: past WARDEN_DIG_TIMEOUT the Warden
        // despawns with no loot (it burrows back underground). A tall wall pens the Warden in so it
        // can't beeline the far player and re-enter hear range before the timeout.
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 1.0, 0.0), Species::Warden);
        // Floor below y=1, plus a 2-tall wall at x>=2 the penned Warden (x≈0) can't cross.
        let world = |p: IVec3| p.y < 1 || (p.x >= 2 && p.y >= 1 && p.y <= 2);
        let far = Vec3::new(40.0, 1.0, 0.0); // > WARDEN_HEAR_RANGE, but < DESPAWN_RADIUS
        let steps = ((WARDEN_EMERGE_TIME + WARDEN_DIG_TIMEOUT + 2.0) / (1.0 / 60.0)) as i32;
        for _ in 0..steps {
            es.update(1.0 / 60.0, far, world, |_| false, |_| false, |_, _| None, |_| true);
            if es.mob_count() == 0 {
                break;
            }
        }
        assert_eq!(es.mob_count(), 0, "a Warden that loses the trail digs away");
        // It left NO loot/XP behind (digging away is not a kill).
        assert_eq!(es.count(), 0, "no loot or XP from a dig-away Warden");
    }

    #[test]
    fn warden_rehunts_player_when_the_trail_goes_cold() {
        // C1 regression: a Warden that has heard a vibration must not cling to that spot forever. Once
        // the trail goes cold (no fresh vibration for WARDEN_TARGET_STALE) it falls back to advancing on
        // a still-in-range player rather than parking on the stale position.
        let mut es = Entities::new();
        es.spawn_mob(Vec3::new(0.0, 1.0, 0.0), Species::Warden);
        let floor = |p: IVec3| p.y < 1; // flat ground everywhere, no walls
        let player = Vec3::new(-12.0, 1.0, 0.0); // within hear range, on the -x side
        // Emerge (no vibration; player close so no dig-away).
        for _ in 0..((WARDEN_EMERGE_TIME / (1.0 / 60.0)) as i32 + 2) {
            es.update(1.0 / 60.0, player, floor, |_| false, |_| false, |_, _| None, |_| true);
        }
        // Phase 1: a +x vibration lures it east, away from the player.
        let vib = Vec3::new(15.0, 1.0, 0.0);
        for _ in 0..60 {
            es.update(1.0 / 60.0, player, floor, |_| false, |_| false, move |_, _| Some(vib), |_| true);
        }
        let lured_x = es.mob_pos(0).x;
        assert!(lured_x > 0.5, "the Warden should have chased the +x vibration (x={lured_x})");
        // Phase 2: silence. After the stale window it reverses and closes on the -x player.
        for _ in 0..((WARDEN_TARGET_STALE / (1.0 / 60.0)) as i32 + 420) {
            es.update(1.0 / 60.0, player, floor, |_| false, |_| false, |_, _| None, |_| true);
        }
        assert_eq!(es.mob_count(), 1, "the in-range player keeps the Warden from digging away");
        let end = es.mob_pos(0);
        assert!(
            (end - player).length() < 5.0,
            "a cold-trail Warden should re-hunt the in-range player, not park on the stale +x spot \
             (lured to x={lured_x}, ended at {end:?})"
        );
    }
}
