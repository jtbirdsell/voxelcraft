//! Player controller: an AABB with swept, axis-separated collision against the voxel world,
//! gravity + jump in walk mode, and free flight in fly mode.

use glam::{IVec3, Vec3};

use crate::block::{self, BlockId};

pub const PLAYER_WIDTH: f32 = 0.6;
pub const PLAYER_HEIGHT: f32 = 1.8;
pub const EYE_HEIGHT: f32 = 1.62;

const HALF_W: f32 = PLAYER_WIDTH * 0.5;
const GRAVITY: f32 = 30.0;
const TERMINAL: f32 = 60.0;
const WALK_SPEED: f32 = 5.0;
const SPRINT_SPEED: f32 = 8.5;
const SNEAK_SPEED: f32 = 1.6;
const JUMP_SPEED: f32 = 9.2;
const FLY_SPEED: f32 = 26.0;
const FLY_SPRINT_SPEED: f32 = 80.0;

// Swimming: buoyant, slower movement; jump key paddles upward.
const SWIM_SPEED: f32 = 4.0;
const WATER_GRAVITY: f32 = 8.0;
const WATER_SINK: f32 = 2.5; // max sink speed
const SWIM_UP: f32 = 4.0;

// Survival tuning (walk mode only; flying is treated as creative and takes no damage/hunger).
const MAX_HEALTH: f32 = 20.0;
const MAX_HUNGER: f32 = 20.0;
const SAFE_FALL: f32 = 3.0; // blocks of fall absorbed before damage; then 1 hp per extra block
// S8 full-vanilla pacing: hunger drains through EXHAUSTION (actions cost; time doesn't), regen
// runs on vanilla cadences paid in exhaustion, and sprinting needs food in the tank.
const EXHAUSTION_LIMIT: f32 = 4.0; // at 4.0: -1 saturation (or -1 hunger once saturation is dry)
const EXH_SPRINT_PER_M: f32 = 0.1;
const EXH_SWIM_PER_M: f32 = 0.01;
const EXH_JUMP: f32 = 0.05;
const EXH_SPRINT_JUMP: f32 = 0.2;
pub const EXH_ATTACK: f32 = 0.1; // per landed swing (app.rs melee site)
pub const EXH_MINE: f32 = 0.005; // per block broken (app.rs mining site)
const EXH_REGEN_PER_HP: f32 = 6.0;
const REGEN_HUNGER: f32 = 18.0; // vanilla threshold (was 17)
const REGEN_INTERVAL: f32 = 4.0; // 1 HP / 4 s at hunger >= 18
const REGEN_SATURATED_INTERVAL: f32 = 0.5; // 1 HP / 0.5 s at full hunger with saturation left
const STARVE_RATE: f32 = 0.25; // vanilla 1 dmg / 4 s, applied continuously
const SPRINT_MIN_HUNGER: f32 = 6.0; // vanilla: can't sprint at 3 haunches or less
const MAX_SATURATION: f32 = 5.0; // hidden food reserve: drains before hunger, fuels faster regen
const PEACEFUL_REGEN: f32 = 1.0; // hp/sec passive health regen on Peaceful (no hunger cost)

// Air + environmental damage.
const MAX_AIR: f32 = 11.0; // ~seconds of breath underwater
const AIR_REFILL: f32 = 4.0; // refills several times faster than it drains
const DROWN_DPS: f32 = 2.0; // hp lost per second once out of air
const LAVA_DPS: f32 = 4.0; // hp/sec while standing in lava

/// XP points needed to advance from `level` to the next.
fn xp_to_next(level: u32) -> f32 {
    10.0 + level as f32 * 4.0
}

/// Per-frame movement intent. Look deltas are accumulated from raw mouse motion.
#[derive(Default)]
pub struct Input {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub sprint: bool,
    pub sneak: bool,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub break_held: bool,
    pub place_pressed: bool,
    /// Right mouse button held (place / eat-hold / use); `place_pressed` is the press edge.
    pub place_held: bool,
    pub drop_pressed: bool,
}

impl Input {
    pub fn take_look(&mut self) -> (f32, f32) {
        let v = (self.yaw_delta, self.pitch_delta);
        self.yaw_delta = 0.0;
        self.pitch_delta = 0.0;
        v
    }
}

pub struct Player {
    /// Feet position: bottom-center of the AABB.
    pub position: Vec3,
    pub velocity: Vec3,
    pub on_ground: bool,
    pub flying: bool,
    /// Creative-mode invulnerability (S1): mirrors `Inventory.creative`, set once at session build
    /// (nothing toggles mode mid-game yet). Creative takes no damage and runs no hunger/air drain
    /// even on foot; survival can never fly, so the old flying-implies-creative shortcut is gone.
    pub creative: bool,
    pub health: f32,
    pub hunger: f32,
    /// Breath remaining underwater (0..MAX_AIR); drives the bubble bar + drowning.
    pub air: f32,
    /// True while the eye is inside water (HUD shows bubbles; gameplay input unaffected).
    pub submerged: bool,
    /// Hidden food saturation; drains before hunger and fuels faster regen.
    saturation: f32,
    /// S8: accumulated action cost (sprint/jump/attack/mine/regen). At 4.0 it spends a point of
    /// saturation-then-hunger. Transient (a 0..4 sawtooth; vanilla persists it — accepted loss).
    exhaustion: f32,
    /// S8: vanilla regen cadence timer (1 HP per 4 s, or per 0.5 s when saturated at full hunger).
    regen_timer: f32,
    /// Experience: total points toward the next level, plus the current level.
    pub xp: f32,
    pub level: u32,
    /// Total equipped-armor defense points, refreshed from the inventory each frame.
    armor_points: u32,
    /// World difficulty, mirrored from the app each frame; scales combat damage + starvation/regen.
    pub difficulty: crate::rules::Difficulty,
    /// Post-hit invulnerability timer (seconds); incoming combat damage is ignored while it's >0.
    hurt_cooldown: f32,
    drown_timer: f32,
    /// Highest y reached since last touching ground, for fall-damage distance.
    air_max_y: f32,
    /// Where `respawn` returns the player on death.
    respawn_point: Vec3,
    /// U11 Darkness effect timer (seconds): set by a sculk shrieker's shriek, decays each tick toward
    /// 0. Drives a screen-dimming pulse (render effect deferred — see notes); the value is exposed so
    /// the renderer can read it when wired up.
    pub darkness: f32,
}

#[inline]
fn comp(v: Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Whether a melee hit lands as a CRITICAL (+50% damage, MC 1.9). The attacker must be falling
/// (airborne + moving down), not sprinting (sprint-attacks can't crit), not in water, and not in
/// creative flight. Vanilla keys on `fallDistance > 0`; with only velocity here we use a downward-
/// velocity proxy, so a hit on the way *up* during a jump doesn't crit (which matches vanilla intent).
#[inline]
pub fn is_critical_hit(on_ground: bool, vel_y: f32, sprint: bool, submerged: bool, flying: bool) -> bool {
    !on_ground && vel_y < 0.0 && !sprint && !submerged && !flying
}

/// Cosine of the half-angle of a raised shield's frontal block arc. `0.0` = the full front hemisphere
/// (±90°), matching Minecraft's "are you facing the source" half-plane test.
pub const SHIELD_BLOCK_COS: f32 = 0.0;

/// Does a RAISED shield (player at `player_pos`, looking along `forward`) intercept a hit from
/// `source`? Horizontal-only — vertical attack angle is ignored (a creeper at your feet still hits the
/// front). Degenerate cases (source directly above/below, or looking straight up/down) don't block.
pub fn shield_blocks(forward: Vec3, player_pos: Vec3, source: Vec3) -> bool {
    let to_src = source - player_pos;
    let to_src_h = Vec3::new(to_src.x, 0.0, to_src.z);
    let fwd_h = Vec3::new(forward.x, 0.0, forward.z);
    if to_src_h.length_squared() < 1e-6 || fwd_h.length_squared() < 1e-6 {
        return false;
    }
    fwd_h.normalize().dot(to_src_h.normalize()) >= SHIELD_BLOCK_COS
}

#[inline]
fn set_comp(v: &mut Vec3, axis: usize, value: f32) {
    match axis {
        0 => v.x = value,
        1 => v.y = value,
        _ => v.z = value,
    }
}

impl Player {
    pub fn new(position: Vec3, flying: bool) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            on_ground: false,
            flying,
            creative: false,
            health: MAX_HEALTH,
            hunger: MAX_HUNGER,
            air: MAX_AIR,
            submerged: false,
            saturation: MAX_SATURATION,
            exhaustion: 0.0,
            regen_timer: 0.0,
            xp: 0.0,
            level: 0,
            armor_points: 0,
            difficulty: crate::rules::Difficulty::Normal,
            hurt_cooldown: 0.0,
            drown_timer: 0.0,
            air_max_y: position.y,
            respawn_point: position,
            darkness: 0.0,
        }
    }

    /// Apply incoming damage, reduced by `armor` defense points (Minecraft-style: each point blocks
    /// 4%, capped at 80%). Environmental sources that ignore armor pass `armor = 0`.
    fn apply_damage(&mut self, raw: f32, armor: u32) {
        let reduction = (armor as f32 * 0.04).min(0.8);
        self.health = (self.health - raw * (1.0 - reduction)).max(0.0);
    }

    /// External combat hit (mob contact / arrow), reduced by armor. A short post-hit invulnerability
    /// window drops further hits so a swarm can't stack a frame of damage into an instant kill.
    pub fn take_hit(&mut self, raw: f32) {
        // Creative mode is invulnerable to combat damage (S1), like Minecraft.
        if self.creative {
            return;
        }
        // All hostile-sourced damage (mob melee, arrows, creeper blasts) scales with difficulty
        // (Peaceful = 0). Environmental damage bypasses take_hit — it calls apply_damage directly —
        // so fall/lava/drowning/starvation are never difficulty-scaled.
        let scaled = raw * self.difficulty.damage_mult();
        if self.hurt_cooldown > 0.0 || scaled <= 0.0 {
            return;
        }
        self.apply_damage(scaled, self.armor_points);
        self.hurt_cooldown = 0.5;
    }

    /// S8: record action cost. Creative/flying never exhausts (survival mechanics don't run there).
    pub fn add_exhaustion(&mut self, amount: f32) {
        if self.creative || self.flying {
            return;
        }
        self.exhaustion += amount;
    }

    /// S8: vanilla sprint gate — sprinting needs more than 3 haunches of hunger.
    pub fn can_sprint(&self) -> bool {
        self.creative || self.flying || self.hunger > SPRINT_MIN_HUNGER
    }

    /// Award experience points, rolling over into levels.
    pub fn add_xp(&mut self, amount: u32) {
        self.xp += amount as f32;
        while self.xp >= xp_to_next(self.level) {
            self.xp -= xp_to_next(self.level);
            self.level += 1;
        }
    }

    /// Progress through the current level, 0..1 (for the HUD XP bar).
    pub fn xp_fraction(&self) -> f32 {
        (self.xp / xp_to_next(self.level)).clamp(0.0, 1.0)
    }

    /// Breath remaining as a 0..1 fraction (for the HUD bubble bar).
    pub fn air_fraction(&self) -> f32 {
        (self.air / MAX_AIR).clamp(0.0, 1.0)
    }

    /// Hidden food saturation (for persistence).
    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    /// Restore persisted survival state on load (M24); values are clamped to valid ranges.
    pub fn restore_state(&mut self, health: f32, hunger: f32, air: f32, saturation: f32, xp: f32, level: u32) {
        self.health = health.clamp(0.0, MAX_HEALTH);
        self.hunger = hunger.clamp(0.0, MAX_HUNGER);
        self.air = air.clamp(0.0, MAX_AIR);
        self.saturation = saturation.clamp(0.0, MAX_SATURATION);
        self.xp = xp.max(0.0);
        self.level = level;
    }

    pub fn is_dead(&self) -> bool {
        self.health <= 0.0
    }

    /// U11: apply a Darkness pulse (seconds) from a sculk shriek — takes the longer of the two so a
    /// repeated shriek refreshes rather than truncates the effect.
    pub fn apply_darkness(&mut self, seconds: f32) {
        self.darkness = self.darkness.max(seconds);
    }

    /// Can the player eat this food now? (Always-edible foods can be eaten at full hunger.)
    pub fn can_eat(&self, always_edible: bool) -> bool {
        always_edible || self.hunger < MAX_HUNGER
    }

    /// Eat a food: restore hunger, then saturation (capped at the new hunger level + the reserve cap,
    /// per Minecraft — saturation never exceeds the current food level).
    pub fn eat(&mut self, hunger: u32, saturation: f32) {
        self.hunger = (self.hunger + hunger as f32).min(MAX_HUNGER);
        self.saturation = (self.saturation + saturation).min(self.hunger).min(MAX_SATURATION);
    }

    /// Reset to the spawn point with full health and hunger (called on death).
    pub fn respawn(&mut self) {
        self.position = self.respawn_point;
        self.velocity = Vec3::ZERO;
        self.health = MAX_HEALTH;
        self.hunger = MAX_HUNGER;
        self.air = MAX_AIR;
        self.submerged = false;
        self.saturation = MAX_SATURATION;
        self.drown_timer = 0.0;
        self.air_max_y = self.respawn_point.y;
        self.on_ground = false;
        self.darkness = 0.0;
    }

    /// S8 vanilla survival tick: spend banked exhaustion into saturation/hunger, regen on the
    /// vanilla cadence (paid in exhaustion), starve at empty. `env_damage` suppresses regen so the
    /// player can't heal mid-lava / mid-drown.
    fn update_survival(&mut self, dt: f32, env_damage: bool) {
        // Peaceful: hunger never depletes and health regenerates passively at no hunger cost (like
        // Minecraft). Environmental damage still applies (handled in update_environment_damage).
        // Exhaustion is discarded so leaving Peaceful can't bank a stale -1 saturation.
        if self.difficulty == crate::rules::Difficulty::Peaceful {
            self.hunger = MAX_HUNGER;
            self.saturation = MAX_SATURATION;
            self.exhaustion = 0.0;
            if !env_damage && self.health < MAX_HEALTH {
                self.health = (self.health + PEACEFUL_REGEN * dt).min(MAX_HEALTH);
            }
            return;
        }

        // Spend banked exhaustion: each 4.0 costs a point of saturation, then hunger.
        while self.exhaustion >= EXHAUSTION_LIMIT {
            self.exhaustion -= EXHAUSTION_LIMIT;
            if self.saturation > 0.0 {
                self.saturation = (self.saturation - 1.0).max(0.0);
            } else {
                self.hunger = (self.hunger - 1.0).max(0.0);
            }
        }

        if !env_damage && self.health < MAX_HEALTH {
            // Vanilla cadence: 1 HP / 0.5 s while saturated at full hunger, 1 HP / 4 s at >= 18.
            let interval = if self.hunger >= MAX_HUNGER && self.saturation > 0.0 {
                Some(REGEN_SATURATED_INTERVAL)
            } else if self.hunger >= REGEN_HUNGER {
                Some(REGEN_INTERVAL)
            } else {
                None
            };
            if let Some(iv) = interval {
                self.regen_timer += dt;
                if self.regen_timer >= iv {
                    self.regen_timer -= iv;
                    self.health = (self.health + 1.0).min(MAX_HEALTH);
                    self.add_exhaustion(EXH_REGEN_PER_HP);
                }
            } else {
                self.regen_timer = 0.0;
            }
        }
        if self.hunger <= 0.0 {
            // Starvation can't push health below the difficulty's floor (Easy 10, Normal 1, Hard 0).
            let floor = self.difficulty.starve_floor();
            if self.health > floor {
                let dmg = (STARVE_RATE * dt).min(self.health - floor);
                self.apply_damage(dmg, 0); // starvation ignores armor
            }
        }
    }

    /// Air (drowning) + lava contact damage, evaluated against the sampled environment each frame.
    /// Returns true while an environmental damage source is active (used to suppress regen).
    fn update_environment_damage(&mut self, dt: f32, submerged: bool, in_lava: bool) -> bool {
        self.submerged = submerged;
        if submerged {
            self.air -= dt;
            if self.air <= 0.0 {
                self.air = 0.0;
                self.drown_timer += dt;
                if self.drown_timer >= 1.0 {
                    self.apply_damage(DROWN_DPS, 0); // drowning ignores armor
                    self.drown_timer -= 1.0; // keep the cadence exact; don't discard the overshoot
                }
            }
        } else {
            self.air = (self.air + AIR_REFILL * dt).min(MAX_AIR);
            self.drown_timer = 0.0;
        }
        if in_lava {
            self.apply_damage(LAVA_DPS * dt, self.armor_points);
        }
        in_lava || (submerged && self.air <= 0.0)
    }

    /// True if any voxel overlapping the player's AABB satisfies `pred` (water/lava tests).
    fn aabb_any(&self, pred: impl Fn(IVec3) -> bool) -> bool {
        let (min, max) = self.aabb();
        let x0 = min.x.floor() as i32;
        let x1 = (max.x - 1e-4).floor() as i32;
        let y0 = min.y.floor() as i32;
        let y1 = (max.y - 1e-4).floor() as i32;
        let z0 = min.z.floor() as i32;
        let z1 = (max.z - 1e-4).floor() as i32;
        for vx in x0..=x1 {
            for vy in y0..=y1 {
                for vz in z0..=z1 {
                    if pred(IVec3::new(vx, vy, vz)) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// The voxel cell containing the eye (for the head-submerged drowning test).
    fn head_cell(&self) -> IVec3 {
        let e = self.eye();
        IVec3::new(e.x.floor() as i32, e.y.floor() as i32, e.z.floor() as i32)
    }

    /// True if a solid surface (full or partial) sits directly beneath the AABB footprint (sneak
    /// ledge test) — a sub-box whose top reaches the player's feet counts as support.
    fn ground_below(&self, block_at: &impl Fn(IVec3) -> (BlockId, u8)) -> bool {
        let (min, max) = self.aabb();
        let foot = min.y - 0.06;
        let y = foot.floor() as i32;
        let x0 = min.x.floor() as i32;
        let x1 = (max.x - 1e-4).floor() as i32;
        let z0 = min.z.floor() as i32;
        let z1 = (max.z - 1e-4).floor() as i32;
        for vx in x0..=x1 {
            for vz in z0..=z1 {
                let (id, st) = block_at(IVec3::new(vx, y, vz));
                for b in block::solid_boxes(id, st) {
                    if y as f32 + b[4] >= foot {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn eye(&self) -> Vec3 {
        self.position + Vec3::new(0.0, EYE_HEIGHT, 0.0)
    }

    fn aabb(&self) -> (Vec3, Vec3) {
        (
            self.position - Vec3::new(HALF_W, 0.0, HALF_W),
            self.position + Vec3::new(HALF_W, PLAYER_HEIGHT, HALF_W),
        )
    }

    /// True if the unit cube at `cell` overlaps the player's AABB.
    pub fn intersects_block(&self, cell: IVec3) -> bool {
        let (min, max) = self.aabb();
        let c = cell.as_vec3();
        min.x < c.x + 1.0
            && max.x > c.x
            && min.y < c.y + 1.0
            && max.y > c.y
            && min.z < c.z + 1.0
            && max.z > c.z
    }

    /// (low offset, high offset) of the AABB relative to `position` along an axis.
    fn offsets(axis: usize) -> (f32, f32) {
        match axis {
            1 => (0.0, PLAYER_HEIGHT),
            _ => (-HALF_W, HALF_W),
        }
    }

    pub fn update(
        &mut self,
        dt: f32,
        yaw: f32,
        input: &Input,
        block_at: impl Fn(IVec3) -> (BlockId, u8),
        armor_points: u32,
    ) {
        self.armor_points = armor_points;
        self.hurt_cooldown = (self.hurt_cooldown - dt).max(0.0);
        self.darkness = (self.darkness - dt).max(0.0); // U11 Darkness decays toward 0 each tick
        let (sy, cy) = yaw.sin_cos();
        let fwd = Vec3::new(cy, 0.0, sy);
        let right = Vec3::new(-sy, 0.0, cy);
        let mut wish = Vec3::ZERO;
        if input.forward {
            wish += fwd;
        }
        if input.back {
            wish -= fwd;
        }
        if input.right {
            wish += right;
        }
        if input.left {
            wish -= right;
        }
        let wish = wish.normalize_or_zero();

        // Sample the surrounding fluid once per frame (skipped in fly mode — treated as creative).
        let in_water = !self.flying && self.aabb_any(|c| block_at(c).0 == block::WATER);
        let in_lava = !self.flying && self.aabb_any(|c| block_at(c).0 == block::LAVA);
        // The head being in *any* fluid stops breathing — you can't breathe in lava either.
        let head_block = if self.flying { block::AIR } else { block_at(self.head_cell()).0 };
        let head_submerged = head_block == block::WATER || head_block == block::LAVA;

        if self.flying {
            let speed = if input.sprint { FLY_SPRINT_SPEED } else { FLY_SPEED };
            let mut v = wish * speed;
            if input.up {
                v.y += speed;
            }
            if input.down {
                v.y -= speed;
            }
            self.velocity = v;
            self.on_ground = false;
        } else if in_water {
            // Buoyant swimming: gentle sink, paddle up with jump, slower horizontal control.
            self.velocity.x = wish.x * SWIM_SPEED;
            self.velocity.z = wish.z * SWIM_SPEED;
            self.velocity.y -= WATER_GRAVITY * dt;
            if input.up {
                self.velocity.y = SWIM_UP;
            }
            self.velocity.y = self.velocity.y.clamp(-WATER_SINK, SWIM_UP);
            if wish != Vec3::ZERO {
                self.add_exhaustion(SWIM_SPEED * dt * EXH_SWIM_PER_M); // S8: 0.01/m swum
            }
        } else {
            // S8: sprinting needs hunger in the tank (the key alone doesn't make you sprint).
            let sprinting = input.sprint && !input.sneak && self.can_sprint();
            let speed = if input.sneak {
                SNEAK_SPEED
            } else if sprinting {
                SPRINT_SPEED
            } else {
                WALK_SPEED
            };
            self.velocity.x = wish.x * speed;
            self.velocity.z = wish.z * speed;
            if sprinting && wish != Vec3::ZERO {
                self.add_exhaustion(SPRINT_SPEED * dt * EXH_SPRINT_PER_M); // S8: 0.1/m sprinted
            }
            self.velocity.y -= GRAVITY * dt;
            self.velocity.y = self.velocity.y.max(-TERMINAL);
            if input.up && self.on_ground {
                self.velocity.y = JUMP_SPEED;
                self.on_ground = false;
                self.add_exhaustion(if sprinting { EXH_SPRINT_JUMP } else { EXH_JUMP }); // S8
            }
        }

        let delta = self.velocity * dt;
        // Sneaking on solid ground refuses any horizontal step that would walk off a ledge.
        let sneak_edge = input.sneak && self.on_ground && !self.flying && !in_water;
        if sneak_edge {
            let bx = self.position.x;
            self.move_axis(0, delta.x, &block_at);
            if !self.ground_below(&block_at) {
                self.position.x = bx;
                self.velocity.x = 0.0;
            }
            let bz = self.position.z;
            self.move_axis(2, delta.z, &block_at);
            if !self.ground_below(&block_at) {
                self.position.z = bz;
                self.velocity.z = 0.0;
            }
        } else {
            self.move_axis(0, delta.x, &block_at);
            self.move_axis(2, delta.z, &block_at);
        }
        let hit_y = self.move_axis(1, delta.y, &block_at);

        if self.flying {
            self.on_ground = false;
            self.air_max_y = self.position.y;
            self.submerged = false;
        } else {
            // Water cushions the fall — no landing damage accrues while submerged.
            let landed = hit_y && delta.y < 0.0;
            if landed && !self.on_ground && !in_water && !self.creative {
                let fall = (self.air_max_y - self.position.y).max(0.0);
                let dmg = (fall - SAFE_FALL).max(0.0);
                self.apply_damage(dmg, 0); // base armor doesn't soften fall damage (matches Minecraft)
            }
            self.on_ground = landed;
            if self.on_ground || in_water {
                self.air_max_y = self.position.y;
            } else {
                self.air_max_y = self.air_max_y.max(self.position.y);
            }
            if self.creative {
                // Creative on foot (S1): submerged still drives the HUD/FOV cue, but air never
                // drains and no environmental or hunger mechanics run — creative is immortal.
                self.submerged = head_submerged;
            } else {
                let env_damage = self.update_environment_damage(dt, head_submerged, in_lava);
                self.update_survival(dt, env_damage);
            }
        }
    }

    /// Move along one axis with substepping (S2): a 0.1 s frame at terminal velocity spans ~6
    /// blocks, so the motion is cut into steps small enough that any crossed face still overlaps
    /// the stepped AABB — high-speed frames can't tunnel. Returns true if a collision clamped the
    /// motion (the remaining distance is discarded — the axis is blocked).
    fn move_axis(&mut self, axis: usize, amount: f32, block_at: &impl Fn(IVec3) -> (BlockId, u8)) -> bool {
        // Safe bound: a crossed obstacle stays inside the stepped AABB while the step is below the
        // smallest body extent (width 0.6); 0.45 leaves margin, and covers the thin sub-boxes too
        // (a crossed door/pane panel ~0.1 thick still sits inside the 0.6-deep trailing AABB).
        const MAX_STEP: f32 = 0.45;
        let mut remaining = amount;
        while remaining != 0.0 {
            let step = remaining.clamp(-MAX_STEP, MAX_STEP);
            remaining -= step;
            if self.move_axis_step(axis, step, block_at) {
                return true;
            }
        }
        false
    }

    /// One bounded step: advance, then clamp the AABB back to the near face of any block sub-box
    /// it actually penetrates (full cubes, slab/stair halves, thin door/pane panels). The box must
    /// truly overlap on ALL three axes — a sub-box that shares the scanned cell but sits outside
    /// the AABB on the moving axis (a fence post behind the player) must not clamp, or walking
    /// away from a fence teleports the player through it to its far face.
    fn move_axis_step(&mut self, axis: usize, amount: f32, block_at: &impl Fn(IVec3) -> (BlockId, u8)) -> bool {
        if amount == 0.0 {
            return false;
        }
        let moved = comp(self.position, axis) + amount;
        set_comp(&mut self.position, axis, moved);

        let (pmin, pmax) = self.aabb();
        let cmin = [pmin.x, pmin.y, pmin.z];
        let cmax = [pmax.x, pmax.y, pmax.z];
        let lo = [cmin[0].floor() as i32, cmin[1].floor() as i32, cmin[2].floor() as i32];
        let hi = [
            (cmax[0] - 1e-4).floor() as i32,
            (cmax[1] - 1e-4).floor() as i32,
            (cmax[2] - 1e-4).floor() as i32,
        ];

        let (off_lo, off_hi) = Self::offsets(axis);
        let mut clamp: Option<f32> = None;
        for vx in lo[0]..=hi[0] {
            for vy in lo[1]..=hi[1] {
                for vz in lo[2]..=hi[2] {
                    let base = [vx as f32, vy as f32, vz as f32];
                    let (id, st) = block_at(IVec3::new(vx, vy, vz));
                    for b in block::solid_boxes(id, st) {
                        // World-space sub-box.
                        let bmin = [base[0] + b[0], base[1] + b[1], base[2] + b[2]];
                        let bmax = [base[0] + b[3], base[1] + b[4], base[2] + b[5]];
                        // A genuine penetration overlaps on every axis (including the moving one).
                        let overlap = (0..3).all(|a| {
                            cmin[a] < bmax[a] - 1e-4 && cmax[a] > bmin[a] + 1e-4
                        });
                        if !overlap {
                            continue;
                        }
                        if amount > 0.0 {
                            let p = bmin[axis] - off_hi; // stop the leading face at the box near face
                            clamp = Some(clamp.map_or(p, |c| c.min(p)));
                        } else {
                            let p = bmax[axis] - off_lo;
                            clamp = Some(clamp.map_or(p, |c| c.max(p)));
                        }
                    }
                }
            }
        }

        if let Some(p) = clamp {
            set_comp(&mut self.position, axis, p);
            set_comp(&mut self.velocity, axis, 0.0);
            true
        } else {
            false
        }
    }

    /// True if any solid collision box strictly overlaps the player AABB (shrunk by 1e-3): the
    /// saved position is genuinely inside blocks (a corrupt/stale save). Standing ON a surface is
    /// contact, not overlap, so a survival save on a slab or inside a dug-out cave room is fine (S2).
    pub fn is_inside_solid(&self, block_at: &impl Fn(IVec3) -> (BlockId, u8)) -> bool {
        let (pmin, pmax) = self.aabb();
        let smin = [pmin.x + 1e-3, pmin.y + 1e-3, pmin.z + 1e-3];
        let smax = [pmax.x - 1e-3, pmax.y - 1e-3, pmax.z - 1e-3];
        for vx in (smin[0].floor() as i32)..=(smax[0].floor() as i32) {
            for vy in (smin[1].floor() as i32)..=(smax[1].floor() as i32) {
                for vz in (smin[2].floor() as i32)..=(smax[2].floor() as i32) {
                    let (id, st) = block_at(IVec3::new(vx, vy, vz));
                    for b in block::solid_boxes(id, st) {
                        let bmin = [vx as f32 + b[0], vy as f32 + b[1], vz as f32 + b[2]];
                        let bmax = [vx as f32 + b[3], vy as f32 + b[4], vz as f32 + b[5]];
                        if (0..3).all(|a| smin[a] < bmax[a] && smax[a] > bmin[a]) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_AIR: fn(IVec3) -> (BlockId, u8) = |_| (block::AIR, 0);

    #[test]
    fn player_falls_and_lands_on_floor() {
        // Solid floor occupying y in 0..10; air above.
        let blocks = |p: IVec3| if (0..10).contains(&p.y) { (block::STONE, 0) } else { (block::AIR, 0) };
        let mut player = Player::new(Vec3::new(0.5, 20.0, 0.5), false);
        let input = Input::default();
        for _ in 0..600 {
            player.update(1.0 / 60.0, 0.0, &input, blocks, 0);
        }
        assert!(player.on_ground, "player should be grounded");
        // Feet rest on the top face of the floor (y = 10).
        assert!(
            (player.position.y - 10.0).abs() < 0.05,
            "rest y = {}",
            player.position.y
        );
    }

    #[test]
    fn player_rests_at_half_height_on_a_slab() {
        // A floor of stone slabs at y=0 (solid only in y 0..0.5); the player must rest at y=0.5,
        // not y=1.0 — proves slabs collide as their true half-box, not a full cube.
        let blocks = |p: IVec3| if p.y == 0 { (block::STONE_SLAB, 0) } else { (block::AIR, 0) };
        let mut p = Player::new(Vec3::new(0.5, 8.0, 0.5), false);
        let input = Input::default();
        for _ in 0..600 {
            p.update(1.0 / 60.0, 0.0, &input, blocks, 0);
        }
        assert!(p.on_ground);
        assert!(
            (p.position.y - 0.5).abs() < 0.05,
            "feet should rest on the slab top (y=0.5), got {}",
            p.position.y
        );
    }

    #[test]
    fn player_cannot_pass_through_wall() {
        // Wall at x >= 5.
        let blocks = |p: IVec3| if p.x >= 5 { (block::STONE, 0) } else { (block::AIR, 0) };
        let mut player = Player::new(Vec3::new(0.5, 50.0, 0.5), true); // fly, no gravity
        let mut input = Input::default();
        input.forward = true; // yaw 0 -> forward is +x
        for _ in 0..600 {
            player.update(1.0 / 60.0, 0.0, &input, blocks, 0);
        }
        // AABB half-width is 0.3, wall face at x = 5, so max feet x = 4.7.
        assert!(player.position.x <= 4.71, "x = {}", player.position.x);
        assert!(player.position.x > 0.5, "player did not move toward the wall");
    }

    /// S2: a terminal-velocity fall through 0.1 s frames (6 blocks/frame) must land on a
    /// 1-block-thick floor, not pass through it (the pre-S2 teleport-then-resolve tunneled here).
    #[test]
    fn terminal_fall_lands_on_thin_floor() {
        let blocks = |p: IVec3| if p.y == 10 { (block::STONE, 0) } else { (block::AIR, 0) };
        let mut p = Player::new(Vec3::new(0.5, 200.0, 0.5), false);
        let input = Input::default();
        for _ in 0..100 {
            p.update(0.1, 0.0, &input, blocks, 0); // the app's max frame dt
        }
        assert!(p.on_ground, "player should land, not tunnel (y = {})", p.position.y);
        assert!((p.position.y - 11.0).abs() < 0.05, "rest y = {}", p.position.y);
    }

    /// S2: a fly-sprint dash (80 blocks/s -> 8 blocks per 0.1 s frame) stops at a 1-block wall.
    #[test]
    fn dash_cannot_tunnel_through_wall() {
        let blocks = |p: IVec3| if p.x == 20 { (block::STONE, 0) } else { (block::AIR, 0) };
        let mut p = Player::new(Vec3::new(0.5, 50.0, 0.5), true); // fly
        let mut input = Input::default();
        input.forward = true; // yaw 0 -> +x
        input.sprint = true; // FLY_SPRINT_SPEED = 80
        for _ in 0..40 {
            p.update(0.1, 0.0, &input, blocks, 0);
        }
        assert!(p.position.x <= 19.71, "dash tunneled the wall: x = {}", p.position.x);
        assert!(p.position.x > 10.0, "player did not move toward the wall");
    }

    /// S2 regression (found in review): walking AWAY from a fence must not clamp the player to the
    /// fence's far face — a sub-box outside the AABB on the moving axis is not a penetration. The
    /// pre-S2 clamp teleported the player THROUGH the fence here.
    #[test]
    fn backing_away_from_fence_does_not_teleport() {
        // Fence in cell x=0 (post box ~x in 0.375..0.625), ground below.
        let blocks = |p: IVec3| {
            if p.y < 0 {
                (block::STONE, 0)
            } else if p.x == 0 && p.y < 2 && p.z == 0 {
                (block::WOODEN_FENCE, 0)
            } else {
                (block::AIR, 0)
            }
        };
        // Player just west of the fence post, walking further west (-x = backward at yaw 0).
        let mut p = Player::new(Vec3::new(-0.05, 0.0, 0.5), false);
        let mut input = Input::default();
        input.back = true;
        for _ in 0..30 {
            p.update(1.0 / 60.0, 0.0, &input, blocks, 0);
        }
        assert!(
            p.position.x < -0.05,
            "player should walk away freely, not clamp to the fence: x = {}",
            p.position.x
        );
    }

    /// S2: the buried-save oracle. Standing ON a floor is contact, not overlap; a head inside
    /// stone is genuinely buried; a dug-out 2-high air pocket underground is a valid position.
    #[test]
    fn is_inside_solid_oracle() {
        let floor = |p: IVec3| if p.y < 10 { (block::STONE, 0) } else { (block::AIR, 0) };
        let p = Player::new(Vec3::new(0.5, 10.0, 0.5), false);
        assert!(!p.is_inside_solid(&floor), "standing on a floor is not buried");
        let p2 = Player::new(Vec3::new(0.5, 9.0, 0.5), false);
        assert!(p2.is_inside_solid(&floor), "feet embedded in the floor is buried");
        // A 2-high air pocket at y 10/11 inside solid rock (a player-dug cave room).
        let pocket = |p: IVec3| {
            if p.x == 0 && p.z == 0 && (p.y == 10 || p.y == 11) {
                (block::AIR, 0)
            } else {
                (block::STONE, 0)
            }
        };
        let p3 = Player::new(Vec3::new(0.5, 10.0, 0.5), false);
        assert!(!p3.is_inside_solid(&pocket), "a dug-out cave room is a valid resume spot");
    }

    #[test]
    fn s8_vanilla_regen_and_sprint_gate() {
        let floor = |q: IVec3| if q.y < 10 { (block::STONE, 0) } else { (block::AIR, 0) };
        let input = Input::default();
        // Saturated fast regen: 1 HP per 0.5 s at full hunger with saturation left.
        let mut p = Player::new(Vec3::new(0.5, 10.0, 0.5), false);
        p.restore_state(10.0, 20.0, 11.0, 5.0, 0.0, 0);
        for _ in 0..28 {
            p.update(1.0 / 60.0, 0.0, &input, floor, 0);
        }
        assert!((p.health - 10.0).abs() < 1e-3, "no heal before the 0.5 s cadence");
        for _ in 0..6 {
            p.update(1.0 / 60.0, 0.0, &input, floor, 0);
        }
        assert!((p.health - 11.0).abs() < 1e-3, "exactly 1 HP at the saturated cadence");
        // Normal regen (hunger 18, no saturation): 1 HP per 4 s — the old 1.2-2.4 HP/s is gone.
        let mut q = Player::new(Vec3::new(0.5, 10.0, 0.5), false);
        q.restore_state(10.0, 18.0, 11.0, 0.0, 0.0, 0);
        for _ in 0..180 {
            q.update(1.0 / 60.0, 0.0, &input, floor, 0);
        }
        assert!((q.health - 10.0).abs() < 1e-3, "no heal before 4 s");
        for _ in 0..66 {
            q.update(1.0 / 60.0, 0.0, &input, floor, 0);
        }
        assert!((q.health - 11.0).abs() < 1e-3, "1 HP per 4 s");
        // Sprint gate: vanilla denies sprint at 3 haunches (hunger 6) or less.
        let mut r = Player::new(Vec3::ZERO, false);
        r.restore_state(20.0, 6.0, 11.0, 0.0, 0.0, 0);
        assert!(!r.can_sprint(), "hunger 6 cannot sprint");
        r.restore_state(20.0, 7.0, 11.0, 0.0, 0.0, 0);
        assert!(r.can_sprint());
    }

    #[test]
    fn air_drains_underwater_and_refills_in_air() {
        let water = |_: IVec3| (block::WATER, 0);
        let mut p = Player::new(Vec3::new(0.5, 10.0, 0.5), false);
        // Fully submerged: air should fall below full.
        for _ in 0..120 {
            p.update(1.0 / 60.0, 0.0, &Input::default(), water, 0);
        }
        assert!(p.submerged, "eye should read as underwater");
        assert!(p.air < MAX_AIR, "air should drain underwater (air = {})", p.air);
        let low = p.air;
        // Back in open air: it refills.
        for _ in 0..120 {
            p.update(1.0 / 60.0, 0.0, &Input::default(), ALL_AIR, 0);
        }
        assert!(!p.submerged);
        assert!(p.air > low, "air should refill out of water");
    }

    #[test]
    fn drowning_damages_when_out_of_air() {
        let water = |_: IVec3| (block::WATER, 0);
        let mut p = Player::new(Vec3::new(0.5, 10.0, 0.5), false);
        for _ in 0..1200 {
            p.update(1.0 / 60.0, 0.0, &Input::default(), water, 0);
        }
        assert_eq!(p.air, 0.0);
        assert!(p.health < MAX_HEALTH, "drowning should hurt (hp = {})", p.health);
    }

    #[test]
    fn xp_accumulates_into_levels() {
        let mut p = Player::new(Vec3::ZERO, true);
        assert_eq!(p.level, 0);
        p.add_xp(10); // exactly xp_to_next(0)
        assert_eq!(p.level, 1);
        assert!(p.xp_fraction() < 0.001);
        p.add_xp(100);
        assert!(p.level > 1);
        assert!((0.0..=1.0).contains(&p.xp_fraction()));
    }

    #[test]
    fn eating_refills_hunger() {
        let mut p = Player::new(Vec3::ZERO, false);
        p.hunger = 6.0;
        p.saturation = 0.0;
        assert!(p.can_eat(false));
        p.eat(8, 12.8); // a steak
        assert_eq!(p.hunger, 14.0, "hunger restored by the food value");
        assert!(p.saturation > 0.0, "saturation gained");
        assert!(p.saturation <= p.hunger, "saturation never exceeds the food level");
        // Full hunger: a normal food can't be eaten.
        p.hunger = MAX_HUNGER;
        assert!(!p.can_eat(false));
        assert!(p.can_eat(true), "always-edible foods ignore the full-hunger gate");
    }

    #[test]
    fn difficulty_starvation_floors_and_peaceful_regen() {
        use crate::rules::Difficulty;
        let input = Input::default();
        // Hard: starvation is lethal (floor 0).
        let mut p = Player::new(Vec3::ZERO, false);
        p.difficulty = Difficulty::Hard;
        (p.health, p.hunger, p.saturation) = (3.0, 0.0, 0.0);
        for _ in 0..1200 {
            p.update_survival(1.0 / 60.0, false);
        }
        assert!(p.health <= 0.0, "Hard starvation should kill, got {}", p.health);
        // Easy: starvation can't drop below 10 HP.
        let mut p = Player::new(Vec3::ZERO, false);
        p.difficulty = Difficulty::Easy;
        (p.health, p.hunger, p.saturation) = (20.0, 0.0, 0.0);
        for _ in 0..3600 {
            p.update_survival(1.0 / 60.0, false);
        }
        assert!((p.health - 10.0).abs() < 0.2, "Easy floors starvation at 10, got {}", p.health);
        // Peaceful: health regenerates with zero hunger, and hunger stays full.
        let mut p = Player::new(Vec3::ZERO, false);
        p.difficulty = Difficulty::Peaceful;
        (p.health, p.hunger, p.saturation) = (5.0, 0.0, 0.0);
        for _ in 0..1200 {
            p.update_survival(1.0 / 60.0, false);
        }
        assert!(p.health > 5.0, "Peaceful should passively regen, got {}", p.health);
        assert_eq!(p.hunger, MAX_HUNGER, "Peaceful keeps hunger full");
    }

    #[test]
    fn difficulty_scales_combat_damage() {
        use crate::rules::Difficulty;
        let mut p = Player::new(Vec3::ZERO, false); // no armor
        p.difficulty = Difficulty::Peaceful;
        p.take_hit(10.0);
        assert_eq!(p.health, MAX_HEALTH, "Peaceful negates combat damage");
        let mut p = Player::new(Vec3::ZERO, false);
        p.difficulty = Difficulty::Hard;
        p.take_hit(4.0); // Hard = 1.5x => 6 damage
        assert!((p.health - (MAX_HEALTH - 6.0)).abs() < 0.01, "Hard scales 1.5x, got {}", p.health);
    }

    #[test]
    fn saturation_drains_before_hunger() {
        // S8: hunger drains through EXHAUSTION (4.0 = one point), saturation absorbing first.
        let mut p = Player::new(Vec3::new(0.5, 10.0, 0.5), false);
        let floor = |q: IVec3| if q.y < 10 { (block::STONE, 0) } else { (block::AIR, 0) };
        let input = Input::default();
        p.add_exhaustion(8.0); // two points banked
        p.update(1.0 / 60.0, 0.0, &input, floor, 0);
        assert!(p.saturation() < MAX_SATURATION, "saturation pays first");
        assert_eq!(p.hunger, MAX_HUNGER, "hunger untouched while saturation lasts");
        p.add_exhaustion(EXHAUSTION_LIMIT * (MAX_SATURATION + 2.0));
        p.update(1.0 / 60.0, 0.0, &input, floor, 0);
        assert_eq!(p.saturation(), 0.0);
        assert!(p.hunger < MAX_HUNGER, "hunger pays once saturation is dry");
        // Plain walking costs nothing now (vanilla): banked exhaustion is the only drain.
        let before = p.hunger;
        let mut walk = Input::default();
        walk.forward = true;
        for _ in 0..240 {
            p.update(1.0 / 60.0, 0.0, &walk, floor, 0);
        }
        assert_eq!(p.hunger, before, "walking alone never drains hunger (S8)")
    }

    #[test]
    fn armor_reduces_damage() {
        let mut bare = Player::new(Vec3::ZERO, true);
        let mut armored = Player::new(Vec3::ZERO, true);
        bare.apply_damage(10.0, 0);
        armored.apply_damage(10.0, 20); // a full diamond set = 20 points = 80% reduction (the cap)
        assert!((bare.health - 10.0).abs() < 1e-3, "bare took full 10 (hp = {})", bare.health);
        assert!((armored.health - 18.0).abs() < 1e-3, "armor cut 10 -> 2 (hp = {})", armored.health);
    }

    #[test]
    fn critical_hit_conditions() {
        // Crit = airborne + falling + not sprinting + not in water + not creative-flying.
        assert!(is_critical_hit(false, -2.0, false, false, false), "falling melee crits");
        assert!(!is_critical_hit(true, -2.0, false, false, false), "grounded never crits");
        assert!(!is_critical_hit(false, 1.0, false, false, false), "rising (jump up) doesn't crit");
        assert!(!is_critical_hit(false, -2.0, true, false, false), "sprint-attacks can't crit");
        assert!(!is_critical_hit(false, -2.0, false, true, false), "no crit underwater");
        assert!(!is_critical_hit(false, -2.0, false, false, true), "no crit while creative-flying");
    }

    #[test]
    fn shield_blocks_frontal_only() {
        let fwd = Vec3::new(1.0, 0.0, 0.0); // looking +X
        let me = Vec3::ZERO;
        assert!(shield_blocks(fwd, me, Vec3::new(5.0, 0.0, 0.0)), "dead ahead is blocked");
        assert!(shield_blocks(fwd, me, Vec3::new(1.0, 0.0, 5.0)), "front-side (in the hemisphere) blocks");
        assert!(shield_blocks(fwd, me, Vec3::new(2.0, 3.0, 0.0)), "vertical offset is ignored (still front)");
        assert!(!shield_blocks(fwd, me, Vec3::new(-5.0, 0.0, 0.0)), "from behind is not blocked");
        assert!(!shield_blocks(fwd, me, Vec3::new(0.0, 5.0, 0.0)), "straight overhead is degenerate");
        // Looking straight up: no horizontal facing -> nothing blocks (known limitation).
        assert!(!shield_blocks(Vec3::new(0.0, 1.0, 0.0), me, Vec3::new(5.0, 0.0, 0.0)));
    }
}
