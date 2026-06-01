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
const HUNGER_DRAIN: f32 = 0.08; // hunger/sec while walking
const HUNGER_SPRINT_MULT: f32 = 2.5;
const REGEN_HUNGER: f32 = 17.0; // health regenerates while hunger is at least this
const REGEN_RATE: f32 = 1.2; // hp/sec regenerated
const REGEN_COST: f32 = 0.6; // hunger/sec spent regenerating
const STARVE_RATE: f32 = 0.8; // hp/sec lost at zero hunger
const MAX_SATURATION: f32 = 5.0; // hidden food reserve: drains before hunger, fuels faster regen

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
    pub health: f32,
    pub hunger: f32,
    /// Breath remaining underwater (0..MAX_AIR); drives the bubble bar + drowning.
    pub air: f32,
    /// True while the eye is inside water (HUD shows bubbles; gameplay input unaffected).
    pub submerged: bool,
    /// Hidden food saturation; drains before hunger and fuels faster regen.
    saturation: f32,
    /// Experience: total points toward the next level, plus the current level.
    pub xp: f32,
    pub level: u32,
    /// Total equipped-armor defense points, refreshed from the inventory each frame.
    armor_points: u32,
    /// Post-hit invulnerability timer (seconds); incoming combat damage is ignored while it's >0.
    hurt_cooldown: f32,
    drown_timer: f32,
    /// Highest y reached since last touching ground, for fall-damage distance.
    air_max_y: f32,
    /// Where `respawn` returns the player on death.
    respawn_point: Vec3,
}

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

impl Player {
    pub fn new(position: Vec3, flying: bool) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            on_ground: false,
            flying,
            health: MAX_HEALTH,
            hunger: MAX_HUNGER,
            air: MAX_AIR,
            submerged: false,
            saturation: MAX_SATURATION,
            xp: 0.0,
            level: 0,
            armor_points: 0,
            hurt_cooldown: 0.0,
            drown_timer: 0.0,
            air_max_y: position.y,
            respawn_point: position,
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
        if self.hurt_cooldown > 0.0 || raw <= 0.0 {
            return;
        }
        let armor = self.armor_points;
        self.apply_damage(raw, armor);
        self.hurt_cooldown = 0.5;
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
    }

    /// Hunger drain, regen when well-fed (faster while saturation lasts), starvation at empty hunger.
    /// `env_damage` suppresses regen so the player can't heal mid-lava / mid-drown.
    fn update_survival(&mut self, dt: f32, input: &Input, env_damage: bool) {
        let moving = input.forward || input.back || input.left || input.right;
        // Hunger only drains through activity (no AFK starvation) — saturation absorbs it first.
        if moving {
            let mult = if input.sprint { HUNGER_SPRINT_MULT } else { 1.0 };
            let drain = HUNGER_DRAIN * mult * dt;
            if self.saturation > 0.0 {
                self.saturation = (self.saturation - drain).max(0.0);
            } else {
                self.hunger = (self.hunger - drain).max(0.0);
            }
        }

        if !env_damage && self.hunger >= REGEN_HUNGER && self.health < MAX_HEALTH {
            let rate = if self.saturation > 0.0 { REGEN_RATE * 2.0 } else { REGEN_RATE };
            self.health = (self.health + rate * dt).min(MAX_HEALTH);
            if self.saturation > 0.0 {
                self.saturation = (self.saturation - REGEN_COST * dt).max(0.0);
            } else {
                self.hunger = (self.hunger - REGEN_COST * dt).max(0.0);
            }
        } else if self.hunger <= 0.0 {
            self.apply_damage(STARVE_RATE * dt, 0); // starvation ignores armor
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
        } else {
            let speed = if input.sneak {
                SNEAK_SPEED
            } else if input.sprint {
                SPRINT_SPEED
            } else {
                WALK_SPEED
            };
            self.velocity.x = wish.x * speed;
            self.velocity.z = wish.z * speed;
            self.velocity.y -= GRAVITY * dt;
            self.velocity.y = self.velocity.y.max(-TERMINAL);
            if input.up && self.on_ground {
                self.velocity.y = JUMP_SPEED;
                self.on_ground = false;
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
            if landed && !self.on_ground && !in_water {
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
            let env_damage = self.update_environment_damage(dt, head_submerged, in_lava);
            self.update_survival(dt, input, env_damage);
        }
    }

    /// Move along one axis, then resolve overlap by clamping the player AABB to the nearest face of
    /// any block sub-box it penetrates (full cubes, but also slab/stair half-boxes). Returns true if
    /// a collision was resolved.
    fn move_axis(&mut self, axis: usize, amount: f32, block_at: &impl Fn(IVec3) -> (BlockId, u8)) -> bool {
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
                        // Only blocks the player overlaps on the two perpendicular axes can stop it.
                        let perp = (0..3).filter(|&a| a != axis).all(|a| {
                            cmin[a] < bmax[a] - 1e-4 && cmax[a] > bmin[a] + 1e-4
                        });
                        if !perp {
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
    fn saturation_drains_before_hunger() {
        // Ground at y<0 (stone) so the player rests at y=0; air (not water) elsewhere.
        let blocks = |p: IVec3| if p.y < 0 { (block::STONE, 0) } else { (block::AIR, 0) };
        let mut p = Player::new(Vec3::new(0.5, 0.0, 0.5), false);
        let full_hunger = p.hunger;
        let mut input = Input::default();
        input.forward = true;
        for _ in 0..120 {
            p.update(1.0 / 60.0, 0.0, &input, blocks, 0);
        }
        // While saturation lasts, hunger is untouched.
        assert_eq!(p.hunger, full_hunger, "hunger should hold while saturation remains");
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
}
