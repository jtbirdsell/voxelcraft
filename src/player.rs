//! Player controller: an AABB with swept, axis-separated collision against the voxel world,
//! gravity + jump in walk mode, and free flight in fly mode.

use glam::{IVec3, Vec3};

pub const PLAYER_WIDTH: f32 = 0.6;
pub const PLAYER_HEIGHT: f32 = 1.8;
pub const EYE_HEIGHT: f32 = 1.62;

const HALF_W: f32 = PLAYER_WIDTH * 0.5;
const GRAVITY: f32 = 30.0;
const TERMINAL: f32 = 60.0;
const WALK_SPEED: f32 = 5.0;
const SPRINT_SPEED: f32 = 8.5;
const JUMP_SPEED: f32 = 9.2;
const FLY_SPEED: f32 = 26.0;
const FLY_SPRINT_SPEED: f32 = 80.0;

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
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub break_pressed: bool,
    pub place_pressed: bool,
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
        }
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
        is_solid: impl Fn(IVec3) -> bool,
    ) {
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
        } else {
            let speed = if input.sprint { SPRINT_SPEED } else { WALK_SPEED };
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
        self.move_axis(0, delta.x, &is_solid);
        self.move_axis(2, delta.z, &is_solid);
        let hit_y = self.move_axis(1, delta.y, &is_solid);
        if !self.flying {
            self.on_ground = hit_y && delta.y < 0.0;
        }
    }

    /// Move along one axis, then resolve any voxel overlap by clamping to the contact face.
    /// Returns true if a collision was resolved.
    fn move_axis(&mut self, axis: usize, amount: f32, is_solid: &impl Fn(IVec3) -> bool) -> bool {
        if amount == 0.0 {
            return false;
        }
        let moved = comp(self.position, axis) + amount;
        set_comp(&mut self.position, axis, moved);

        let (min, max) = self.aabb();
        let x0 = min.x.floor() as i32;
        let x1 = (max.x - 1e-4).floor() as i32;
        let y0 = min.y.floor() as i32;
        let y1 = (max.y - 1e-4).floor() as i32;
        let z0 = min.z.floor() as i32;
        let z1 = (max.z - 1e-4).floor() as i32;

        let (lo, hi) = Self::offsets(axis);
        let mut clamp: Option<f32> = None;
        for vx in x0..=x1 {
            for vy in y0..=y1 {
                for vz in z0..=z1 {
                    if is_solid(IVec3::new(vx, vy, vz)) {
                        let vcoord = match axis {
                            0 => vx,
                            1 => vy,
                            _ => vz,
                        } as f32;
                        if amount > 0.0 {
                            let p = vcoord - hi;
                            clamp = Some(clamp.map_or(p, |c| c.min(p)));
                        } else {
                            let p = (vcoord + 1.0) - lo;
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

    #[test]
    fn player_falls_and_lands_on_floor() {
        // Solid floor occupying y in 0..10; air above.
        let is_solid = |p: IVec3| p.y >= 0 && p.y < 10;
        let mut player = Player::new(Vec3::new(0.5, 20.0, 0.5), false);
        let input = Input::default();
        for _ in 0..600 {
            player.update(1.0 / 60.0, 0.0, &input, is_solid);
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
    fn player_cannot_pass_through_wall() {
        // Wall at x >= 5.
        let is_solid = |p: IVec3| p.x >= 5;
        let mut player = Player::new(Vec3::new(0.5, 50.0, 0.5), true); // fly, no gravity
        let mut input = Input::default();
        input.forward = true; // yaw 0 -> forward is +x
        for _ in 0..600 {
            player.update(1.0 / 60.0, 0.0, &input, is_solid);
        }
        // AABB half-width is 0.3, wall face at x = 5, so max feet x = 4.7.
        assert!(player.position.x <= 4.71, "x = {}", player.position.x);
        assert!(player.position.x > 0.5, "player did not move toward the wall");
    }
}
