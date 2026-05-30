//! Voxel ray traversal (Amanatides & Woo 3D-DDA) for block targeting.

use glam::{IVec3, Vec3};

pub struct RayHit {
    /// The solid block that was hit.
    pub block: IVec3,
    /// The face normal of the hit (points back toward the ray origin); the adjacent empty
    /// cell for placement is `block + normal`.
    pub normal: IVec3,
}

/// Step along `dir` from `origin` up to `max_dist` blocks, returning the first cell for which
/// `is_solid` is true.
pub fn cast(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    is_solid: impl Fn(IVec3) -> bool,
) -> Option<RayHit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    let mut voxel = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );
    let step = IVec3::new(
        dir.x.signum() as i32,
        dir.y.signum() as i32,
        dir.z.signum() as i32,
    );

    // Distance (in t) to cross one full cell along each axis.
    let t_delta = Vec3::new(
        if dir.x != 0.0 { (1.0 / dir.x).abs() } else { f32::INFINITY },
        if dir.y != 0.0 { (1.0 / dir.y).abs() } else { f32::INFINITY },
        if dir.z != 0.0 { (1.0 / dir.z).abs() } else { f32::INFINITY },
    );

    // Distance (in t) to the first cell boundary along each axis.
    let next_boundary = |o: f32, d: f32, v: i32| -> f32 {
        if d > 0.0 {
            (v as f32 + 1.0 - o) / d
        } else if d < 0.0 {
            (o - v as f32) / -d
        } else {
            f32::INFINITY
        }
    };
    let mut t_max = Vec3::new(
        next_boundary(origin.x, dir.x, voxel.x),
        next_boundary(origin.y, dir.y, voxel.y),
        next_boundary(origin.z, dir.z, voxel.z),
    );

    // Assigned by every loop branch before it is read; the starting cell below returns ZERO directly.
    let mut normal: IVec3;
    let mut t = 0.0f32;
    // Check the starting cell too.
    if is_solid(voxel) {
        return Some(RayHit { block: voxel, normal: IVec3::ZERO });
    }
    while t <= max_dist {
        if t_max.x < t_max.y && t_max.x < t_max.z {
            voxel.x += step.x;
            t = t_max.x;
            t_max.x += t_delta.x;
            normal = IVec3::new(-step.x, 0, 0);
        } else if t_max.y < t_max.z {
            voxel.y += step.y;
            t = t_max.y;
            t_max.y += t_delta.y;
            normal = IVec3::new(0, -step.y, 0);
        } else {
            voxel.z += step.z;
            t = t_max.z;
            t_max.z += t_delta.z;
            normal = IVec3::new(0, 0, -step.z);
        }
        if t > max_dist {
            break;
        }
        if is_solid(voxel) {
            return Some(RayHit { block: voxel, normal });
        }
    }
    None
}
