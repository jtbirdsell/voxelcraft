//! View-frustum culling via Gribb–Hartmann plane extraction from the view-projection matrix.
//! Assumes wgpu/D3D clip space (z in [0, 1]).

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

pub struct Frustum {
    /// Planes as (a, b, c, d) with the normal (a,b,c) pointing inward; a point is inside the
    /// plane when a*x + b*y + c*z + d >= 0.
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_view_proj(m: Mat4) -> Self {
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        let r3 = m.row(3);

        let mut planes = [
            r3 + r0, // left
            r3 - r0, // right
            r3 + r1, // bottom
            r3 - r1, // top
            r2,      // near (z in [0,1])
            r3 - r2, // far
        ];
        for p in &mut planes {
            let len = p.xyz().length();
            if len > 0.0 {
                *p /= len;
            }
        }
        Self { planes }
    }

    /// True if the axis-aligned box [min, max] is at least partially inside the frustum.
    pub fn aabb_visible(&self, min: Vec3, max: Vec3) -> bool {
        for p in &self.planes {
            let n = p.xyz();
            // Pick the box corner farthest along the plane normal (the "positive vertex").
            let positive = Vec3::new(
                if n.x >= 0.0 { max.x } else { min.x },
                if n.y >= 0.0 { max.y } else { min.y },
                if n.z >= 0.0 { max.z } else { min.z },
            );
            if n.dot(positive) + p.w < 0.0 {
                // Entirely on the outside of this plane.
                return false;
            }
        }
        true
    }
}
