// Hardware ray-tracing spike (compute). Proves the RTX 4090's RT cores are reachable through wgpu's
// Vulkan backend: per pixel we cast a primary ray into a hardware BLAS/TLAS via `rayQuery`, then a
// second hardware shadow ray toward the sun. If this renders a scene with correct hard shadows, the
// acceleration-structure build + ray queries are working on real silicon — the prerequisite for moving
// the software DDA tracer (rtx_common.wgsl) onto the RT cores.

enable wgpu_ray_query;

struct Uniforms {
    eye: vec4<f32>,
    ll: vec4<f32>,    // direction to the lower-left of the image plane (eye-relative)
    horiz: vec4<f32>, // spans left->right across the plane
    vert: vec4<f32>,  // spans bottom->top across the plane
    sun: vec4<f32>,   // normalized sun direction (toward the sun)
    dims: vec4<u32>,  // width, height, _, _
};

// Per-triangle attributes, indexed by the committed intersection's primitive_index. We bake the whole
// scene (ground + boxes) into ONE BLAS with an identity-transform TLAS instance, so primitive_index is
// a global triangle id and the geometric normal needs no instance-transform correction.
struct Tri {
    normal: vec4<f32>,
    albedo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var acc: acceleration_structure;
@group(0) @binding(2) var<storage, read> tris: array<Tri>;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba8unorm, write>;

fn sky(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.78, 0.86, 0.95);
    let zenith = vec3<f32>(0.32, 0.52, 0.86);
    return mix(horizon, zenith, t);
}

// Returns true if `origin + sun*t` is occluded before reaching the light (a hardware shadow ray).
fn in_shadow(origin: vec3<f32>, sun: vec3<f32>) -> bool {
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 1000.0, origin, sun));
    while (rayQueryProceed(&rq)) {}
    let hit = rayQueryGetCommittedIntersection(&rq);
    return hit.kind != RAY_QUERY_INTERSECTION_NONE;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u.dims.x;
    let h = u.dims.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }

    let px = (f32(gid.x) + 0.5) / f32(w);
    let py = 1.0 - (f32(gid.y) + 0.5) / f32(h); // image row 0 is the top; py=1 is the top of the plane
    let dir = normalize(u.ll.xyz + u.horiz.xyz * px + u.vert.xyz * py);
    let origin = u.eye.xyz;

    var color = sky(dir);

    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 1000.0, origin, dir));
    while (rayQueryProceed(&rq)) {}
    let hit = rayQueryGetCommittedIntersection(&rq);

    if (hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        let p = origin + dir * hit.t;
        let tri = tris[hit.primitive_index];
        var n = tri.normal.xyz;
        if (dot(n, dir) > 0.0) {
            n = -n; // face the camera
        }
        let sun = normalize(u.sun.xyz);
        let ndl = max(dot(n, sun), 0.0);

        var shade = 0.0;
        if (ndl > 0.0) {
            // Offset along the normal to avoid self-intersection (shadow acne).
            let shadowed = in_shadow(p + n * 0.002, sun);
            shade = select(ndl, 0.0, shadowed);
        }

        let ambient = 0.18;
        color = tri.albedo.xyz * (ambient + 0.95 * shade);
    }

    // Gamma-encode for the unorm target so the PNG reads correctly.
    let srgb = pow(color, vec3<f32>(1.0 / 2.2));
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(srgb, 1.0));
}
