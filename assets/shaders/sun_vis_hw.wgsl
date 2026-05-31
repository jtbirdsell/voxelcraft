// Hardware ray-query trace + sun shadow (M33-G5). Appended by the renderer in place of the DDA
// trace/sun_visibility when VOXELCRAFT_TRACER=hwrt and the backend exposes hardware ray query.
// `rt_acc` (the world TLAS) is bound at group 3; `voxel_id` / `Hit` come from rtx_common.
//
// Hybrid model, chosen for parity with the DDA: hardware ray query finds the closest hit on the RT
// cores, then we recover the material + face normal from the *voxel volume* at the hit point — the
// same voxel_color / voxel_emission the DDA reads — so GI and water reflections match the software
// path closely while the (expensive) traversal moves to hardware.

fn trace(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32, max_steps: i32) -> Hit {
    var h: Hit;
    h.hit = false;
    var rq: ray_query;
    rayQueryInitialize(&rq, rt_acc, RayDesc(0u, 0xFFu, 0.001, max_dist, origin, dir));
    while (rayQueryProceed(&rq)) {}
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        return h;
    }
    let p = origin + dir * hit.t;
    // The solid voxel sits just past the hit along the ray, the air voxel just before it; their
    // integer difference is the entered face normal (voxels are axis-aligned).
    let solid = vec3<i32>(floor(p + dir * 0.05));
    let air = vec3<i32>(floor(p - dir * 0.05));
    let dn = vec3<f32>(air - solid);
    h.hit = true;
    h.pos = p;
    h.id = voxel_id(solid);
    // `dn` is the entered face normal. On rare grazing rays where both samples land in the same
    // voxel (dn=0), fall back to the face perpendicular to the ray's dominant travel axis, not -dir
    // (which is not an axis-aligned normal and skews the bounce term). (review H2)
    let ad = abs(dir);
    var fallback = vec3<f32>(0.0, 0.0, -sign(dir.z));
    if (ad.x >= ad.y && ad.x >= ad.z) {
        fallback = vec3<f32>(-sign(dir.x), 0.0, 0.0);
    } else if (ad.y >= ad.z) {
        fallback = vec3<f32>(0.0, -sign(dir.y), 0.0);
    }
    h.normal = select(fallback, normalize(dn), dot(dn, dn) > 0.5);
    return h;
}

fn sun_visibility(world_pos: vec3<f32>, n: vec3<f32>, max_dist: f32) -> f32 {
    let sun = normalize(camera.sun_dir.xyz);
    if (sun.y <= 0.02) {
        return 1.0;
    }
    let origin = world_pos + n * 0.06 + sun * 0.02;
    var rq: ray_query;
    // flags = 0x4 (TERMINATE_ON_FIRST_HIT): any occluder shadows; cull_mask 0xFF.
    rayQueryInitialize(&rq, rt_acc, RayDesc(0x4u, 0xFFu, 0.0, max_dist, origin, sun));
    while (rayQueryProceed(&rq)) {}
    let hit = rayQueryGetCommittedIntersection(&rq);
    return select(1.0, 0.0, hit.kind != RAY_QUERY_INTERSECTION_NONE);
}
