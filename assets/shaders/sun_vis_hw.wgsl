// Hardware sun-shadow ray (M33-G5a): an inline ray-query any-hit against the world TLAS (`rt_acc`,
// bound at group 3). Appended by the renderer in place of the DDA `sun_visibility` when
// VOXELCRAFT_TRACER=hwrt and the backend exposes hardware ray query. Same signature/semantics as the
// DDA version (1 lit, 0 shadowed), so the rest of the shader is unchanged.
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
