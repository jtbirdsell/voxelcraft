// Chunk lighting: ray-traced sun shadows + ambient occlusion + one-bounce colored global
// illumination, composed with the direct sun term and distance fog. rtx_common.wgsl is prepended
// at pipeline build time and provides the Camera/Volume bindings, the vertex stage, voxel_color(),
// trace() and sun_visibility().

// Branchless orthonormal basis (Duff et al. 2017) with `n` as the third (z) column.
fn onb(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bt, n);
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

// Ambient occlusion + one-bounce colored GI: cosine-weighted hemisphere rays gather sky radiance
// when they escape and bounced (sun-lit) material color when they hit. Returns indirect irradiance.
fn gather_gi(world_pos: vec3<f32>, n: vec3<f32>, px: vec2<f32>) -> vec3<f32> {
    let rays = i32(volume.params.z);
    if (rays <= 0) {
        return vec3<f32>(camera.params.z);
    }
    let gi_dist = volume.paramsf.x;
    let gi_strength = volume.paramsf.y;
    let sky_boost = volume.paramsf.z;
    // A diagonal ray crosses up to ~sqrt(3) voxel boundaries per block travelled, so size the
    // step budget above gi_dist to keep AO range isotropic rather than clipping diagonal rays.
    let max_steps = i32(gi_dist * 1.8) + 2;

    let sun = normalize(camera.sun_dir.xyz);
    let ambient = camera.params.z;
    let sun_intensity = camera.params.w;
    let sky = camera.sky_color.rgb * sky_boost;

    let basis = onb(n);
    let rnd = hash2(px);
    let origin = world_pos + n * 0.06;

    var accum = vec3<f32>(0.0);
    for (var i = 0; i < rays; i = i + 1) {
        // Stratified-ish cosine-weighted hemisphere sample via golden-ratio additive recurrence,
        // Cranley–Patterson rotated per-pixel so neighbouring pixels don't share a pattern.
        let u1 = fract(rnd.x + f32(i) * 0.7548776662);
        let u2 = fract(rnd.y + f32(i) * 0.5698402909);
        let r = sqrt(u1);
        let phi = 6.28318530718 * u2;
        let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        let dir = normalize(basis * local);

        let h = trace(origin, dir, gi_dist, max_steps);
        if (h.hit) {
            let alb = voxel_color(h.id);
            let ndl_h = max(dot(h.normal, sun), 0.0);
            var vis = 1.0;
            if (ndl_h > 0.0) {
                vis = sun_visibility(h.pos, h.normal, gi_dist);
            }
            // Bounced radiance leaving the hit surface: its own ambient + direct sun term, plus any
            // self-emission (lava) so emissive blocks cast colored indirect light.
            accum += alb * (ambient + sun_intensity * ndl_h * vis) + voxel_emission(h.id);
        } else {
            accum += sky;
        }
    }
    return (accum / f32(rays)) * gi_strength;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let emission = in.shade.x;
    // Lava crust flows: scroll + gently wobble its tile UV over time (torch/glowstone stay static).
    var tuv = in.uv;
    if (in.tile == 12u) {
        tuv.y = tuv.y - camera.time.x * 0.10;
        tuv.x = tuv.x + sin(camera.time.x * 0.7 + in.world_pos.x) * 0.03;
    }
    let texel = sample_tile(in.tile, tuv);
    // Alpha cutout for cross-billboard plants (atlas alpha 0 outside the plant shape).
    if (texel.a < 0.5) {
        discard;
    }
    let albedo = texel.rgb;
    let sun = normalize(camera.sun_dir.xyz);
    let ambient = camera.params.z;
    let sun_intensity = camera.params.w;
    let ndl = max(dot(n, sun), 0.0);

    var shadow = 1.0;
    if (volume.params.y >= 1u) {
        shadow = sun_visibility(in.world_pos, n, 96.0);
    }
    let direct = sun_intensity * ndl * shadow;

    var indirect = vec3<f32>(ambient);
    if (volume.params.y >= 2u) {
        indirect = gather_gi(in.world_pos, n, in.clip_pos.xy);
    }
    // Block-light + skylight (M14): skylight gates the sky-driven indirect so caves go dark, while
    // block light (torch/glowstone/lava grid) adds warm local light that survives underground.
    let sky = in.light.x;
    let blockl = in.light.y;
    indirect = indirect * sky;
    let warm = blockl * vec3<f32>(1.15, 0.92, 0.55);
    // Lit albedo plus self-emission (lava glows regardless of sun/ambient).
    var rgb = albedo * (indirect + vec3<f32>(direct) + warm) + albedo * (emission * 2.5);

    let dist = length(in.world_pos - camera.cam_pos.xyz);
    let fog = smoothstep(camera.params.x, camera.params.y, dist);
    rgb = mix(rgb, camera.fog_color.rgb, fog);

    // Mob hurt-flash (M29): `shade.y` carries a small fractional hurt value (0..~0.35) on mob
    // geometry. Terrain reuses `shade.y` as an integer tint_class (0/1/2), so only an in-between
    // fractional value triggers the red flash — terrain is left byte-for-byte unchanged.
    let flash = select(0.0, in.shade.y, in.shade.y > 0.0 && in.shade.y < 0.5);
    rgb = mix(rgb, vec3<f32>(1.0, 0.18, 0.13), flash * 1.6);

    return vec4<f32>(rgb, 1.0);
}
