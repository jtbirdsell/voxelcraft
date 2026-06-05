// Chunk lighting: ray-traced sun shadows + ambient occlusion + one-bounce colored global
// illumination, composed with the direct sun term and distance fog. rtx_common.wgsl + atlas.wgsl
// are prepended at pipeline build time and provide the Camera/Volume bindings, the vertex stage,
// voxel_color(), gather_gi(), sample_tile(), trace() and sun_visibility().
//
// M33-G6: `DEFER_GI` (a const injected by the renderer per VOXELCRAFT_GI) selects WHERE the
// hemisphere GI is gathered. When false (VOXELCRAFT_GI=fragment, the parity oracle) it is gathered
// inline here, exactly as in G5b. When true (default) the indirect term is left to the GI compute
// pass (gi_compute.wgsl) + composite (gi_composite.wgsl): this shader writes only the direct +
// block-light + emissive color (fogged + hurt-flashed), and the composite additively blends
// albedo * irradiance * skylight * (1-fog) * (1-flash) back in — algebraically identical.

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(1) gnormal: vec4<f32>,  // world normal .xyz + emission .w (G-buffer, M33-G3)
    @location(2) gmotion: vec2<f32>,  // screen-space motion (cur_uv - prev_uv)
    @location(3) gpos: vec4<f32>,     // world-space surface position .xyz + hurt-flash .w (M33-G6)
    @location(4) galbedo: vec4<f32>,  // albedo .rgb + skylight .a
    @location(5) gdepth: f32,         // linear view depth (clip.w) — DLSS RR depth guide (M33-G8)
};

// Clip-space position -> screen UV (origin top-left). Guards w≈0 just behind the eye plane.
fn ndc_to_uv(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / max(abs(clip.w), 1e-6);
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// M35 visual fix: a small ambient floor so deep, sun-shadowed, zero-skylight surfaces — the bottom of
// a pit, an overhang, an enclosed concave corner — never collapse to PURE BLACK. The lit color's only
// sky-driven fill term is `ambient * sky`, which vanishes at sky=0; with the sun shadowed and no block
// light that leaves `albedo * 0` = black, and because the baked skylight is interpolated per-vertex the
// black has a smooth gradient halo (the "weird blend out"). This term lifts ONLY low-sky surfaces (the
// `(1 - sky)` weight leaves lit terrain byte-identical) and scales with the day/night `ambient` (so
// nights stay dark). It is added to the ALWAYS-written base color, so the deferred-GI composite and the
// in-fragment GI oracle remain in exact parity (the composite only adds the unchanged sky-gated GI).
// `AMBIENT_FLOOR_FRAC` is injected by the renderer (default 0.25; VOXELCRAFT_AMBIENT_FLOOR overrides,
// 0 = the old pure-black behaviour) alongside DEFER_GI.

@fragment
fn fs_main(in: VsOut) -> FragOut {
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
        // Primary sun-shadow ray; range is a tier knob (P21, paramsg.x; maxed tier = the old 96.0).
        shadow = sun_visibility(in.world_pos, n, volume.paramsg.x);
    }
    let direct = sun_intensity * ndl * shadow;

    // Block-light + skylight (M14): skylight gates the sky-driven indirect so caves go dark, while
    // block light (torch/glowstone/lava grid) adds warm local light that survives underground.
    let sky = in.light.x;
    let blockl = in.light.y;
    // M35-SL: a slightly brighter/warmer block-light tint so a torch's glow pool survives daylight.
    let warm = blockl * vec3<f32>(1.45, 1.0, 0.55);

    // Indirect (AO + one-bounce GI). DEFER_GI=false (VOXELCRAFT_GI=fragment): gather inline, the
    // G5b parity oracle. DEFER_GI=true (default): the GI compute pass produces the irradiance and
    // the composite additively blends `albedo * irradiance * sky * (1-fog) * (1-flash)`, so the
    // indirect term stays 0 here. When GI is off (rtx_mode < 2) the baseline is the ambient term —
    // the compute pass writes that same baseline, so both paths agree across all rtx modes.
    var indirect = vec3<f32>(0.0);
    if (!DEFER_GI) {
        var irr = vec3<f32>(ambient);
        if (volume.params.y >= 2u) {
            irr = gather_gi(in.world_pos, n, in.clip_pos.xy);
        }
        indirect = irr * sky;
    }
    // Lit albedo plus self-emission (lava glows regardless of sun/ambient). `amb_floor` keeps
    // fully-shadowed, sky-occluded surfaces from going pure black (see AMBIENT_FLOOR_FRAC above).
    let amb_floor = ambient * AMBIENT_FLOOR_FRAC * (1.0 - sky);
    var rgb = albedo * (indirect + vec3<f32>(direct) + warm + vec3<f32>(amb_floor)) + albedo * (emission * 2.5);

    let dist = length(in.world_pos - camera.cam_pos.xyz);
    let fog = smoothstep(camera.params.x, camera.params.y, dist);
    rgb = mix(rgb, camera.fog_color.rgb, fog);

    // Mob hurt-flash (M29): `shade.y` carries a small fractional hurt value (0..~0.35) on mob
    // geometry. Terrain reuses `shade.y` as an integer tint_class (0/1/2), so only an in-between
    // fractional value triggers the red flash — terrain is left byte-for-byte unchanged.
    let flash = select(0.0, in.shade.y, in.shade.y > 0.0 && in.shade.y < 0.5);
    rgb = mix(rgb, vec3<f32>(1.0, 0.18, 0.13), flash * 1.6);

    var out: FragOut;
    out.color = vec4<f32>(rgb, 1.0);
    out.gnormal = vec4<f32>(n, emission);
    out.gmotion = ndc_to_uv(in.cur_clip) - ndc_to_uv(in.prev_clip);
    // .w carries the hurt-flash amount so the GI composite can attenuate the deferred indirect by
    // (1 - flash*1.6), matching the flash mix applied above. 0 for all terrain.
    out.gpos = vec4<f32>(in.world_pos, flash);
    out.galbedo = vec4<f32>(albedo, in.light.x);
    // Linear view depth for DLSS RR: clip.w is the perspective divisor = positive view-space depth,
    // and is unaffected by the x/y projection jitter. 0 in the in-fragment (non-deferred) path is fine.
    out.gdepth = in.cur_clip.w;
    return out;
}
