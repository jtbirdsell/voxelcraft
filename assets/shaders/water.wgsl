// Water surface: translucent base shading (sun + ambient + fog) over the already-rendered floor,
// plus a ray-traced reflection blended in by a Schlick-Fresnel term. The key correctness rule:
// the water draws with ALPHA_BLENDING over opaque terrain, so `rgb` is the water's OWN color and
// `alpha` is how much of the underwater floor it hides. Clarity must therefore be driven by water
// DEPTH (how much water the column holds), not by view angle. rtx_common.wgsl (prepended at build
// time) supplies Camera/Volume, the vertex stage, trace(), voxel_color(), voxel_emission() and
// sun_visibility().

// Beer-Lambert extinction per block, per channel: red is absorbed fastest so deep water trends
// blue-green. `WATER_TINT` is the body color seen once the floor is fully absorbed.
const WATER_EXTINCTION: vec3<f32> = vec3<f32>(0.42, 0.20, 0.12);
const WATER_TINT:       vec3<f32> = vec3<f32>(0.06, 0.22, 0.30);
const DEPTH_MAX:        f32 = 24.0;   // cap on the downward depth march (deep just saturates)
const REFL_MAX:         f32 = 80.0;   // reflection ray range
const RIPPLE_AMP:       f32 = 0.06;   // tangent-space tilt of the static ripple (small => calm)
const RIPPLE_SCALE:     f32 = 0.55;   // spatial frequency of the dominant wave

// Static sum-of-sines height field on world XZ, returning its analytic xz-gradient
// (dh/dx, dh/dz). No time uniform exists, so this is frozen in world space: it reads as calm
// textured water and, crucially, breaks the perfectly-flat mirror so reflection hit/miss edges
// dither across a few pixels instead of snapping into hard blotches.
fn ripple_grad(p: vec2<f32>) -> vec2<f32> {
    let d0 = vec2<f32>( 0.80,  0.60); let f0 = RIPPLE_SCALE * 1.0; let a0 = 1.00;
    let d1 = vec2<f32>(-0.50,  0.87); let f1 = RIPPLE_SCALE * 1.7; let a1 = 0.55;
    let d2 = vec2<f32>( 0.20, -0.98); let f2 = RIPPLE_SCALE * 2.9; let a2 = 0.28;
    var g = a0 * f0 * cos(dot(d0, p) * f0) * d0;
    g += a1 * f1 * cos(dot(d1, p) * f1) * d1;
    g += a2 * f2 * cos(dot(d2, p) * f2) * d2;
    return g;
}

// Perturb the flat +Y water normal by the ripple gradient. Side faces (base_n.y ~ 0) are left
// alone so vertical water edges keep their true normal.
fn ripple_normal(world_pos: vec3<f32>, base_n: vec3<f32>) -> vec3<f32> {
    let g = ripple_grad(world_pos.xz);
    let perturbed = normalize(vec3<f32>(-g.x * RIPPLE_AMP, 1.0, -g.y * RIPPLE_AMP));
    let upness = clamp(base_n.y, 0.0, 1.0);
    return normalize(mix(base_n, perturbed, upness));
}

fn sky_radiance(dir: vec3<f32>, sun: vec3<f32>, sun_intensity: f32) -> vec3<f32> {
    // Broad, daylight-gated sun lobe (was pow 200 * 3 -> a razor specular that aliased into hard
    // bright specks). A wider exponent + lower gain spreads the glint over many fragments.
    let day = clamp(sun.y, 0.0, 1.0);
    let glint = pow(max(dot(dir, sun), 0.0), 60.0) * sun_intensity * day * 1.0;
    return camera.sky_color.rgb + vec3<f32>(glint);
}

// Colour along the mirror ray: lit terrain on a hit (faded toward sky over the far end of its
// range so distant grazing silhouettes dissolve instead of forming hard hit/miss blotches),
// otherwise sky. Reflected radiance is clamped so one bright lava/snow hit can't spike a pixel.
fn reflection_color(world_pos: vec3<f32>, n: vec3<f32>, incident: vec3<f32>) -> vec3<f32> {
    let sun = normalize(camera.sun_dir.xyz);
    let sun_intensity = camera.params.w;
    let ambient = camera.params.z;
    let refl = reflect(incident, n);
    let origin = world_pos + n * 0.05 + refl * 0.05;
    let sky = sky_radiance(refl, sun, sun_intensity);
    let h = trace(origin, refl, REFL_MAX, 180);
    if (h.hit) {
        let ndl = max(dot(h.normal, sun), 0.0);
        var vis = 1.0;
        if (ndl > 0.0) {
            vis = sun_visibility(h.pos, h.normal, 64.0);
        }
        var lit = voxel_color(h.id) * (ambient + sun_intensity * ndl * vis) + voxel_emission(h.id);
        lit = min(lit, vec3<f32>(2.0)); // tame emissive spikes
        let d = length(h.pos - origin);
        let fade = smoothstep(REFL_MAX * 0.6, REFL_MAX, d);
        return mix(lit, sky, fade);
    }
    return sky;
}

// Water-column depth below `p`, in blocks: march the opaque volume straight DOWN. Water is not
// opaque (id 7 is absent from the volume), so the ray passes through the column and stops at the
// first floor voxel; that hit distance IS the depth. Marching straight down (rather than along the
// view ray) keeps depth a stable per-location property, so it adds no per-pixel direction noise.
fn water_depth(p: vec3<f32>) -> f32 {
    let origin = p + vec3<f32>(0.0, -0.05, 0.0); // nudge just below the surface
    let h = trace(origin, vec3<f32>(0.0, -1.0, 0.0), DEPTH_MAX, i32(DEPTH_MAX * 2.0) + 4);
    if (h.hit) {
        return clamp(length(origin - h.pos), 0.0, DEPTH_MAX);
    }
    return DEPTH_MAX; // no floor within range -> treat as deep
}

// 1 at/beyond the voxel volume's horizontal boundary, 0 well inside it. The downward depth march
// only has floor data inside the 256-block volume, so out-of-volume water would hard-cut to opaque
// along the volume's axis-aligned edge (a sharp right-angle seam). This drives a smooth fade.
fn volume_edge_fade(p: vec3<f32>) -> f32 {
    let size = f32(volume.params.x);
    let lx = p.x - f32(volume.origin.x);
    let lz = p.z - f32(volume.origin.z);
    let edge = min(min(lx, size - lx), min(lz, size - lz)); // blocks to the nearest xz edge
    return 1.0 - smoothstep(0.0, 24.0, edge);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base_n = normalize(in.normal);
    let sun = normalize(camera.sun_dir.xyz);
    let ambient = camera.params.z;
    let sun_intensity = camera.params.w;
    let rtx = volume.params.y >= 1u;

    // Lighting, Fresnel and opacity all use the TRUE flat normal, so clarity stays smooth and
    // consistent. The ripple is applied ONLY to the reflection ray (below) to dither the mirror —
    // feeding it into Fresnel here made opacity patchy (Fresnel is very sensitive near grazing).
    let ndl = max(dot(base_n, sun), 0.0);

    // Shadow ray uses the TRUE flat normal/origin to avoid ripple-induced self-intersection acne.
    var shadow = 1.0;
    if (rtx) {
        shadow = sun_visibility(in.world_pos, base_n, 96.0);
    }
    let lit = ambient + sun_intensity * ndl * shadow;

    let view_dir = normalize(camera.cam_pos.xyz - in.world_pos); // surface -> camera

    // ---- OPACITY FROM DEPTH (Beer-Lambert), NOT FROM VIEW ANGLE -------------------------------
    // The same depth of water reads identically near and far, so the sheet is one coherent
    // surface. Top faces measure the column below them; thin vertical side faces would read ~1
    // block (too clear), so floor their depth so edges still read as water.
    var depth = water_depth(in.world_pos);
    if (base_n.y < 0.5) {
        depth = max(depth, 4.0);
    }
    let transmit = exp(-WATER_EXTINCTION * depth);          // per-channel surviving-floor fraction
    var clarity = (transmit.r + transmit.g + transmit.b) / 3.0;
    // Out-of-volume water has no depth data and would hard-cut to opaque at the volume's edge; fade
    // it to deep/opaque approaching that edge so it reads as a smooth haze, not a right-angle seam.
    clarity = mix(clarity, 0.0, volume_edge_fade(in.world_pos));
    // The water's own color: its tinted albedo over shallow water, trending to WATER_TINT as the
    // floor is absorbed. Both lit the same way so the surface shades consistently.
    let shallow_col = in.color.rgb * lit;
    let deep_col = WATER_TINT * lit;
    var rgb = mix(deep_col, shallow_col, clarity);
    var alpha = clamp(1.0 - clarity, 0.12, 0.95);           // shallow -> floor shows, deep -> opaque

    if (rtx) {
        // Fresnel from the FLAT normal (smooth across the surface) sets how much reflection is
        // mixed into the color. The ripple perturbs ONLY the reflection ray, so the mirror edges
        // dither without making the Fresnel — and therefore the opacity — patchy. Cap keeps far
        // water a strong-but-not-pure mirror.
        let cos_t = max(dot(base_n, view_dir), 0.0);
        let fres = min(clamp(0.02 + 0.98 * pow(1.0 - cos_t, 5.0), 0.0, 1.0), 0.5);
        let refl = reflection_color(in.world_pos, ripple_normal(in.world_pos, base_n), -view_dir);
        rgb = mix(rgb, refl, fres);
        // Opacity stays DEPTH-driven; a reflective grazing surface gets only a small, smooth bump,
        // so view-angle opacity can't reintroduce the patchy clear/opaque inconsistency.
        alpha = clamp(max(alpha, fres * 0.3), 0.12, 0.95);
    } else {
        let halfway = normalize(sun + view_dir);
        let spec = pow(max(dot(base_n, halfway), 0.0), 64.0) * sun_intensity * 0.6;
        rgb = rgb + vec3<f32>(spec);
    }

    let dist = length(in.world_pos - camera.cam_pos.xyz);
    let fog = smoothstep(camera.params.x, camera.params.y, dist);
    rgb = mix(rgb, camera.fog_color.rgb, fog);

    return vec4<f32>(rgb, alpha);
}
