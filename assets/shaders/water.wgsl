// Water surface: translucent base shading (sun + ambient + fog) plus a ray-traced reflection —
// a mirror ray marched through the voxel volume — blended in by a Schlick-Fresnel term so the
// water mirrors the sky and shoreline at grazing angles and shows its own tint when looked into.
// rtx_common.wgsl (prepended at build time) supplies Camera/Volume, the vertex stage, trace(),
// voxel_color() and sun_visibility().

fn sky_radiance(dir: vec3<f32>, sun: vec3<f32>, sun_intensity: f32) -> vec3<f32> {
    // Base sky tint plus a sharp sun glint when the reflected ray points near the sun.
    let glint = pow(max(dot(dir, sun), 0.0), 200.0) * sun_intensity * 3.0;
    return camera.sky_color.rgb + vec3<f32>(glint);
}

// Colour seen along the mirror ray: lit material on a hit, sky (with sun glint) on a miss.
fn reflection_color(world_pos: vec3<f32>, n: vec3<f32>, incident: vec3<f32>) -> vec3<f32> {
    let sun = normalize(camera.sun_dir.xyz);
    let sun_intensity = camera.params.w;
    let ambient = camera.params.z;
    let refl = reflect(incident, n);
    let origin = world_pos + n * 0.05 + refl * 0.05;
    let h = trace(origin, refl, 80.0, 180);
    if (h.hit) {
        let ndl = max(dot(h.normal, sun), 0.0);
        var vis = 1.0;
        if (ndl > 0.0) {
            vis = sun_visibility(h.pos, h.normal, 64.0);
        }
        return voxel_color(h.id) * (ambient + sun_intensity * ndl * vis);
    }
    return sky_radiance(refl, sun, sun_intensity);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let sun = normalize(camera.sun_dir.xyz);
    let ambient = camera.params.z;
    let sun_intensity = camera.params.w;
    let ndl = max(dot(n, sun), 0.0);

    // Base translucent water shading, sun-shadowed when ray-traced lighting is on.
    var shadow = 1.0;
    if (volume.params.y >= 1u) {
        shadow = sun_visibility(in.world_pos, n, 96.0);
    }
    let base = in.color * (ambient + sun_intensity * ndl * shadow);

    let view_dir = normalize(camera.cam_pos.xyz - in.world_pos); // surface -> camera
    var rgb = base;
    var alpha = 0.7;

    if (volume.params.y >= 1u) {
        // Schlick Fresnel, F0 ~ 0.02 for water: grazing angles reflect, head-on shows water tint.
        let cos_t = max(dot(n, view_dir), 0.0);
        let fres = clamp(0.02 + 0.98 * pow(1.0 - cos_t, 5.0), 0.0, 1.0);
        let refl = reflection_color(in.world_pos, n, -view_dir);
        rgb = mix(base, refl, fres);
        // A reflective sheet reads more opaque, especially at grazing angles.
        alpha = mix(0.7, 0.95, fres);
    } else {
        // Legacy specular sheen when ray-traced lighting is off.
        let halfway = normalize(sun + view_dir);
        let spec = pow(max(dot(n, halfway), 0.0), 64.0) * sun_intensity * 0.6;
        rgb = base + vec3<f32>(spec);
    }

    let dist = length(in.world_pos - camera.cam_pos.xyz);
    let fog = smoothstep(camera.params.x, camera.params.y, dist);
    rgb = mix(rgb, camera.fog_color.rgb, fog);

    return vec4<f32>(rgb, alpha);
}
